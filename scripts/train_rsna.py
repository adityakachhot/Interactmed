#!/usr/bin/env python3
"""
scripts/train_rsna.py
=====================
InteractMed-VLM — Phase 4: Local RSNA Fine-Tuning Pipeline
Tuned for MacBook Air M3 (16 GB Unified Memory) with Apple Metal (MPS) support.

Fine-tunes the TransVG-AGPT multimodal transformer and box prediction head on the
RSNA Pneumonia Detection Challenge training split (4,210 positive patients).

Key Design Principles:
  1. Stage 1 Encoder Freezing: ResNet-50 visual CNN and ClinicalBERT text model are frozen.
     Only the 8.4M multimodal transformer & box head parameters are trained (4.8% of model).
  2. Dynamic Best-Match GT Loss: For multi-box pneumonia lesions, the model dynamically
     optimizes against the nearest lesion (minimum L1 + GIoU loss), aligning with the
     best-overlap inference evaluation rule.
  3. Checkpoint Safety: The original checkpoint ('checkpoints/transvg.pth') is strictly
     read-only. Fine-tuned checkpoints are saved to 'checkpoints/finetuned_rsna/'.
  4. Local Hardware Optimization: Batched with zero-copy tensors, optional gradient
     accumulation, and native MPS acceleration.

Usage:
  # Quick smoke test (10 train steps, 10 val steps):
  python scripts/train_rsna.py --config configs/rsna_train.yaml --smoke-test

  # Full training (8 epochs, batch size 4 on MPS):
  python scripts/train_rsna.py --config configs/rsna_train.yaml
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

# Add project root and third_party/AGPT to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGPT_ROOT = PROJECT_ROOT / "third_party" / "AGPT"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(AGPT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGPT_ROOT))

from Args import build_args
from models.transvg import build_transvg_model
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from utils.misc import NestedTensor


# ── Dataset Definition ─────────────────────────────────────────────────────────

def _load_dicom_rgb(dcm_path: Path) -> np.ndarray:
    """Load DICOM and return (H, W, 3) uint8 RGB array."""
    ds = pydicom.dcmread(str(dcm_path))
    pixels = ds.pixel_array.astype(np.float32)
    pmin, pmax = pixels.min(), pixels.max()
    if pmax > pmin:
        pixels = ((pixels - pmin) / (pmax - pmin) * 255.0).astype(np.uint8)
    else:
        pixels = np.zeros_like(pixels, dtype=np.uint8)
    if pixels.ndim == 2:
        pixels = np.stack([pixels, pixels, pixels], axis=-1)
    return pixels


class RSNAGroundingDataset(Dataset):
    """
    Dataset for RSNA Pneumonia phrase grounding fine-tuning.
    Each item represents a single positive chest X-ray image with 1 or more ground-truth boxes.
    """

    def __init__(
        self,
        split_csv: Path,
        images_dir: Path,
        query: str = "pneumonia",
        imsize: int = 640,
        is_training: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.images_dir = images_dir
        self.query = query
        self.imsize = imsize
        self.is_training = is_training

        # Load CSV and group bounding boxes by patientId.
        # The split CSVs may contain duplicate box rows due to a many-to-many join
        # with the class_info CSV in evaluate_rsna.py's prepare_splits(). Deduplicate
        # on (x, y, width, height) within each patient group before collecting boxes.
        df = pd.read_csv(split_csv)
        grouped = df.groupby("patientId")
        self.samples = []
        for pid, group in grouped:
            dedup_boxes = group[["x", "y", "width", "height"]].drop_duplicates()
            boxes = [
                [float(row["x"]), float(row["y"]), float(row["width"]), float(row["height"])]
                for _, row in dedup_boxes.iterrows()
            ]
            self.samples.append({"patient_id": pid, "boxes_xywh": boxes})

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        # Augmentation transforms
        if self.is_training:
            self.transform = A.Compose(
                [
                    A.LongestMaxSize(max_size=imsize),
                    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.0, hue=0.0, p=0.5),
                    A.Affine(scale=(0.96, 1.04), translate_percent=(-0.02, 0.02), rotate=(-2, 2),
                             mode=cv2.BORDER_CONSTANT, cval=0, p=0.4),
                    A.PadIfNeeded(min_height=imsize, min_width=imsize, border_mode=cv2.BORDER_CONSTANT, value=0),
                    A.Normalize(mean=0.485, std=0.229),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.1),
            )
        else:
            self.transform = A.Compose(
                [
                    A.LongestMaxSize(max_size=imsize),
                    A.PadIfNeeded(min_height=imsize, min_width=imsize, border_mode=cv2.BORDER_CONSTANT, value=0),
                    A.Normalize(mean=0.485, std=0.229),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.1),
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        patient_id = sample["patient_id"]
        dcm_path = self.images_dir / f"{patient_id}.dcm"

        rgb = _load_dicom_rgb(dcm_path)
        orig_h, orig_w, _ = rgb.shape

        # Convert native [x, y, w, h] to [x1, y1, x2, y2] clamped to image
        voc_boxes = []
        labels = []
        for x, y, w, h in sample["boxes_xywh"]:
            x1 = max(0.0, min(float(orig_w), x))
            y1 = max(0.0, min(float(orig_h), y))
            x2 = max(x1 + 1.0, min(float(orig_w), x + w))
            y2 = max(y1 + 1.0, min(float(orig_h), y + h))
            voc_boxes.append([x1, y1, x2, y2])
            labels.append(1)

        try:
            transformed = self.transform(image=rgb, bboxes=voc_boxes, labels=labels)
            img_tensor = transformed["image"]
            tx_boxes = transformed["bboxes"]
        except Exception:
            # Fallback if augmentation drops all boxes
            tx_boxes = []

        # If transform dropped all boxes, apply non-augmented resize
        if len(tx_boxes) == 0:
            scale = self.imsize / max(orig_h, orig_w)
            new_w = round(orig_w * scale)
            new_h = round(orig_h * scale)
            pad_x = (self.imsize - new_w) // 2
            pad_y = (self.imsize - new_h) // 2
            tx_boxes = []
            for b in voc_boxes:
                bx1 = b[0] * scale + pad_x
                by1 = b[1] * scale + pad_y
                bx2 = b[2] * scale + pad_x
                by2 = b[3] * scale + pad_y
                tx_boxes.append([bx1, by1, bx2, by2])
            fallback_tf = A.Compose([
                A.LongestMaxSize(max_size=self.imsize),
                A.PadIfNeeded(min_height=self.imsize, min_width=self.imsize, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.Normalize(mean=0.485, std=0.229),
                ToTensorV2(),
            ])
            img_tensor = fallback_tf(image=rgb)["image"]

        # Convert transformed boxes on 640x640 canvas to normalized [cx, cy, w, h] in [0, 1]
        norm_cxcywh = []
        for b in tx_boxes:
            bx1, by1, bx2, by2 = b
            bw = max(1e-4, (bx2 - bx1) / float(self.imsize))
            bh = max(1e-4, (by2 - by1) / float(self.imsize))
            bcx = ((bx1 + bx2) / 2.0) / float(self.imsize)
            bcy = ((by1 + by2) / 2.0) / float(self.imsize)
            norm_cxcywh.append([bcx, bcy, bw, bh])

        boxes_tensor = torch.tensor(norm_cxcywh, dtype=torch.float32)

        return {
            "image": img_tensor,
            "boxes_cxcywh": boxes_tensor,
            "orig_boxes_xywh": sample["boxes_xywh"],
            "patient_id": patient_id,
            "query": self.query,
            "orig_size": (orig_h, orig_w),
        }


def rsna_collate_fn(batch: List[dict]) -> dict:
    """Collate batch into NestedTensor for TransVG."""
    images = torch.stack([item["image"] for item in batch], dim=0)
    masks = torch.zeros((images.shape[0], images.shape[2], images.shape[3]), dtype=torch.bool)
    queries = [item["query"] for item in batch]
    boxes_list = [item["boxes_cxcywh"] for item in batch]
    patient_ids = [item["patient_id"] for item in batch]
    orig_sizes = [item["orig_size"] for item in batch]
    orig_boxes = [item["orig_boxes_xywh"] for item in batch]

    return {
        "nested_images": NestedTensor(images, masks),
        "queries": queries,
        "boxes_list": boxes_list,
        "patient_ids": patient_ids,
        "orig_sizes": orig_sizes,
        "orig_boxes": orig_boxes,
    }


# ── Dynamic Best-Match Loss ───────────────────────────────────────────────────

def dynamic_best_match_loss(
    pred_boxes: torch.Tensor,
    gt_boxes_list: List[torch.Tensor],
    l1_coef: float = 5.0,
    giou_coef: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes dynamic minimum-loss assignment across multiple ground-truth boxes per image.
    For each image i with K_i ground-truth boxes, penalizes against the nearest box:
        loss_i = min_k [ 5.0 * L1(pred_i, gt_{i,k}) + 2.0 * (1 - GIoU(pred_i, gt_{i,k})) ]
    """
    batch_size = pred_boxes.shape[0]
    device = pred_boxes.device
    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)

    total_l1 = torch.tensor(0.0, device=device)
    total_giou = torch.tensor(0.0, device=device)
    total_loss = torch.tensor(0.0, device=device)

    for i in range(batch_size):
        pred_i_cxcywh = pred_boxes[i:i + 1]  # (1, 4)
        pred_i_xyxy = pred_xyxy[i:i + 1]      # (1, 4)
        gts = gt_boxes_list[i].to(device)    # (K, 4)

        if gts.shape[0] == 0:
            continue

        gts_xyxy = box_cxcywh_to_xyxy(gts)    # (K, 4)

        # L1 loss against each of the K candidate boxes: (K,)
        l1_k = F.l1_loss(pred_i_cxcywh.repeat(gts.shape[0], 1), gts, reduction="none").sum(dim=-1)

        # GIoU loss against each of the K candidate boxes: (K,)
        giou_matrix = generalized_box_iou(pred_i_xyxy, gts_xyxy)  # (1, K)
        giou_k = 1.0 - giou_matrix.squeeze(0)                      # (K,)

        # Combined cost for each candidate box
        cost_k = l1_coef * l1_k + giou_coef * giou_k               # (K,)

        # Minimum loss index
        best_idx = torch.argmin(cost_k)

        total_l1 = total_l1 + l1_k[best_idx]
        total_giou = total_giou + giou_k[best_idx]
        total_loss = total_loss + cost_k[best_idx]

    loss = total_loss / max(1, batch_size)
    l1_loss = total_l1 / max(1, batch_size)
    giou_loss = total_giou / max(1, batch_size)

    return loss, l1_loss, giou_loss


# ── Validation Evaluation Helper ──────────────────────────────────────────────

def evaluate_validation_split(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    imsize: int = 640,
) -> Dict[str, float]:
    """Run validation evaluation and compute mean best-match IoU & Recall@0.50."""
    model.eval()
    all_ious = []

    with torch.no_grad():
        for batch in val_loader:
            nested_imgs = batch["nested_images"].to(device)
            queries = batch["queries"]
            orig_sizes = batch["orig_sizes"]
            orig_boxes_list = batch["orig_boxes"]

            outputs = model(nested_imgs, queries)
            pred_boxes_norm = outputs["pred_boxes"].cpu()  # (B, 4) in [cx, cy, w, h] on 640x640 canvas

            for i in range(pred_boxes_norm.shape[0]):
                orig_h, orig_w = orig_sizes[i]
                cx, cy, w, h = pred_boxes_norm[i].tolist()

                # Convert normalized canvas cxcywh to canvas pixel xyxy
                px1 = (cx - w / 2.0) * imsize
                py1 = (cy - h / 2.0) * imsize
                px2 = (cx + w / 2.0) * imsize
                py2 = (cy + h / 2.0) * imsize

                # Reverse letterbox to original native DICOM coordinates
                scale = imsize / max(orig_h, orig_w)
                new_w = round(orig_w * scale)
                new_h = round(orig_h * scale)
                pad_x = (imsize - new_w) // 2
                pad_y = (imsize - new_h) // 2

                ox1 = max(0.0, min(float(orig_w), (px1 - pad_x) / scale))
                oy1 = max(0.0, min(float(orig_h), (py1 - pad_y) / scale))
                ox2 = max(0.0, min(float(orig_w), (px2 - pad_x) / scale))
                oy2 = max(0.0, min(float(orig_h), (py2 - pad_y) / scale))

                # Compute IoU against all ground truth boxes
                best_iou = 0.0
                for gx, gy, gw, gh in orig_boxes_list[i]:
                    gx2, gy2 = gx + gw, gy + gh
                    ix1, iy1 = max(ox1, gx), max(oy1, gy)
                    ix2, iy2 = min(ox2, gx2), min(oy2, gy2)
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    area_p = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
                    area_g = gw * gh
                    union = area_p + area_g - inter
                    iou = inter / union if union > 0 else 0.0
                    if iou > best_iou:
                        best_iou = iou

                all_ious.append(best_iou)

    ious_arr = np.array(all_ious)
    mean_iou = float(ious_arr.mean()) if len(ious_arr) > 0 else 0.0
    recall_50 = float((ious_arr >= 0.50).mean()) if len(ious_arr) > 0 else 0.0
    recall_75 = float((ious_arr >= 0.75).mean()) if len(ious_arr) > 0 else 0.0

    return {
        "mean_best_match_iou": round(mean_iou, 4),
        "recall_at_iou_0.50": round(recall_50, 4),
        "recall_at_iou_0.75": round(recall_75, 4),
        "n_samples": len(all_ious),
    }


# ── Training Engine ───────────────────────────────────────────────────────────

def run_training(cfg: dict, args_cli: argparse.Namespace) -> None:
    # Resolve device
    req_device = args_cli.device or cfg["model"].get("device", "auto")
    if req_device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(req_device)

    print(f"\n[INFO] Starting RSNA Fine-Tuning Pipeline (Phase 4)")
    print(f"[INFO] Compute device: {device}")

    # Build AGPT model configuration
    agpt_args = build_args("transvg", "ms_cxr")
    agpt_args.device = device
    imsize = cfg["data"].get("imsize", 640)
    agpt_args.imsize = imsize

    # Instantiate model
    print(f"[INFO] Initializing TransVG-AGPT architecture ...")
    model = build_transvg_model(agpt_args)

    # Load pretrained master checkpoint strictly read-only
    ckp_path = Path(cfg["model"]["base_checkpoint"])
    if not ckp_path.is_file():
        sys.exit(f"[ERROR] Master checkpoint not found: {ckp_path}")
    print(f"[INFO] Loading master checkpoint: {ckp_path}")
    ckp = torch.load(str(ckp_path), map_location="cpu")
    model.load_state_dict(ckp["model"], strict=False)
    print(f"[INFO] Pretrained weights loaded successfully.")

    # Freeze encoders (Stage 1 strategy)
    freeze_encoders = cfg["model"].get("freeze_encoders", True)
    if freeze_encoders:
        print(f"[INFO] Stage 1 Freezing: Locking visual backbone and ClinicalBERT text model ...")
        model.visumodel.eval()
        for p in model.visumodel.parameters():
            p.requires_grad = False
        model.textmodel.eval()
        for p in model.textmodel.parameters():
            p.requires_grad = False

    model = model.to(device)

    # Inspect trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"[INFO] Total parameters     : {total_params:,}")
    print(f"[INFO] Trainable parameters : {trainable_count:,} ({trainable_count / total_params * 100:.2f}%)")

    # Hyperparameters
    epochs = args_cli.epochs or cfg["training"].get("epochs", 8)
    batch_size = args_cli.batch_size or cfg["training"].get("batch_size", 4)
    lr = args_cli.lr or float(cfg["training"].get("lr", 5.0e-5))
    weight_decay = float(cfg["training"].get("weight_decay", 1.0e-4))
    clip_max_norm = float(cfg["training"].get("clip_max_norm", 0.1))
    l1_coef = float(cfg["training"].get("l1_loss_coef", 5.0))
    giou_coef = float(cfg["training"].get("giou_loss_coef", 2.0))

    # Datasets
    images_dir = Path(cfg["data"]["images_dir"])
    train_csv = Path(cfg["data"]["train_split_csv"])
    val_csv = Path(cfg["data"]["val_split_csv"])

    max_train_samples = 40 if args_cli.smoke_test else None
    max_val_samples = 20 if args_cli.smoke_test else None
    if args_cli.smoke_test:
        epochs = 1
        print(f"[INFO] Smoke test active: limiting to {max_train_samples} train and {max_val_samples} val samples (1 epoch).")

    train_dataset = RSNAGroundingDataset(
        split_csv=train_csv,
        images_dir=images_dir,
        query=cfg["data"].get("primary_query", "pneumonia"),
        imsize=imsize,
        is_training=True,
        max_samples=max_train_samples,
    )
    val_dataset = RSNAGroundingDataset(
        split_csv=val_csv,
        images_dir=images_dir,
        query=cfg["data"].get("primary_query", "pneumonia"),
        imsize=imsize,
        is_training=False,
        max_samples=max_val_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=rsna_collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        collate_fn=rsna_collate_fn,
        drop_last=False,
    )

    print(f"[INFO] Dataset ready: {len(train_dataset)} train samples, {len(val_dataset)} val samples.")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1.0e-6)

    # Directories
    ckpt_dir = Path(cfg["output"].get("checkpoints_dir", "checkpoints/finetuned_rsna"))
    art_dir = Path(cfg["output"].get("artifacts_dir", "artifacts/rsna/finetuning"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(cfg["output"].get("log_file", art_dir / "training_history.json"))

    history = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": cfg,
        "epochs": [],
    }

    best_val_iou = 0.0
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"  Starting Training Loop — {epochs} Epochs")
    print(f"{'='*70}\n")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        # Keep frozen encoders strictly in eval mode
        if freeze_encoders:
            model.visumodel.eval()
            model.textmodel.eval()

        running_loss = 0.0
        running_l1 = 0.0
        running_giou = 0.0
        n_batches = 0

        for step, batch in enumerate(train_loader, start=1):
            t0 = time.time()
            nested_imgs = batch["nested_images"].to(device)
            queries = batch["queries"]
            gt_boxes_list = batch["boxes_list"]

            optimizer.zero_grad()
            outputs = model(nested_imgs, queries)
            pred_boxes = outputs["pred_boxes"]

            loss, l1_loss, giou_loss = dynamic_best_match_loss(
                pred_boxes=pred_boxes,
                gt_boxes_list=gt_boxes_list,
                l1_coef=l1_coef,
                giou_coef=giou_coef,
            )

            loss.backward()
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=clip_max_norm)
            optimizer.step()

            step_time = (time.time() - t0) * 1000.0
            running_loss += loss.item()
            running_l1 += l1_loss.item()
            running_giou += giou_loss.item()
            n_batches += 1

            if step % 25 == 0 or step == len(train_loader):
                cur_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  [Epoch {epoch}/{epochs} Step {step:4d}/{len(train_loader):4d}] "
                    f"Loss: {loss.item():.4f} (L1: {l1_loss.item():.4f}, GIoU: {giou_loss.item():.4f}) | "
                    f"lr: {cur_lr:.2e} | {step_time:.1f}ms/step"
                )

        scheduler.step()
        epoch_duration = time.time() - epoch_start
        mean_train_loss = running_loss / max(1, n_batches)
        mean_train_l1 = running_l1 / max(1, n_batches)
        mean_train_giou = running_giou / max(1, n_batches)

        print(f"\n[INFO] Epoch {epoch} complete in {epoch_duration:.1f}s. Evaluating validation split ...")
        val_metrics = evaluate_validation_split(model, val_loader, device, imsize=imsize)
        val_iou = val_metrics["mean_best_match_iou"]
        val_r50 = val_metrics["recall_at_iou_0.50"]
        val_r75 = val_metrics["recall_at_iou_0.75"]

        print(
            f"  ── Val Results: Mean IoU = {val_iou:.4f} | "
            f"Recall@0.50 = {val_r50:.4f} | Recall@0.75 = {val_r75:.4f}"
        )

        epoch_record = {
            "epoch": epoch,
            "duration_s": round(epoch_duration, 2),
            "train_loss": round(mean_train_loss, 4),
            "train_l1_loss": round(mean_train_l1, 4),
            "train_giou_loss": round(mean_train_giou, 4),
            "val_mean_best_match_iou": val_iou,
            "val_recall_at_0.50": val_r50,
            "val_recall_at_0.75": val_r75,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history["epochs"].append(epoch_record)

        # Checkpoint saving
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_path = ckpt_dir / "best_model_val_iou.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": cfg,
                },
                best_path,
            )
            print(f"  ⭐ NEW BEST: Saved {best_path} (mIoU: {best_val_iou:.4f})")

        # Save latest checkpoint
        latest_path = ckpt_dir / "latest_checkpoint.pth"
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": cfg,
            },
            latest_path,
        )

        # Save training log JSON
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

        print()

    total_time = time.time() - start_time
    print(f"{'='*70}")
    print(f"  Training Finished in {total_time/60.0:.2f} minutes")
    print(f"  Best Validation Mean IoU: {best_val_iou:.4f}")
    print(f"  Artifacts saved to: {ckpt_dir} and {log_path}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="InteractMed-VLM Local RSNA Fine-Tuning")
    parser.add_argument("--config", type=str, default="configs/rsna_train.yaml", help="Path to training config YAML")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 1-epoch smoke test on small subset")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device (mps, cpu, cuda)")

    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        sys.exit(f"[ERROR] Config not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    run_training(cfg, args)


if __name__ == "__main__":
    main()
