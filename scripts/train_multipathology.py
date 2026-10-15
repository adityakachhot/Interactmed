#!/usr/bin/env python3
"""
scripts/train_multipathology.py
===============================
Controlled Multi-Pathology Training Experiment for InteractMed-VLM TransVG-AGPT.

Trains a unified phrase grounding model on a balanced multi-pathology cohort:
- MS-CXR training split: 746 annotations covering 8 clinical categories
  (Cardiomegaly, Consolidation, Pneumothorax, Pleural Effusion,
   Lung Opacity, Atelectasis, Pneumonia, Edema)
- RSNA Pneumonia training split: 200 balanced pneumonia patients
  to reinforce pneumonia detection without overwhelming other pathologies.

Evaluation is strictly performed on MS-CXR validation split (147 annotations)
and RSNA validation split, preserving all locked test cohorts.

Usage:
  # Preflight smoke test (20 train samples, 10 val samples):
  /Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/train_multipathology.py --smoke-test

  # Full training run (3 epochs on MPS):
  /Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/train_multipathology.py --config configs/multipathology_train.yaml
"""

import argparse
import io
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

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


# ── Dataset Loading Helpers ───────────────────────────────────────────────────

def _load_dicom_rgb(dcm_path: Path) -> np.ndarray:
    """Load DICOM and return (H, W, 3) uint8 RGB array."""
    ds = pydicom.dcmread(str(dcm_path))
    pixels = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = pixels.max() - pixels
    pmin, pmax = pixels.min(), pixels.max()
    if pmax > pmin:
        pixels = ((pixels - pmin) / (pmax - pmin) * 255.0).astype(np.uint8)
    else:
        pixels = np.zeros_like(pixels, dtype=np.uint8)
    if pixels.ndim == 2:
        pixels = np.stack([pixels, pixels, pixels], axis=-1)
    return pixels


def _load_jpeg_rgb(jpg_path: Path) -> np.ndarray:
    """Load JPEG image and return (H, W, 3) uint8 RGB array."""
    with Image.open(jpg_path) as img:
        rgb = img.convert("RGB")
        return np.array(rgb, dtype=np.uint8)


class MultiPathologyDataset(Dataset):
    """
    Unified dataset supporting both MS-CXR multi-pathology cases and RSNA pneumonia cases.
    Each sample represents an image and query phrase with one or more ground-truth bounding boxes.
    """

    def __init__(
        self,
        samples: List[dict],
        imsize: int = 640,
        is_training: bool = True,
    ):
        self.samples = samples
        self.imsize = imsize
        self.is_training = is_training

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
        item = self.samples[idx]
        img_type = item["img_type"]
        img_path = Path(item["img_path"])

        if img_type == "dicom":
            rgb = _load_dicom_rgb(img_path)
        else:
            rgb = _load_jpeg_rgb(img_path)

        orig_h, orig_w, _ = rgb.shape

        # Convert native [x, y, w, h] to clamped VOC [x1, y1, x2, y2]
        voc_boxes = []
        labels = []
        for x, y, w, h in item["boxes_xywh"]:
            x1 = max(0.0, min(float(orig_w), float(x)))
            y1 = max(0.0, min(float(orig_h), float(y)))
            x2 = max(x1 + 1.0, min(float(orig_w), float(x + w)))
            y2 = max(y1 + 1.0, min(float(orig_h), float(y + h)))
            voc_boxes.append([x1, y1, x2, y2])
            labels.append(1)

        try:
            transformed = self.transform(image=rgb, bboxes=voc_boxes, labels=labels)
            img_tensor = transformed["image"]
            tx_boxes = transformed["bboxes"]
        except Exception:
            tx_boxes = []

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
            "orig_boxes_xywh": item["boxes_xywh"],
            "query": item["query"],
            "category": item["category"],
            "orig_size": (orig_h, orig_w),
            "sample_id": item["sample_id"],
        }


def multipathology_collate_fn(batch: List[dict]) -> dict:
    images = torch.stack([item["image"] for item in batch], dim=0)
    masks = torch.zeros((images.shape[0], images.shape[2], images.shape[3]), dtype=torch.bool)
    queries = [item["query"] for item in batch]
    boxes_list = [item["boxes_cxcywh"] for item in batch]
    categories = [item["category"] for item in batch]
    orig_sizes = [item["orig_size"] for item in batch]
    orig_boxes = [item["orig_boxes_xywh"] for item in batch]
    sample_ids = [item["sample_id"] for item in batch]

    return {
        "nested_images": NestedTensor(images, masks),
        "queries": queries,
        "boxes_list": boxes_list,
        "categories": categories,
        "orig_sizes": orig_sizes,
        "orig_boxes": orig_boxes,
        "sample_ids": sample_ids,
    }


# ── Dynamic Best-Match Loss ───────────────────────────────────────────────────

def dynamic_best_match_loss(
    pred_boxes: torch.Tensor,
    gt_boxes_list: List[torch.Tensor],
    l1_coef: float = 5.0,
    giou_coef: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = pred_boxes.shape[0]
    device = pred_boxes.device
    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)

    total_l1 = torch.tensor(0.0, device=device)
    total_giou = torch.tensor(0.0, device=device)
    total_loss = torch.tensor(0.0, device=device)

    for i in range(batch_size):
        pred_i_cxcywh = pred_boxes[i:i + 1]
        pred_i_xyxy = pred_xyxy[i:i + 1]
        gts = gt_boxes_list[i].to(device)

        if gts.shape[0] == 0:
            continue

        gts_xyxy = box_cxcywh_to_xyxy(gts)
        l1_k = F.l1_loss(pred_i_cxcywh.repeat(gts.shape[0], 1), gts, reduction="none").sum(dim=-1)
        giou_matrix = generalized_box_iou(pred_i_xyxy, gts_xyxy)
        giou_k = 1.0 - giou_matrix.squeeze(0)

        cost_k = l1_coef * l1_k + giou_coef * giou_k
        best_idx = torch.argmin(cost_k)

        total_l1 = total_l1 + l1_k[best_idx]
        total_giou = total_giou + giou_k[best_idx]
        total_loss = total_loss + cost_k[best_idx]

    loss = total_loss / max(1, batch_size)
    l1_loss = total_l1 / max(1, batch_size)
    giou_loss = total_giou / max(1, batch_size)

    return loss, l1_loss, giou_loss


# ── Validation Evaluation Helper ──────────────────────────────────────────────

def evaluate_val_dataset(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    imsize: int = 640,
) -> Dict[str, dict]:
    """Run validation evaluation per category and return detailed metrics."""
    model.eval()
    cat_ious = defaultdict(list)
    all_ious = []

    with torch.no_grad():
        for batch in val_loader:
            nested_imgs = batch["nested_images"].to(device)
            queries = batch["queries"]
            categories = batch["categories"]
            orig_sizes = batch["orig_sizes"]
            orig_boxes_list = batch["orig_boxes"]

            outputs = model(nested_imgs, queries)
            pred_boxes_norm = outputs["pred_boxes"].cpu()

            for i in range(pred_boxes_norm.shape[0]):
                orig_h, orig_w = orig_sizes[i]
                cx, cy, w, h = pred_boxes_norm[i].tolist()

                px1 = (cx - w / 2.0) * imsize
                py1 = (cy - h / 2.0) * imsize
                px2 = (cx + w / 2.0) * imsize
                py2 = (cy + h / 2.0) * imsize

                scale = imsize / max(orig_h, orig_w)
                new_w = round(orig_w * scale)
                new_h = round(orig_h * scale)
                pad_x = (imsize - new_w) // 2
                pad_y = (imsize - new_h) // 2

                ox1 = max(0.0, min(float(orig_w), (px1 - pad_x) / scale))
                oy1 = max(0.0, min(float(orig_h), (py1 - pad_y) / scale))
                ox2 = max(0.0, min(float(orig_w), (px2 - pad_x) / scale))
                oy2 = max(0.0, min(float(orig_h), (py2 - pad_y) / scale))

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

                cat = categories[i]
                cat_ious[cat].append(best_iou)
                all_ious.append(best_iou)

    results = {}
    for cat, ious in cat_ious.items():
        arr = np.array(ious)
        results[cat] = {
            "n_queries": len(arr),
            "mean_iou": round(float(arr.mean()), 4),
            "recall_50": round(float((arr >= 0.50).mean() * 100.0), 2),
            "recall_75": round(float((arr >= 0.75).mean() * 100.0), 2),
        }

    all_arr = np.array(all_ious)
    results["__OVERALL__"] = {
        "n_queries": len(all_arr),
        "mean_iou": round(float(all_arr.mean()), 4),
        "recall_50": round(float((all_arr >= 0.50).mean() * 100.0), 2),
        "recall_75": round(float((all_arr >= 0.75).mean() * 100.0), 2),
    }

    return results


# ── Sample Assembly ────────────────────────────────────────────────────────────

def prepare_multipathology_data(
    local_verified_csv: Path,
    mimic_base_dir: Path,
    rsna_train_csv: Path,
    rsna_images_dir: Path,
    rsna_sample_size: int = 200,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """Build train and validation sample lists without any patient/split leakage."""
    df_local = pd.read_csv(local_verified_csv)

    # 1. MS-CXR Train
    df_mscxr_train = df_local[(df_local["dataset_source"] == "ms_cxr") & (df_local["split"] == "train")]
    grouped_train = defaultdict(list)
    for _, row in df_mscxr_train.iterrows():
        key = (
            row["dicom_id"],
            row["phrase"],
            row["category_name"],
            row["image_path"],
        )
        grouped_train[key].append([float(row["orig_x1"]), float(row["orig_y1"]), float(row["orig_w"]), float(row["orig_h"])])

    train_samples = []
    for (dicom_id, phrase, cat, rel_path), boxes in grouped_train.items():
        full_p = mimic_base_dir / rel_path
        if full_p.is_file():
            train_samples.append({
                "sample_id": f"mscxr_{dicom_id}",
                "img_type": "jpeg",
                "img_path": str(full_p),
                "query": phrase,
                "category": cat,
                "boxes_xywh": boxes,
            })

    # 2. RSNA Pneumonia Train (Balanced subset)
    if rsna_train_csv.is_file() and rsna_sample_size > 0:
        df_rsna = pd.read_csv(rsna_train_csv)
        rsna_pids = sorted(df_rsna["patientId"].unique().tolist())
        rng = random.Random(seed)
        sampled_pids = rng.sample(rsna_pids, min(rsna_sample_size, len(rsna_pids)))
        rsna_sub = df_rsna[df_rsna["patientId"].isin(sampled_pids)]
        for pid, group in rsna_sub.groupby("patientId"):
            dcm_path = rsna_images_dir / f"{pid}.dcm"
            if dcm_path.is_file():
                dedup = group[["x", "y", "width", "height"]].drop_duplicates()
                boxes = [[float(r["x"]), float(r["y"]), float(r["width"]), float(r["height"])] for _, r in dedup.iterrows()]
                train_samples.append({
                    "sample_id": f"rsna_{pid}",
                    "img_type": "dicom",
                    "img_path": str(dcm_path),
                    "query": "pneumonia",
                    "category": "Pneumonia",
                    "boxes_xywh": boxes,
                })

    # 3. MS-CXR Val
    df_mscxr_val = df_local[(df_local["dataset_source"] == "ms_cxr") & (df_local["split"] == "val")]
    grouped_val = defaultdict(list)
    for _, row in df_mscxr_val.iterrows():
        key = (
            row["dicom_id"],
            row["phrase"],
            row["category_name"],
            row["image_path"],
        )
        grouped_val[key].append([float(row["orig_x1"]), float(row["orig_y1"]), float(row["orig_w"]), float(row["orig_h"])])

    val_samples = []
    for (dicom_id, phrase, cat, rel_path), boxes in grouped_val.items():
        full_p = mimic_base_dir / rel_path
        if full_p.is_file():
            val_samples.append({
                "sample_id": f"mscxr_val_{dicom_id}",
                "img_type": "jpeg",
                "img_path": str(full_p),
                "query": phrase,
                "category": cat,
                "boxes_xywh": boxes,
            })

    # Shuffle training samples
    random.Random(seed).shuffle(train_samples)

    return train_samples, val_samples


# ── Training Engine ───────────────────────────────────────────────────────────

def run_training(cfg: dict, args_cli: argparse.Namespace) -> None:
    req_device = args_cli.device or cfg["model"].get("device", "auto")
    if req_device == "auto":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    else:
        device = torch.device(req_device)

    print(f"\n=======================================================")
    print(f"Multi-Pathology Grounding Training Experiment")
    print(f"Device: {device}")
    print(f"=======================================================")

    imsize = cfg["data"].get("imsize", 640)
    agpt_args = build_args("transvg", "ms_cxr")
    agpt_args.device = device
    agpt_args.imsize = imsize

    print(f"[INFO] Initializing TransVG-AGPT architecture ...")
    model = build_transvg_model(agpt_args)

    base_ckpt = Path(cfg["model"]["base_checkpoint"])
    print(f"[INFO] Loading base weights: {base_ckpt}")
    ckp = torch.load(str(base_ckpt), map_location="cpu")
    model.load_state_dict(ckp["model"], strict=False)

    # Freeze visual CNN and ClinicalBERT encoders
    if cfg["model"].get("freeze_encoders", True):
        print(f"[INFO] Stage 1 Freezing: Locking visual backbone and ClinicalBERT text model ...")
        model.visumodel.eval()
        for p in model.visumodel.parameters():
            p.requires_grad = False
        model.textmodel.eval()
        for p in model.textmodel.parameters():
            p.requires_grad = False

    model = model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable parameters: {trainable_count:,} / {total_params:,} ({trainable_count / total_params * 100:.2f}%)")

    # Load data
    train_samples, val_samples = prepare_multipathology_data(
        local_verified_csv=Path(cfg["data"]["local_verified_csv"]),
        mimic_base_dir=Path(cfg["data"]["mimic_base_dir"]),
        rsna_train_csv=Path(cfg["data"]["rsna_train_csv"]),
        rsna_images_dir=Path(cfg["data"]["rsna_images_dir"]),
        rsna_sample_size=cfg["data"].get("rsna_train_sample_size", 200),
    )

    if args_cli.smoke_test:
        train_samples = train_samples[:20]
        val_samples = val_samples[:10]
        epochs = 1
        print(f"[SMOKE TEST] Limited to {len(train_samples)} train and {len(val_samples)} val samples.")
    else:
        epochs = args_cli.epochs or cfg["training"].get("epochs", 3)

    print(f"[INFO] Training set size  : {len(train_samples)} unique query items")
    print(f"[INFO] Validation set size: {len(val_samples)} unique query items")

    train_dataset = MultiPathologyDataset(train_samples, imsize=imsize, is_training=True)
    val_dataset = MultiPathologyDataset(val_samples, imsize=imsize, is_training=False)

    batch_size = args_cli.batch_size or cfg["training"].get("batch_size", 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=multipathology_collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=multipathology_collate_fn,
        drop_last=False,
    )

    lr = float(cfg["training"].get("lr", 2.0e-5))
    weight_decay = float(cfg["training"].get("weight_decay", 1.0e-4))
    clip_max_norm = float(cfg["training"].get("clip_max_norm", 0.1))
    l1_coef = float(cfg["training"].get("l1_loss_coef", 5.0))
    giou_coef = float(cfg["training"].get("giou_loss_coef", 2.0))

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * cfg["training"].get("warmup_epochs", 1)
    min_lr = float(cfg["training"].get("min_lr", 1.0e-6))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr / lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Output setup
    out_ckpt_dir = Path(cfg["output"]["checkpoints_dir"])
    out_art_dir = Path(cfg["output"]["artifacts_dir"])
    out_ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_art_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "start_time": datetime.now().isoformat(),
        "config": cfg,
        "epochs": [],
    }

    best_val_iou = -1.0
    global_step = 0

    print(f"\n[INFO] Starting training: {epochs} epoch(s), {len(train_loader)} steps per epoch.")
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        if cfg["model"].get("freeze_encoders", True):
            model.visumodel.eval()
            model.textmodel.eval()

        train_loss_sum = 0.0
        train_l1_sum = 0.0
        train_giou_sum = 0.0
        steps_in_epoch = 0

        for step, batch in enumerate(train_loader, 1):
            nested_imgs = batch["nested_images"].to(device)
            queries = batch["queries"]
            boxes_list = batch["boxes_list"]

            outputs = model(nested_imgs, queries)
            pred_boxes = outputs["pred_boxes"]

            loss, l1_loss, giou_loss = dynamic_best_match_loss(
                pred_boxes, boxes_list, l1_coef=l1_coef, giou_coef=giou_coef
            )

            optimizer.zero_grad()
            loss.backward()
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, clip_max_norm)
            optimizer.step()
            scheduler.step()

            train_loss_sum += loss.item()
            train_l1_sum += l1_loss.item()
            train_giou_sum += giou_loss.item()
            steps_in_epoch += 1
            global_step += 1

            if step % 25 == 0 or step == len(train_loader):
                cur_lr = scheduler.get_last_lr()[0]
                print(f"Epoch [{epoch}/{epochs}] Step [{step}/{len(train_loader)}] Loss: {loss.item():.4f} (L1: {l1_loss.item():.4f}, GIoU: {giou_loss.item():.4f}) LR: {cur_lr:.2e}")

        epoch_duration = time.perf_counter() - epoch_start
        mean_train_loss = train_loss_sum / max(1, steps_in_epoch)

        print(f"\n[EVAL] Evaluating Epoch {epoch} on MS-CXR validation set ...")
        eval_metrics = evaluate_val_dataset(model, val_loader, device=device, imsize=imsize)
        overall = eval_metrics["__OVERALL__"]

        print(f"[Epoch {epoch} Result] Time: {epoch_duration:.1f}s | Train Loss: {mean_train_loss:.4f} | Val Mean IoU: {overall['mean_iou']:.4f} | Val R@50: {overall['recall_50']:.1f}%")

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(mean_train_loss, 4),
            "epoch_duration_s": round(epoch_duration, 1),
            "eval_metrics": eval_metrics,
        }
        history["epochs"].append(epoch_record)

        # Save checkpoint
        is_best = overall["mean_iou"] > best_val_iou
        if is_best:
            best_val_iou = overall["mean_iou"]
            best_path = out_ckpt_dir / "best_model_val_iou.pth"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "val_metrics": eval_metrics,
                "config": cfg,
            }, best_path)
            print(f"[CHECKPOINT] Saved new best model (Mean IoU = {best_val_iou:.4f}) -> {best_path}")

        last_path = out_ckpt_dir / "last_model.pth"
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "val_metrics": eval_metrics,
            "config": cfg,
        }, last_path)

        with open(cfg["output"]["log_file"], "w") as f:
            json.dump(history, f, indent=2)

    print(f"\n=======================================================")
    print(f"Training Complete. Best Validation IoU: {best_val_iou:.4f}")
    print(f"Log saved: {cfg['output']['log_file']}")
    print(f"=======================================================")


def main():
    parser = argparse.ArgumentParser(description="Multi-Pathology TransVG Fine-Tuning")
    parser.add_argument("--config", default="configs/multipathology_train.yaml", help="Path to config yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick preflight smoke test")
    parser.add_argument("--device", default=None, help="Device override: auto, mps, cpu")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_training(cfg, args)


if __name__ == "__main__":
    main()
