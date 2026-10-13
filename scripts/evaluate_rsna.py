#!/usr/bin/env python3
"""
scripts/evaluate_rsna.py
========================
InteractMed-VLM — RSNA External Grounding Evaluation

Evaluates TransVG-AGPT checkpoints (pretrained zero-shot baseline or fine-tuned
models) on the RSNA Pneumonia Detection Challenge 2018 dataset.

Evaluation scope
----------------
- Pneumonia localization (external grounding) only.
- Positive images (class == "Lung Opacity", Target == 1) are the evaluation set.
- Negative images are processed separately and reported separately; they do NOT
  contribute to recall or IoU metrics.
- A single grounding box is predicted per image per query.
- Multi-GT images (multi-focal pneumonia) use the best-overlap matching rule:
  IoU(pred, gt*) = max_i IoU(pred, gt_i).

Key evaluation rules (frozen before test split is evaluated)
------------------------------------------------------------
- Primary query: "pneumonia"
- Matching: best-overlap (max IoU over all GT boxes for the same image)
- Recall@0.50: fraction of positive images with best-match IoU >= 0.50
- Recall@0.75: fraction of positive images with best-match IoU >= 0.75

Usage
-----
# Step 1: Prepare deterministic splits (run once)
python scripts/evaluate_rsna.py --config configs/rsna_eval.yaml --prepare-splits

# Step 2: Smoke test (5 images, verify pipeline end-to-end)
python scripts/evaluate_rsna.py --config configs/rsna_eval.yaml --smoke-test

# Step 3: Evaluate pretrained baseline on validation split
python scripts/evaluate_rsna.py --config configs/rsna_eval.yaml --split val \
    --output-dir artifacts/rsna/runs/original_baseline \
    --run-name baseline_validation \
    --model-label "Pretrained Baseline"

# Step 4: Evaluate fine-tuned checkpoint on validation split
python scripts/evaluate_rsna.py --config configs/rsna_eval.yaml --split val \
    --checkpoint checkpoints/finetuned_rsna/run_01_3ep/best_model_val_iou.pth \
    --output-dir artifacts/rsna/runs/run_01_3ep \
    --run-name finetuned_validation \
    --model-label "Fine-Tuned Run 01 (3 epochs)"
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class DualLogger:
    """Tee stdout to both terminal and a log file."""
    def __init__(self, filepath: Path):
        self.terminal = sys.stdout
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        try:
            self.log_file.write(message)
            self.log_file.flush()
        except Exception:
            pass

    def flush(self):
        self.terminal.flush()
        try:
            self.log_file.flush()
        except Exception:
            pass

    def close(self):
        if hasattr(self, "log_file") and not self.log_file.closed:
            self.log_file.close()

# ── Guard against loading before checking data presence ───────────────────────

def _check_rsna_data(cfg: dict) -> None:
    labels_csv = Path(cfg["data"]["labels_csv"])
    images_dir = Path(cfg["data"]["images_dir"])
    if not labels_csv.is_file():
        sys.exit(
            f"\n[ERROR] RSNA labels CSV not found: {labels_csv}\n"
            "  → Follow docs/RSNA_DOWNLOAD.md to download and extract the dataset.\n"
            "  → Then run: python scripts/validate_rsna.py --rsna-root data/raw/rsna\n"
        )
    if not images_dir.is_dir():
        sys.exit(
            f"\n[ERROR] RSNA train images directory not found: {images_dir}\n"
            "  → Follow docs/RSNA_DOWNLOAD.md to extract stage_2_train_images/\n"
        )


# ── Coordinate Mapping ────────────────────────────────────────────────────────

def compute_letterbox_params(orig_h: int, orig_w: int, canvas: int = 640) -> Tuple[float, int, int]:
    """
    Compute the scale factor and top-left padding offsets applied by AGPT's
    LongestMaxSize + PadIfNeeded (zero-padding) pipeline.

    Returns:
        scale  : float, scale factor applied (canvas / max(orig_h, orig_w))
        pad_x  : int, horizontal padding added to left side (pixels)
        pad_y  : int, vertical padding added to top side (pixels)
    """
    scale = canvas / max(orig_h, orig_w)
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))
    # PadIfNeeded centers the image (albumentations default)
    pad_y = (canvas - new_h) // 2
    pad_x = (canvas - new_w) // 2
    return scale, pad_x, pad_y


def pred_norm_to_pixel(
    pred_cx_cy_wh_norm: np.ndarray,
    orig_h: int,
    orig_w: int,
    canvas: int = 640,
) -> np.ndarray:
    """
    Map model's normalized predicted box (cx, cy, w, h in [0,1])
    back to [x1, y1, x2, y2] in original pixel space.

    The model predicts on a 640×640 letterboxed canvas.
    We reverse the letterbox to recover original pixel coordinates.

    Parameters
    ----------
    pred_cx_cy_wh_norm : shape (4,) — [cx, cy, w, h] normalized in [0,1]
    orig_h, orig_w     : original DICOM image dimensions
    canvas             : AGPT canvas size (640)

    Returns
    -------
    np.ndarray shape (4,) — [x1, y1, x2, y2] in original pixel space (float)
    """
    cx_n, cy_n, w_n, h_n = pred_cx_cy_wh_norm

    # Step 1: convert to canvas pixel coords (cx, cy, w, h → x1,y1,x2,y2)
    cx_c = cx_n * canvas
    cy_c = cy_n * canvas
    w_c  = w_n  * canvas
    h_c  = h_n  * canvas
    x1_c = cx_c - 0.5 * w_c
    y1_c = cy_c - 0.5 * h_c
    x2_c = cx_c + 0.5 * w_c
    y2_c = cy_c + 0.5 * h_c

    # Step 2: reverse letterbox padding
    scale, pad_x, pad_y = compute_letterbox_params(orig_h, orig_w, canvas)
    x1_p = (x1_c - pad_x) / scale
    y1_p = (y1_c - pad_y) / scale
    x2_p = (x2_c - pad_x) / scale
    y2_p = (y2_c - pad_y) / scale

    # Step 3: clamp to original image bounds
    x1_p = max(0.0, min(orig_w, x1_p))
    y1_p = max(0.0, min(orig_h, y1_p))
    x2_p = max(0.0, min(orig_w, x2_p))
    y2_p = max(0.0, min(orig_h, y2_p))

    return np.array([x1_p, y1_p, x2_p, y2_p], dtype=np.float32)


def pred_norm_to_canvas_xyxy(pred_cx_cy_wh_norm: np.ndarray, canvas: int = 640) -> np.ndarray:
    """Convert normalized (cx,cy,w,h) → canvas-pixel (x1,y1,x2,y2)."""
    cx_n, cy_n, w_n, h_n = pred_cx_cy_wh_norm
    x1 = (cx_n - 0.5 * w_n) * canvas
    y1 = (cy_n - 0.5 * h_n) * canvas
    x2 = (cx_n + 0.5 * w_n) * canvas
    y2 = (cy_n + 0.5 * h_n) * canvas
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ── IoU Calculation ───────────────────────────────────────────────────────────

def compute_iou(pred_xyxy: np.ndarray, gt_xyxy: np.ndarray) -> float:
    """
    Compute IoU between a single predicted box and a single GT box.
    Both in [x1, y1, x2, y2] pixel format.
    """
    inter_x1 = max(pred_xyxy[0], gt_xyxy[0])
    inter_y1 = max(pred_xyxy[1], gt_xyxy[1])
    inter_x2 = min(pred_xyxy[2], gt_xyxy[2])
    inter_y2 = min(pred_xyxy[3], gt_xyxy[3])

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    pred_area = max(0.0, pred_xyxy[2] - pred_xyxy[0]) * max(0.0, pred_xyxy[3] - pred_xyxy[1])
    gt_area   = max(0.0, gt_xyxy[2]   - gt_xyxy[0])   * max(0.0, gt_xyxy[3]   - gt_xyxy[1])
    union_area = pred_area + gt_area - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def best_match_iou(pred_xyxy: np.ndarray, gt_boxes_pixel: List[np.ndarray]) -> Tuple[float, int]:
    """
    Best-overlap matching rule: match the prediction to the GT box with
    the highest IoU. Returns (best_iou, best_gt_index).

    This is the frozen multi-GT matching rule defined in configs/rsna_eval.yaml.
    """
    if not gt_boxes_pixel:
        return 0.0, -1
    ious = [compute_iou(pred_xyxy, gt) for gt in gt_boxes_pixel]
    best_idx = int(np.argmax(ious))
    return ious[best_idx], best_idx


# ── RSNA Data Loading & Splitting ─────────────────────────────────────────────

def load_rsna_annotations(cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load RSNA labels and class info. Returns:
      - positives_df: rows with class=="Lung Opacity" and Target==1
      - negatives_df: all other rows (one row per patient for negatives)
    """
    labels_df = pd.read_csv(cfg["data"]["labels_csv"])
    class_df  = pd.read_csv(cfg["data"]["class_info_csv"])

    # Merge class info into labels
    merged = labels_df.merge(class_df, on="patientId", how="left")

    # Positive images: annotated with bounding boxes for Lung Opacity
    positives = merged[(merged["Target"] == 1) & (merged["class"] == "Lung Opacity")].copy()
    # Negative images: all patients NOT in positives
    pos_patient_ids = set(positives["patientId"].unique())
    negatives = merged[~merged["patientId"].isin(pos_patient_ids)].drop_duplicates("patientId").copy()

    return positives, negatives


def prepare_splits(cfg: dict, verbose: bool = True) -> dict:
    """
    Create deterministic val/test/train splits for positive images.
    Saves split CSVs to data/processed/rsna/.
    Returns a dict mapping split_name → list of patient_ids.
    """
    import random

    splits_dir = Path(cfg["data"]["processed_splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)
    meta_path = splits_dir / "split_metadata.json"

    # If splits already exist, load and return them
    if meta_path.is_file():
        with open(meta_path) as f:
            meta = json.load(f)
        if verbose:
            print(f"[INFO] Loading existing splits from {splits_dir}/")
            for split_name, ids in meta["split_counts"].items():
                print(f"       {split_name}: {ids} images")
        splits = {}
        for split_name in ["train", "val", "test"]:
            csv_path = splits_dir / f"rsna_{split_name}_positive.csv"
            if csv_path.is_file():
                splits[split_name] = pd.read_csv(csv_path)
        return splits

    positives, negatives = load_rsna_annotations(cfg)

    # Group by patient (unique images)
    unique_patients = positives.groupby("patientId").agg(
        n_boxes=("x", "count"),
        boxes=("x", lambda s: list(zip(s, positives.loc[s.index, "y"],
                                       positives.loc[s.index, "width"],
                                       positives.loc[s.index, "height"])))
    ).reset_index()

    n_total = len(unique_patients)
    rng = random.Random(cfg["data"]["split_seed"])
    order = list(range(n_total))
    rng.shuffle(order)

    n_val  = max(1, int(n_total * cfg["data"]["val_fraction"]))
    n_test = max(1, int(n_total * cfg["data"]["test_fraction"]))

    val_idxs   = order[:n_val]
    test_idxs  = order[n_val:n_val + n_test]
    train_idxs = order[n_val + n_test:]

    val_patients   = positives[positives["patientId"].isin(unique_patients.iloc[val_idxs]["patientId"])]
    test_patients  = positives[positives["patientId"].isin(unique_patients.iloc[test_idxs]["patientId"])]
    train_patients = positives[positives["patientId"].isin(unique_patients.iloc[train_idxs]["patientId"])]

    splits = {"train": train_patients, "val": val_patients, "test": test_patients}
    for name, df in splits.items():
        df.to_csv(splits_dir / f"rsna_{name}_positive.csv", index=False)
    negatives.to_csv(splits_dir / "rsna_negatives.csv", index=False)

    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "split_seed": cfg["data"]["split_seed"],
        "val_fraction": cfg["data"]["val_fraction"],
        "test_fraction": cfg["data"]["test_fraction"],
        "total_positive_patients": n_total,
        "split_counts": {
            "train": len(train_patients["patientId"].unique()),
            "val":   len(val_patients["patientId"].unique()),
            "test":  len(test_patients["patientId"].unique()),
        },
        "negative_patients": len(negatives),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"[INFO] Splits created and saved to {splits_dir}/")
        for name, count in meta["split_counts"].items():
            print(f"       {name}: {count} unique positive patients")
        print(f"       negatives: {meta['negative_patients']} patients")

    return splits


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_agpt_model(cfg: dict, ckpt_path: Optional[Path] = None):
    """
    Load the TransVG-AGPT model (pretrained or fine-tuned) using AGPT interfaces.
    """
    import torch

    # Add AGPT repo to sys.path for import resolution
    agpt_root = Path(cfg.get("agpt_root", "third_party/AGPT")).resolve()
    if str(agpt_root) not in sys.path:
        sys.path.insert(0, str(agpt_root))

    from Args import build_args
    from models.transvg import build_transvg_model

    # Select device
    device_cfg = cfg["model"].get("device", "auto")
    if device_cfg == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_cfg)

    print(f"[INFO] Using device: {device}")

    # Build args using the official AGPT factory
    args = build_args("transvg", "ms_cxr")
    args.device = device

    # Build model
    model = build_transvg_model(args)

    # Load checkpoint (strict=False as used in demo.py)
    if ckpt_path is None:
        ckpt_val = (
            cfg.get("model", {}).get("checkpoint")
            or cfg.get("experiment", {}).get("checkpoint")
            or cfg.get("experiment", {}).get("default_checkpoint", "checkpoints/transvg.pth")
        )
        ckpt_path = Path(ckpt_val).resolve()
    else:
        ckpt_path = Path(ckpt_path).resolve()

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device)
    model.eval()

    print(f"[INFO] Model loaded successfully. Parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    return model, args, device, ckpt_path


# ── Single-Image Inference ────────────────────────────────────────────────────

def infer_single_image(
    model,
    args,
    device,
    dcm_path: Path,
    query: str,
    canvas: int = 640,
) -> Tuple[Optional[np.ndarray], float, int, int]:
    """
    Run a single forward pass of TransVG-AGPT on one DICOM image.

    Returns:
        pred_norm : np.ndarray shape (4,) [cx,cy,w,h] in [0,1], or None on failure
        latency_ms: float, wall-clock inference time in milliseconds
        orig_h    : original image height (pixels)
        orig_w    : original image width (pixels)
    """
    import torch
    import pydicom
    from PIL import Image
    from dataloader.mscxr_dataloader import get_transforms
    from utils.misc import NestedTensor

    try:
        # Load DICOM
        t0 = time.perf_counter()
        ds = pydicom.dcmread(str(dcm_path))
        pixel_array = ds.pixel_array

        # Normalize to uint8 [0,255] if needed
        if pixel_array.dtype != np.uint8:
            pmin, pmax = pixel_array.min(), pixel_array.max()
            if pmax > pmin:
                pixel_array = ((pixel_array - pmin) / (pmax - pmin) * 255).astype(np.uint8)
            else:
                pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)

        # Convert grayscale → RGB (3 channels by repeating)
        if pixel_array.ndim == 2:
            pixel_array = np.stack([pixel_array] * 3, axis=-1)
        elif pixel_array.shape[2] == 1:
            pixel_array = np.concatenate([pixel_array] * 3, axis=-1)

        orig_h, orig_w = pixel_array.shape[:2]

        # Apply AGPT val transforms (no bboxes needed for inference)
        transform = get_transforms(args, "val", with_bboxes=False)
        transformed = transform(image=pixel_array)
        image_tensor = transformed["image"].unsqueeze(0).to(device)  # (1, 3, 640, 640)
        image_mask = torch.zeros_like(image_tensor[:, 0, :, :], dtype=torch.bool)
        image_nested = NestedTensor(image_tensor, image_mask)

        # Forward pass
        with torch.no_grad():
            outputs = model(image_nested, [query])

        # outputs["pred_boxes"] shape: (1, 4) — normalized (cx, cy, w, h) in [0,1]
        pred_boxes = outputs["pred_boxes"].squeeze(0).detach().cpu().numpy()  # (4,)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return pred_boxes, latency_ms, orig_h, orig_w

    except Exception as exc:
        warnings.warn(f"Inference failed for {dcm_path.name}: {exc}")
        return None, 0.0, 0, 0


# ── Per-Image Evaluation ──────────────────────────────────────────────────────

def evaluate_image(
    patient_id: str,
    split: str,
    dcm_path: Path,
    gt_boxes_xywh: List[List[float]],
    model,
    args,
    device,
    query: str,
    canvas: int = 640,
) -> dict:
    """
    Evaluate a single positive image. Returns a per-image result dict.
    """
    # Run inference
    pred_norm, latency_ms, orig_h, orig_w = infer_single_image(
        model, args, device, dcm_path, query, canvas
    )

    if pred_norm is None:
        return {
            "patient_id": patient_id,
            "split": split,
            "dcm_path": str(dcm_path),
            "image_width_px": orig_w,
            "image_height_px": orig_h,
            "query": query,
            "n_gt_boxes": len(gt_boxes_xywh),
            "gt_boxes_pixel": json.dumps(gt_boxes_xywh),
            "pred_box_norm": None,
            "pred_box_canvas": None,
            "pred_box_pixel": None,
            "best_match_iou": None,
            "iou_at_0_50": None,
            "iou_at_0_75": None,
            "inference_time_ms": latency_ms,
            "inference_success": False,
            "exclusion_reason": "inference_exception",
        }

    # Convert GT boxes from [x, y, w, h] pixel → [x1, y1, x2, y2] pixel
    gt_xyxy_list = []
    for (gx, gy, gw, gh) in gt_boxes_xywh:
        gt_xyxy_list.append(np.array([gx, gy, gx + gw, gy + gh], dtype=np.float32))

    # Map prediction back to original pixel space
    pred_pixel_xyxy  = pred_norm_to_pixel(pred_norm, orig_h, orig_w, canvas)
    pred_canvas_xyxy = pred_norm_to_canvas_xyxy(pred_norm, canvas)

    # Compute best-match IoU (frozen multi-GT matching rule)
    bm_iou, bm_idx = best_match_iou(pred_pixel_xyxy, gt_xyxy_list)

    return {
        "patient_id": patient_id,
        "split": split,
        "dcm_path": str(dcm_path),
        "image_width_px": orig_w,
        "image_height_px": orig_h,
        "query": query,
        "n_gt_boxes": len(gt_boxes_xywh),
        "gt_boxes_pixel": json.dumps([[round(v, 2) for v in b] for b in gt_boxes_xywh]),
        "pred_box_norm": json.dumps([round(float(v), 6) for v in pred_norm]),
        "pred_box_canvas": json.dumps([round(float(v), 2) for v in pred_canvas_xyxy]),
        "pred_box_pixel": json.dumps([round(float(v), 2) for v in pred_pixel_xyxy]),
        "best_match_iou": round(bm_iou, 6),
        "iou_at_0_50": int(bm_iou >= 0.50),
        "iou_at_0_75": int(bm_iou >= 0.75),
        "inference_time_ms": round(latency_ms, 2),
        "inference_success": True,
        "exclusion_reason": "",
    }


# ── Split-Level Evaluation ────────────────────────────────────────────────────

def run_split_evaluation(
    split_df: pd.DataFrame,
    split: str,
    images_dir: Path,
    model,
    args,
    device,
    query: str,
    canvas: int = 640,
    save_viz: bool = False,
    viz_dir: Optional[Path] = None,
    max_images: Optional[int] = None,
    verbose: bool = True,
) -> List[dict]:
    """
    Evaluate all positive images in a split.
    Returns a list of per-image result dicts.
    """
    # Group rows by patientId to collect all GT boxes per image
    grouped = split_df.groupby("patientId")
    patient_ids = list(grouped.groups.keys())

    if max_images is not None:
        patient_ids = patient_ids[:max_images]

    results = []
    n_total = len(patient_ids)
    n_success = 0
    n_skip = 0

    print(f"\n[INFO] Evaluating {n_total} positive images in split='{split}', query='{query}'")

    for i, patient_id in enumerate(patient_ids):
        group = grouped.get_group(patient_id)

        dcm_path = images_dir / f"{patient_id}.dcm"
        if not dcm_path.is_file():
            result = {
                "patient_id": patient_id,
                "split": split,
                "dcm_path": str(dcm_path),
                "image_width_px": 0,
                "image_height_px": 0,
                "query": query,
                "n_gt_boxes": len(group),
                "gt_boxes_pixel": json.dumps(
                    group[["x", "y", "width", "height"]].values.tolist()
                ),
                "pred_box_norm": None,
                "pred_box_canvas": None,
                "pred_box_pixel": None,
                "best_match_iou": None,
                "iou_at_0_50": None,
                "iou_at_0_75": None,
                "inference_time_ms": 0.0,
                "inference_success": False,
                "exclusion_reason": "dcm_file_not_found",
            }
            results.append(result)
            n_skip += 1
            if verbose and (i % 50 == 0 or i < 5):
                print(f"  [{i+1:>5}/{n_total}] SKIP {patient_id} — DICOM not found")
            continue

        # Collect GT boxes
        gt_boxes_xywh = group[["x", "y", "width", "height"]].values.tolist()

        result = evaluate_image(
            patient_id=patient_id,
            split=split,
            dcm_path=dcm_path,
            gt_boxes_xywh=gt_boxes_xywh,
            model=model,
            args=args,
            device=device,
            query=query,
            canvas=canvas,
        )
        results.append(result)

        if result["inference_success"]:
            n_success += 1
            if verbose and (i % 50 == 0 or i < 5):
                print(
                    f"  [{i+1:>5}/{n_total}] OK  {patient_id}  "
                    f"IoU={result['best_match_iou']:.4f}  "
                    f"lat={result['inference_time_ms']:.0f}ms"
                )
        else:
            n_skip += 1
            if verbose:
                print(f"  [{i+1:>5}/{n_total}] FAIL {patient_id} — {result['exclusion_reason']}")

        # Optional visualization
        if save_viz and viz_dir and result["inference_success"]:
            try:
                try:
                    from scripts.visualize_rsna_predictions import save_visualization
                except ModuleNotFoundError:
                    from visualize_rsna_predictions import save_visualization
                save_visualization(
                    dcm_path=dcm_path,
                    gt_boxes_xywh=gt_boxes_xywh,
                    pred_box_pixel_xyxy=json.loads(result["pred_box_pixel"]),
                    patient_id=patient_id,
                    iou=result["best_match_iou"],
                    query=query,
                    out_dir=viz_dir,
                )
            except Exception as viz_exc:
                warnings.warn(f"Visualization failed for {patient_id}: {viz_exc}")

    print(f"\n[INFO] Done. {n_success}/{n_total} successful, {n_skip} excluded.")
    return results


# ── Aggregate Metrics ─────────────────────────────────────────────────────────

def compute_metrics(results: List[dict], split: str, query: str) -> dict:
    """
    Compute aggregate evaluation metrics from per-image results.

    Only images with inference_success=True and no exclusion_reason contribute
    to IoU and recall metrics.
    """
    all_df = pd.DataFrame(results)
    evaluable = all_df[all_df["inference_success"] & (all_df["exclusion_reason"] == "")]

    n_total = len(all_df)
    n_evaluable = len(evaluable)
    n_excluded = n_total - n_evaluable

    if n_evaluable == 0:
        return {
            "split": split,
            "query": query,
            "n_total_positive_images": n_total,
            "n_evaluable": 0,
            "n_excluded": n_excluded,
            "mean_best_match_iou": None,
            "recall_at_iou_0.50": None,
            "recall_at_iou_0.75": None,
            "inference_success_rate": 0.0,
            "mean_latency_ms": None,
        }

    mean_iou = float(evaluable["best_match_iou"].mean())
    recall_50 = float(evaluable["iou_at_0_50"].mean())
    recall_75 = float(evaluable["iou_at_0_75"].mean())
    success_rate = n_evaluable / n_total if n_total > 0 else 0.0
    mean_lat = float(evaluable["inference_time_ms"].mean())

    return {
        "split": split,
        "query": query,
        "n_total_positive_images": n_total,
        "n_evaluable": n_evaluable,
        "n_excluded": n_excluded,
        "exclusion_reasons": all_df[all_df["exclusion_reason"] != ""]["exclusion_reason"].value_counts().to_dict(),
        "mean_best_match_iou": round(mean_iou, 4),
        "recall_at_iou_0.50": round(recall_50, 4),
        "recall_at_iou_0.75": round(recall_75, 4),
        "inference_success_rate": round(success_rate, 4),
        "mean_latency_ms": round(mean_lat, 2),
        "matching_rule": "best_overlap_max_iou_gt",
        "multi_gt_rule": "max(IoU(pred, gt_i)) over all gt_i for same image",
    }


# ── Main Entrypoint ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RSNA external evaluation of TransVG-AGPT (pretrained baseline & fine-tuned).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  --prepare-splits         Create deterministic val/test/train splits (run once).
  --smoke-test             Run on 5 images to verify pipeline end-to-end.
  --split val              Evaluate primary + secondary query on val split.
  --split test             Evaluate primary query on held-out test split (run once).
  --save-viz               Save per-image prediction visualizations inside run directory.

Examples:
  # Pretrained baseline evaluation:
  python scripts/evaluate_rsna.py --config configs/rsna_eval.yaml --split val \\
      --output-dir artifacts/rsna/runs/original_baseline \\
      --run-name baseline_validation \\
      --model-label "Pretrained Baseline"

  # Fine-tuned checkpoint evaluation:
  python scripts/evaluate_rsna.py --config configs/rsna_eval.yaml --split val \\
      --checkpoint checkpoints/finetuned_rsna/run_01_3ep/best_model_val_iou.pth \\
      --output-dir artifacts/rsna/runs/run_01_3ep \\
      --run-name finetuned_validation \\
      --model-label "Fine-Tuned Run 01 (3 epochs)"
        """,
    )
    parser.add_argument("--config", default="configs/rsna_eval.yaml",
                        help="Path to rsna_eval.yaml (default: configs/rsna_eval.yaml)")
    parser.add_argument("--prepare-splits", action="store_true",
                        help="Create deterministic splits and exit.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run on a small subset (n=5) to verify pipeline.")
    parser.add_argument("--split", choices=["val", "test"], default=None,
                        help="Evaluate val or test split.")
    parser.add_argument("--save-viz", action="store_true",
                        help="Save per-image visualization overlays inside output directory.")
    parser.add_argument("--query", default=None,
                        help="Override query (for secondary-query val runs only).")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of images (for testing).")
    parser.add_argument("--agpt-root", default="third_party/AGPT",
                        help="Path to AGPT repo (default: third_party/AGPT)")
    parser.add_argument("--checkpoint", default=None,
                        help="Override checkpoint path (for evaluating fine-tuned models).")
    parser.add_argument("--output-dir", default=None,
                        help="Custom output directory for artifacts (CSV, JSON, log, viz).")
    parser.add_argument("--run-name", default=None,
                        help="Custom run name (used for filenames and summary records).")
    parser.add_argument("--model-label", default=None,
                        help="Explicit model label for logging/reporting (e.g. 'Pretrained Baseline').")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting existing evaluation files in output directory.")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        sys.exit(f"[ERROR] Config not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["agpt_root"] = args.agpt_root

    # ── Check RSNA data is present ───────────────────────────────────────────
    _check_rsna_data(cfg)

    # ── Prepare splits ───────────────────────────────────────────────────────
    if args.prepare_splits:
        prepare_splits(cfg, verbose=True)
        print("\n[INFO] Splits prepared. Next step:")
        print("  python scripts/evaluate_rsna.py --config configs/rsna_eval.yaml --smoke-test")
        return

    # ── Validate split or mode selection ─────────────────────────────────────
    if not args.smoke_test and args.split is None:
        parser.print_help()
        sys.exit("\n[ERROR] Specify --prepare-splits, --smoke-test, or --split {val|test}.")

    # ── Resolve Checkpoint ───────────────────────────────────────────────────
    ckpt_raw = (
        args.checkpoint
        or cfg.get("model", {}).get("checkpoint")
        or cfg.get("experiment", {}).get("checkpoint")
        or cfg.get("experiment", {}).get("default_checkpoint", "checkpoints/transvg.pth")
    )
    ckpt_path = Path(ckpt_raw).resolve()
    if not ckpt_path.is_file():
        sys.exit(f"\n[ERROR] Checkpoint file does not exist: {ckpt_path}\n")

    is_finetuned = "finetuned" in str(ckpt_path).lower()

    # ── Resolve Model Label ──────────────────────────────────────────────────
    if args.model_label:
        model_label = args.model_label
    elif is_finetuned:
        model_label = f"Fine-Tuned TransVG ({ckpt_path.parent.name})"
    elif "transvg.pth" in str(ckpt_path).lower():
        model_label = "Pretrained Baseline"
    else:
        model_label = f"TransVG ({ckpt_path.stem})"

    # ── Determine query ──────────────────────────────────────────────────────
    query = args.query if args.query else cfg["query"]["primary"]
    if args.query and args.split == "test":
        sys.exit("[ERROR] --query override is not permitted on the test split. "
                 "The test split must use the primary query only.")

    # ── Smoke test ───────────────────────────────────────────────────────────
    if args.smoke_test:
        if args.output_dir:
            smoke_dir = Path(args.output_dir).resolve()
        else:
            smoke_dir = Path(cfg["smoke_test"]["output_dir"]).resolve()

        smoke_dir.mkdir(parents=True, exist_ok=True)
        smoke_run_name = args.run_name or "smoke_test"
        smoke_csv = smoke_dir / f"{smoke_run_name}_results.csv"
        smoke_json = smoke_dir / f"{smoke_run_name}_summary.json"
        smoke_log = smoke_dir / f"{smoke_run_name}.log"

        if (smoke_csv.exists() or smoke_json.exists()) and not args.overwrite:
            sys.exit(
                f"\n[ERROR] Smoke test outputs already exist in {smoke_dir}:\n"
                f"  CSV : {smoke_csv} (exists: {smoke_csv.exists()})\n"
                f"  JSON: {smoke_json} (exists: {smoke_json.exists()})\n"
                f"\n[SAFETY] To prevent accidental overwriting, pass --overwrite.\n"
            )

        dual_logger = DualLogger(smoke_log)
        orig_stdout = sys.stdout
        sys.stdout = dual_logger

        try:
            print("═" * 65)
            print(f"  RSNA Evaluation Smoke Test — {model_label}")
            print("═" * 65)
            print(f"  Checkpoint  : {ckpt_path}")
            print(f"  Query       : '{query}'")
            print(f"  Output Dir  : {smoke_dir}")
            print(f"  Model Label : {model_label}")
            print("═" * 65)

            splits = prepare_splits(cfg, verbose=False)
            val_df = splits.get("val")
            if val_df is None or len(val_df) == 0:
                sys.exit("[ERROR] Val split is empty. Run --prepare-splits first.")

            print("\n[INFO] Loading TransVG-AGPT model ...")
            model, agpt_args, device, _ = load_agpt_model(cfg, ckpt_path=ckpt_path)
            canvas = cfg["model"]["imsize"]
            images_dir = Path(cfg["data"]["images_dir"])

            print(f"\n[INFO] Running smoke test (n={cfg['smoke_test']['n_images']} positive images)...")
            results = run_split_evaluation(
                split_df=val_df,
                split="smoke_test",
                images_dir=images_dir,
                model=model,
                args=agpt_args,
                device=device,
                query=query,
                canvas=canvas,
                save_viz=cfg["smoke_test"]["save_visualizations"],
                viz_dir=smoke_dir,
                max_images=cfg["smoke_test"]["n_images"],
                verbose=True,
            )

            df = pd.DataFrame(results)
            print("\n── Smoke Test Results ──────────────────────────────────────")
            print(df[["patient_id", "inference_success", "best_match_iou",
                       "iou_at_0_50", "inference_time_ms"]].to_string(index=False))
            metrics = compute_metrics(results, "smoke_test", query)
            print(f"\n  mean IoU     : {metrics['mean_best_match_iou']}")
            print(f"  Recall@0.50  : {metrics['recall_at_iou_0.50']}")
            print(f"  Recall@0.75  : {metrics['recall_at_iou_0.75']}")
            print(f"  Success rate : {metrics['inference_success_rate']}")
            print(f"  Mean latency : {metrics['mean_latency_ms']} ms")

            df.to_csv(smoke_csv, index=False)
            smoke_summary = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "model_label": model_label,
                "checkpoint_path": str(ckpt_path),
                "query": query,
                "metrics": metrics,
            }
            with open(smoke_json, "w") as f:
                json.dump(smoke_summary, f, indent=2)
            print(f"\n  Results saved to: {smoke_csv}")
            print(f"  Summary saved to: {smoke_json}")
            print(f"  Log saved to    : {smoke_log}")
            return
        finally:
            sys.stdout = orig_stdout
            dual_logger.close()

    # ── Resolve Run Name & Output Directory ──────────────────────────────────
    if args.run_name:
        run_name = args.run_name
    else:
        prefix = "finetuned" if is_finetuned else "baseline"
        run_name = f"{prefix}_{args.split}"

    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        runs_base = Path(cfg["output"].get("runs_dir", Path(cfg["output"]["artifacts_dir"]) / "runs"))
        out_dir = (runs_base / run_name).resolve()

    out_csv = out_dir / f"{run_name}.csv"
    out_json = out_dir / f"{run_name}.json"
    out_log = out_dir / f"{run_name}.log"
    viz_dir = out_dir / "visualizations"

    # ── Guard Against Silently Overwriting Previous Results ──────────────────
    if (out_csv.exists() or out_json.exists()) and not args.overwrite:
        sys.exit(
            f"\n[ERROR] Evaluation results already exist in destination directory:\n"
            f"  Directory: {out_dir}\n"
            f"  CSV      : {out_csv} (exists: {out_csv.exists()})\n"
            f"  JSON     : {out_json} (exists: {out_json.exists()})\n"
            f"\n[SAFETY] To prevent accidental overwriting of completed evaluations, execution halted.\n"
            f"  → Specify a different --output-dir or --run-name, or\n"
            f"  → Pass --overwrite to explicitly overwrite existing files.\n"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

    dual_logger = DualLogger(out_log)
    orig_stdout = sys.stdout
    sys.stdout = dual_logger

    try:
        print("═" * 65)
        print(f"  RSNA Evaluation ({model_label}) — {args.split.upper()} SPLIT")
        print("═" * 65)
        print(f"  Timestamp        : {datetime.utcnow().isoformat()}Z")
        print(f"  Split            : {args.split}")
        print(f"  Run Name         : {run_name}")
        print(f"  Model Label      : {model_label}")
        print(f"  Checkpoint       : {ckpt_path}")
        print(f"  Output Directory : {out_dir}")
        print(f"  Results CSV      : {out_csv}")
        print(f"  Summary JSON     : {out_json}")
        print(f"  Log File         : {out_log}")
        print(f"  Save Viz         : {args.save_viz} (dir: {viz_dir if args.save_viz else 'None'})")
        print(f"  Primary Query    : {cfg['query']['primary']}")
        print(f"  Query Used       : {query}")
        print("═" * 65)

        splits = prepare_splits(cfg, verbose=False)
        split_df = splits.get(args.split)
        if split_df is None or len(split_df) == 0:
            sys.exit(f"[ERROR] Split '{args.split}' not found or empty. Run --prepare-splits first.")

        # ── Load Model ───────────────────────────────────────────────────────
        print("\n[INFO] Loading TransVG-AGPT model ...")
        model, agpt_args, device, _ = load_agpt_model(cfg, ckpt_path=ckpt_path)
        canvas = cfg["model"]["imsize"]
        images_dir = Path(cfg["data"]["images_dir"])

        # ── Run Evaluation ───────────────────────────────────────────────────
        print(f"\n[INFO] Starting '{args.split}' split evaluation ...")
        t_start = time.perf_counter()

        results = run_split_evaluation(
            split_df=split_df,
            split=args.split,
            images_dir=images_dir,
            model=model,
            args=agpt_args,
            device=device,
            query=query,
            canvas=canvas,
            save_viz=args.save_viz,
            viz_dir=viz_dir if args.save_viz else None,
            max_images=args.max_images,
            verbose=True,
        )

        total_time_s = time.perf_counter() - t_start
        metrics = compute_metrics(results, args.split, query)

        # ── If val split: also run secondary query for reporting ─────────────
        secondary_metrics = None
        if args.split == "val" and not args.query:
            secondary_query = cfg["query"]["secondary"]
            print(f"\n[INFO] Running secondary query (val only): '{secondary_query}'")
            secondary_results = run_split_evaluation(
                split_df=split_df,
                split=args.split + "_secondary",
                images_dir=images_dir,
                model=model,
                args=agpt_args,
                device=device,
                query=secondary_query,
                canvas=canvas,
                save_viz=False,
                max_images=args.max_images,
                verbose=False,
            )
            secondary_metrics = compute_metrics(secondary_results, args.split, secondary_query)

        # ── Save per-image CSV ───────────────────────────────────────────────
        results_df = pd.DataFrame(results)
        results_df.to_csv(out_csv, index=False)
        print(f"\n[INFO] Per-image results saved: {out_csv}")

        # ── Save JSON summary ─────────────────────────────────────────────────
        summary = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_label": model_label,
            "checkpoint_path": str(ckpt_path),
            "checkpoint_name": ckpt_path.name,
            "split": args.split,
            "run_name": run_name,
            "primary_query": cfg["query"]["primary"],
            "query_used": query,
            "total_wall_time_s": round(total_time_s, 2),
            "primary_metrics": metrics,
            "secondary_metrics_val_only": secondary_metrics,
            "matching_rule": cfg["matching"],
            "output_directory": str(out_dir),
            "files": {
                "per_image_csv": str(out_csv),
                "summary_json": str(out_json),
                "log_file": str(out_log),
                "visualizations_dir": str(viz_dir) if args.save_viz else None,
            },
            "experiment": {
                "name": run_name,
                "description": f"RSNA evaluation of {model_label} on {args.split} split",
                "model_label": model_label,
                "checkpoint": str(ckpt_path),
                "task": cfg.get("experiment", {}).get("task", "pneumonia_localization"),
                "phase": "4B" if is_finetuned else "3B",
            },
        }
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[INFO] Summary JSON saved: {out_json}")

        # ── Print summary ─────────────────────────────────────────────────────
        print("\n" + "═" * 65)
        print(f"  RSNA Evaluation ({model_label}) — {args.split.upper()} SPLIT")
        print("═" * 65)
        print(f"  Run Name                : {run_name}")
        print(f"  Model Label             : {model_label}")
        print(f"  Checkpoint              : {ckpt_path}")
        print(f"  Output Directory        : {out_dir}")
        print(f"  Query (primary)         : '{query}'")
        print(f"  Total positive images   : {metrics['n_total_positive_images']}")
        print(f"  Evaluable               : {metrics['n_evaluable']}")
        print(f"  Excluded                : {metrics['n_excluded']}")
        print(f"  Inference success rate  : {metrics['inference_success_rate']:.1%}")
        print(f"  Mean best-match IoU     : {metrics['mean_best_match_iou']}")
        print(f"  Recall @ IoU ≥ 0.50     : {metrics['recall_at_iou_0.50']}")
        print(f"  Recall @ IoU ≥ 0.75     : {metrics['recall_at_iou_0.75']}")
        print(f"  Mean inference latency  : {metrics['mean_latency_ms']} ms")
        if secondary_metrics:
            print(f"\n  Secondary query (val only): '{secondary_query}'")
            print(f"  Mean best-match IoU     : {secondary_metrics['mean_best_match_iou']}")
            print(f"  Recall @ IoU ≥ 0.50     : {secondary_metrics['recall_at_iou_0.50']}")
            print(f"  Recall @ IoU ≥ 0.75     : {secondary_metrics['recall_at_iou_0.75']}")
        print("═" * 65)

        if args.split == "test":
            print("\n  ⚠️  TEST SPLIT EVALUATED. These results are final.")
            print("  Do NOT re-run with different queries or preprocessing settings.")
    finally:
        sys.stdout = orig_stdout
        dual_logger.close()


if __name__ == "__main__":
    main()
