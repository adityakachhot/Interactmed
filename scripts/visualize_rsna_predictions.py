#!/usr/bin/env python3
"""
scripts/visualize_rsna_predictions.py
======================================
InteractMed-VLM — Phase 3B: RSNA Prediction Visualization

Renders per-image overlays showing:
  - Ground-truth bounding boxes (red, solid)
  - Predicted bounding box (blue, dashed)
  - Best-matched GT box highlighted (green, solid)
  - IoU score and patient ID in the title

Usage (standalone batch):
  python scripts/visualize_rsna_predictions.py \
      --config configs/rsna_eval.yaml \
      --results-csv artifacts/rsna/baseline_validation.csv \
      --split val \
      --max-images 50

The save_visualization() function is also called internally by evaluate_rsna.py
when --save-viz is passed.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def _load_dicom_rgb(dcm_path: Path) -> Optional[np.ndarray]:
    """Load a DICOM file and return an 8-bit RGB numpy array, or None on failure."""
    try:
        import pydicom
        ds = pydicom.dcmread(str(dcm_path))
        pixels = ds.pixel_array
        if pixels.dtype != np.uint8:
            pmin, pmax = pixels.min(), pixels.max()
            if pmax > pmin:
                pixels = ((pixels - pmin) / (pmax - pmin) * 255).astype(np.uint8)
            else:
                pixels = np.zeros_like(pixels, dtype=np.uint8)
        if pixels.ndim == 2:
            pixels = np.stack([pixels] * 3, axis=-1)
        elif pixels.shape[2] == 1:
            pixels = np.concatenate([pixels] * 3, axis=-1)
        return pixels
    except Exception as exc:
        warnings.warn(f"Failed to load DICOM {dcm_path.name}: {exc}")
        return None


def save_visualization(
    dcm_path: Path,
    gt_boxes_xywh: List[List[float]],
    pred_box_pixel_xyxy: List[float],
    patient_id: str,
    iou: float,
    query: str,
    out_dir: Path,
    best_gt_idx: int = -1,
) -> Optional[Path]:
    """
    Render and save an overlay visualization for one image.

    Boxes:
      - All GT boxes: red solid rectangles, labeled GT-0, GT-1, ...
      - Best-matched GT box (if best_gt_idx >= 0): green solid rectangle
      - Prediction box: blue dashed rectangle

    Parameters
    ----------
    dcm_path           : Path to DICOM file
    gt_boxes_xywh      : list of [x, y, w, h] in original pixel coords
    pred_box_pixel_xyxy: [x1, y1, x2, y2] in original pixel coords
    patient_id         : RSNA patient UUID
    iou                : best-match IoU value
    query              : grounding query string used
    out_dir            : directory to save PNG
    best_gt_idx        : index of best-matched GT box (-1 = unknown)

    Returns
    -------
    Path to saved PNG, or None if visualization failed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        pixels = _load_dicom_rgb(dcm_path)
        if pixels is None:
            return None

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(pixels, cmap="gray" if pixels.ndim == 2 else None, aspect="equal")

        # Draw all GT boxes
        for idx, (gx, gy, gw, gh) in enumerate(gt_boxes_xywh):
            is_best = (idx == best_gt_idx)
            color = "limegreen" if is_best else "red"
            lw = 2.5 if is_best else 1.5
            label = f"GT-{idx} (best)" if is_best else f"GT-{idx}"
            rect = mpatches.FancyBboxPatch(
                (gx, gy), gw, gh,
                linewidth=lw, edgecolor=color, facecolor="none",
                linestyle="solid",
                boxstyle="square,pad=0",
            )
            ax.add_patch(rect)
            ax.text(gx + 2, gy - 4, label, color=color, fontsize=7,
                    fontweight="bold", clip_on=True,
                    bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"))

        # Draw predicted box
        px1, py1, px2, py2 = pred_box_pixel_xyxy
        pw = px2 - px1
        ph = py2 - py1
        pred_rect = mpatches.FancyBboxPatch(
            (px1, py1), pw, ph,
            linewidth=2.5, edgecolor="dodgerblue", facecolor="none",
            linestyle="dashed",
            boxstyle="square,pad=0",
        )
        ax.add_patch(pred_rect)
        ax.text(px1 + 2, py2 + 12, "Pred", color="dodgerblue", fontsize=7,
                fontweight="bold", clip_on=True,
                bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"))

        # Title
        iou_str = f"{iou:.4f}" if iou is not None else "N/A"
        thresh_str = "[IoU >= 0.50]" if (iou is not None and iou >= 0.5) else "[IoU < 0.50]"
        title = (
            f"{patient_id}\n"
            f'Query: "{query}"  |  Best-match IoU: {iou_str} {thresh_str}\n'
            f"GT boxes: {len(gt_boxes_xywh)}  |  "
            f"Red=GT  Green=Best-GT  Blue(dashed)=Pred"
        )
        ax.set_title(title, fontsize=8, pad=6)
        ax.axis("off")

        out_dir.mkdir(parents=True, exist_ok=True)
        iou_tag = f"iou{int(iou * 100):03d}" if iou is not None else "iouNA"
        out_path = out_dir / f"{patient_id}_{iou_tag}.png"
        fig.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor="black")
        plt.close(fig)
        return out_path

    except Exception as exc:
        warnings.warn(f"Visualization failed for {patient_id}: {exc}")
        return None


def batch_visualize(
    results_csv: Path,
    images_dir: Path,
    out_dir: Path,
    query: str,
    max_images: Optional[int] = None,
    verbose: bool = True,
) -> None:
    """
    Generate visualizations for all successfully inferred images in a results CSV.
    Called standalone (not from evaluate_rsna.py).
    """
    import pandas as pd

    if not results_csv.is_file():
        sys.exit(f"[ERROR] Results CSV not found: {results_csv}")

    df = pd.read_csv(results_csv)
    df = df[df["inference_success"] == True]  # noqa: E712

    if max_images is not None:
        df = df.head(max_images)

    n = len(df)
    print(f"[INFO] Generating {n} visualizations → {out_dir}/")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    for i, row in df.iterrows():
        dcm_path = Path(row["dcm_path"])
        if not dcm_path.is_file():
            if verbose:
                print(f"  SKIP {row['patient_id']} — DICOM not found")
            continue

        try:
            gt_boxes_xywh = json.loads(row["gt_boxes_pixel"])
        except Exception:
            gt_boxes_xywh = []

        try:
            pred_box = json.loads(row["pred_box_pixel"])
        except Exception:
            pred_box = [0, 0, 100, 100]

        iou = float(row["best_match_iou"]) if pd.notna(row["best_match_iou"]) else None
        q = row.get("query", query)

        out_path = save_visualization(
            dcm_path=dcm_path,
            gt_boxes_xywh=gt_boxes_xywh,
            pred_box_pixel_xyxy=pred_box,
            patient_id=row["patient_id"],
            iou=iou if iou is not None else 0.0,
            query=q,
            out_dir=out_dir,
        )

        if out_path:
            n_ok += 1
            if verbose and (i % 50 == 0 or i < 5):
                print(f"  [{n_ok:>4}/{n}] Saved: {out_path.name}")

    print(f"[INFO] Done. {n_ok}/{n} visualizations saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Generate RSNA prediction visualization overlays from a results CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize all val results
  python scripts/visualize_rsna_predictions.py \\
      --config configs/rsna_eval.yaml \\
      --results-csv artifacts/rsna/baseline_validation.csv \\
      --split val

  # Visualize top 50 test results
  python scripts/visualize_rsna_predictions.py \\
      --config configs/rsna_eval.yaml \\
      --results-csv artifacts/rsna/baseline_test.csv \\
      --split test --max-images 50
        """,
    )
    parser.add_argument("--config", default="configs/rsna_eval.yaml")
    parser.add_argument("--results-csv", required=True,
                        help="Path to per-image results CSV from evaluate_rsna.py")
    parser.add_argument("--split", choices=["val", "test", "smoke_test"], default="val")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--out-dir", default=None,
                        help="Override output directory (default: artifacts/rsna/visualizations/<split>)")
    args = parser.parse_args()

    import yaml
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        sys.exit(f"[ERROR] Config not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    images_dir = Path(cfg["data"]["images_dir"])
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(cfg["output"]["visualizations_dir"]) / args.split
    )
    query = cfg["query"]["primary"]

    batch_visualize(
        results_csv=Path(args.results_csv),
        images_dir=images_dir,
        out_dir=out_dir,
        query=query,
        max_images=args.max_images,
        verbose=True,
    )


if __name__ == "__main__":
    main()
