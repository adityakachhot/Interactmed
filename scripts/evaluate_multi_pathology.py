#!/usr/bin/env python3
"""
scripts/evaluate_multi_pathology.py
===================================
Evaluates TransVG checkpoints (Pretrained Baseline vs RSNA Fine-Tuned)
across all 8 supported clinical categories in the MS-CXR validation split:
1. Cardiomegaly
2. Pneumothorax
3. Pleural Effusion
4. Pneumonia
5. Consolidation
6. Atelectasis
7. Edema
8. Lung Opacity

Metrics reported per category:
- N images, N annotations
- Mean best-match IoU
- Recall @ IoU >= 0.50
- Recall @ IoU >= 0.75
- Mean inference latency (ms)
- Representative failures
"""

import argparse
import io
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from src.grounding_service import GroundingService, compute_letterbox_params, pred_norm_to_pixel
from scripts.evaluate_rsna import compute_iou, best_match_iou


def evaluate_checkpoint_on_mscxr(
    ckpt_name: str,
    ckpt_path: str,
    df_val: pd.DataFrame,
    images_base_dir: Path,
    device: str = "auto",
) -> Dict[str, dict]:
    """
    Evaluate a TransVG checkpoint on MS-CXR validation split per category.
    """
    print(f"\n=======================================================")
    print(f"Loading {ckpt_name}: {ckpt_path}")
    print(f"=======================================================")

    service = GroundingService(checkpoint=ckpt_path, device=device)
    service.ensure_loaded()

    # Group by (dicom_id, phrase, category_name) to get all GT boxes per query
    grouped = defaultdict(list)
    for _, row in df_val.iterrows():
        key = (row["dicom_id"], row["phrase"], row["category_name"], row["image_path"], row["image_width"], row["image_height"])
        x1 = float(row["orig_x1"])
        y1 = float(row["orig_y1"])
        x2 = x1 + float(row["orig_w"])
        y2 = y1 + float(row["orig_h"])
        grouped[key].append(np.array([x1, y1, x2, y2], dtype=np.float32))

    print(f"Total unique queries in validation set: {len(grouped)} across {len(df_val)} annotations")

    cat_results = defaultdict(list)
    cat_latencies = defaultdict(list)
    cat_failures = defaultdict(list)

    total_queries = len(grouped)
    for idx, (key, gt_boxes) in enumerate(grouped.items(), 1):
        dicom_id, phrase, cat, rel_path, img_w, img_h = key
        full_img_path = images_base_dir / rel_path

        if not full_img_path.is_file():
            print(f"[WARN] Missing image: {full_img_path}")
            continue

        with open(full_img_path, "rb") as f:
            img_bytes = f.read()

        t0 = time.perf_counter()
        res = service.predict(img_bytes, query=phrase, image_filename=full_img_path.name)
        lat = (time.perf_counter() - t0) * 1000.0

        if not res["success"] or res["pred_box_pixel"] is None:
            best_iou = 0.0
            cat_failures[cat].append({
                "dicom_id": dicom_id,
                "phrase": phrase,
                "reason": res.get("error", "Prediction failed"),
            })
        else:
            pred_box = np.array(res["pred_box_pixel"], dtype=np.float32)
            best_iou, best_idx = best_match_iou(pred_box, gt_boxes)
            if best_iou < 0.20:
                cat_failures[cat].append({
                    "dicom_id": dicom_id,
                    "phrase": phrase,
                    "pred_box": [round(float(x), 1) for x in pred_box],
                    "gt_box": [round(float(x), 1) for x in gt_boxes[best_idx]],
                    "iou": round(float(best_iou), 4),
                })

        cat_results[cat].append(best_iou)
        cat_latencies[cat].append(lat)

        if idx % 25 == 0 or idx == total_queries:
            print(f"Processed {idx}/{total_queries} queries...")

    # Aggregate metrics
    summary = {}
    for cat in sorted(cat_results.keys()):
        ious = np.array(cat_results[cat])
        lats = np.array(cat_latencies[cat])
        n_queries = len(ious)
        mean_iou = float(np.mean(ious))
        rec_50 = float(np.mean(ious >= 0.50)) * 100.0
        rec_75 = float(np.mean(ious >= 0.75)) * 100.0
        mean_lat = float(np.mean(lats))

        summary[cat] = {
            "n_queries": n_queries,
            "mean_iou": round(mean_iou, 4),
            "recall_50": round(rec_50, 2),
            "recall_75": round(rec_75, 2),
            "mean_latency_ms": round(mean_lat, 2),
            "n_failures": len(cat_failures[cat]),
            "representative_failures": cat_failures[cat][:3],
        }

    # Overall across all categories
    all_ious = [iou for sub in cat_results.values() for iou in sub]
    all_lats = [lat for sub in cat_latencies.values() for lat in sub]
    summary["__OVERALL__"] = {
        "n_queries": len(all_ious),
        "mean_iou": round(float(np.mean(all_ious)), 4),
        "recall_50": round(float(np.mean(np.array(all_ious) >= 0.50)) * 100.0, 2),
        "recall_75": round(float(np.mean(np.array(all_ious) >= 0.75)) * 100.0, 2),
        "mean_latency_ms": round(float(np.mean(all_lats)), 2),
    }

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Dataset split: val or test")
    parser.add_argument("--device", default="auto", help="Compute device: auto, mps, or cpu")
    parser.add_argument("--output-dir", default="artifacts/multi_pathology_eval", help="Output directory")
    parser.add_argument("--checkpoint", default=None, help="Custom checkpoint path to evaluate")
    parser.add_argument("--checkpoint-name", default=None, help="Display name for custom checkpoint")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_local = pd.read_csv("data/processed/local_grounding_verified.csv")
    df_eval = df_local[(df_local["dataset_source"] == "ms_cxr") & (df_local["split"] == args.split)].copy()
    mimic_base = Path("/Users/adityakachhot/Downloads/data/mimic-cxr")

    print(f"Loaded {len(df_eval)} MS-CXR {args.split} annotations.")

    if args.checkpoint:
        ckpt_name = args.checkpoint_name or Path(args.checkpoint).stem
        checkpoints = [(ckpt_name, args.checkpoint)]
    else:
        checkpoints = [
            ("Pretrained Baseline (MS-CXR)", "checkpoints/transvg.pth"),
            ("Fine-Tuned RSNA (Pneumonia)", "checkpoints/finetuned_rsna/run_01_3ep/best_model_val_iou.pth"),
            ("Multi-Pathology Unified", "checkpoints/multipathology/run_01_multicxr/best_model_val_iou.pth"),
        ]

    all_summaries = {}
    for name, path in checkpoints:
        summary = evaluate_checkpoint_on_mscxr(
            ckpt_name=name,
            ckpt_path=path,
            df_val=df_eval,
            images_base_dir=mimic_base,
            device=args.device,
        )
        all_summaries[name] = summary

    # Save to JSON
    out_filename = "locked_test_results.json" if args.split == "test" else "baseline_vs_finetuned_val.json"
    json_path = out_dir / out_filename
    with open(json_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved evaluation results to {json_path}")

    # Print results table
    if len(all_summaries) == 1:
        name = list(all_summaries.keys())[0]
        summary = all_summaries[name]
        print("\n" + "=" * 75)
        print(f"Results for: {name}")
        print(f"{'Category':<20} | {'N':<4} | {'Mean IoU':<10} | {'Recall@50':<10} | {'Recall@75':<10}")
        print("-" * 75)
        for cat in sorted(summary.keys()):
            if cat == "__OVERALL__":
                continue
            c_data = summary[cat]
            print(f"{cat:<20} | {c_data['n_queries']:<4} | {c_data['mean_iou']:<10.4f} | {c_data['recall_50']:<9.1f}% | {c_data['recall_75']:<9.1f}%")
        print("-" * 75)
        tot = summary["__OVERALL__"]
        print(f"{'OVERALL':<20} | {tot['n_queries']:<4} | {tot['mean_iou']:<10.4f} | {tot['recall_50']:<9.1f}% | {tot['recall_75']:<9.1f}%")
        print("=" * 75)
    elif "Pretrained Baseline (MS-CXR)" in all_summaries and "Multi-Pathology Unified" in all_summaries:
        print("\n" + "=" * 115)
        print(f"{'Category':<18} | {'N':<4} | {'Baseline mIoU':<14} | {'RSNA mIoU':<12} | {'Unified mIoU':<12} | {'Base R@50':<9} | {'RSNA R@50':<9} | {'Unified R@50':<11}")
        print("-" * 115)

        base = all_summaries["Pretrained Baseline (MS-CXR)"]
        fine = all_summaries.get("Fine-Tuned RSNA (Pneumonia)", base)
        unif = all_summaries["Multi-Pathology Unified"]

        for cat in sorted(base.keys()):
            if cat == "__OVERALL__":
                continue
            n = base[cat]["n_queries"]
            b_iou = f"{base[cat]['mean_iou']:.3f}"
            f_iou = f"{fine[cat]['mean_iou']:.3f}"
            u_iou = f"{unif[cat]['mean_iou']:.3f}"
            b_r50 = f"{base[cat]['recall_50']:.1f}%"
            f_r50 = f"{fine[cat]['recall_50']:.1f}%"
            u_r50 = f"{unif[cat]['recall_50']:.1f}%"
            print(f"{cat:<18} | {n:<4} | {b_iou:<14} | {f_iou:<12} | {u_iou:<12} | {b_r50:<9} | {f_r50:<9} | {u_r50:<11}")

        print("-" * 115)
        b_tot = base["__OVERALL__"]
        f_tot = fine["__OVERALL__"]
        u_tot = unif["__OVERALL__"]
        print(f"{'OVERALL':<18} | {b_tot['n_queries']:<4} | {b_tot['mean_iou']:<14.3f} | {f_tot['mean_iou']:<12.3f} | {u_tot['mean_iou']:<12.3f} | {b_tot['recall_50']:<8.1f}% | {f_tot['recall_50']:<8.1f}% | {u_tot['recall_50']:<10.1f}%")
        print("=" * 115)


if __name__ == "__main__":
    main()
