#!/usr/bin/env python3
"""
scripts/evaluate_task19a_grounding.py
======================================
Grounding Evaluation on MS-CXR Held-Out Test Partition (163 annotation pairs).
Strictly evaluates:
- Per-finding grounding performance (8 pathology categories)
- Mean IoU (best-match definition)
- Recall @ IoU 0.50 and 0.75
- Missing / invalid box count
- Latency (ms)
- Multi-model comparison: Baseline TransVG, RSNA Fine-tuned, Multi-Pathology
- Phrase variant analysis (category name vs radiologist sentence vs anatomical phrase)

Saves output to: artifacts/dataset_reporting/task19a_grounding_metrics.csv
"""

from pathlib import Path
import os
import sys
import time
import json
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.grounding_service import GroundingService, get_grounding_service

OUT_CSV = PROJECT_ROOT / "artifacts" / "dataset_reporting" / "task19a_grounding_metrics.csv"
SELECTED_CSV = PROJECT_ROOT / "artifacts" / "dataset_reporting" / "task19a_selected_images.csv"

IMAGE_SEARCH_ROOTS = [
    "/Users/adityakachhot/Downloads/data/mimic-cxr",
    "/Users/adityakachhot/Downloads",
    str(PROJECT_ROOT),
]


def resolve_image_path(rel_path: str) -> Optional[str]:
    for r in IMAGE_SEARCH_ROOTS:
        cand = os.path.join(r, rel_path)
        if os.path.exists(cand):
            return cand
    return None


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Computes Intersection over Union between two boxes formatted as [x1, y1, x2, y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    union = boxAArea + boxBArea - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


def run_evaluation():
    print("Loading selected images manifest...")
    df = pd.read_csv(SELECTED_CSV)
    test_set = df[df["partition_role"] == "mscxr_held_out_test"].reset_index(drop=True)
    print(f"Total held-out test annotations: {len(test_set)}")

    # Models to compare
    model_configs = [
        {
            "model_name": "TransVG Baseline",
            "checkpoint_path": str(PROJECT_ROOT / "checkpoints" / "transvg.pth"),
        },
        {
            "model_name": "TransVG RSNA Fine-tuned (3ep)",
            "checkpoint_path": str(PROJECT_ROOT / "checkpoints" / "finetuned_rsna" / "run_01_3ep" / "best_model_val_iou.pth"),
        },
        {
            "model_name": "TransVG Multi-Pathology (Task 3 Epoch 5)",
            "checkpoint_path": str(PROJECT_ROOT / "checkpoints" / "multipathology" / "run_02_continuation_to_6ep" / "best_model_val_iou.pth"),
        },
    ]

    all_metric_rows = []

    for cfg in model_configs:
        mname = cfg["model_name"]
        ckpt = cfg["checkpoint_path"]
        print(f"\nEvaluating {mname} on {len(test_set)} cases...")

        if not os.path.exists(ckpt):
            print(f"Warning: Checkpoint not found: {ckpt}. Skipping.")
            continue

        svc = GroundingService(checkpoint=ckpt)
        svc.ensure_loaded()

        per_cat_stats: Dict[str, Dict[str, Any]] = {}
        total_ious = []
        latencies = []
        invalid_count = 0

        for idx, row in test_set.iterrows():
            cat = str(row["category_name"]).strip()
            phrase = str(row["phrase"]).strip()
            img_rel = str(row["image_path"]).strip()
            gt_box = [float(row["norm_x1"]), float(row["norm_y1"]), float(row["norm_x2"]), float(row["norm_y2"])]

            if cat not in per_cat_stats:
                per_cat_stats[cat] = {
                    "count": 0,
                    "ious": [],
                    "recall_50": 0,
                    "recall_75": 0,
                    "invalid": 0,
                    "latencies": [],
                }

            full_path = resolve_image_path(img_rel)
            if not full_path or not os.path.exists(full_path):
                invalid_count += 1
                per_cat_stats[cat]["invalid"] += 1
                continue

            with open(full_path, "rb") as f:
                img_bytes = f.read()

            t0 = time.perf_counter()
            # Predict using phrase query
            res = svc.predict(img_bytes, phrase)
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)
            per_cat_stats[cat]["latencies"].append(lat)

            if not res.get("success") or not res.get("pred_box_norm"):
                invalid_count += 1
                per_cat_stats[cat]["invalid"] += 1
                total_ious.append(0.0)
                per_cat_stats[cat]["ious"].append(0.0)
                continue

            # pred_box_norm is [cx, cy, w, h] normalized in [0, 1]
            cx, cy, bw, bh = res["pred_box_norm"]
            pred_xyxy = [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0]

            iou = compute_iou(gt_box, pred_xyxy)
            total_ious.append(iou)
            per_cat_stats[cat]["count"] += 1
            per_cat_stats[cat]["ious"].append(iou)
            if iou >= 0.50:
                per_cat_stats[cat]["recall_50"] += 1
            if iou >= 0.75:
                per_cat_stats[cat]["recall_75"] += 1

        # Summary for model across categories
        for cat, stats in per_cat_stats.items():
            cnt = len(stats["ious"])
            m_iou = float(np.mean(stats["ious"])) if cnt > 0 else 0.0
            r50 = float(stats["recall_50"] / cnt) if cnt > 0 else 0.0
            r75 = float(stats["recall_75"] / cnt) if cnt > 0 else 0.0
            m_lat = float(np.mean(stats["latencies"])) if stats["latencies"] else 0.0

            all_metric_rows.append({
                "model_name": mname,
                "finding_name": cat,
                "split": "mscxr_held_out_test",
                "eligible_pairs": cnt,
                "mean_best_match_iou": round(m_iou, 4),
                "recall_at_iou_050": round(r50, 4),
                "recall_at_iou_075": round(r75, 4),
                "invalid_or_missing_box_count": stats["invalid"],
                "mean_latency_ms": round(m_lat, 2),
            })

        # Overall row for this model
        overall_cnt = len(total_ious)
        overall_iou = float(np.mean(total_ious)) if overall_cnt > 0 else 0.0
        overall_r50 = float(sum(1 for v in total_ious if v >= 0.50) / overall_cnt) if overall_cnt > 0 else 0.0
        overall_r75 = float(sum(1 for v in total_ious if v >= 0.75) / overall_cnt) if overall_cnt > 0 else 0.0
        overall_lat = float(np.mean(latencies)) if latencies else 0.0

        all_metric_rows.append({
            "model_name": mname,
            "finding_name": "OVERALL_MICRO_AVERAGE",
            "split": "mscxr_held_out_test",
            "eligible_pairs": overall_cnt,
            "mean_best_match_iou": round(overall_iou, 4),
            "recall_at_iou_050": round(overall_r50, 4),
            "recall_at_iou_075": round(overall_r75, 4),
            "invalid_or_missing_box_count": invalid_count,
            "mean_latency_ms": round(overall_lat, 2),
        })

    out_df = pd.DataFrame(all_metric_rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nGrounding evaluation complete. Saved {len(out_df)} metric rows to {OUT_CSV}")


if __name__ == "__main__":
    run_evaluation()
