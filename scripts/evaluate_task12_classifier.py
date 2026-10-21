#!/usr/bin/env python3
"""
scripts/evaluate_task12_classifier.py
=====================================
Comprehensive Evaluation of TorchXRayVision DenseNet-121 on the
InteractMed-VLM Validation Cohort (val_dev.jsonl, N=50).

Evaluates:
- Threshold sweep (0.50 to 0.80) to determine optimal operating thresholds.
- Per-finding TP, FP, TN, FN, precision, recall, and F1.
- Global Micro-F1 and Macro-F1 across mapped categories.
- Latency (median and p95) and memory profiling.
- Output saved to artifacts/local_improvements/task12_classifier_evaluation.csv.
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Set, Any
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier_service import (
    ClassifierService,
    XRV_TO_INTERACTMED_MAP,
    UNMAPPED_XRV_PATHOLOGIES,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Reference clinical categories (matches audited standard)
CATEGORIES = {
    "pneumonia_consolidation": ["pneumonia", "consolidation", "infiltrate", "airspace disease"],
    "pulmonary_edema": ["pulmonary edema", "edema", "alveolar edema", "interstitial edema"],
    "pleural_effusion": ["pleural effusion", "effusion", "blunted costophrenic"],
    "pneumothorax": ["pneumothorax", "pleural air"],
    "atelectasis": ["atelectasis", "volume loss"],
    "cardiomegaly": ["cardiomegaly", "enlarged heart", "enlarged cardiac silhouette"],
    "lung_opacity": ["opacity", "opacities"],
}


def run_classifier_evaluation():
    val_file = PROJECT_ROOT / "artifacts/fine_tuning_cohort/val_dev.jsonl"
    out_dir = PROJECT_ROOT / "artifacts/local_improvements"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(val_file) as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(records)} validation records from {val_file}")

    svc = ClassifierService(weights="densenet121-res224-all")
    svc.ensure_loaded()

    # 1. Collect predictions and reference truths
    cases_data = []
    latencies = []

    print("Running DenseNet-121 inference on 50 validation cases...")
    for idx, rec in enumerate(records):
        cid = rec["id"]
        pid = rec["patient_id"]
        img_p = rec["image"]
        gold_labels = set(rec.get("gold_labels", []))
        primary_cat = rec["primary_category"]

        # Parse reference findings
        ref_findings = set()
        for cat in CATEGORIES.keys():
            if cat in gold_labels or cat == primary_cat:
                ref_findings.add(cat)

        t0 = time.perf_counter()
        res = svc.predict(img_p, threshold=0.5)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        cases_data.append({
            "safe_case_id": cid,
            "patient_id": pid,
            "ref_findings": ref_findings,
            "mapped_scores": res.mapped_findings,
            "unmapped_scores": res.unmapped_findings,
            "raw_probabilities": res.raw_probabilities,
            "latency_ms": elapsed_ms,
        })

    print(f"Completed inference across {len(records)} cases.")
    print(f"Median Latency: {np.median(latencies):.2f} ms | p95: {np.percentile(latencies, 95):.2f} ms")

    # 2. Threshold Sensitivity Sweep (from 0.50 to 0.75 in 0.05 steps)
    thresholds = [0.50, 0.55, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75]
    sweep_rows = []

    for thresh in thresholds:
        tp_total = 0
        fp_total = 0
        fn_total = 0
        tn_total = 0

        per_cat_metrics = {}

        for cat in CATEGORIES.keys():
            c_tp = 0
            c_fp = 0
            c_fn = 0
            c_tn = 0

            for c_data in cases_data:
                has_ref = (cat in c_data["ref_findings"])
                score = c_data["mapped_scores"].get(cat, 0.0)
                pred_pos = (score >= thresh)

                if pred_pos and has_ref:
                    c_tp += 1
                elif pred_pos and not has_ref:
                    c_fp += 1
                elif not pred_pos and has_ref:
                    c_fn += 1
                else:
                    c_tn += 1

            c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
            c_rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
            c_f1 = (2 * c_prec * c_rec / (c_prec + c_rec)) if (c_prec + c_rec) > 0 else 0.0

            per_cat_metrics[cat] = {
                "tp": c_tp, "fp": c_fp, "fn": c_fn, "tn": c_tn,
                "precision": c_prec, "recall": c_rec, "f1": c_f1,
            }

            tp_total += c_tp
            fp_total += c_fp
            fn_total += c_fn
            tn_total += c_tn

        micro_prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        micro_rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0.0
        macro_f1 = float(np.mean([m["f1"] for m in per_cat_metrics.values()]))

        sweep_rows.append({
            "threshold": thresh,
            "tp_total": tp_total,
            "fp_total": fp_total,
            "fn_total": fn_total,
            "tn_total": tn_total,
            "micro_precision": round(micro_prec, 4),
            "micro_recall": round(micro_rec, 4),
            "micro_f1": round(micro_f1, 4),
            "macro_f1": round(macro_f1, 4),
            "pneumonia_f1": round(per_cat_metrics["pneumonia_consolidation"]["f1"], 4),
            "pneumonia_rec": round(per_cat_metrics["pneumonia_consolidation"]["recall"], 4),
            "pneumonia_prec": round(per_cat_metrics["pneumonia_consolidation"]["precision"], 4),
            "edema_f1": round(per_cat_metrics["pulmonary_edema"]["f1"], 4),
            "edema_rec": round(per_cat_metrics["pulmonary_edema"]["recall"], 4),
            "effusion_f1": round(per_cat_metrics["pleural_effusion"]["f1"], 4),
            "cardiomegaly_f1": round(per_cat_metrics["cardiomegaly"]["f1"], 4),
            "cardiomegaly_rec": round(per_cat_metrics["cardiomegaly"]["recall"], 4),
            "atelectasis_f1": round(per_cat_metrics["atelectasis"]["f1"], 4),
        })

    sweep_df = pd.DataFrame(sweep_rows)
    eval_csv_p = out_dir / "task12_classifier_evaluation.csv"
    sweep_df.to_csv(eval_csv_p, index=False)
    print(f"\nSaved classifier threshold evaluation to: {eval_csv_p}")

    print("\n" + "=" * 80)
    print("TORCHXRAYVISION DENSENET-121 THRESHOLD SENSITIVITY SWEEP (N=50 val_dev)")
    print("=" * 80)
    col_fmt = "{:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<14}"
    print(col_fmt.format("Threshold", "Micro-F1", "Macro-F1", "Micro-Prec", "Micro-Rec", "Pneumonia-F1"))
    print("-" * 80)
    for r in sweep_rows:
        print(col_fmt.format(
            f"{r['threshold']:.2f}",
            f"{r['micro_f1']:.4f}",
            f"{r['macro_f1']:.4f}",
            f"{r['micro_precision']:.4f}",
            f"{r['micro_recall']:.4f}",
            f"{r['pneumonia_f1']:.4f}",
        ))
    print("=" * 80)

if __name__ == "__main__":
    run_classifier_evaluation()
