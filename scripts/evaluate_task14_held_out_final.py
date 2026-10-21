#!/usr/bin/env python3
"""
scripts/evaluate_task14_held_out_final.py
=========================================
Final Project-Held-Out Evaluation across the 117-case quarantined cohort (expanded_test).
Strictly adheres to frozen configuration (task14_frozen_configuration.yaml).
Outputs:
- task14_held_out_case_predictions.csv
- task14_held_out_metrics.csv
- task14_held_out_per_category_metrics.csv
"""

from pathlib import Path
import json
import time
import sys
from typing import Dict, List, Set, Any, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "artifacts" / "local_improvements" / "task14_final_evaluation_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from src.classifier_service import get_classifier_service

CATEGORIES = [
    "pneumonia_consolidation",
    "pulmonary_edema",
    "pleural_effusion",
    "pneumothorax",
    "atelectasis",
    "cardiomegaly"
]

def compute_wilson_ci(k: int, n: int) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.95996
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return round(max(0.0, float(centre - half)), 4), round(min(1.0, float(centre + half)), 4)

def run_held_out_evaluation():
    manifest_df = pd.read_csv(PROJECT_ROOT / "artifacts/report_evaluation/expanded_cohort/expanded_manifest.csv")
    test_df = manifest_df[manifest_df["split"] == "expanded_test"].reset_index(drop=True)
    with open(PROJECT_ROOT / "artifacts/report_evaluation/expanded_cohort/expanded_reference_store.json") as f:
        ref_store = json.load(f)

    svc = get_classifier_service()
    svc.ensure_loaded()

    print(f"Executing frozen project-held-out evaluation across {len(test_df)} cases...")

    case_rows = []
    t_start = time.perf_counter()

    for idx, row in test_df.iterrows():
        cid = row["safe_case_id"]
        ref_item = ref_store[cid]
        gl = set(ref_item.get("gold_affirmed_labels", []))
        pc = ref_item.get("primary_category", "")
        is_abnormal = (pc != "normal")
        img_p = ref_item["image_path"]

        # Ground truth findings for the 6 target categories
        gold_findings = set()
        for cat in CATEGORIES:
            if cat in gl or cat == pc:
                gold_findings.add(cat)

        t0 = time.perf_counter()
        res = svc.predict(img_p, threshold=0.50)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        scores = res.mapped_findings
        alerts_062 = {k for k, v in scores.items() if k in CATEGORIES and v >= 0.62}
        alerts_064 = {k for k, v in scores.items() if k in CATEGORIES and v >= 0.64}

        tp_062 = len(alerts_062 & gold_findings)
        fp_062 = len(alerts_062 - gold_findings)
        fn_062 = len(gold_findings - alerts_062)

        tp_064 = len(alerts_064 & gold_findings)
        fp_064 = len(alerts_064 - gold_findings)
        fn_064 = len(gold_findings - alerts_064)

        is_fn_case_062 = is_abnormal and (len(alerts_062) == 0)
        is_fn_case_064 = is_abnormal and (len(alerts_064) == 0)

        case_rows.append({
            "safe_case_id": cid,
            "patient_id": row["study_uid_hash"][:12], # anonymized study identifier
            "is_abnormal": is_abnormal,
            "primary_category": pc,
            "reference_findings": "; ".join(sorted(list(gold_findings))) or "normal",
            "alerts_tau_062": "; ".join(sorted(list(alerts_062))) or "none",
            "alerts_tau_064": "; ".join(sorted(list(alerts_064))) or "none",
            "tp_062": tp_062, "fp_062": fp_062, "fn_062": fn_062,
            "tp_064": tp_064, "fp_064": fp_064, "fn_064": fn_064,
            "false_normal_062": is_fn_case_062,
            "false_normal_064": is_fn_case_064,
            "latency_ms": round(elapsed_ms, 2),
            **{f"score_{c}": round(scores.get(c, 0.0), 4) for c in CATEGORIES}
        })

    total_time = time.perf_counter() - t_start
    print(f"Inference completed in {total_time:.2f}s ({total_time/len(test_df)*1000:.1f} ms/case).")

    cases_df = pd.DataFrame(case_rows)
    cases_csv_p = OUT_DIR / "task14_held_out_case_predictions.csv"
    cases_df.to_csv(cases_csv_p, index=False)
    print(f"Saved per-case predictions to: {cases_csv_p}")

    # Compute Overall Metrics across thresholds 0.62 and 0.64
    abnormal_tot = sum(1 for c in case_rows if c["is_abnormal"])
    normal_tot = len(case_rows) - abnormal_tot
    total_refs = sum(len(c["reference_findings"].split("; ")) for c in case_rows if c["reference_findings"] != "normal")

    summary_metrics = []
    for th, pfx in [(0.62, "062"), (0.64, "064")]:
        tp_tot = sum(c[f"tp_{pfx}"] for c in case_rows)
        fp_tot = sum(c[f"fp_{pfx}"] for c in case_rows)
        fn_tot = sum(c[f"fn_{pfx}"] for c in case_rows)
        fn_cases = sum(1 for c in case_rows if c[f"false_normal_{pfx}"])
        fp_cases = sum(1 for c in case_rows if c[f"fp_{pfx}"] > 0)

        prec = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 0
        rec = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        prec_ci = compute_wilson_ci(tp_tot, tp_tot + fp_tot)
        rec_ci = compute_wilson_ci(tp_tot, tp_tot + fn_tot)
        fn_rate_ci = compute_wilson_ci(fn_cases, abnormal_tot)

        summary_metrics.append({
            "threshold": th,
            "policy_role": "High-Sensitivity Review Advisory" if th == 0.62 else "Balanced Diagnostic Alert",
            "total_cases": len(case_rows),
            "abnormal_cases": abnormal_tot,
            "normal_controls": normal_tot,
            "total_reference_findings": total_refs,
            "tp_findings": tp_tot,
            "fp_findings": fp_tot,
            "fn_findings": fn_tot,
            "micro_precision": round(prec, 4),
            "precision_ci_95": f"[{prec_ci[0]}, {prec_ci[1]}]",
            "micro_recall": round(rec, 4),
            "recall_ci_95": f"[{rec_ci[0]}, {rec_ci[1]}]",
            "micro_f1": round(f1, 4),
            "case_false_normal_rate": f"{fn_cases}/{abnormal_tot} ({fn_cases/abnormal_tot*100:.1f}%)",
            "case_false_normal_ci_95": f"[{fn_rate_ci[0]}, {fn_rate_ci[1]}]",
            "finding_omission_rate": f"{fn_tot}/{total_refs} ({fn_tot/total_refs*100:.1f}%)",
            "cases_with_false_alert": f"{fp_cases}/{len(case_rows)} ({fp_cases/len(case_rows)*100:.1f}%)",
            "median_latency_ms": round(float(np.median([c["latency_ms"] for c in case_rows])), 1),
            "p95_latency_ms": round(float(np.percentile([c["latency_ms"] for c in case_rows], 95)), 1),
        })

    summary_df = pd.DataFrame(summary_metrics)
    summary_csv_p = OUT_DIR / "task14_held_out_metrics.csv"
    summary_df.to_csv(summary_csv_p, index=False)
    print(f"Saved summary metrics to: {summary_csv_p}")
    print("\n" + summary_df[["threshold", "policy_role", "micro_precision", "micro_recall", "micro_f1", "case_false_normal_rate", "cases_with_false_alert"]].to_string())

    # Compute Per-Category Breakdown at both thresholds
    per_cat_rows = []
    for cat in CATEGORIES:
        for th, pfx in [(0.62, "062"), (0.64, "064")]:
            c_tp, c_fp, c_fn = 0, 0, 0
            for c in case_rows:
                refs = set(c["reference_findings"].split("; "))
                alerts = set(c[f"alerts_tau_{pfx}"].split("; "))
                has_ref = (cat in refs)
                has_alert = (cat in alerts)
                if has_alert and has_ref:
                    c_tp += 1
                elif has_alert and not has_ref:
                    c_fp += 1
                elif not has_alert and has_ref:
                    c_fn += 1

            cp = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0
            cr = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0
            cf1 = 2 * cp * cr / (cp + cr) if (cp + cr) > 0 else 0

            per_cat_rows.append({
                "finding_category": cat,
                "threshold": th,
                "support_positives": c_tp + c_fn,
                "tp": c_tp, "fp": c_fp, "fn": c_fn,
                "precision": round(cp, 4),
                "recall": round(cr, 4),
                "f1": round(cf1, 4),
            })

    per_cat_df = pd.DataFrame(per_cat_rows)
    per_cat_csv_p = OUT_DIR / "task14_held_out_per_category_metrics.csv"
    per_cat_df.to_csv(per_cat_csv_p, index=False)
    print(f"Saved per-category metrics to: {per_cat_csv_p}")

if __name__ == "__main__":
    run_held_out_evaluation()
