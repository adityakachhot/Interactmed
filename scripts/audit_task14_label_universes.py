#!/usr/bin/env python3
"""
scripts/audit_task14_label_universes.py
=======================================
Generates task14_label_universe_audit.csv, resolving the label-count discrepancies:
1. Universe A (Core Chest Pathology Matrix): 79 Positives / 221 Negatives / 300 Total Instances.
2. Universe B (Full Clinical Report Reference Findings): 114 Reference Positives across 50 cases.
3. Universe C (Structured Dataset Ground Truth / Gold Labels): 76 Gold Findings.
"""

from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "artifacts" / "local_improvements" / "task14_final_evaluation_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_label_universe_audit():
    val_file = PROJECT_ROOT / "artifacts/fine_tuning_cohort/val_dev.jsonl"
    with open(val_file) as f:
        records = [json.loads(line) for line in f if line.strip()]

    base_df = pd.read_csv(PROJECT_ROOT / "artifacts/medgemma_finetuning/task11_cloud_run_01/per_case_predictions.csv")

    # 1. Universe A: Core Chest Pathology Matrix (6 Categories, Closed Binary Matrix)
    # Categories: pneumonia_consolidation, pulmonary_edema, pleural_effusion, pneumothorax, atelectasis, cardiomegaly
    cats_uA = ["pneumonia_consolidation", "pulmonary_edema", "pleural_effusion", "pneumothorax", "atelectasis", "cardiomegaly"]
    pos_uA = 0
    neg_uA = 0
    uA_breakdown = {}
    for c in cats_uA:
        c_pos = 0
        c_neg = 0
        for idx, row in base_df.iterrows():
            refs = set(str(row["ref_findings"]).split("; ")) if str(row["ref_findings"]) != "normal" else set()
            if c in refs:
                c_pos += 1
            else:
                c_neg += 1
        pos_uA += c_pos
        neg_uA += c_neg
        uA_breakdown[c] = {"pos": c_pos, "neg": c_neg}

    # 2. Universe B: Full Clinical Report Reference Findings (7 Categories + Lung Opacity)
    pos_uB = 0
    uB_breakdown = {}
    for idx, row in base_df.iterrows():
        refs = set(str(row["ref_findings"]).split("; ")) if str(row["ref_findings"]) != "normal" else set()
        pos_uB += len(refs)
        for r in refs:
            uB_breakdown[r] = uB_breakdown.get(r, 0) + 1

    # 3. Universe C: Structured Chest ImaGenome Gold Labels
    cats_uC = ["pneumonia_consolidation", "pulmonary_edema", "pleural_effusion", "pneumothorax", "atelectasis", "cardiomegaly"]
    pos_uC = 0
    uC_breakdown = {c: 0 for c in cats_uC}
    for rec in records:
        gl = set(rec.get("gold_labels", []))
        pc = rec.get("primary_category", "")
        for c in cats_uC:
            if c in gl or c == pc:
                pos_uC += 1
                uC_breakdown[c] += 1

    rows = [
        {
            "universe_id": "UNIVERSE_A",
            "universe_name": "Core Chest Pathology Matrix (Disease-Specific Diagnostic Space)",
            "evaluation_target": "Auxiliary Pretrained Classifier Continuous Calibration & Score Thresholding",
            "categories_evaluated": "6 core pathologies: pneumonia_consolidation, pulmonary_edema, pleural_effusion, pneumothorax, atelectasis, cardiomegaly",
            "num_categories": 6,
            "total_sample_cases": 50,
            "total_candidate_instances": 300,
            "positive_instances": pos_uA, # 79
            "negative_instances": neg_uA, # 221
            "support_devices_included": "No (Excluded; classifier only detects native parenchymal/cardiac pathologies)",
            "composite_label_handling": "Pneumonia = max(Pneumonia, Consolidation, Infiltration)",
            "unmapped_findings_handling": "Recorded separately in unmapped_findings dictionary; not scored in 6-class matrix",
            "negative_label_semantics": "Explicit absence in clinical reference (negative control within 50x6 matrix)",
            "category_breakdown": str(uA_breakdown),
            "primary_metric": "Threshold-sweep Alert Micro-F1, Alert Precision, Alert Recall, ROC/PR curves",
        },
        {
            "universe_id": "UNIVERSE_B",
            "universe_name": "Full Clinical Report Reference Findings (Narrative Text Extraction Space)",
            "evaluation_target": "End-to-End VLM Report Generation & Multi-Component Hybrid Pipeline",
            "categories_evaluated": "7 core findings + lung_opacity: pneumonia, edema, effusion, pneumothorax, atelectasis, cardiomegaly, support_devices, lung_opacity",
            "num_categories": 8,
            "total_sample_cases": 50,
            "total_candidate_instances": "Variable (Set-based set-difference per case)",
            "positive_instances": pos_uB, # 114
            "negative_instances": "Implicit (Categories not asserted in case findings list)",
            "support_devices_included": f"Yes ({uB_breakdown.get('support_devices', 0)} instances; captured by specialized device JSON extractor)",
            "composite_label_handling": "NLP regex matching pneumonia, consolidation, infiltrate, bronchogram, airspace disease",
            "unmapped_findings_handling": "Surfaced as incidental advisory warnings in consistency banner",
            "negative_label_semantics": "Absence from NLP-extracted affirmed findings list",
            "category_breakdown": str(uB_breakdown),
            "primary_metric": "Micro-F1, Macro-F1, Finding-Level Omission Rate, Case-Level False-Normal Rate, Unsupported Assertion Rate",
        },
        {
            "universe_id": "UNIVERSE_C",
            "universe_name": "Structured Ground Truth Annotations (Chest ImaGenome Gold Labels)",
            "evaluation_target": "Task 12 Baseline Ground Truth Annotation Alignment",
            "categories_evaluated": "6 core gold finding categories: pneumonia, edema, effusion, pneumothorax, atelectasis, cardiomegaly",
            "num_categories": 6,
            "total_sample_cases": 50,
            "total_candidate_instances": 300,
            "positive_instances": pos_uC, # 76
            "negative_instances": 300 - pos_uC, # 224
            "support_devices_included": "No (Support devices annotated under hardware attribute, not gold pathology)",
            "composite_label_handling": "Direct mapping to Chest ImaGenome normalized finding category",
            "unmapped_findings_handling": "Excluded from gold label list",
            "negative_label_semantics": "Absence from gold_labels attribute in val_dev.jsonl",
            "category_breakdown": str(uC_breakdown),
            "primary_metric": "Task 12 Initial Classifier Evaluation Table (tp=76, fp=157 at tau=0.62)",
        },
    ]

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "task14_label_universe_audit.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(f"Universe A Positives: {pos_uA}, Negatives: {neg_uA}, Total: {pos_uA + neg_uA}")
    print(f"Universe B Positives: {pos_uB}")
    print(f"Universe C Positives: {pos_uC}")

if __name__ == "__main__":
    generate_label_universe_audit()
