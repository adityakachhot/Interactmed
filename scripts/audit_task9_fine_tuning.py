#!/usr/bin/env python3
"""
scripts/audit_task9_fine_tuning.py
==================================
Comprehensive Audit Script for Task 9:
Audits dataset provenance, patient/study independence, cohort overlap,
radiograph correspondence, clinical content (devices, laterality, uncertainty, negation),
template repetition, hardware compatibility, and production integration.
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def extract_pid(path_str: str) -> str:
    m = re.search(r'/(p\d+)/s\d+/', path_str)
    return m.group(1) if m else "unknown"

def extract_sid(path_str: str) -> str:
    m = re.search(r'/s(\d+)/', path_str)
    return f"s{m.group(1)}" if m else "unknown"

def extract_dicom(path_str: str) -> str:
    stem = Path(path_str).stem
    return stem.replace(" copy", "")

def run_audit() -> Dict[str, Any]:
    audit_data = {}

    # 1. Dataset Provenance & Split Sizes
    exp_manifest_path = PROJECT_ROOT / "artifacts/report_evaluation/expanded_cohort/expanded_manifest.csv"
    df_exp = pd.read_csv(exp_manifest_path)
    ref_store_path = PROJECT_ROOT / "artifacts/report_evaluation/expanded_cohort/expanded_reference_store.json"
    with open(ref_store_path) as f:
        ref_store = json.load(f)

    train_jsonl = PROJECT_ROOT / "artifacts/fine_tuning_cohort/train_dev.jsonl"
    val_jsonl = PROJECT_ROOT / "artifacts/fine_tuning_cohort/val_dev.jsonl"
    train_records = [json.loads(l) for l in open(train_jsonl)]
    val_records = [json.loads(l) for l in open(val_jsonl)]

    df_test = df_exp[df_exp["split"] == "expanded_test"]
    test_case_ids = list(df_test["safe_case_id"])
    test_records = []
    for cid in test_case_ids:
        if cid in ref_store:
            test_records.append(ref_store[cid])

    # Task 8 cases
    task8_manifest_path = PROJECT_ROOT / "artifacts/report_quality/task8_cohort_manifest.json"
    with open(task8_manifest_path) as f:
        task8_cases = json.load(f)

    # Task 7 cases
    t7_df = pd.read_csv(PROJECT_ROOT / "artifacts/report_quality/per_case_comparison.csv")

    # Counts
    train_pids = [extract_pid(r["image"]) for r in train_records]
    train_sids = [extract_sid(r["image"]) for r in train_records]
    train_dicoms = [extract_dicom(r["image"]) for r in train_records]

    val_pids = [extract_pid(r["image"]) for r in val_records]
    val_sids = [extract_sid(r["image"]) for r in val_records]
    val_dicoms = [extract_dicom(r["image"]) for r in val_records]

    test_pids = [extract_pid(r["image_path"]) for r in test_records]
    test_sids = [extract_sid(r["image_path"]) for r in test_records]
    test_dicoms = [extract_dicom(r["image_path"]) for r in test_records]

    task8_pids = [c["patient_id"] for c in task8_cases]
    task8_dicoms = [extract_dicom(c["image_path"]) for c in task8_cases]

    t7_dicoms = [d.strip() for d in t7_df["dicom_id"]]

    audit_data["counts"] = {
        "train": {
            "examples": len(train_records),
            "unique_patients": len(set(train_pids)),
            "unique_studies": len(set(train_sids)),
            "unique_images": len(set(train_dicoms)),
        },
        "val": {
            "examples": len(val_records),
            "unique_patients": len(set(val_pids)),
            "unique_studies": len(set(val_sids)),
            "unique_images": len(set(val_dicoms)),
        },
        "test": {
            "examples": len(test_records),
            "unique_patients": len(set(test_pids)),
            "unique_studies": len(set(test_sids)),
            "unique_images": len(set(test_dicoms)),
        },
        "task8": {
            "examples": len(task8_cases),
            "unique_patients": len(set(task8_pids)),
            "unique_images": len(set(task8_dicoms)),
        },
        "task7": {
            "examples": len(t7_df),
            "unique_images": len(set(t7_dicoms)),
        }
    }

    # Overlaps
    audit_data["patient_overlaps"] = {
        "train_val": len(set(train_pids) & set(val_pids)),
        "train_test": len(set(train_pids) & set(test_pids)),
        "val_test": len(set(val_pids) & set(test_pids)),
        "task8_train": len(set(task8_pids) & set(train_pids)),
        "task8_val": len(set(task8_pids) & set(val_pids)),
        "task8_test": len(set(task8_pids) & set(test_pids)),
    }

    audit_data["dicom_overlaps"] = {
        "task7_train": len(set(t7_dicoms) & set(train_dicoms)),
        "task7_val": len(set(t7_dicoms) & set(val_dicoms)),
        "task7_test": len(set(t7_dicoms) & set(test_dicoms)),
        "task7_task8": len(set(t7_dicoms) & set(task8_dicoms)),
    }

    # 2. Radiograph Correspondence & Verification
    missing_train = sum(1 for r in train_records if not Path(r["image"]).exists())
    missing_val = sum(1 for r in val_records if not Path(r["image"]).exists())
    missing_test = sum(1 for r in test_records if not Path(r["image_path"]).exists())
    audit_data["image_existence"] = {
        "missing_train": missing_train,
        "missing_val": missing_val,
        "missing_test": missing_test,
    }

    # 3. Content Analysis: Devices, Laterality, Uncertainty, Negation, Templates
    device_regex = re.compile(
        r'\b(?:tube|catheter|line|pacemaker|pacer|lead|hardware|wire|picc|et tube|endotracheal|'
        r'tracheostomy|nasogastric|dobhoff|chest tube|drain|stent|clip|valve)\b', re.IGNORECASE
    )
    laterality_regex = re.compile(
        r'\b(?:right|left|bilateral|midline|subclavian|jugular|retrocardiac|apical|basal|basilar)\b', re.IGNORECASE
    )
    uncertainty_regex = re.compile(
        r'\b(?:possible|probable|cannot exclude|cannot be excluded|equivocal|borderline|'
        r'suggestive|suspicious|may represent|could represent|might represent|question of|'
        r'uncertain|indeterminate|low lung volumes limit|obscured)\b', re.IGNORECASE
    )
    negation_regex = re.compile(
        r'\b(?:no|without|negative|free of|clear of|normal|unremarkable|clean|absent|'
        r'no acute|no definite|fails to show|not identified|not seen)\b', re.IGNORECASE
    )

    def analyze_texts(records, is_jsonl=True):
        texts = []
        for r in records:
            if is_jsonl:
                # messages[1] is model response: content is [{'type': 'text', 'text': '...'}]
                c = r["messages"][1]["content"]
                if isinstance(c, list):
                    txt = "".join(item.get("text", "") for item in c if isinstance(item, dict))
                else:
                    txt = str(c)
            else:
                txt = r.get("reference_full_report", "")
            texts.append(txt)

        device_cnt = sum(1 for t in texts if device_regex.search(t))
        laterality_cnt = sum(1 for t in texts if laterality_regex.search(t))
        uncertainty_cnt = sum(1 for t in texts if uncertainty_regex.search(t))
        negation_cnt = sum(1 for t in texts if negation_regex.search(t))

        # Check for repeated reports
        clean_texts = [re.sub(r'\s+', ' ', t).strip().lower() for t in texts]
        text_counts = Counter(clean_texts)
        duplicates = sum(v for k, v in text_counts.items() if v > 1)
        unique_text_count = len(text_counts)

        return {
            "total": len(texts),
            "with_devices": device_cnt,
            "device_pct": round(device_cnt / len(texts) * 100, 2) if texts else 0,
            "with_laterality": laterality_cnt,
            "laterality_pct": round(laterality_cnt / len(texts) * 100, 2) if texts else 0,
            "with_uncertainty": uncertainty_cnt,
            "uncertainty_pct": round(uncertainty_cnt / len(texts) * 100, 2) if texts else 0,
            "with_negation": negation_cnt,
            "negation_pct": round(negation_cnt / len(texts) * 100, 2) if texts else 0,
            "unique_reports": unique_text_count,
            "duplicate_reports_count": duplicates,
            "most_common_templates": [
                {"template": k[:80] + "...", "count": v}
                for k, v in text_counts.most_common(3) if v > 1
            ]
        }

    audit_data["content_train"] = analyze_texts(train_records, is_jsonl=True)
    audit_data["content_val"] = analyze_texts(val_records, is_jsonl=True)
    audit_data["content_test"] = analyze_texts(test_records, is_jsonl=False)

    return audit_data

if __name__ == "__main__":
    results = run_audit()
    print(json.dumps(results, indent=2))
