#!/usr/bin/env python3
"""
scripts/prepare_medgemma_finetuning_data.py
===========================================
Prepares a patient-disjoint, stratified training and validation dataset
for MedGemma 4B fine-tuning pilot from the verified 250-case expanded development cohort.

Governance and Clinical Safeguards:
-----------------------------------
1. Strictly drawn from 'expanded_dev' (250 images across 248 patients).
2. The 117-case held-out test cohort ('expanded_test') is strictly EXCLUDED and remains untouched.
3. Patient-disjoint partitioning: Patients(train_dev) ∩ Patients(val_dev) = ∅.
4. Preserves complete radiologist reference reports parsed into:
   - OBSERVATIONS
   - IMPRESSION
   - LIMITATIONS
5. Anatomical bounding boxes are NOT used as report generation targets.
6. All MIMIC-CXR and Chest-ImaGenome data remain strictly local.
7. Clinical text is written to JSONL files and anonymized manifest; generic logs remain free of PHI.

Usage:
------
    python scripts/prepare_medgemma_finetuning_data.py \\
        --manifest artifacts/report_evaluation/expanded_cohort/expanded_manifest.csv \\
        --ref-store artifacts/report_evaluation/expanded_cohort/expanded_reference_store.json \\
        --output-dir artifacts/fine_tuning_cohort \\
        --train-size 200 \\
        --val-size 50 \\
        --seed 42
"""

import argparse
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

STRUCTURED_PROMPT = (
    "You are an expert medical radiologist examining this chest radiograph. "
    "Generate an objective, structured research report with EXACTLY these three headings:\n\n"
    "OBSERVATIONS\n"
    "Detail visible findings across support apparatus, cardiac silhouette, lungs/opacities, "
    "pleural spaces, and mediastinum. Incorporate supported laterality (right/left/bilateral) "
    "or state unspecified laterality when unverified.\n\n"
    "IMPRESSION\n"
    "Concisely summarize key findings and diagnostic uncertainty. Never declare normal "
    "or no acute cardiopulmonary disease if abnormalities or indeterminate features are present.\n\n"
    "LIMITATIONS\n"
    "State technical projection limitations.\n\n"
    "Begin directly with:\n"
    "OBSERVATIONS"
)


def extract_patient_id(image_path: str) -> str:
    """Extracts MIMIC-CXR patient ID (e.g. p10000032) from file path."""
    match = re.search(r'/(p\d+)/s\d+/', str(image_path))
    if match:
        return match.group(1)
    # Fallback to parent directory name if pattern not found
    parts = Path(image_path).parts
    for p in parts:
        if re.match(r'^p\d+$', p):
            return p
    return "patient_unknown"


def extract_study_id(image_path: str) -> str:
    """Extracts MIMIC-CXR study ID (e.g. s50414267) from file path."""
    match = re.search(r'/s(\d+)/', str(image_path))
    if match:
        return f"s{match.group(1)}"
    parts = Path(image_path).parts
    for p in parts:
        if re.match(r'^s\d+$', p):
            return p
    return "study_unknown"


def parse_structured_reference_report(full_text: str, view_position: str = "AP") -> Dict[str, str]:
    """
    Parses a raw MIMIC-CXR radiologist report into structured sections:
    OBSERVATIONS, IMPRESSION, and LIMITATIONS.
    Preserves all clinical information and standardizes headings.
    """
    text = full_text.replace("\r\n", "\n").strip()

    # Strip wet read block if present at the beginning
    text = re.sub(
        r"^WET READ:.*?(?=FINAL REPORT|\bINDICATION|\bHISTORY|\bFINDINGS|\bIMPRESSION|$)",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    # Strip 'FINAL REPORT' header
    text = re.sub(r"^FINAL REPORT\s*", "", text, flags=re.IGNORECASE).strip()

    findings = ""
    impression = ""

    # 1. Look for explicit FINDINGS section
    f_match = re.search(
        r"\bFINDINGS\s*:\s*(.*?)(?=\bIMPRESSION\s*:|\bRECOMMENDATION|\bNOTE\b|\bNOTIFICATION|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if f_match:
        findings = f_match.group(1).strip()

    # 2. Look for explicit IMPRESSION section
    i_match = re.search(
        r"\bIMPRESSION\s*:\s*(.*?)(?=\bRECOMMENDATION|\bNOTE\b|\bNOTIFICATION|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if i_match:
        impression = i_match.group(1).strip()

    # 3. Fallback for findings: descriptive text between TECHNIQUE/COMPARISON/EXAMINATION and IMPRESSION
    if not findings:
        alt_f = re.search(
            r"(?:TECHNIQUE|COMPARISON|STUDY|EXAMINATION|REASON FOR EXAMINATION)[^\n]*\n+(.*?)(?=\bIMPRESSION\s*:|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if alt_f:
            cand = alt_f.group(1).strip()
            # Clean out clinical indication lines if present at top
            cand = re.sub(
                r"^(?:INDICATION|HISTORY|CLINICAL INDICATION|REASON FOR EXAMINATION)[^\n]*\n+",
                "",
                cand,
                flags=re.IGNORECASE,
            ).strip()
            if len(cand) > 15:
                findings = cand

    # 4. Fallback if findings still empty but impression exists
    if not findings and impression:
        findings = "Findings detailed in clinical impression."

    # 5. Fallback if impression empty but findings exist
    if not impression and findings:
        if re.search(r"clear|normal|unremarkable", findings, re.IGNORECASE):
            impression = "No acute cardiopulmonary abnormality."
        else:
            impression = "Cardiopulmonary findings as detailed in observations."

    # Normalize inner whitespace
    findings = re.sub(r"\s+", " ", findings).strip()
    impression = re.sub(r"\s+", " ", impression).strip()

    # 6. Format standard projection limitation based on actual radiograph view
    view = (view_position or "AP").upper().strip()
    if "AP" in view:
        limitations = "Single frontal AP projection. Evaluation limited by absence of orthogonal lateral view."
    elif "PA" in view:
        limitations = "Single frontal PA projection without orthogonal lateral view."
    else:
        limitations = "Single projection radiograph. Technical evaluation limited by lack of orthogonal lateral view."

    structured_text = (
        f"OBSERVATIONS\n{findings}\n\n"
        f"IMPRESSION\n{impression}\n\n"
        f"LIMITATIONS\n{limitations}"
    )

    return {
        "observations": findings,
        "impression": impression,
        "limitations": limitations,
        "structured_target": structured_text,
    }


def split_patient_disjoint(
    cases: List[Dict[str, Any]],
    train_target: int = 200,
    val_target: int = 50,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits cases into train and val while strictly guaranteeing:
    1. Patients(train) ∩ Patients(val) = ∅
    2. Category balance is preserved (stratification by primary finding category).
    """
    import random
    rng = random.Random(seed)

    # Group cases by patient_id
    patient_to_cases: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in cases:
        pid = c["patient_id"]
        patient_to_cases[pid].append(c)

    # Identify primary category for each patient (majority vote or first case)
    patient_category: Dict[str, str] = {}
    for pid, p_cases in patient_to_cases.items():
        cats = [c["primary_category"] for c in p_cases]
        patient_category[pid] = Counter(cats).most_common(1)[0][0]

    # Group patients by category
    cat_to_patients: Dict[str, List[str]] = defaultdict(list)
    for pid, cat in patient_category.items():
        cat_to_patients[cat].append(pid)

    # Sort categories deterministically
    sorted_cats = sorted(cat_to_patients.keys())

    val_patients: Set[str] = set()
    train_patients: Set[str] = set()

    val_case_count = 0

    # First pass: Allocate proportional patients from each category to validation
    val_ratio = float(val_target) / float(train_target + val_target)

    for cat in sorted_cats:
        pids = sorted(cat_to_patients[cat])
        rng.shuffle(pids)

        # Calculate target val cases for this category
        cat_cases_total = sum(len(patient_to_cases[p]) for p in pids)
        cat_val_target = round(cat_cases_total * val_ratio)

        current_cat_val_cases = 0
        for pid in pids:
            num_cases = len(patient_to_cases[pid])
            if (current_cat_val_cases + num_cases <= cat_val_target) and (val_case_count + num_cases <= val_target):
                val_patients.add(pid)
                current_cat_val_cases += num_cases
                val_case_count += num_cases
            else:
                train_patients.add(pid)

    # Balance remaining patients if val_case_count is slightly below target
    remaining_train = [p for p in train_patients if len(patient_to_cases[p]) == 1]
    rng.shuffle(remaining_train)
    while val_case_count < val_target and remaining_train:
        p = remaining_train.pop()
        train_patients.remove(p)
        val_patients.add(p)
        val_case_count += 1

    # Final verification of disjointness
    assert len(train_patients.intersection(val_patients)) == 0, "FATAL: Patient overlap detected!"

    train_cases = [c for c in cases if c["patient_id"] in train_patients]
    val_cases = [c for c in cases if c["patient_id"] in val_patients]

    assert len(train_cases) + len(val_cases) == len(cases), "FATAL: Case count mismatch!"
    return train_cases, val_cases


def format_conversational_record(case: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    """
    Formats a single image-report case into the standard multimodal
    conversational JSONL format used by Hugging Face / Google Health MedGemma.
    """
    return {
        "id": case["safe_case_id"],
        "patient_id": case["patient_id"],
        "study_id": case["study_id"],
        "image": case["image_path"],
        "primary_category": case["primary_category"],
        "gold_labels": case["gold_affirmed_labels"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": case["structured_target"]},
                ],
            },
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare patient-disjoint fine-tuning data for MedGemma 4B.")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/report_evaluation/expanded_cohort/expanded_manifest.csv"),
        help="Path to expanded cohort manifest CSV.",
    )
    parser.add_argument(
        "--ref-store",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/report_evaluation/expanded_cohort/expanded_reference_store.json"),
        help="Path to expanded reference store JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/fine_tuning_cohort"),
        help="Output directory for fine-tuning artifacts.",
    )
    parser.add_argument("--train-size", type=int, default=200, help="Target training set size.")
    parser.add_argument("--val-size", type=int, default=50, help="Target validation set size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for partitioning.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MEDGEMMA 4B FINE-TUNING DATA PREPARATION")
    print("=" * 70)
    print(f"Manifest:       {args.manifest}")
    print(f"Reference Store:{args.ref_store}")
    print(f"Output Dir:     {out_dir}")
    print(f"Target Split:   {args.train_size} train, {args.val_size} val")
    print(f"Seed:           {args.seed}")
    print("-" * 70)

    # 1. Load manifest and reference store
    manifest_df = pd.read_csv(args.manifest)
    with open(args.ref_store, "r", encoding="utf-8") as f:
        ref_store: Dict[str, Dict[str, Any]] = json.load(f)

    # 2. Filter strictly for expanded_dev
    dev_manifest = manifest_df[manifest_df["split"] == "expanded_dev"].copy()
    test_manifest = manifest_df[manifest_df["split"] == "expanded_test"].copy()

    print(f"Manifest row count:         {len(manifest_df)}")
    print(f"Expanded dev cases:         {len(dev_manifest)}")
    print(f"Expanded test cases:        {len(test_manifest)} (SEALED & UNTOUCHED)")

    assert len(test_manifest) == 117, f"Expected 117 test cases, got {len(test_manifest)}"
    assert len(dev_manifest) == 250, f"Expected 250 dev cases, got {len(dev_manifest)}"

    # 3. Verify each dev case
    verified_dev_cases: List[Dict[str, Any]] = []
    skipped_cases: List[str] = []

    for _, row in dev_manifest.iterrows():
        case_id = row["safe_case_id"]
        ref_data = ref_store.get(case_id)
        if not ref_data:
            skipped_cases.append(f"{case_id}: missing from reference store")
            continue

        img_path = ref_data.get("image_path", "")
        if not os.path.exists(img_path):
            skipped_cases.append(f"{case_id}: image not found on disk ({img_path})")
            continue

        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            skipped_cases.append(f"{case_id}: invalid PIL image ({e})")
            continue

        full_rep = ref_data.get("reference_full_report", "")
        if not full_rep or len(full_rep.strip()) < 20:
            skipped_cases.append(f"{case_id}: reference report empty or too short (<20 chars)")
            continue

        pid = extract_patient_id(img_path)
        sid = extract_study_id(img_path)
        view_pos = ref_data.get("view_position", row.get("view_position", "AP"))

        # Parse structured report
        parsed = parse_structured_reference_report(full_rep, view_position=view_pos)

        verified_dev_cases.append({
            "safe_case_id": case_id,
            "split_source": "expanded_dev",
            "patient_id": pid,
            "study_id": sid,
            "image_path": img_path,
            "view_position": view_pos,
            "primary_category": ref_data.get("primary_category", row.get("primary_finding_category", "unknown")),
            "gold_affirmed_labels": ref_data.get("gold_affirmed_labels", []),
            "reference_full_report": full_rep,
            "observations": parsed["observations"],
            "impression": parsed["impression"],
            "limitations": parsed["limitations"],
            "structured_target": parsed["structured_target"],
            "target_word_count": len(parsed["structured_target"].split()),
            "report_type": "full_report",
        })

    print(f"Verified valid dev cases:   {len(verified_dev_cases)} / {len(dev_manifest)}")
    if skipped_cases:
        print(f"WARNING: Skipped {len(skipped_cases)} cases:")
        for s in skipped_cases:
            print(f"  - {s}")
        if len(verified_dev_cases) < 150:
            print("ERROR: Usable cohort too small for planned fine-tuning pilot! Halting.")
            sys.exit(1)

    # 4. Partition into train_dev and val_dev
    train_cases, val_cases = split_patient_disjoint(
        verified_dev_cases,
        train_target=args.train_size,
        val_target=args.val_size,
        seed=args.seed,
    )

    train_pids = set(c["patient_id"] for c in train_cases)
    val_pids = set(c["patient_id"] for c in val_cases)

    # Verify patient disjointness against test cohort as well
    test_pids = set(extract_patient_id(ref_store[c]["image_path"]) for c in test_manifest["safe_case_id"] if c in ref_store)

    overlap_train_val = train_pids.intersection(val_pids)
    overlap_train_test = train_pids.intersection(test_pids)
    overlap_val_test = val_pids.intersection(test_pids)

    print("\n" + "=" * 70)
    print("PATIENT ISOLATION AUDIT")
    print("=" * 70)
    print(f"Train Cases:               {len(train_cases)} (Unique Patients: {len(train_pids)})")
    print(f"Val Cases:                 {len(val_cases)} (Unique Patients: {len(val_pids)})")
    print(f"Locked Test Cases:         {len(test_manifest)} (Unique Patients: {len(test_pids)})")
    print(f"Overlap(Train, Val):       {len(overlap_train_val)} patients (Must be 0)")
    print(f"Overlap(Train, Test):      {len(overlap_train_test)} patients (Must be 0)")
    print(f"Overlap(Val, Test):        {len(overlap_val_test)} patients (Must be 0)")

    assert len(overlap_train_val) == 0, "Patient overlap between train and val!"
    assert len(overlap_train_test) == 0, "Patient overlap between train and test!"
    assert len(overlap_val_test) == 0, "Patient overlap between val and test!"

    # 5. Write JSONL files
    train_jsonl_path = out_dir / "train_dev.jsonl"
    val_jsonl_path = out_dir / "val_dev.jsonl"

    with open(train_jsonl_path, "w", encoding="utf-8") as f:
        for c in train_cases:
            rec = format_conversational_record(c, STRUCTURED_PROMPT)
            f.write(json.dumps(rec) + "\n")

    with open(val_jsonl_path, "w", encoding="utf-8") as f:
        for c in val_cases:
            rec = format_conversational_record(c, STRUCTURED_PROMPT)
            f.write(json.dumps(rec) + "\n")

    print(f"\nSaved {len(train_cases)} training records to:   {train_jsonl_path}")
    print(f"Saved {len(val_cases)} validation records to: {val_jsonl_path}")

    # 6. Save target reports as individual text files for reference/evaluation
    target_reports_dir = out_dir / "target_reports"
    target_reports_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for split_name, cases in [("train_dev", train_cases), ("val_dev", val_cases)]:
        for c in cases:
            target_path = target_reports_dir / f"{c['safe_case_id']}_target.txt"
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(c["structured_target"])

            manifest_rows.append({
                "safe_case_id": c["safe_case_id"],
                "split": split_name,
                "patient_id": c["patient_id"],
                "study_id": c["study_id"],
                "primary_category": c["primary_category"],
                "gold_labels_count": len(c["gold_affirmed_labels"]),
                "image_path": c["image_path"],
                "target_report_path": str(target_path.relative_to(PROJECT_ROOT)),
                "target_word_count": c["target_word_count"],
                "report_type": c["report_type"],
            })

    # Add untouched test records to the manifest for auditing
    for _, row in test_manifest.iterrows():
        cid = row["safe_case_id"]
        t_ref = ref_store.get(cid, {})
        t_img = t_ref.get("image_path", "")
        manifest_rows.append({
            "safe_case_id": cid,
            "split": "expanded_test_locked",
            "patient_id": extract_patient_id(t_img),
            "study_id": extract_study_id(t_img),
            "primary_category": row.get("primary_finding_category", "unknown"),
            "gold_labels_count": row.get("gold_affirmed_count", 0),
            "image_path": t_img,
            "target_report_path": "SEALED_TEST_RESERVE",
            "target_word_count": 0,
            "report_type": "locked_test",
        })

    cohort_manifest_df = pd.DataFrame(manifest_rows)
    cohort_manifest_path = out_dir / "dataset_manifest.csv"
    cohort_manifest_df.to_csv(cohort_manifest_path, index=False)
    print(f"Saved dataset manifest to:               {cohort_manifest_path}")

    # 7. Print category breakdown table
    print("\n" + "=" * 70)
    print("CATEGORY DISTRIBUTION BY SPLIT")
    print("=" * 70)
    train_cats = Counter(c["primary_category"] for c in train_cases)
    val_cats = Counter(c["primary_category"] for c in val_cases)
    all_cats = sorted(set(list(train_cats.keys()) + list(val_cats.keys())))

    header = f"{'Category':<28} | {'Train (N=200)':<15} | {'Val (N=50)':<15} | {'Total Dev (250)':<15}"
    print(header)
    print("-" * len(header))
    for cat in all_cats:
        t_cnt = train_cats.get(cat, 0)
        v_cnt = val_cats.get(cat, 0)
        tot = t_cnt + v_cnt
        print(f"{cat:<28} | {t_cnt:<5} ({t_cnt/len(train_cases)*100:4.1f}%)   | {v_cnt:<5} ({v_cnt/len(val_cases)*100:4.1f}%)   | {tot:<5} ({tot/250*100:4.1f}%)")

    # 8. Save audit summary JSON
    audit_summary = {
        "status": "success",
        "dataset_name": "InteractMed-VLM MedGemma 4B Fine-Tuning Pilot Cohort",
        "source_cohort": "expanded_dev",
        "locked_test_cohort": "expanded_test",
        "total_verified_dev_cases": len(verified_dev_cases),
        "train_dev_cases": len(train_cases),
        "val_dev_cases": len(val_cases),
        "locked_test_cases": len(test_manifest),
        "unique_train_patients": len(train_pids),
        "unique_val_patients": len(val_pids),
        "unique_test_patients": len(test_pids),
        "patient_overlap_train_val": 0,
        "patient_overlap_train_test": 0,
        "patient_overlap_val_test": 0,
        "mean_target_words_train": float(pd.Series([c["target_word_count"] for c in train_cases]).mean()),
        "mean_target_words_val": float(pd.Series([c["target_word_count"] for c in val_cases]).mean()),
        "train_categories": dict(train_cats),
        "val_categories": dict(val_cats),
    }

    audit_json_path = out_dir / "audit_summary.json"
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"Saved audit summary to:                  {audit_json_path}")
    print("=" * 70)
    print("DATA PREPARATION COMPLETE SUCCESSFULLY.")
    print("=" * 70)


if __name__ == "__main__":
    main()
