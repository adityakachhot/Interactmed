#!/usr/bin/env python3
"""
scripts/validate_rsna.py
========================
InteractMed-VLM — Phase 3A: RSNA Pneumonia Detection Challenge Validation

Validates a locally extracted RSNA Pneumonia Detection Challenge dataset:
  - Checks for required files and directories
  - Validates annotation CSV schema and statistics
  - Samples DICOM files for readability via pydicom
  - Validates bounding box geometry (range, positivity)
  - Outputs a structured JSON report

Usage:
    /Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/validate_rsna.py \
        --rsna-root data/raw/rsna \
        --output docs/rsna_validation_report.json \
        --verbose

"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Expected constants from the official RSNA Pneumonia Detection Challenge
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_LABEL_COLS = {"patientId", "x", "y", "width", "height", "Target"}
EXPECTED_CLASS_COLS = {"patientId", "class"}
EXPECTED_CLASSES = {"Normal", "Lung Opacity", "No Lung Opacity / Not Normal"}

# Approximate expected counts (±5% tolerance used in checks)
EXPECTED_TRAIN_IMAGES = 26684
EXPECTED_POSITIVE_CASES = 6012
EXPECTED_NEGATIVE_CASES = 20672
EXPECTED_TOTAL_LABEL_ROWS = 30227

# DICOM files are named as {patientId}.dcm
TRAIN_IMAGES_DIR = "stage_2_train_images"
TEST_IMAGES_DIR = "stage_2_test_images"
LABELS_CSV = "stage_2_train_labels.csv"
CLASS_INFO_CSV = "stage_2_detailed_class_info.csv"
SAMPLE_SUBMISSION_CSV = "stage_2_sample_submission.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────


def within_tolerance(actual: int, expected: int, pct: float = 0.10) -> bool:
    """Check that `actual` is within `pct`% of `expected`."""
    return abs(actual - expected) / max(expected, 1) <= pct


def print_sep(char: str = "─", width: int = 70) -> None:
    print(char * width)


def print_header(title: str, width: int = 70) -> None:
    print_sep("═", width)
    print(f"  {title}")
    print_sep("═", width)


def print_section(title: str, width: int = 70) -> None:
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def fmt(val, ok: bool) -> str:
    status = "✅" if ok else "❌"
    return f"{status}  {val}"


# ─────────────────────────────────────────────────────────────────────────────
# Main validation logic
# ─────────────────────────────────────────────────────────────────────────────


def validate_rsna(rsna_root: Path, verbose: bool = True, n_dicom_samples: int = 10) -> dict:
    """
    Run all validation checks and return a structured results dictionary.

    Parameters
    ----------
    rsna_root : Path
        Path to the extracted RSNA dataset root directory.
    verbose : bool
        If True, print detailed progress and results to stdout.
    n_dicom_samples : int
        Number of random DICOM files to attempt reading for integrity check.

    Returns
    -------
    dict
        Structured validation results (suitable for JSON serialisation).
    """

    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "rsna_root": str(rsna_root.resolve()),
        "validation_checks": {},
        "statistics": {},
        "errors": [],
        "warnings": [],
        "overall_status": "PASS",
    }

    checks = results["validation_checks"]
    stats = results["statistics"]

    if verbose:
        print_header("RSNA Pneumonia Dataset Validation — InteractMed-VLM Phase 3A")
        print(f"  rsna_root : {rsna_root.resolve()}")
        print(f"  timestamp : {results['timestamp']}\n")

    # ── 1. Directory Existence ───────────────────────────────────────────────

    if verbose:
        print_section("1. File & Directory Existence")

    root_exists = rsna_root.is_dir()
    checks["rsna_root_exists"] = root_exists

    if not root_exists:
        msg = (
            f"RSNA root directory not found: {rsna_root.resolve()}\n"
            "  → Run extraction first: unzip rsna-pneumonia-detection-challenge.zip"
        )
        results["errors"].append(msg)
        results["overall_status"] = "FAIL"
        if verbose:
            print(f"  ❌  rsna_root_exists : {rsna_root.resolve()}")
            print(f"\n  FATAL: {msg}")
        return results

    labels_path = rsna_root / LABELS_CSV
    class_info_path = rsna_root / CLASS_INFO_CSV
    sample_sub_path = rsna_root / SAMPLE_SUBMISSION_CSV
    train_images_path = rsna_root / TRAIN_IMAGES_DIR
    test_images_path = rsna_root / TEST_IMAGES_DIR

    file_checks = {
        "labels_csv_exists": labels_path.is_file(),
        "class_info_csv_exists": class_info_path.is_file(),
        "sample_submission_exists": sample_sub_path.is_file(),
        "train_images_dir_exists": train_images_path.is_dir(),
        "test_images_dir_exists": test_images_path.is_dir(),
    }
    checks.update(file_checks)

    for name, ok in file_checks.items():
        if not ok:
            results["errors"].append(f"Missing required file/directory: {name}")
            results["overall_status"] = "FAIL"
        if verbose:
            label = name.replace("_", " ")
            print(f"  {fmt(label, ok)}")

    if not file_checks["labels_csv_exists"]:
        results["errors"].append(f"Labels CSV not found at: {labels_path}")
        if verbose:
            print("\n  FATAL: Cannot proceed without labels CSV.")
        return results

    # ── 2. Labels CSV Schema ─────────────────────────────────────────────────

    if verbose:
        print_section("2. Labels CSV Schema & Statistics")

    try:
        labels_df = pd.read_csv(labels_path)
        actual_cols = set(labels_df.columns.tolist())
        missing_cols = EXPECTED_LABEL_COLS - actual_cols
        extra_cols = actual_cols - EXPECTED_LABEL_COLS

        col_ok = len(missing_cols) == 0
        checks["label_columns_valid"] = col_ok

        if verbose:
            print(f"  {fmt('label_columns_valid', col_ok)}")
            print(f"       found columns : {sorted(actual_cols)}")
            if missing_cols:
                print(f"       MISSING       : {sorted(missing_cols)}")
            if extra_cols:
                print(f"       extra cols    : {sorted(extra_cols)}")

        if not col_ok:
            results["errors"].append(f"Labels CSV missing columns: {sorted(missing_cols)}")
            results["overall_status"] = "FAIL"

    except Exception as exc:
        results["errors"].append(f"Failed to read labels CSV: {exc}")
        results["overall_status"] = "FAIL"
        if verbose:
            print(f"  ❌  Failed to read {LABELS_CSV}: {exc}")
        return results

    # ── 3. Annotation Statistics ─────────────────────────────────────────────

    total_rows = len(labels_df)
    unique_patients = labels_df["patientId"].nunique()
    positive_rows = labels_df[labels_df["Target"] == 1]
    negative_rows = labels_df[labels_df["Target"] == 0]
    positive_patients = positive_rows["patientId"].nunique()
    negative_patients = negative_rows["patientId"].nunique()

    # Multi-box analysis
    pos_box_counts = positive_rows.groupby("patientId").size()
    multi_box_patients = int((pos_box_counts > 1).sum())
    max_boxes = int(pos_box_counts.max()) if len(pos_box_counts) > 0 else 0

    stats.update({
        "label_rows_total": total_rows,
        "unique_train_patients": unique_patients,
        "positive_cases": positive_patients,
        "negative_cases": negative_patients,
        "multi_box_patients": multi_box_patients,
        "max_boxes_per_patient": max_boxes,
    })

    rows_ok = within_tolerance(total_rows, EXPECTED_TOTAL_LABEL_ROWS)
    patients_ok = within_tolerance(unique_patients, EXPECTED_TRAIN_IMAGES)
    pos_ok = within_tolerance(positive_patients, EXPECTED_POSITIVE_CASES)
    neg_ok = within_tolerance(negative_patients, EXPECTED_NEGATIVE_CASES)

    checks["label_rows_within_expected"] = rows_ok
    checks["patient_count_within_expected"] = patients_ok
    checks["positive_count_within_expected"] = pos_ok
    checks["negative_count_within_expected"] = neg_ok

    if verbose:
        print(f"\n  {fmt(f'label_rows_total         : {total_rows:>8,}  (expected ~{EXPECTED_TOTAL_LABEL_ROWS:,})', rows_ok)}")
        print(f"  {fmt(f'unique_train_patients    : {unique_patients:>8,}  (expected ~{EXPECTED_TRAIN_IMAGES:,})', patients_ok)}")
        print(f"  {fmt(f'positive_cases           : {positive_patients:>8,}  (expected ~{EXPECTED_POSITIVE_CASES:,})', pos_ok)}")
        print(f"  {fmt(f'negative_cases           : {negative_patients:>8,}  (expected ~{EXPECTED_NEGATIVE_CASES:,})', neg_ok)}")
        print(f"  ℹ️   multi_box_patients       : {multi_box_patients:>8,}")
        print(f"  ℹ️   max_boxes_per_patient    : {max_boxes:>8}")

    for check_name, ok in [
        ("label_rows_within_expected", rows_ok),
        ("patient_count_within_expected", patients_ok),
        ("positive_count_within_expected", pos_ok),
        ("negative_count_within_expected", neg_ok),
    ]:
        if not ok:
            results["warnings"].append(
                f"{check_name} is outside expected range (±10%). "
                "This may indicate a partial download or alternate dataset version."
            )

    # ── 4. Bounding Box Geometry Validation ──────────────────────────────────

    if verbose:
        print_section("3. Bounding Box Geometry (Positive Cases Only)")

    if len(positive_rows) > 0:
        bb_cols = ["x", "y", "width", "height"]
        has_bb_cols = all(c in labels_df.columns for c in bb_cols)

        if has_bb_cols:
            pos_df = positive_rows[bb_cols].dropna()

            # All values must be non-negative
            non_neg = (pos_df >= 0).all().all()
            # Width and height must be positive
            positive_dims = ((pos_df["width"] > 0) & (pos_df["height"] > 0)).all()
            # Coordinates must be within a reasonable pixel range (0–2048)
            within_range = (
                (pos_df["x"] < 2048)
                & (pos_df["y"] < 2048)
                & (pos_df["width"] <= 2048)
                & (pos_df["height"] <= 2048)
            ).all()

            checks["bbox_non_negative"] = bool(non_neg)
            checks["bbox_positive_dimensions"] = bool(positive_dims)
            checks["bbox_within_pixel_range"] = bool(within_range)

            x2 = pos_df["x"] + pos_df["width"]
            y2 = pos_df["y"] + pos_df["height"]
            bbox_range_ok = bool(non_neg and positive_dims and within_range)

            stats["bbox_x_range"] = [float(pos_df["x"].min()), float(x2.max())]
            stats["bbox_y_range"] = [float(pos_df["y"].min()), float(y2.max())]
            stats["bbox_width_range"] = [float(pos_df["width"].min()), float(pos_df["width"].max())]
            stats["bbox_height_range"] = [float(pos_df["height"].min()), float(pos_df["height"].max())]

            if verbose:
                print(f"  {fmt(f'bbox_non_negative              : {non_neg}', bool(non_neg))}")
                print(f"  {fmt(f'bbox_positive_dimensions       : {positive_dims}', bool(positive_dims))}")
                print(f"  {fmt(f'bbox_within_pixel_range (2048) : {within_range}', bool(within_range))}")
                print(f"  ℹ️   x  range : [{pos_df['x'].min():.1f}, {x2.max():.1f}]")
                print(f"  ℹ️   y  range : [{pos_df['y'].min():.1f}, {y2.max():.1f}]")
                print(f"  ℹ️   w  range : [{pos_df['width'].min():.1f}, {pos_df['width'].max():.1f}]")
                print(f"  ℹ️   h  range : [{pos_df['height'].min():.1f}, {pos_df['height'].max():.1f}]")

            for chk, ok in [
                ("bbox_non_negative", bool(non_neg)),
                ("bbox_positive_dimensions", bool(positive_dims)),
                ("bbox_within_pixel_range", bool(within_range)),
            ]:
                if not ok:
                    results["errors"].append(f"Bounding box check failed: {chk}")
                    results["overall_status"] = "FAIL"
        else:
            results["warnings"].append("Cannot validate bounding boxes: missing x/y/width/height columns.")
    else:
        results["warnings"].append("No positive (Target=1) rows found — skipping bbox geometry checks.")
        if verbose:
            print("  ⚠️   No positive rows found — skipping bbox checks.")

    # ── 5. Detailed Class Info CSV ────────────────────────────────────────────

    if verbose:
        print_section("4. Detailed Class Info CSV")

    if class_info_path.is_file():
        try:
            class_df = pd.read_csv(class_info_path)
            class_cols_ok = EXPECTED_CLASS_COLS.issubset(set(class_df.columns))
            actual_classes = set(class_df["class"].unique()) if "class" in class_df.columns else set()
            classes_ok = actual_classes.issubset(EXPECTED_CLASSES)

            checks["class_info_columns_valid"] = class_cols_ok
            checks["class_info_classes_valid"] = classes_ok

            stats["class_distribution"] = (
                class_df["class"].value_counts().to_dict() if "class" in class_df.columns else {}
            )

            if verbose:
                print(f"  {fmt('class_info_columns_valid', class_cols_ok)}")
                print(f"  {fmt('class_info_classes_valid', classes_ok)}")
                if "class" in class_df.columns:
                    for cls, cnt in class_df["class"].value_counts().items():
                        print(f"  ℹ️   {cls:<40} : {cnt:,}")
                if not classes_ok:
                    unexpected = actual_classes - EXPECTED_CLASSES
                    print(f"  ⚠️   Unexpected classes: {unexpected}")

            if not class_cols_ok:
                results["errors"].append("class_info CSV missing required columns.")
                results["overall_status"] = "FAIL"

        except Exception as exc:
            results["errors"].append(f"Failed to read class_info CSV: {exc}")
            if verbose:
                print(f"  ❌  Failed to read {CLASS_INFO_CSV}: {exc}")
    else:
        results["warnings"].append(f"{CLASS_INFO_CSV} not found — skipping class distribution check.")
        if verbose:
            print(f"  ⚠️   {CLASS_INFO_CSV} not found.")

    # ── 6. Train Image Directory File Count ──────────────────────────────────

    if verbose:
        print_section("5. Training Image Directory File Count")

    if train_images_path.is_dir():
        train_dcm_files = list(train_images_path.glob("*.dcm"))
        n_train_dcm = len(train_dcm_files)
        train_count_ok = within_tolerance(n_train_dcm, EXPECTED_TRAIN_IMAGES)
        checks["train_dicom_count_within_expected"] = train_count_ok
        stats["train_dicom_count"] = n_train_dcm

        if verbose:
            print(f"  {fmt(f'train_dicom_count : {n_train_dcm:,}  (expected ~{EXPECTED_TRAIN_IMAGES:,})', train_count_ok)}")

        if not train_count_ok:
            results["warnings"].append(
                f"Train DICOM count ({n_train_dcm}) outside expected range. "
                "Possible partial download."
            )
    else:
        n_train_dcm = 0
        train_dcm_files = []
        results["warnings"].append("train_images directory not found — skipping DICOM count.")
        if verbose:
            print(f"  ⚠️   {TRAIN_IMAGES_DIR} not found.")

    # ── 7. Test Image Directory File Count ───────────────────────────────────

    if test_images_path.is_dir():
        test_dcm_files = list(test_images_path.glob("*.dcm"))
        n_test_dcm = len(test_dcm_files)
        stats["test_dicom_count"] = n_test_dcm
        if verbose:
            print(f"  ℹ️   test_dicom_count  : {n_test_dcm:,}")
    else:
        stats["test_dicom_count"] = 0
        results["warnings"].append("test_images directory not found.")

    # ── 8. DICOM Sample Integrity Check ──────────────────────────────────────

    if verbose:
        print_section(f"6. DICOM Sample Integrity Check (n={n_dicom_samples})")

    try:
        import pydicom  # type: ignore
        pydicom_available = True
    except ImportError:
        pydicom_available = False
        results["warnings"].append(
            "pydicom is not installed. Skipping DICOM integrity check. "
            "Install with: pip install pydicom"
        )
        if verbose:
            print("  ⚠️   pydicom not installed — skipping DICOM integrity check.")

    dicom_samples_ok = None
    if pydicom_available and len(train_dcm_files) > 0:
        sample_files = random.sample(
            train_dcm_files, min(n_dicom_samples, len(train_dcm_files))
        )
        dicom_ok_count = 0
        dicom_errors = []

        for dcm_path in sample_files:
            try:
                ds = pydicom.dcmread(str(dcm_path))
                _ = ds.pixel_array  # Force pixel data decode
                dicom_ok_count += 1
            except Exception as exc:
                dicom_errors.append(f"{dcm_path.name}: {exc}")

        dicom_samples_ok = dicom_ok_count == len(sample_files)
        checks["dicom_sample_readable"] = dicom_samples_ok
        stats["dicom_sample_n"] = len(sample_files)
        stats["dicom_sample_ok_count"] = dicom_ok_count
        stats["dicom_sample_errors"] = dicom_errors

        if verbose:
            print(f"  {fmt(f'dicom_sample_readable : {dicom_ok_count}/{len(sample_files)} DICOMs OK', dicom_samples_ok)}")
            for err in dicom_errors:
                print(f"  ⚠️   DICOM error: {err}")

        if not dicom_samples_ok:
            results["errors"].append(
                f"DICOM sample integrity FAILED: {len(dicom_errors)} files unreadable. "
                f"Details: {dicom_errors}"
            )
            results["overall_status"] = "FAIL"

    # ── 9. PatientId ↔ DICOM Filename Cross-Check ───────────────────────────

    if verbose:
        print_section("7. PatientId ↔ DICOM Filename Cross-Check")

    if len(train_dcm_files) > 0 and len(labels_df) > 0:
        dcm_stems = {f.stem for f in train_dcm_files}
        label_patient_ids = set(labels_df["patientId"].dropna().unique())

        in_labels_not_dcm = label_patient_ids - dcm_stems
        in_dcm_not_labels = dcm_stems - label_patient_ids

        cross_check_ok = len(in_labels_not_dcm) == 0 and len(in_dcm_not_labels) == 0
        checks["patient_id_dicom_cross_check"] = cross_check_ok

        stats["label_ids_without_dcm"] = len(in_labels_not_dcm)
        stats["dcm_files_without_label"] = len(in_dcm_not_labels)

        if verbose:
            print(f"  {fmt('patient_id_dicom_cross_check', cross_check_ok)}")
            print(f"  ℹ️   label IDs without DICOM  : {len(in_labels_not_dcm)}")
            print(f"  ℹ️   DICOM files without label : {len(in_dcm_not_labels)}")
            if len(in_labels_not_dcm) > 0:
                ex = list(in_labels_not_dcm)[:5]
                print(f"  ⚠️   Examples without DICOM: {ex}")

        if not cross_check_ok:
            results["warnings"].append(
                f"PatientId cross-check: {len(in_labels_not_dcm)} label IDs have no DICOM, "
                f"{len(in_dcm_not_labels)} DICOMs have no label entry."
            )
    else:
        if verbose:
            print("  ⚠️   Skipping cross-check (no DICOMs or labels found).")

    # ── 10. Final Summary ────────────────────────────────────────────────────

    if verbose:
        print_sep("═")
        status_icon = "✅" if results["overall_status"] == "PASS" else "❌"
        print(f"\n  OVERALL STATUS: {status_icon} {results['overall_status']}")
        if results["errors"]:
            print(f"\n  Errors ({len(results['errors'])}):")
            for err in results["errors"]:
                print(f"    ❌  {err}")
        if results["warnings"]:
            print(f"\n  Warnings ({len(results['warnings'])}):")
            for warn in results["warnings"]:
                print(f"    ⚠️   {warn}")
        print()
        print_sep("═")
        print()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate an extracted RSNA Pneumonia Detection Challenge dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_rsna.py --rsna-root data/raw/rsna --verbose
  python scripts/validate_rsna.py --rsna-root data/raw/rsna \\
      --output docs/rsna_validation_report.json
        """,
    )
    parser.add_argument(
        "--rsna-root",
        type=Path,
        default=Path("data/raw/rsna"),
        help="Path to the extracted RSNA dataset root directory (default: data/raw/rsna)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional: path to write the JSON validation report",
    )
    parser.add_argument(
        "--dicom-samples",
        type=int,
        default=10,
        help="Number of DICOM files to randomly sample for integrity check (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress (default: True)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    results = validate_rsna(
        rsna_root=args.rsna_root,
        verbose=verbose,
        n_dicom_samples=args.dicom_samples,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        if verbose:
            print(f"  Validation report written to: {args.output}")

    return 0 if results["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
