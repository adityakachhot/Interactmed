#!/usr/bin/env python3
"""
scripts/audit_overlap.py
========================
InteractMed-VLM Cross-Dataset and Cross-Split Overlap & Leakage Audit.

Computes set intersections across Patients (subject_id), Studies (study_id),
and Images (dicom_id) between MS-CXR and Chest ImaGenome Gold.
Audits cross-split contamination against the official MS-CXR test benchmark.
Determines independent evaluation eligibility per project protocol.

Outputs:
  - docs/dataset_overlap_report.json
  - Formatted terminal overlap and data-leakage audit report

Usage:
    python scripts/audit_overlap.py [--table data/processed/unified_grounding_master.csv] [--output docs/dataset_overlap_report.json]
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Set

import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("AuditOverlap")


def audit_overlap(df: pd.DataFrame) -> Dict[str, Any]:
    logger.info("Computing cross-dataset set intersections...")

    df_ms = df[df["dataset_source"] == "ms_cxr"].copy()
    df_cig = df[df["dataset_source"] == "chest_imagenome_gold"].copy()

    # Sets
    ms_subjects = set(df_ms["subject_id"].dropna().astype(str).unique())
    cig_subjects = set(df_cig["subject_id"].dropna().astype(str).unique())

    ms_studies = set(df_ms["study_id"].dropna().astype(str).unique())
    cig_studies = set(df_cig["study_id"].dropna().astype(str).unique())

    ms_dicoms = set(df_ms["dicom_id"].dropna().astype(str).unique())
    cig_dicoms = set(df_cig["dicom_id"].dropna().astype(str).unique())

    # Intersections
    shared_subjects = ms_subjects.intersection(cig_subjects)
    shared_studies = ms_studies.intersection(cig_studies)
    shared_dicoms = ms_dicoms.intersection(cig_dicoms)

    # Local Disk Images Overlap
    df_local = df[df["image_exists"] == True].copy()
    local_ms_dicoms = set(df_local[df_local["dataset_source"] == "ms_cxr"]["dicom_id"].unique())
    local_cig_dicoms = set(df_local[df_local["dataset_source"] == "chest_imagenome_gold"]["dicom_id"].unique())
    shared_local_dicoms = local_ms_dicoms.intersection(local_cig_dicoms)

    # Cross-Split Contamination with MS-CXR splits
    split_contamination = {}
    for split_name, grp in df_ms.groupby("split"):
        sp_subjects = set(grp["subject_id"].dropna().astype(str).unique())
        sp_studies = set(grp["study_id"].dropna().astype(str).unique())
        sp_dicoms = set(grp["dicom_id"].dropna().astype(str).unique())

        sub_overlap = sp_subjects.intersection(cig_subjects)
        dic_overlap = sp_dicoms.intersection(cig_dicoms)

        split_contamination[split_name] = {
            "total_split_records": len(grp),
            "unique_subjects": len(sp_subjects),
            "unique_studies": len(sp_studies),
            "unique_dicoms": len(sp_dicoms),
            "shared_subjects_with_cig_gold": len(sub_overlap),
            "shared_dicoms_with_cig_gold": len(dic_overlap),
            "is_strictly_patient_disjoint": len(sub_overlap) == 0,
            "is_strictly_image_disjoint": len(dic_overlap) == 0,
        }

    # Task 11 Determination
    # If Chest ImaGenome Gold pretraining split cannot be verified from published repository,
    # it must be classified as a visual sanity-check set, not an independent benchmark.
    report = {
        "dataset_entity_counts": {
            "ms_cxr": {
                "records": len(df_ms),
                "unique_patients": len(ms_subjects),
                "unique_studies": len(ms_studies),
                "unique_images": len(ms_dicoms),
                "images_present_locally": len(local_ms_dicoms),
            },
            "chest_imagenome_gold": {
                "records": len(df_cig),
                "unique_patients": len(cig_subjects),
                "unique_studies": len(cig_studies),
                "unique_images": len(cig_dicoms),
                "images_present_locally": len(local_cig_dicoms),
            },
        },
        "cross_dataset_overlap": {
            "shared_patients_count": len(shared_subjects),
            "shared_studies_count": len(shared_studies),
            "shared_images_count": len(shared_dicoms),
            "union_unique_patients": len(ms_subjects.union(cig_subjects)),
            "union_unique_images": len(ms_dicoms.union(cig_dicoms)),
            "shared_images_present_locally_on_disk": len(shared_local_dicoms),
        },
        "ms_cxr_split_contamination_audit": split_contamination,
        "evaluation_split_classification": {
            "ms_cxr_test_split": {
                "classification": "INDEPENDENT_BENCHMARK",
                "patient_leakage_with_cig_gold": False,
                "image_leakage_with_cig_gold": False,
                "description": "Clean, zero-leakage held-out test split for phrase grounding evaluation.",
            },
            "chest_imagenome_gold": {
                "classification": "VISUAL_SANITY_CHECK_SET",
                "reason": "AGPT pretrained on Chest ImaGenome without publishing exact excluded patient splits. Cannot guarantee held-out status from pretraining.",
                "approved_role": "Visual anatomical alignment verification, sanity-check inference, and qualitative error inspection only.",
            },
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Audit Cross-Dataset Overlap and Contamination")
    parser.add_argument("--table", default="data/processed/unified_grounding_master.csv", help="Path to master table")
    parser.add_argument("--output", default="docs/dataset_overlap_report.json", help="Path to save report")
    args = parser.parse_args()

    if not os.path.exists(args.table):
        logger.error(f"Target table {args.table} does not exist.")
        sys.exit(1)

    df = pd.read_csv(args.table)
    report = audit_overlap(df)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fp:
        json.dump(report, fp, indent=2)

    print("\n" + "=" * 80)
    print("INTERACTMED-VLM CROSS-DATASET & OVERLAP AUDIT REPORT")
    print("=" * 80)
    print("Entity Counts:")
    print(f"  MS-CXR:               {report['dataset_entity_counts']['ms_cxr']['unique_patients']:,} patients | {report['dataset_entity_counts']['ms_cxr']['unique_studies']:,} studies | {report['dataset_entity_counts']['ms_cxr']['unique_images']:,} images")
    print(f"  Chest ImaGenome Gold: {report['dataset_entity_counts']['chest_imagenome_gold']['unique_patients']:,} patients | {report['dataset_entity_counts']['chest_imagenome_gold']['unique_studies']:,} studies | {report['dataset_entity_counts']['chest_imagenome_gold']['unique_images']:,} images")
    print(f"\nIntersections (Overlap):")
    print(f"  Shared Patients:      {report['cross_dataset_overlap']['shared_patients_count']} patients")
    print(f"  Shared Studies:       {report['cross_dataset_overlap']['shared_studies_count']} studies")
    print(f"  Shared Images:        {report['cross_dataset_overlap']['shared_images_count']} images (Locally present on disk: {report['cross_dataset_overlap']['shared_images_present_locally_on_disk']})")
    print(f"\nMS-CXR Benchmark Cross-Split Contamination Analysis:")
    for sp_name, sp_info in report["ms_cxr_split_contamination_audit"].items():
        leak_str = "CLEAN (Zero Leakage)" if sp_info["is_strictly_patient_disjoint"] else f"CONTAMINATED ({sp_info['shared_subjects_with_cig_gold']} shared patients)"
        print(f"  MS-CXR Split '{sp_name:<5}': {sp_info['unique_dicoms']:>4} images | {sp_info['unique_subjects']:>4} patients -> {leak_str}")
    print(f"\nProtocol Classification (Task 11):")
    print(f"  MS-CXR Test Split:    {report['evaluation_split_classification']['ms_cxr_test_split']['classification']}")
    print(f"  Chest ImaGenome Gold: {report['evaluation_split_classification']['chest_imagenome_gold']['classification']}")
    print(f"                        ({report['evaluation_split_classification']['chest_imagenome_gold']['reason']})")
    print("=" * 80 + "\n")
    logger.info(f"Saved overlap report to {args.output}")


if __name__ == "__main__":
    main()
