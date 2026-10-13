#!/usr/bin/env python3
"""
scripts/prepare_local_metadata.py
=================================
InteractMed-VLM Local Metadata Preparation and Normalization Pipeline.

Builds a validated, unified local data layer from existing MS-CXR and
Chest ImaGenome Gold annotations, normalizes coordinate conventions,
merges patient demographic context, and matches against local MIMIC images
without leaking report text, image pixels, or PHI.

Outputs:
  - data/processed/unified_grounding_master.csv: Full catalog with image_exists flag.
  - data/processed/local_grounding_verified.csv: Verified subset where image_exists == True.
  - data/processed/metadata_summary.json: Aggregate counts and split statistics.

Usage:
    python scripts/prepare_local_metadata.py [--config configs/data_paths.yaml]
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import yaml

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("PrepareMetadata")

# ------------------------------------------------------------------------------
# Anatomical Region Mappings (Fine-grained to Coarse Organ Regions)
# ------------------------------------------------------------------------------
MS_CXR_PATHOLOGY_TO_ORGAN = {
    "cardiomegaly": "heart",
    "pneumothorax": "pleura",
    "pleural effusion": "pleura",
    "pneumonia": "lungs",
    "consolidation": "lungs",
    "lung opacity": "lungs",
    "atelectasis": "lungs",
    "edema": "lungs",
}

CIG_ANATOMY_TO_ORGAN = {
    # Heart and mediastinum
    "cardiac silhouette": "heart",
    "aortic arch": "heart",
    "mediastinum": "heart",
    "upper mediastinum": "heart",
    "carina": "heart",
    "svc": "heart",
    "cavoatrial junction": "heart",
    # Pleura and diaphragmatic recesses
    "left costophrenic angle": "pleura",
    "right costophrenic angle": "pleura",
    "left hemidiaphragm": "pleura",
    "right hemidiaphragm": "pleura",
    # Lungs
    "left lung": "lungs",
    "right lung": "lungs",
    "left apical zone": "lungs",
    "right apical zone": "lungs",
    "left upper lung zone": "lungs",
    "right upper lung zone": "lungs",
    "left mid lung zone": "lungs",
    "right mid lung zone": "lungs",
    "left lower lung zone": "lungs",
    "right lower lung zone": "lungs",
    "left hilar structures": "lungs",
    "right hilar structures": "lungs",
    "trachea": "lungs",
    "spine": "spine",
    "rib": "ribs",
}


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def scan_local_image_inventory(mimic_root: str) -> Dict[str, Tuple[str, Tuple[int, int]]]:
    """
    Scans local MIMIC image directory and extracts relative path and (width, height)
    via JPEG EXIF header without decompressing pixel arrays.
    Returns: dict mapping dicom_id -> (relative_path, (width, height))
    """
    inventory = {}
    if not os.path.exists(mimic_root):
        logger.warning(f"MIMIC image root not found at {mimic_root}")
        return inventory

    logger.info(f"Scanning local image headers from {mimic_root}...")
    for root, _, files in os.walk(mimic_root):
        for f in files:
            if f.lower().endswith(".jpg") and not f.startswith("."):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, mimic_root)
                dicom_id = os.path.splitext(f)[0]
                try:
                    with Image.open(full_path) as img:
                        w, h = img.size
                    inventory[dicom_id] = (rel_path, (w, h))
                except Exception as e:
                    logger.warning(f"Failed to read image header for {rel_path}: {e}")
                    inventory[dicom_id] = (rel_path, (0, 0))

    logger.info(f"Discovered {len(inventory)} valid local image files on disk.")
    return inventory


def parse_ms_cxr(
    csv_path: str,
    local_inventory: Dict[str, Tuple[str, Tuple[int, int]]],
) -> pd.DataFrame:
    """
    Parses MS-CXR dataset:
    - Input coordinates: [x, y, w, h] in native pixels.
    - Converts to [x1, y1, x2, y2].
    - Normalizes to [0, 1].
    - Extracts patient and study IDs from path.
    """
    logger.info(f"Parsing MS-CXR dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    path_pattern = re.compile(r"files/p\d+/p(\d+)/s(\d+)/([a-f0-9-]+)\.jpg")

    records = []
    for _, row in df.iterrows():
        path_str = str(row["path"])
        match = path_pattern.search(path_str)
        if match:
            subject_id = match.group(1)
            study_id = match.group(2)
        else:
            subject_id = "unknown"
            study_id = "unknown"

        dicom_id = str(row["dicom_id"]).strip()
        category = str(row["category_name"]).strip()
        phrase = str(row["label_text"]).strip()
        split = str(row["split"]).strip()

        # Coordinates
        x = float(row["x"])
        y = float(row["y"])
        w = float(row["w"])
        h = float(row["h"])
        img_w = float(row["image_width"])
        img_h = float(row["image_height"])

        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

        # Sanitize / Validate
        val_error = "none"
        if x2 <= x1 or y2 <= y1 or w <= 0 or h <= 0:
            val_error = "non_positive_dimensions"
            is_valid = False
        else:
            is_valid = True

        norm_x1 = max(0.0, min(1.0, x1 / max(1.0, img_w)))
        norm_y1 = max(0.0, min(1.0, y1 / max(1.0, img_h)))
        norm_x2 = max(0.0, min(1.0, x2 / max(1.0, img_w)))
        norm_y2 = max(0.0, min(1.0, y2 / max(1.0, img_h)))

        # Check local image existence
        exists = dicom_id in local_inventory
        local_rel_path = local_inventory[dicom_id][0] if exists else path_str

        organ = MS_CXR_PATHOLOGY_TO_ORGAN.get(category.lower(), "lungs")

        records.append({
            "dataset_source": "ms_cxr",
            "label_type": "pathology",
            "dicom_id": dicom_id,
            "subject_id": subject_id,
            "study_id": study_id,
            "split": split,
            "phrase": phrase,
            "category_name": category,
            "organ_region": organ,
            "image_width": int(img_w),
            "image_height": int(img_h),
            "orig_coord_format": "xywh",
            "orig_x1": round(x1, 2),
            "orig_y1": round(y1, 2),
            "orig_x2": round(x2, 2),
            "orig_y2": round(y2, 2),
            "orig_w": round(w, 2),
            "orig_h": round(h, 2),
            "norm_x1": round(norm_x1, 5),
            "norm_y1": round(norm_y1, 5),
            "norm_x2": round(norm_x2, 5),
            "norm_y2": round(norm_y2, 5),
            "is_valid_bbox": is_valid,
            "validation_error": val_error,
            "image_exists": exists,
            "image_path": local_rel_path if exists else None,
            "gender": None,
            "age_decile": None,
            "view_position": None,
        })

    out_df = pd.DataFrame(records)
    logger.info(f"Parsed {len(out_df)} MS-CXR records (Matched local images: {out_df['image_exists'].sum()}).")
    return out_df


def parse_chest_imagenome_gold(
    bbox_csv: str,
    record_list_csv: str,
    study_list_csv: str,
    scaling_csv: str,
    local_inventory: Dict[str, Tuple[str, Tuple[int, int]]],
) -> pd.DataFrame:
    """
    Parses Chest ImaGenome Gold dataset:
    - Input coordinates: [original_x1, original_y1, original_x2, original_y2] and 224x224 scaled.
    - Merges patient demographics (gender, age_decile) from study_list.
    - Merges view position and path from record_list.
    - Cleans flipped coordinates and boundary padding offsets.
    - Marks split as 'gold_sanity_check' per Task 11.
    """
    logger.info(f"Parsing Chest ImaGenome Gold from {bbox_csv}...")
    df_bbox = pd.read_csv(bbox_csv)
    df_bbox["dicom_id"] = df_bbox["image_id"].astype(str).str.replace(".dcm", "", regex=False).str.strip()

    # Load master metadata for linking
    df_rec = pd.read_csv(record_list_csv)
    df_rec["dicom_id"] = df_rec["dicom_id"].astype(str).str.strip()
    df_rec["path"] = df_rec["path"].str.replace(".dcm", ".jpg", regex=False)

    df_study = pd.read_csv(study_list_csv)

    # Merge
    merged = pd.merge(
        df_bbox,
        df_rec[["dicom_id", "subject_id", "study_id", "path", "ViewPosition"]],
        on="dicom_id",
        how="left",
    )
    merged = pd.merge(
        merged,
        df_study[["study_id", "gender", "age_decile"]],
        on="study_id",
        how="left",
    )

    records = []
    for _, row in merged.iterrows():
        dicom_id = str(row["dicom_id"])
        subject_id = str(row.get("subject_id", "unknown"))
        study_id = str(row.get("study_id", "unknown"))
        bbox_name = str(row["bbox_name"]).strip()
        path_str = str(row.get("path", ""))

        exists = dicom_id in local_inventory
        local_rel_path = local_inventory[dicom_id][0] if exists else path_str

        # Get image dimensions: prefer true image dimensions if file exists, else use 224 or scaling
        if exists:
            img_w, img_h = local_inventory[dicom_id][1]
        else:
            # Estimate from native original bbox extents or fallback
            img_w = 224
            img_h = 224

        # Native coordinates in CIG
        ox1 = float(row["original_x1"])
        oy1 = float(row["original_y1"])
        ox2 = float(row["original_x2"])
        oy2 = float(row["original_y2"])

        # 224 coordinates in CIG (pre-scaled and padded)
        x1_224 = float(row["x1"])
        y1_224 = float(row["y1"])
        x2_224 = float(row["x2"])
        y2_224 = float(row["y2"])

        val_error = "none"

        # Check for flipped coordinates (e.g. annotator flipped x1 and x2)
        if ox2 < ox1:
            ox1, ox2 = ox2, ox1
            x1_224, x2_224 = x2_224, x1_224
            val_error = "flipped_x_corrected"
        if oy2 < oy1:
            oy1, oy2 = oy2, oy1
            y1_224, y2_224 = y2_224, y1_224
            val_error = "flipped_y_corrected" if val_error == "none" else val_error + "_and_y"

        # Check for negative offsets caused by aspect ratio padding
        if ox1 < 0:
            ox1 = 0.0
            val_error = "negative_x1_clamped"
        if oy1 < 0:
            oy1 = 0.0
            val_error = "negative_y1_clamped"

        # Width and height
        orig_w = ox2 - ox1
        orig_h = oy2 - oy1

        is_valid = (orig_w > 0) and (orig_h > 0) and (x2_224 > x1_224) and (y2_224 > y1_224)
        if not is_valid and val_error == "none":
            val_error = "zero_or_negative_area"

        # Normalized coordinates: computed from the 224 square bounding box
        norm_x1 = max(0.0, min(1.0, x1_224 / 224.0))
        norm_y1 = max(0.0, min(1.0, y1_224 / 224.0))
        norm_x2 = max(0.0, min(1.0, x2_224 / 224.0))
        norm_y2 = max(0.0, min(1.0, y2_224 / 224.0))

        organ = CIG_ANATOMY_TO_ORGAN.get(bbox_name.lower(), "lungs")

        records.append({
            "dataset_source": "chest_imagenome_gold",
            "label_type": "anatomy",
            "dicom_id": dicom_id,
            "subject_id": subject_id,
            "study_id": study_id,
            "split": "gold_sanity_check",  # Classified as sanity-check per Task 11
            "phrase": bbox_name,
            "category_name": bbox_name,
            "organ_region": organ,
            "image_width": int(img_w),
            "image_height": int(img_h),
            "orig_coord_format": "xyxy",
            "orig_x1": round(ox1, 2),
            "orig_y1": round(oy1, 2),
            "orig_x2": round(ox2, 2),
            "orig_y2": round(oy2, 2),
            "orig_w": round(orig_w, 2),
            "orig_h": round(orig_h, 2),
            "norm_x1": round(norm_x1, 5),
            "norm_y1": round(norm_y1, 5),
            "norm_x2": round(norm_x2, 5),
            "norm_y2": round(norm_y2, 5),
            "is_valid_bbox": is_valid,
            "validation_error": val_error,
            "image_exists": exists,
            "image_path": local_rel_path if exists else None,
            "gender": row.get("gender"),
            "age_decile": row.get("age_decile"),
            "view_position": row.get("ViewPosition"),
        })

    out_df = pd.DataFrame(records)
    logger.info(f"Parsed {len(out_df)} Chest ImaGenome Gold records (Matched local images: {out_df['image_exists'].sum()}).")
    return out_df


def main():
    parser = argparse.ArgumentParser(description="Prepare validated local metadata for InteractMed-VLM")
    parser.add_argument("--config", default="configs/data_paths.yaml", help="Path to data paths configuration")
    parser.add_argument("--output-dir", default="data/processed", help="Directory to save processed metadata")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    mimic_dir = cfg.get("mimic_cxr", {}).get("root_dir", "/Users/adityakachhot/Downloads/data/mimic-cxr")
    ms_csv = cfg.get("ms_cxr", {}).get("files", {}).get("csv", "/Users/adityakachhot/Downloads/data/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.csv")

    cig_bbox_csv = cfg.get("chest_imagenome", {}).get("files", {}).get("gold_bbox_csv", "/Users/adityakachhot/Downloads/data/chest-imagenome/gold_dataset/gold_bbox_coordinate_annotations_1000images.csv")
    cig_rec_csv = cfg.get("chest_imagenome", {}).get("files", {}).get("cxr_record_list_view", "/Users/adityakachhot/Downloads/chest-imagenome-dataset-1.0.0/utils/cxr-record-list_view.csv")
    cig_study_csv = cfg.get("chest_imagenome", {}).get("files", {}).get("cxr_study_list_v2", "/Users/adityakachhot/Downloads/chest-imagenome-dataset-1.0.0/utils/cxr-study-list_v2.csv")
    cig_scale_csv = cfg.get("chest_imagenome", {}).get("files", {}).get("gold_scaling_csv", "/Users/adityakachhot/Downloads/data/chest-imagenome/gold_dataset/gold_bbox_scaling_factors_original_to_224x224.csv")

    # 1. Scan Local Image Inventory
    local_inventory = scan_local_image_inventory(mimic_dir)

    # 2. Parse Datasets
    df_ms = parse_ms_cxr(ms_csv, local_inventory)
    df_cig = parse_chest_imagenome_gold(cig_bbox_csv, cig_rec_csv, cig_study_csv, cig_scale_csv, local_inventory)

    # 3. Concatenate Master Catalog
    master_df = pd.concat([df_ms, df_cig], ignore_index=True)
    master_df["row_id"] = master_df.index

    # 4. Filter Verified Local Records
    local_df = master_df[master_df["image_exists"] == True].copy().reset_index(drop=True)

    # 5. Save Tables
    master_path = os.path.join(args.output_dir, "unified_grounding_master.csv")
    local_path = os.path.join(args.output_dir, "local_grounding_verified.csv")

    master_df.to_csv(master_path, index=False)
    local_df.to_csv(local_path, index=False)
    logger.info(f"Saved master catalog to {master_path} ({len(master_df):,} rows)")
    logger.info(f"Saved local verified subset to {local_path} ({len(local_df):,} rows)")

    # 6. Generate Summary Statistics
    summary = {
        "execution_summary": {
            "total_master_records": len(master_df),
            "total_local_verified_records": len(local_df),
            "unmatched_records_lacking_images": len(master_df) - len(local_df),
            "local_match_percentage": round(100.0 * len(local_df) / max(1, len(master_df)), 2),
        },
        "by_dataset": {
            "ms_cxr": {
                "total_records": len(df_ms),
                "matched_local_records": int(df_ms["image_exists"].sum()),
                "unmatched_records": int((~df_ms["image_exists"]).sum()),
                "unique_dicoms_total": df_ms["dicom_id"].nunique(),
                "unique_dicoms_local": df_ms[df_ms["image_exists"]]["dicom_id"].nunique(),
                "unique_subjects": df_ms["subject_id"].nunique(),
                "valid_bboxes_count": int(df_ms["is_valid_bbox"].sum()),
                "splits": df_ms.groupby("split")["image_exists"].agg(["count", "sum"]).rename(columns={"count": "total", "sum": "local"}).to_dict(orient="index"),
            },
            "chest_imagenome_gold": {
                "total_records": len(df_cig),
                "matched_local_records": int(df_cig["image_exists"].sum()),
                "unmatched_records": int((~df_cig["image_exists"]).sum()),
                "unique_dicoms_total": df_cig["dicom_id"].nunique(),
                "unique_dicoms_local": df_cig[df_cig["image_exists"]]["dicom_id"].nunique(),
                "unique_subjects": df_cig["subject_id"].nunique(),
                "valid_bboxes_count": int(df_cig["is_valid_bbox"].sum()),
                "classification": "visual_sanity_check_set",
            },
        },
        "coordinate_validity": {
            "total_valid_bboxes": int(master_df["is_valid_bbox"].sum()),
            "total_anomalous_bboxes": int((~master_df["is_valid_bbox"]).sum()),
            "validation_errors_distribution": master_df["validation_error"].value_counts().to_dict(),
        },
        "patient_demographics_coverage": {
            "records_with_gender": int(master_df["gender"].notna().sum()),
            "records_with_age_decile": int(master_df["age_decile"].notna().sum()),
        },
    }

    summary_path = os.path.join(args.output_dir, "metadata_summary.json")
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    logger.info(f"Saved metadata summary to {summary_path}")

    # Terminal output report
    print("\n" + "=" * 80)
    print("INTERACTMED-VLM LOCAL DATA LAYER PREPARATION SUMMARY")
    print("=" * 80)
    print(f"Master Catalog Records:           {len(master_df):,}")
    print(f"Local Verified Records on Disk:   {len(local_df):,} ({summary['execution_summary']['local_match_percentage']}%)")
    print(f"Unmatched Records Lacking Images: {len(master_df) - len(local_df):,}")
    print("\nBreakdown by Dataset:")
    print(f"  MS-CXR:")
    print(f"    Total Records:                {len(df_ms):,} ({df_ms['dicom_id'].nunique():,} unique images)")
    print(f"    Matched Local Records:        {df_ms['image_exists'].sum():,} ({df_ms[df_ms['image_exists']]['dicom_id'].nunique():,} unique images)")
    print(f"    Valid BBoxes:                 {df_ms['is_valid_bbox'].sum():,} / {len(df_ms):,}")
    print(f"  Chest ImaGenome Gold:")
    print(f"    Total Records:                {len(df_cig):,} ({df_cig['dicom_id'].nunique():,} unique images)")
    print(f"    Matched Local Records:        {df_cig['image_exists'].sum():,} ({df_cig[df_cig['image_exists']]['dicom_id'].nunique():,} unique images)")
    print(f"    Valid BBoxes:                 {df_cig['is_valid_bbox'].sum():,} / {len(df_cig):,}")
    print(f"    Split Classification:         visual_sanity_check_set (unverified pretraining split)")
    print(f"\nCoordinate Integrity:")
    print(f"    Anomalies Handled/Flagged:    {summary['coordinate_validity']['validation_errors_distribution']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
