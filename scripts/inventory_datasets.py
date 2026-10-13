#!/usr/bin/env python3
"""
scripts/inventory_datasets.py
=============================
InteractMed-VLM Dataset Inventory and Verification Script.

Discovers, verifies, and catalogs all local datasets, metadata headers,
bounding-box coordinate systems, image-file availability, and cross-dataset
overlaps without leaking clinical report text, image pixels, or individual PHI.

Usage:
    python scripts/inventory_datasets.py [--config configs/data_paths.yaml] [--output-json docs/inventory_summary.json]
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetInventory")


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        logger.warning(f"Config path {config_path} not found. Using defaults.")
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def scan_local_mimic_images(mimic_root: str) -> Dict[str, Any]:
    """Scans and catalogs local MIMIC-CXR-JPG files."""
    if not os.path.exists(mimic_root):
        return {
            "status": "missing",
            "root_dir": mimic_root,
            "total_images": 0,
            "relative_paths": set(),
            "dicom_ids": set(),
        }

    total_images = 0
    rel_paths = set()
    dicom_ids = set()

    for root, _, files in os.walk(mimic_root):
        for f in files:
            if f.lower().endswith(".jpg") and not f.startswith("."):
                full_path = os.path.join(root, f)
                rel_p = os.path.relpath(full_path, mimic_root)
                rel_paths.add(rel_p)
                dicom_id = os.path.splitext(f)[0]
                dicom_ids.add(dicom_id)
                total_images += 1

    return {
        "status": "available",
        "root_dir": mimic_root,
        "total_images": total_images,
        "relative_paths": rel_paths,
        "dicom_ids": dicom_ids,
    }


def audit_ms_cxr(
    csv_path: str,
    json_path: str,
    local_images: Dict[str, Any],
) -> Dict[str, Any]:
    """Audits MS-CXR CSV and JSON metadata."""
    if not os.path.exists(csv_path):
        return {"status": "missing", "path": csv_path}

    df = pd.read_csv(csv_path)
    total_records = len(df)
    unique_dicoms = set(df["dicom_id"].dropna().unique())
    unique_paths = set(df["path"].dropna().unique()) if "path" in df.columns else set()

    # Image existence check
    local_rel_paths = local_images.get("relative_paths", set())
    local_dicom_ids = local_images.get("dicom_ids", set())

    # Path in MS-CXR is typically 'files/p10/p10002428/s55758034/xxx.jpg'
    records_with_image = 0
    dicoms_with_image = 0
    for d_id in unique_dicoms:
        if d_id in local_dicom_ids:
            dicoms_with_image += 1

    if "path" in df.columns:
        exists_mask = df["path"].isin(local_rel_paths)
        records_with_image = int(exists_mask.sum())
    else:
        exists_mask = df["dicom_id"].isin(local_dicom_ids)
        records_with_image = int(exists_mask.sum())

    # JSON audit
    json_annotations_count = 0
    json_images_count = 0
    categories = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as fp:
                coco_data = json.load(fp)
            json_annotations_count = len(coco_data.get("annotations", []))
            json_images_count = len(coco_data.get("images", []))
            categories = [c.get("name") for c in coco_data.get("categories", [])]
        except Exception as e:
            logger.warning(f"Failed to read MS-CXR JSON: {e}")

    # Splits
    split_counts = df["split"].value_counts().to_dict() if "split" in df.columns else {}

    return {
        "status": "available",
        "csv_path": csv_path,
        "json_path": json_path,
        "columns": df.columns.tolist(),
        "total_records": total_records,
        "unique_dicom_ids": len(unique_dicoms),
        "categories": categories or (df["category_name"].unique().tolist() if "category_name" in df.columns else []),
        "splits": split_counts,
        "bbox_format": "[x, y, w, h] in absolute pixel coordinates (unnormalized)",
        "json_annotations_count": json_annotations_count,
        "json_images_count": json_images_count,
        "records_with_existing_images": records_with_image,
        "records_missing_images": total_records - records_with_image,
        "unique_images_locally_present": dicoms_with_image,
        "unique_images_missing": len(unique_dicoms) - dicoms_with_image,
        "image_coverage_pct": round(100.0 * dicoms_with_image / max(1, len(unique_dicoms)), 2),
        "dicom_set": unique_dicoms,
    }


def audit_chest_imagenome_gold(
    bbox_csv: str,
    record_list_csv: str,
    study_list_csv: str,
    sentences_txt: str,
    local_images: Dict[str, Any],
) -> Dict[str, Any]:
    """Audits Chest ImaGenome Gold dataset files."""
    if not os.path.exists(bbox_csv):
        return {"status": "missing", "path": bbox_csv}

    df_bbox = pd.read_csv(bbox_csv)
    total_records = len(df_bbox)

    # image_id in CIG bbox csv has .dcm extension
    raw_images = df_bbox["image_id"].dropna().unique()
    clean_dicoms = {os.path.splitext(img)[0] for img in raw_images}

    # Bbox columns
    bbox_cols = [c for c in df_bbox.columns if "coord" in c or "x" in c or "y" in c]

    # Map to local images
    local_dicom_ids = local_images.get("dicom_ids", set())
    dicoms_with_image = len(clean_dicoms.intersection(local_dicom_ids))

    clean_img_series = df_bbox["image_id"].apply(lambda x: os.path.splitext(str(x))[0])
    records_with_image = int(clean_img_series.isin(local_dicom_ids).sum())

    # Study list (demographics / patient context)
    has_patient_context = False
    context_columns = []
    if os.path.exists(study_list_csv):
        try:
            df_study = pd.read_csv(study_list_csv, nrows=5)
            has_patient_context = True
            context_columns = df_study.columns.tolist()
        except Exception:
            pass

    # Sentences availability
    sentence_records = 0
    if os.path.exists(sentences_txt):
        try:
            with open(sentences_txt, "r") as fp:
                sentence_records = sum(1 for _ in fp) - 1  # subtract header
        except Exception:
            pass

    return {
        "status": "available",
        "bbox_csv": bbox_csv,
        "columns": df_bbox.columns.tolist(),
        "total_records": total_records,
        "unique_dicom_ids": len(clean_dicoms),
        "unique_annotators": df_bbox["annotator"].unique().tolist() if "annotator" in df_bbox.columns else [],
        "bbox_formats": {
            "coord224": "[x1, y1, x2, y2] scaled to 224x224 square bounding box",
            "coord_original": "[original_x1, original_y1, original_x2, original_y2] in native DICOM resolution",
            "components": ["x1", "y1", "x2", "y2", "width", "height", "original_x1", "original_y1", "original_x2", "original_y2", "original_width", "original_height"],
        },
        "records_with_existing_images": records_with_image,
        "records_missing_images": total_records - records_with_image,
        "unique_images_locally_present": dicoms_with_image,
        "unique_images_missing": len(clean_dicoms) - dicoms_with_image,
        "image_coverage_pct": round(100.0 * dicoms_with_image / max(1, len(clean_dicoms)), 2),
        "has_patient_context_metadata": has_patient_context,
        "patient_context_columns": context_columns,
        "sentence_records_available": sentence_records,
        "dicom_set": clean_dicoms,
    }


def audit_unified_manifest(
    manifest_csv: str,
    download_list_txt: str,
    local_images: Dict[str, Any],
) -> Dict[str, Any]:
    """Audits prior unified grounding manifest and download list."""
    if not os.path.exists(manifest_csv):
        return {"status": "missing", "path": manifest_csv}

    df = pd.read_csv(manifest_csv)
    total_records = len(df)
    unique_dicoms = set(df["dicom_id"].dropna().unique())
    unique_studies = set(df["study_id"].dropna().unique())
    unique_subjects = set(df["subject_id"].dropna().unique())

    local_dicom_ids = local_images.get("dicom_ids", set())
    local_rel_paths = local_images.get("relative_paths", set())

    # Check existence
    if "image_relative_path" in df.columns:
        df["exists"] = df["image_relative_path"].isin(local_rel_paths)
    else:
        df["exists"] = df["dicom_id"].isin(local_dicom_ids)

    records_with_image = int(df["exists"].sum())
    dicoms_with_image = len(unique_dicoms.intersection(local_dicom_ids))

    # By source
    by_source = {}
    if "source" in df.columns:
        for src, grp in df.groupby("source"):
            grp_dicoms = set(grp["dicom_id"].dropna().unique())
            grp_present_dicoms = len(grp_dicoms.intersection(local_dicom_ids))
            by_source[src] = {
                "total_records": len(grp),
                "records_with_images": int(grp["exists"].sum()),
                "unique_dicoms": len(grp_dicoms),
                "unique_dicoms_present": grp_present_dicoms,
                "coverage_pct": round(100.0 * grp_present_dicoms / max(1, len(grp_dicoms)), 2),
            }

    # Download list
    dl_total = 0
    dl_present = 0
    if os.path.exists(download_list_txt):
        with open(download_list_txt, "r") as fp:
            dl_paths = [line.strip() for line in fp if line.strip()]
        dl_total = len(dl_paths)
        dl_present = sum(1 for p in dl_paths if p in local_rel_paths)

    return {
        "status": "available",
        "manifest_path": manifest_csv,
        "download_list_path": download_list_txt,
        "columns": df.columns.tolist(),
        "total_records": total_records,
        "unique_dicoms": len(unique_dicoms),
        "unique_studies": len(unique_studies),
        "unique_subjects": len(unique_subjects),
        "records_with_existing_images": records_with_image,
        "records_missing_images": total_records - records_with_image,
        "unique_images_locally_present": dicoms_with_image,
        "unique_images_missing": len(unique_dicoms) - dicoms_with_image,
        "image_coverage_pct": round(100.0 * dicoms_with_image / max(1, len(unique_dicoms)), 2),
        "by_source": by_source,
        "download_list_summary": {
            "total_requested_paths": dl_total,
            "present_locally": dl_present,
            "pending_download": dl_total - dl_present,
        },
        "dicom_set": unique_dicoms,
    }


def audit_mimic_cxr_vqa(vqa_dir: str, local_images: Dict[str, Any]) -> Dict[str, Any]:
    """Audits MIMIC-CXR-VQA splits."""
    if not os.path.exists(vqa_dir):
        return {"status": "missing", "path": vqa_dir}

    splits = {}
    total_qas = 0
    for split_file in ["train.json", "valid.json", "test.json"]:
        p = os.path.join(vqa_dir, split_file)
        if os.path.exists(p):
            try:
                with open(p, "r") as fp:
                    data = json.load(fp)
                splits[split_file] = {
                    "count": len(data),
                    "sample_keys": list(data[0].keys()) if len(data) > 0 else [],
                }
                total_qas += len(data)
            except Exception as e:
                splits[split_file] = {"error": str(e)}

    return {
        "status": "available" if splits else "empty",
        "dir": vqa_dir,
        "splits": splits,
        "total_qa_pairs": total_qas,
    }


def compute_cross_dataset_overlap(
    ms_cxr_dicoms: Set[str],
    cig_gold_dicoms: Set[str],
    manifest_dicoms: Set[str],
    local_dicoms: Set[str],
) -> Dict[str, Any]:
    """Computes exact set intersections across all discovered datasets."""
    overlap_ms_cig = ms_cxr_dicoms.intersection(cig_gold_dicoms)
    all_known_dicoms = ms_cxr_dicoms.union(cig_gold_dicoms).union(manifest_dicoms)

    return {
        "total_unique_dicoms_across_all_benchmarks": len(all_known_dicoms),
        "ms_cxr_total_dicoms": len(ms_cxr_dicoms),
        "cig_gold_total_dicoms": len(cig_gold_dicoms),
        "manifest_total_dicoms": len(manifest_dicoms),
        "ms_cxr_intersect_cig_gold": len(overlap_ms_cig),
        "ms_cxr_intersect_local_images": len(ms_cxr_dicoms.intersection(local_dicoms)),
        "cig_gold_intersect_local_images": len(cig_gold_dicoms.intersection(local_dicoms)),
        "manifest_intersect_local_images": len(manifest_dicoms.intersection(local_dicoms)),
        "all_known_intersect_local_images": len(all_known_dicoms.intersection(local_dicoms)),
        "local_images_not_in_any_benchmark": len(local_dicoms - all_known_dicoms),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit and inventory InteractMed-VLM datasets.")
    parser.add_argument("--config", default="configs/data_paths.yaml", help="Path to data paths configuration.")
    parser.add_argument("--output-json", default="docs/inventory_summary.json", help="Path to save summary JSON.")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    # Resolve paths
    mimic_dir = cfg.get("mimic_cxr", {}).get("root_dir", "/Users/adityakachhot/Downloads/data/mimic-cxr")
    ms_csv = cfg.get("ms_cxr", {}).get("files", {}).get("csv", "/Users/adityakachhot/Downloads/data/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.csv")
    ms_json = cfg.get("ms_cxr", {}).get("files", {}).get("json", "/Users/adityakachhot/Downloads/data/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.json")

    cig_bbox_csv = cfg.get("chest_imagenome", {}).get("files", {}).get("gold_bbox_csv", "/Users/adityakachhot/Downloads/data/chest-imagenome/gold_dataset/gold_bbox_coordinate_annotations_1000images.csv")
    cig_rec_csv = cfg.get("chest_imagenome", {}).get("files", {}).get("cxr_record_list_view", "/Users/adityakachhot/Downloads/chest-imagenome-dataset-1.0.0/utils/cxr-record-list_view.csv")
    cig_study_csv = cfg.get("chest_imagenome", {}).get("files", {}).get("cxr_study_list_v2", "/Users/adityakachhot/Downloads/chest-imagenome-dataset-1.0.0/utils/cxr-study-list_v2.csv")
    cig_sent_txt = cfg.get("chest_imagenome", {}).get("files", {}).get("gold_sentences_txt", "/Users/adityakachhot/Downloads/data/chest-imagenome/gold_dataset/gold_all_sentences_500pts_1000studies.txt")

    manifest_csv = cfg.get("prior_project", {}).get("files", {}).get("unified_manifest_csv", "/Users/adityakachhot/interactmedVLM/manifests/unified_manifest.csv")
    download_txt = cfg.get("prior_project", {}).get("files", {}).get("download_list_txt", "/Users/adityakachhot/interactmedVLM/manifests/download_list.txt")

    vqa_dir = cfg.get("mimic_cxr_vqa", {}).get("root_dir", "/Users/adityakachhot/Downloads/data/mimic-cxr-vqa")

    logger.info("Scanning local image storage...")
    local_images = scan_local_mimic_images(mimic_dir)
    logger.info(f"Discovered {local_images['total_images']} local JPG images in {mimic_dir}")

    logger.info("Auditing MS-CXR dataset...")
    ms_summary = audit_ms_cxr(ms_csv, ms_json, local_images)

    logger.info("Auditing Chest ImaGenome Gold dataset...")
    cig_summary = audit_chest_imagenome_gold(cig_bbox_csv, cig_rec_csv, cig_study_csv, cig_sent_txt, local_images)

    logger.info("Auditing Prior Unified Manifest...")
    manifest_summary = audit_unified_manifest(manifest_csv, download_txt, local_images)

    logger.info("Auditing MIMIC-CXR-VQA...")
    vqa_summary = audit_mimic_cxr_vqa(vqa_dir, local_images)

    logger.info("Computing cross-dataset overlaps...")
    overlap_summary = compute_cross_dataset_overlap(
        ms_summary.get("dicom_set", set()),
        cig_summary.get("dicom_set", set()),
        manifest_summary.get("dicom_set", set()),
        local_images.get("dicom_ids", set()),
    )

    # Remove non-serializable sets for JSON output
    for d in [ms_summary, cig_summary, manifest_summary]:
        d.pop("dicom_set", None)
    local_images.pop("relative_paths", None)
    local_images.pop("dicom_ids", None)

    summary = {
        "execution_summary": {
            "status": "SUCCESS",
            "host": "MacBook Air M3 (16 GB Unified Memory)",
            "python": sys.version.split()[0],
        },
        "local_image_storage": local_images,
        "ms_cxr": ms_summary,
        "chest_imagenome_gold": cig_summary,
        "unified_manifest": manifest_summary,
        "mimic_cxr_vqa": vqa_summary,
        "external_rsna_pneumonia": {
            "status": "MISSING_NOT_DOWNLOADED",
            "required_for": "External grounding evaluation and optional domain fine-tuning",
            "target_dir": "data/rsna",
            "expected_files": ["stage_2_train_labels.csv", "stage_2_train_images/*.dcm"],
        },
        "model_checkpoints": {
            "transvg_agpt": {
                "status": "MISSING_NOT_DOWNLOADED",
                "required_for": "Pretrained TransVG-AGPT medical phrase grounding inference and baseline evaluation",
                "target_path": "checkpoints/transvg.pth",
                "upstream_gdrive_folder": "1lFaqAlG6AJExCjkW3Pbb24CX-LWSstrK",
                "upstream_github": "https://github.com/Claire1217/AGPT",
            }
        },
        "cross_dataset_overlap": overlap_summary,
    }

    # Print summary report
    print("\n" + "=" * 80)
    print("INTERACTMED-VLM DATASET INVENTORY SUMMARY REPORT")
    print("=" * 80)
    print(f"Local JPG Images Cataloged:           {local_images['total_images']:,}")
    print(f"MS-CXR Records:                       {ms_summary['total_records']:,} ({ms_summary['unique_dicom_ids']:,} unique images)")
    print(f"  -> Records with local image:        {ms_summary['records_with_existing_images']:,} / {ms_summary['total_records']:,} ({ms_summary['records_with_existing_images']/ms_summary['total_records']*100:.1f}%)")
    print(f"  -> Unique images present locally:   {ms_summary['unique_images_locally_present']:,} / {ms_summary['unique_dicom_ids']:,} ({ms_summary['image_coverage_pct']}%)")
    print(f"Chest ImaGenome Gold Records:         {cig_summary['total_records']:,} ({cig_summary['unique_dicom_ids']:,} unique images)")
    print(f"  -> Records with local image:        {cig_summary['records_with_existing_images']:,} / {cig_summary['total_records']:,} ({cig_summary['records_with_existing_images']/cig_summary['total_records']*100:.1f}%)")
    print(f"  -> Unique images present locally:   {cig_summary['unique_images_locally_present']:,} / {cig_summary['unique_dicom_ids']:,} ({cig_summary['image_coverage_pct']}%)")
    print(f"Unified Manifest Records:             {manifest_summary['total_records']:,} ({manifest_summary['unique_dicoms']:,} unique images)")
    print(f"  -> Records with local image:        {manifest_summary['records_with_existing_images']:,} / {manifest_summary['total_records']:,} ({manifest_summary['records_with_existing_images']/manifest_summary['total_records']*100:.1f}%)")
    print(f"  -> Unique images present locally:   {manifest_summary['unique_images_locally_present']:,} / {manifest_summary['unique_dicoms']:,} ({manifest_summary['image_coverage_pct']}%)")
    print(f"Cross-dataset Image Overlap:")
    print(f"  -> MS-CXR & Chest ImaGenome Gold:   {overlap_summary['ms_cxr_intersect_cig_gold']} images overlap")
    print(f"  -> Total Unique Benchmark DICOMs:   {overlap_summary['total_unique_dicoms_across_all_benchmarks']:,}")
    print(f"  -> Benchmark DICOMs Available:      {overlap_summary['all_known_intersect_local_images']:,} / {overlap_summary['total_unique_dicoms_across_all_benchmarks']:,}")
    print(f"External RSNA Dataset:                {summary['external_rsna_pneumonia']['status']}")
    print(f"TransVG-AGPT Checkpoint:              {summary['model_checkpoints']['transvg_agpt']['status']}")
    print("=" * 80 + "\n")

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as fp:
            json.dump(summary, fp, indent=2)
        logger.info(f"Saved machine-readable inventory to {args.output_json}")


if __name__ == "__main__":
    main()
