"""
build_manifest.py
=================
Unified dataset manifest builder and GCS image download list generator for InteractMed-VLM.

Harmonizes annotations from:
1. MS-CXR: phrase-grounded bounding boxes with COCO coordinates.
2. Chest ImaGenome Gold: expert-annotated anatomical scene graph bounding boxes and sentences.

Outputs:
- manifests/unified_manifest.csv: Harmonized annotations with standard schema:
    [dicom_id, subject_id, study_id, image_relative_path, region_label, bbox, finding_text, split, source]
- manifests/download_list.txt: Deduplicated GCS object paths for targeted MIMIC-CXR-JPG download:
    files/p{prefix}/p{subject_id}/s{study_id}/{dicom_id}.jpg
"""

import argparse
import ast
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Anatomical Region Mapping Dictionary
# Maps detailed anatomical entities to high-level regions: 'heart', 'lungs', 'pleura'
# ------------------------------------------------------------------------------
MS_CXR_CATEGORY_TO_REGION = {
    "cardiomegaly": "heart",
    "pneumothorax": "pleura",
    "pleural effusion": "pleura",
    "pneumonia": "lungs",
    "consolidation": "lungs",
    "lung opacity": "lungs",
    "atelectasis": "lungs",
    "edema": "lungs",
}

IMAGENOME_ANATOMY_TO_REGION = {
    # Heart & mediastinum
    "cardiac silhouette": "heart",
    "aortic arch": "heart",
    "mediastinum": "heart",
    "upper mediastinum": "heart",
    "carina": "heart",
    "svc": "heart",
    # Pleural space
    "left costophrenic angle": "pleura",
    "right costophrenic angle": "pleura",
    "left hemidiaphragm": "pleura",
    "right hemidiaphragm": "pleura",
    # Lungs & parenchyma
    "left lung": "lungs",
    "right lung": "lungs",
    "left upper lung zone": "lungs",
    "right upper lung zone": "lungs",
    "left mid lung zone": "lungs",
    "right mid lung zone": "lungs",
    "left lower lung zone": "lungs",
    "right lower lung zone": "lungs",
    "left hilar structures": "lungs",
    "right hilar structures": "lungs",
    "left apical zone": "lungs",
    "right apical zone": "lungs",
    "trachea": "lungs",
}


def map_region_label(raw_label: str) -> str:
    """
    Maps fine-grained anatomical labels or disease categories to one of
    the three primary pipeline regions: 'heart', 'lungs', 'pleura'.
    Defaults to 'lungs' if not recognized.
    """
    clean = str(raw_label).strip().lower()
    if clean in MS_CXR_CATEGORY_TO_REGION:
        return MS_CXR_CATEGORY_TO_REGION[clean]
    if clean in IMAGENOME_ANATOMY_TO_REGION:
        return IMAGENOME_ANATOMY_TO_REGION[clean]

    # Substring heuristics
    if any(k in clean for k in ["cardiac", "heart", "aort", "mediastin"]):
        return "heart"
    if any(k in clean for k in ["pleur", "effusion", "costophrenic", "pneumothorax"]):
        return "pleura"
    return "lungs"


def format_gcs_relative_path(subject_id: Union[int, str], study_id: Union[int, str], dicom_id: str) -> str:
    """
    Formats the canonical MIMIC-CXR-JPG object relative path:
    files/p{prefix}/p{subject_id}/s{study_id}/{dicom_id}.jpg
    """
    subj = str(subject_id).strip().lstrip("p")
    study = str(study_id).strip().lstrip("s")
    dicom = str(dicom_id).strip().replace(".dcm", "").replace(".jpg", "")
    prefix = subj[:2] if len(subj) >= 2 else "10"
    return f"files/p{prefix}/p{subj}/s{study}/{dicom}.jpg"


def parse_ms_cxr_path(path_str: str) -> Tuple[str, str, str]:
    """
    Extracts (subject_id, study_id, dicom_id) from MS-CXR path string:
    e.g., 'files/p10/p10233088/s54276838/675d792f-a3521e48-5eec8573-1e81d644-e60c34f8.jpg'
    """
    parts = str(path_str).split("/")
    subj = ""
    study = ""
    dicom = ""
    for part in parts:
        if part.startswith("p") and len(part) > 3 and part[1:].isdigit():
            subj = part.lstrip("p")
        elif part.startswith("s") and part[1:].isdigit():
            study = part.lstrip("s")
        elif ".jpg" in part:
            dicom = part.replace(".jpg", "")

    return subj, study, dicom


def load_ms_cxr(
    csv_path: str,
    json_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Loads MS-CXR CSV/JSON annotations and formats into intermediate DataFrame.
    Expected CSV columns:
      ['dicom_id', 'category_name', 'label_text', 'path', 'x', 'y', 'w', 'h', 'image_width', 'image_height', 'split']
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"MS-CXR CSV not found: {csv_path}")

    logger.info(f"Loading MS-CXR from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = ["dicom_id", "category_name", "label_text", "path", "x", "y", "w", "h"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required MS-CXR column: {col}. Found: {df.columns.tolist()}")

    records = []
    for _, row in df.iterrows():
        subj, study, dicom_from_path = parse_ms_cxr_path(row["path"])
        dicom = str(row["dicom_id"]).strip() or dicom_from_path

        # Standardize bbox as [x, y, w, h]
        bbox = [
            round(float(row["x"]), 2),
            round(float(row["y"]), 2),
            round(float(row["w"]), 2),
            round(float(row["h"]), 2),
        ]

        region_label = map_region_label(row["category_name"])
        split = str(row["split"]) if "split" in row and pd.notna(row["split"]) else "train"
        rel_path = format_gcs_relative_path(subj, study, dicom)

        records.append({
            "dicom_id": dicom,
            "subject_id": subj,
            "study_id": study,
            "image_relative_path": rel_path,
            "region_label": region_label,
            "raw_region": str(row["category_name"]),
            "bbox": str(bbox),
            "bbox_x": bbox[0],
            "bbox_y": bbox[1],
            "bbox_w": bbox[2],
            "bbox_h": bbox[3],
            "phrase": str(row["label_text"]),
            "finding_text": str(row["label_text"]),
            "split": split,
            "source": "ms_cxr",
        })

    logger.info(f"Loaded {len(records)} MS-CXR records.")
    return pd.DataFrame(records)


def parse_bbox_string(coord_str: Any) -> Optional[List[float]]:
    """Parses a string like '[1350, 1241, 2571, 2210]' into a list of floats."""
    if isinstance(coord_str, (list, tuple)):
        return [float(x) for x in coord_str]
    if pd.isna(coord_str):
        return None
    try:
        val = ast.literal_eval(str(coord_str).strip())
        if isinstance(val, (list, tuple)) and len(val) == 4:
            return [float(x) for x in val]
    except Exception:
        pass
    return None


def load_chest_imagenome_gold(
    gold_dir_or_file: str,
) -> pd.DataFrame:
    """
    Loads Chest ImaGenome gold_dataset annotations.
    Reads 'gold_object_attribute_with_coordinates.txt' (or direct file if passed).
    Expected columns:
      ['patient_id', 'study_id', 'image_id', 'bbox', 'coord_original', 'sentence', 'label_name', ...]
    """
    if os.path.isdir(gold_dir_or_file):
        target_file = os.path.join(gold_dir_or_file, "gold_object_attribute_with_coordinates.txt")
        if not os.path.exists(target_file):
            # Fallback to any coordinate file in the folder
            for candidate in ["gold_baseline_object_attribute_with_coordinate.txt", "gold_bbox_coordinate_annotations_1000images.csv"]:
                c_path = os.path.join(gold_dir_or_file, candidate)
                if os.path.exists(c_path):
                    target_file = c_path
                    break
    else:
        target_file = gold_dir_or_file

    if not os.path.exists(target_file):
        raise FileNotFoundError(f"Chest ImaGenome gold file not found: {target_file}")

    logger.info(f"Loading Chest ImaGenome Gold from: {target_file}")
    sep = "," if target_file.endswith(".csv") else "\t"
    df = pd.read_csv(target_file, sep=sep)

    # Harmonize column names
    patient_col = "patient_id" if "patient_id" in df.columns else "subject_id"
    image_col = "image_id" if "image_id" in df.columns else "dicom_id"
    study_col = "study_id"
    bbox_name_col = "bbox" if "bbox" in df.columns else "bbox_name"
    sentence_col = "sentence" if "sentence" in df.columns else ("label_name" if "label_name" in df.columns else "attribute")
    coord_col = "coord_original" if "coord_original" in df.columns else None

    records = []
    for _, row in df.iterrows():
        subj = str(row[patient_col]).strip()
        study = str(row[study_col]).strip()
        raw_img = str(row[image_col]).strip()
        dicom = raw_img.replace(".dcm", "").replace(".jpg", "")

        # Extract bbox
        # In Chest ImaGenome, coord_original is [x1, y1, x2, y2]
        coord = None
        if coord_col and coord_col in row:
            coord = parse_bbox_string(row[coord_col])
        elif all(k in row for k in ["original_x1", "original_y1", "original_x2", "original_y2"]):
            coord = [
                float(row["original_x1"]),
                float(row["original_y1"]),
                float(row["original_x2"]),
                float(row["original_y2"]),
            ]

        if not coord or len(coord) != 4:
            continue

        # Convert [x1, y1, x2, y2] -> COCO [x, y, w, h]
        x1, y1, x2, y2 = coord
        x = round(float(x1), 2)
        y = round(float(y1), 2)
        w = round(float(max(0.0, x2 - x1)), 2)
        h = round(float(max(0.0, y2 - y1)), 2)
        bbox = [x, y, w, h]

        raw_anatomy = str(row[bbox_name_col]) if bbox_name_col in row else "lungs"
        region_label = map_region_label(raw_anatomy)

        sentence = str(row[sentence_col]).strip() if sentence_col in row and pd.notna(row[sentence_col]) else ""
        if not sentence and "label_name" in row and pd.notna(row["label_name"]):
            sentence = str(row["label_name"]).strip()

        rel_path = format_gcs_relative_path(subj, study, dicom)

        records.append({
            "dicom_id": dicom,
            "subject_id": subj,
            "study_id": study,
            "image_relative_path": rel_path,
            "region_label": region_label,
            "raw_region": raw_anatomy,
            "bbox": str(bbox),
            "bbox_x": bbox[0],
            "bbox_y": bbox[1],
            "bbox_w": bbox[2],
            "bbox_h": bbox[3],
            "phrase": sentence,
            "finding_text": sentence,
            "split": "gold_test",
            "source": "chest_imagenome_gold",
        })

    logger.info(f"Loaded {len(records)} Chest ImaGenome Gold records.")
    return pd.DataFrame(records)


def unify_manifests(
    df_ms_cxr: pd.DataFrame,
    df_gold: pd.DataFrame,
    join_mode: str = "union",
) -> pd.DataFrame:
    """
    Joins MS-CXR and Chest ImaGenome Gold annotations on subject_id/study_id/dicom_id.
    Modes:
      - 'union': Outer union concatenating both datasets into the unified schema.
      - 'inner': Restricts to studies/dicoms that appear in BOTH datasets.
    """
    canonical_columns = [
        "dicom_id",
        "subject_id",
        "study_id",
        "image_relative_path",
        "region_label",
        "bbox",
        "phrase",
        "finding_text",
        "split",
        "source",
        "raw_region",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
    ]

    if join_mode == "inner":
        # Keep only dicoms present in both
        common_dicoms = set(df_ms_cxr["dicom_id"]).intersection(set(df_gold["dicom_id"]))
        logger.info(f"Inner join selected: {len(common_dicoms)} overlapping dicoms.")
        df_ms_sub = df_ms_cxr[df_ms_cxr["dicom_id"].isin(common_dicoms)]
        df_gold_sub = df_gold[df_gold["dicom_id"].isin(common_dicoms)]
        unified = pd.concat([df_ms_sub, df_gold_sub], ignore_index=True)
    else:
        logger.info(f"Union mode: combining {len(df_ms_cxr)} MS-CXR and {len(df_gold)} Gold records.")
        unified = pd.concat([df_ms_cxr, df_gold], ignore_index=True)

    # Ensure all canonical columns exist
    for col in canonical_columns:
        if col not in unified.columns:
            unified[col] = ""

    # Sort deterministically
    unified = unified.sort_values(by=["subject_id", "study_id", "dicom_id", "source"]).reset_index(drop=True)
    return unified[canonical_columns]


def generate_download_list(unified_df: pd.DataFrame, output_path: str) -> List[str]:
    """
    Generates a deduplicated list of MIMIC-CXR-JPG GCS object paths:
    files/p{prefix}/p{subject_id}/s{study_id}/{dicom_id}.jpg
    """
    unique_paths = sorted(unified_df["image_relative_path"].dropna().unique().tolist())
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in unique_paths:
            f.write(f"{p}\n")

    logger.info(f"Wrote {len(unique_paths)} deduplicated image download paths to: {output_path}")
    return unique_paths


def build_manifest_cli() -> None:
    parser = argparse.ArgumentParser(description="Build unified manifest and download list for InteractMed-VLM.")
    parser.add_argument(
        "--ms-cxr-csv",
        type=str,
        default="data/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.csv",
        help="Path to MS-CXR CSV file",
    )
    parser.add_argument(
        "--ms-cxr-json",
        type=str,
        default="data/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.json",
        help="Path to MS-CXR COCO JSON file",
    )
    parser.add_argument(
        "--gold-dir",
        type=str,
        default="data/chest-imagenome/gold_dataset",
        help="Path to Chest ImaGenome gold dataset directory or file",
    )
    parser.add_argument(
        "--output-manifest",
        type=str,
        default="manifests/unified_manifest.csv",
        help="Destination path for unified CSV manifest",
    )
    parser.add_argument(
        "--output-download-list",
        type=str,
        default="manifests/download_list.txt",
        help="Destination path for download list txt file",
    )
    parser.add_argument(
        "--join-mode",
        type=str,
        choices=["union", "inner"],
        default="union",
        help="Join mode across MS-CXR and Gold ('union' to preserve all, 'inner' for overlap)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick testing",
    )

    args = parser.parse_args()

    # Load MS-CXR
    df_ms = load_ms_cxr(args.ms_cxr_csv, args.ms_cxr_json)
    if args.limit:
        df_ms = df_ms.head(args.limit)

    # Load Chest ImaGenome Gold
    df_gold = load_chest_imagenome_gold(args.gold_dir)
    if args.limit:
        df_gold = df_gold.head(args.limit)

    # Unify
    unified_df = unify_manifests(df_ms, df_gold, join_mode=args.join_mode)

    # Save manifest
    os.makedirs(os.path.dirname(os.path.abspath(args.output_manifest)), exist_ok=True)
    unified_df.to_csv(args.output_manifest, index=False)
    logger.info(f"Saved unified manifest ({len(unified_df)} rows) to: {args.output_manifest}")

    # Save download list
    generate_download_list(unified_df, args.output_download_list)


if __name__ == "__main__":
    build_manifest_cli()
