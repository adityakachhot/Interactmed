#!/usr/bin/env python3
"""
scripts/audit_coordinates.py
============================
InteractMed-VLM Geometric Coordinate and Bounding-Box Audit Script.

Audits bounding box validity, coordinate systems, mathematical transforms,
aspect ratio anomalies, and boundary violations across normalized master tables.

Outputs:
  - docs/coordinate_audit_report.json
  - Formatted terminal audit summary

Usage:
    python scripts/audit_coordinates.py [--table data/processed/unified_grounding_master.csv] [--output docs/coordinate_audit_report.json]
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("AuditCoordinates")


def audit_table_coordinates(df: pd.DataFrame) -> Dict[str, Any]:
    total_records = len(df)
    logger.info(f"Auditing coordinates across {total_records:,} records...")

    # 1. Bounding Box Dimensions
    x1 = df["norm_x1"]
    y1 = df["norm_y1"]
    x2 = df["norm_x2"]
    y2 = df["norm_y2"]

    w_norm = x2 - x1
    h_norm = y2 - y1

    eps = 1e-4
    valid_width = (w_norm > eps)
    valid_height = (h_norm > eps)
    valid_area = valid_width & valid_height

    # 2. Boundary Constraints in [0, 1]
    in_bounds_x = (x1 >= -eps) & (x2 <= 1.0 + eps)
    in_bounds_y = (y1 >= -eps) & (y2 <= 1.0 + eps)
    strictly_in_unit_square = in_bounds_x & in_bounds_y

    # 3. Native Coordinate Consistency
    ox1 = df["orig_x1"]
    oy1 = df["orig_y1"]
    ox2 = df["orig_x2"]
    oy2 = df["orig_y2"]
    ow = df["orig_w"]
    oh = df["orig_h"]

    diff_w = (np.abs((ox2 - ox1) - ow)).max()
    diff_h = (np.abs((oy2 - oy1) - oh)).max()

    # 4. Aspect Ratio & Box Size Analysis
    aspect_ratios = w_norm / np.maximum(h_norm, 1e-6)
    extreme_aspect_ratio_count = int(((aspect_ratios > 10.0) | (aspect_ratios < 0.1)).sum())

    # Box Area (% of image)
    box_areas_pct = (w_norm * h_norm) * 100.0

    # 5. Breakdown by Dataset
    breakdown = {}
    for src, grp in df.groupby("dataset_source"):
        grp_w = grp["norm_x2"] - grp["norm_x1"]
        grp_h = grp["norm_y2"] - grp["norm_y1"]
        grp_val = (grp_w > eps) & (grp_h > eps)
        breakdown[src] = {
            "total_records": len(grp),
            "valid_boxes_count": int(grp_val.sum()),
            "invalid_boxes_count": int((~grp_val).sum()),
            "mean_box_width_pct": round(float(grp_w.mean() * 100), 2),
            "mean_box_height_pct": round(float(grp_h.mean() * 100), 2),
            "mean_box_area_pct": round(float((grp_w * grp_h).mean() * 100), 2),
            "min_box_area_pct": round(float((grp_w * grp_h).min() * 100), 4),
            "max_box_area_pct": round(float((grp_w * grp_h).max() * 100), 2),
            "validation_errors": grp["validation_error"].value_counts().to_dict(),
        }

    report = {
        "summary": {
            "total_records_audited": total_records,
            "strictly_in_unit_square_count": int(strictly_in_unit_square.sum()),
            "strictly_in_unit_square_pct": round(100.0 * strictly_in_unit_square.sum() / max(1, total_records), 3),
            "valid_positive_area_count": int(valid_area.sum()),
            "valid_positive_area_pct": round(100.0 * valid_area.sum() / max(1, total_records), 3),
            "degenerate_zero_area_count": int((~valid_area).sum()),
            "extreme_aspect_ratio_count": extreme_aspect_ratio_count,
            "max_native_width_reconstruction_diff_px": round(float(diff_w), 4),
            "max_native_height_reconstruction_diff_px": round(float(diff_h), 4),
        },
        "area_distribution_pct": {
            "p10": round(float(np.percentile(box_areas_pct, 10)), 3),
            "p25": round(float(np.percentile(box_areas_pct, 25)), 3),
            "median": round(float(np.median(box_areas_pct)), 3),
            "p75": round(float(np.percentile(box_areas_pct, 75)), 3),
            "p90": round(float(np.percentile(box_areas_pct, 90)), 3),
            "mean": round(float(np.mean(box_areas_pct)), 3),
        },
        "by_dataset": breakdown,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Audit Bounding Box Coordinates")
    parser.add_argument("--table", default="data/processed/unified_grounding_master.csv", help="Path to master table")
    parser.add_argument("--output", default="docs/coordinate_audit_report.json", help="Path to save audit report")
    args = parser.parse_args()

    if not os.path.exists(args.table):
        logger.error(f"Target table {args.table} does not exist. Run scripts/prepare_local_metadata.py first.")
        sys.exit(1)

    df = pd.read_csv(args.table)
    report = audit_table_coordinates(df)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fp:
        json.dump(report, fp, indent=2)

    print("\n" + "=" * 80)
    print("INTERACTMED-VLM COORDINATE & BOUNDING-BOX AUDIT REPORT")
    print("=" * 80)
    print(f"Total Records Audited:                 {report['summary']['total_records_audited']:,}")
    print(f"Bounded in [0, 1] Unit Square:         {report['summary']['strictly_in_unit_square_count']:,} ({report['summary']['strictly_in_unit_square_pct']}%)")
    print(f"Valid Positive Bounding Box Area:      {report['summary']['valid_positive_area_count']:,} ({report['summary']['valid_positive_area_pct']}%)")
    print(f"Degenerate / Non-Positive Area Boxes:  {report['summary']['degenerate_zero_area_count']}")
    print(f"Extreme Aspect Ratios (>10:1 or <1:10):{report['summary']['extreme_aspect_ratio_count']:,}")
    print(f"Native Transform Equivalence Diff:     {report['summary']['max_native_width_reconstruction_diff_px']} px")
    print("\nBox Area Distribution (% of image):")
    print(f"  10th percentile: {report['area_distribution_pct']['p10']}% | Median: {report['area_distribution_pct']['median']}% | 90th percentile: {report['area_distribution_pct']['p90']}%")
    print("\nBreakdown by Dataset:")
    for ds, d_info in report["by_dataset"].items():
        print(f"  {ds.upper():<22} Valid: {d_info['valid_boxes_count']:,} / {d_info['total_records']:,} | Mean Area: {d_info['mean_box_area_pct']}% | Errors: {d_info['validation_errors']}")
    print("=" * 80 + "\n")
    logger.info(f"Saved audit report to {args.output}")


if __name__ == "__main__":
    main()
