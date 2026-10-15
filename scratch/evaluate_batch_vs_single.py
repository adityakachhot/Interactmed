"""
scratch/evaluate_batch_vs_single.py
===================================
Run real-image quality evaluation comparing Single-Image vs Batch pathways
across different clinical appearances:
  1. Reference Normal
  2. Lung Opacity / Consolidation
  3. Visible Support Apparatus (Endotracheal tube)
  4. Non-Opacity Thoracic Abnormality / Ambiguous
"""

import hashlib
import io
import json
import time
from pathlib import Path
from PIL import Image

from src.report_service import (
    _load_to_rgb_array,
    research_report,
    patient_friendly_report,
    PROMPT_VERSION_RESEARCH,
)
from src.batch_service import get_batch_service


def main():
    test_cases = [
        {
            "name": "Reference Normal",
            "path": "data/raw/rsna/stage_2_train_images/003d8fa0-6bf1-40ed-b54c-ac657f8495c5.dcm",
            "ref_label": "Normal",
            "expected_findings": "Unremarkable lungs, normal cardiac silhouette, no apparatus",
        },
        {
            "name": "Annotated Lung Opacity",
            "path": "data/raw/rsna/stage_2_train_images/00436515-870c-4b36-a041-de91049b9ab4.dcm",
            "ref_label": "Lung Opacity",
            "expected_findings": "Focal opacity or consolidation in lung parenchyma",
        },
        {
            "name": "Visible Support Apparatus",
            "path": "artifacts/rsna/smoke_test/010ccb9f-6d46-4380-af11-84f87397a1b8_iou058.png",
            "ref_label": "Support Apparatus + Lung Opacity",
            "expected_findings": "Endotracheal tube / invasive line observed",
        },
        {
            "name": "Non-Opacity Thoracic Variation",
            "path": "data/raw/rsna/stage_2_train_images/0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm",
            "ref_label": "No Lung Opacity / Not Normal",
            "expected_findings": "Cardiomegaly, pleural change, or technical variation",
        },
    ]

    results = []
    svc = get_batch_service()

    # Load file bytes
    loaded_files = []
    for tc in test_cases:
        p = Path(tc["path"])
        if not p.exists():
            print(f"File not found: {p}")
            continue
        with open(p, "rb") as f:
            data = f.read()
        loaded_files.append((p.name, data, tc))

    print(f"Loaded {len(loaded_files)} test images for live evaluation.")

    # 1. Run Single-Image Reporting
    print("\n--- Running Single-Image Reports ---")
    single_reports = []
    for fname, data, tc in loaded_files:
        print(f"Generating single-image report for {tc['name']} ({fname})...")
        t0 = time.time()
        res = research_report(image_bytes=data, image_filename=fname)
        dt = time.time() - t0
        print(f"  Done in {dt:.1f}s. Model: {res.get('model_used')}. Triage: {res.get('triage_result', {}).get('matched_rule_id')}")
        single_reports.append((tc, fname, data, res))

    # 2. Run Batch Reporting
    print("\n--- Running Batch Reporting ---")
    batch_input = [(fname, data) for fname, data, tc in loaded_files]
    t0_b = time.time()
    batch = svc.create_batch(batch_input, style="research")
    print(f"Created batch {batch.batch_id} with {batch.total_items} items. Waiting for completion...")

    # Wait for batch completion
    while batch.status in ("queued", "processing"):
        time.sleep(2)
        b_cur = svc.get_batch(batch.batch_id)
        if b_cur:
            batch = b_cur
            print(f"  Batch status: {batch.status} ({batch.completed_count}/{batch.total_items} completed, {batch.failed_count} failed)")

    total_batch_time = time.time() - t0_b
    print(f"Batch completed in {total_batch_time:.1f}s.")

    # 3. Compare Single vs Batch for each image
    print("\n--- Comparing Single-Image vs Batch Results ---")
    for i, (tc, fname, data, s_rep) in enumerate(single_reports):
        b_item = batch.items[i]
        
        # Verify identity
        bytes_match = (s_rep["bytes_sha256"] == b_item.bytes_sha256)
        pixel_match = (s_rep["pixel_sha256"] == b_item.pixel_sha256)
        orig_dims_match = (s_rep["orig_dims"] == b_item.orig_dims)
        prep_dims_match = (s_rep["prep_dims"] == b_item.prep_dims)

        s_triage = s_rep.get("triage_result", {}).get("matched_rule_id")
        b_triage = b_item.triage_result.get("matched_rule_id") if b_item.triage_result else None

        # Check section differences
        s_sections = s_rep.get("sections", {})
        b_sections = b_item.sections or {}
        section_keys_match = (set(s_sections.keys()) == set(b_sections.keys()))

        has_raw_markdown_s = "**" in s_rep.get("report_text", "")
        has_raw_markdown_b = "**" in (b_item.report_text or "")

        comp = {
            "test_case": tc["name"],
            "filename": fname,
            "ref_label": tc["ref_label"],
            "expected": tc["expected_findings"],
            "bytes_sha256": s_rep["bytes_sha256"],
            "pixel_sha256": s_rep["pixel_sha256"],
            "orig_dims": s_rep["orig_dims"],
            "prep_dims": s_rep["prep_dims"],
            "identity_match": {
                "bytes": bytes_match,
                "pixel": pixel_match,
                "orig_dims": orig_dims_match,
                "prep_dims": prep_dims_match,
            },
            "single_image": {
                "triage_rule": s_triage,
                "latency_ms": s_rep.get("latency_ms"),
                "tokens_per_sec": s_rep.get("tokens_per_second"),
                "has_markdown_artifacts": has_raw_markdown_s,
                "sections": s_sections,
                "report_text": s_rep.get("report_text"),
            },
            "batch_item": {
                "job_id": b_item.job_id,
                "status": b_item.status,
                "triage_rule": b_triage,
                "latency_ms": b_item.latency_ms,
                "tokens_per_sec": b_item.tokens_per_second,
                "has_markdown_artifacts": has_raw_markdown_b,
                "sections": b_sections,
                "report_text": b_item.report_text,
            },
            "triage_match": (s_triage == b_triage),
            "section_keys_match": section_keys_match,
        }
        results.append(comp)

        print(f"\n[{tc['name']}] ({fname})")
        print(f"  Reference: {tc['ref_label']}")
        print(f"  Bytes SHA-256: {s_rep['bytes_sha256'][:16]}... (Match: {bytes_match})")
        print(f"  Pixel SHA-256: {s_rep['pixel_sha256'][:16]}... (Match: {pixel_match})")
        print(f"  Orig Dims: {s_rep['orig_dims']}, Prep Dims: {s_rep['prep_dims']}")
        print(f"  Single Triage: {s_triage} | Batch Triage: {b_triage}")
        print(f"  Apparatus in Batch: {b_sections.get('SUPPORT APPARATUS')}")
        print(f"  Lungs in Batch: {b_sections.get('LUNGS AND OPACITIES')}")
        print(f"  Impression in Batch: {b_sections.get('OVERALL IMPRESSION')}")
        print(f"  Markdown Artifacts Leak: Single={has_raw_markdown_s}, Batch={has_raw_markdown_b}")

    out_file = Path("scratch/evaluation_results.json")
    out_file.parent.mkdir(exist_ok=True, parents=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed evaluation results to {out_file}")


if __name__ == "__main__":
    main()
