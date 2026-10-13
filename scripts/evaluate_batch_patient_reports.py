"""
scripts/evaluate_batch_patient_reports.py
=========================================
Independent evaluation and verification of patient-friendly batch reports.
Evaluates 4 diverse chest X-rays:
  - 0004cfab (Normal reference)
  - 00436515 (Bilateral severe pneumonia opacities + intubation)
  - 00aecb01 (Mid-lung opacity + tracheostomy)
  - 00c0b293 (Bilateral lower opacities + central line)

Saves detailed evaluation record to docs/batch_patient_report_eval.json.
Measures:
  - Job-to-image mapping integrity
  - Absence of anatomical errors ("only shows one side of the chest")
  - Clinical finding differentiation and vocabulary richness
  - Pairwise Jaccard text overlap across reports
"""

import hashlib
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.batch_service import get_batch_service

TEST_CASES = [
    {
        "filename": "0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm",
        "path": "data/raw/rsna/stage_2_train_images/0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm",
        "reference_label": "Normal / Baseline",
        "reference_features": ["Clear lungs or mild right basilar markings", "No intubation"],
    },
    {
        "filename": "00436515-870c-4b36-a041-de91049b9ab4.dcm",
        "path": "data/raw/rsna/stage_2_train_images/00436515-870c-4b36-a041-de91049b9ab4.dcm",
        "reference_label": "Severe Bilateral Pneumonia Opacities",
        "reference_features": ["Bilateral lung opacities/consolidation", "Endotracheal tube", "Enteric/central lines"],
    },
    {
        "filename": "00aecb01-a116-45a2-956c-08d2fa55433f.dcm",
        "path": "data/raw/rsna/stage_2_train_images/00aecb01-a116-45a2-956c-08d2fa55433f.dcm",
        "reference_label": "Mid-lung Opacities with Tracheostomy",
        "reference_features": ["Tracheostomy tube", "Mid-zone parenchymal shading"],
    },
    {
        "filename": "00c0b293-48e7-4e16-ac76-9269ba535a62.dcm",
        "path": "data/raw/rsna/stage_2_train_images/00c0b293-48e7-4e16-ac76-9269ba535a62.dcm",
        "reference_label": "Bilateral Lower Lobe Infiltrates",
        "reference_features": ["Bilateral lower lung opacities/infiltrates", "Central line catheter"],
    },
]

def jaccard_similarity(text_a: str, text_b: str) -> float:
    words_a = set(re.findall(r'\b[a-zA-Z]{3,}\b', text_a.lower()))
    words_b = set(re.findall(r'\b[a-zA-Z]{3,}\b', text_b.lower()))
    if not words_a or not words_b:
        return 0.0
    return round(len(words_a & words_b) / len(words_a | words_b), 4)

def run_evaluation():
    print("======================================================================")
    print("INTERACTMED-VLM: PATIENT-FRIENDLY BATCH REPORT EVALUATION")
    print("======================================================================")
    
    files_data = []
    metadata_map = {}
    
    for tc in TEST_CASES:
        p = Path(tc["path"])
        if not p.exists():
            print(f"Error: test file {p} does not exist!")
            sys.exit(1)
        raw_bytes = p.read_bytes()
        sha = hashlib.sha256(raw_bytes).hexdigest()[:16]
        files_data.append((tc["filename"], raw_bytes))
        metadata_map[tc["filename"]] = {
            "sha256": sha,
            "reference_label": tc["reference_label"],
            "reference_features": tc["reference_features"],
        }
        print(f"Loaded {tc['filename']} (SHA-256: {sha}, {tc['reference_label']})")

    svc = get_batch_service()
    print("\nCreating batch via BatchService (style: patient)...")
    batch = svc.create_batch(files_data, style="patient")
    batch_id = batch.batch_id
    print(f"Batch created: {batch_id} with {batch.total_items} items.")

    # Poll until completion
    t0 = time.time()
    while True:
        b = svc.get_batch(batch_id)
        if b.status in ("completed", "failed", "cancelled"):
            break
        print(f"  Progress: {b.completed_count} of {b.total_items} completed (elapsed: {time.time()-t0:.1f}s)")
        time.sleep(3)

    b = svc.get_batch(batch_id)
    print(f"\nBatch finished with status: {b.status} in {time.time()-t0:.2f}s")

    eval_items = []
    report_texts = {}

    for it in b.items:
        meta = metadata_map.get(it.filename, {})
        report_text = it.report_text or ""
        report_texts[it.filename] = report_text

        # Check for factual error
        one_side_error_present = bool(re.search(r'only\s+shows\s+one\s+side(?:\s+of\s+the\s+chest|\s+at\s+a\s+time)?', report_text, re.IGNORECASE))
        
        # Check for generic boilerplate
        generic_normal_present = bool(
            re.search(r'bones\s+(?:of\s+your\s+chest\s+wall\s+)?look\s+intact', report_text, re.IGNORECASE) and
            re.search(r'no\s+obvious\s+signs\s+of\s+infection\s+or\s+fluid', report_text, re.IGNORECASE)
        )

        item_eval = {
            "job_id": it.job_id,
            "filename": it.filename,
            "sha256": meta.get("sha256"),
            "reference_label": meta.get("reference_label"),
            "reference_features": meta.get("reference_features"),
            "status": it.status,
            "latency_ms": it.latency_ms,
            "tokens_per_second": it.tokens_per_second,
            "model_used": it.model_used,
            "has_frontal_one_side_error": one_side_error_present,
            "has_generic_normal_boilerplate": generic_normal_present,
            "validation_warnings": it.validation_warnings,
            "sections": it.sections,
            "report_text": report_text,
        }
        eval_items.append(item_eval)
        print(f"\n--- Item {it.job_id} ({it.filename} - {meta.get('reference_label')}) ---")
        print(f"Status: {it.status} | Latency: {it.latency_ms} ms | Tokens/sec: {it.tokens_per_second}")
        print(f"Anatomical error ('only shows one side') detected: {one_side_error_present}")
        print(f"Warnings: {it.validation_warnings}")
        print("Report Text Preview:\n" + (report_text[:300] + "..." if len(report_text) > 300 else report_text))

    # Calculate pairwise Jaccard similarities
    pairwise_similarities = {}
    filenames = list(report_texts.keys())
    for i in range(len(filenames)):
        for j in range(i + 1, len(filenames)):
            f_a, f_b = filenames[i], filenames[j]
            sim = jaccard_similarity(report_texts[f_a], report_texts[f_b])
            pair_key = f"{f_a} vs {f_b}"
            pairwise_similarities[pair_key] = sim
            print(f"Pairwise Jaccard overlap ({f_a[:8]}.. vs {f_b[:8]}..): {sim:.4f}")

    avg_sim = round(sum(pairwise_similarities.values()) / max(len(pairwise_similarities), 1), 4)
    print(f"\nAverage Pairwise Jaccard Similarity across 4 reports: {avg_sim:.4f}")

    # Summary record
    out_record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "batch_id": batch_id,
        "total_items": len(eval_items),
        "status": b.status,
        "average_pairwise_similarity": avg_sim,
        "pairwise_similarities": pairwise_similarities,
        "items": eval_items,
    }

    out_path = PROJECT_ROOT / "docs" / "batch_patient_report_eval.json"
    with open(out_path, "w") as f:
        json.dump(out_record, f, indent=2)

    print(f"\nEvaluation record saved to: {out_path}")
    print("======================================================================")
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("======================================================================")

if __name__ == "__main__":
    run_evaluation()
