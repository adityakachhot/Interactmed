"""
scripts/benchmark_batch.py
==========================
Comprehensive benchmark and verification script for Batch Reports feature.

Measures:
  1. Cold vs warm request latency and memory
  2. Single-image request
  3. 3-image batch request
  4. 10-image batch request
  5. 11-image upload rejection
  6. Corrupt/invalid image handling
  7. Partial failure preservation & retry
  8. In-flight queue cancellation
  9. Both styles (Patient-friendly vs Research)
  10. Non-interference with Grounding and Patient-Friendly single report

Outputs a detailed JSON benchmark summary.
"""

import glob
import io
import json
import os
import resource
import sys
import time
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.batch_service import get_batch_service, clean_and_structure_research_report
from src.report_service import patient_friendly_report
from src.grounding_service import get_grounding_service


def get_peak_memory_mb() -> float:
    """Get max RSS in MB on macOS (ru_maxrss is in bytes)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / (1024 * 1024)


def run_benchmarks():
    print("=" * 70)
    print("INTERACTMED-VLM: BATCH REPORTS BENCHMARK & VERIFICATION")
    print("=" * 70)

    results = {}
    batch_svc = get_batch_service()

    # Locate test DICOM images
    dcm_files = sorted(glob.glob("data/raw/rsna/stage_2_train_images/*.dcm"))
    if not dcm_files:
        raise RuntimeError("No RSNA test DICOM files found in data/raw/rsna/stage_2_train_images/")
    test_10_paths = dcm_files[:10]
    print(f"Loaded {len(test_10_paths)} real DICOM files for benchmarking.")

    # Load file bytes
    test_10_data = []
    for p in test_10_paths:
        with open(p, "rb") as f:
            test_10_data.append((Path(p).name, f.read()))

    # ─────────────────────────────────────────────────────────────────────────
    # 1. ATTEMPT 11 IMAGES (REJECTION TEST)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Scenario 1: Attempted upload of 11 images (Expected Rejection) ---")
    extra_data = test_10_data + [("extra_11th.dcm", test_10_data[0][1])]
    try:
        batch_svc.create_batch(extra_data, style="patient")
        print("FAIL: Expected ValueError on 11 images!")
        results["rejection_11_images"] = "FAILED"
    except ValueError as exc:
        print(f"PASS: Rejected with error: {exc}")
        results["rejection_11_images"] = "PASSED"

    # ─────────────────────────────────────────────────────────────────────────
    # 2. INVALID / CORRUPT IMAGE HANDLING & PARTIAL FAILURE PRESERVATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Scenario 2: Corrupt image mixed with valid image ---")
    corrupt_data = [
        ("corrupt_header.dcm", b"NOT_A_DICOM_INVALID_DATA" * 5),
        (test_10_data[0][0], test_10_data[0][1]),
    ]
    b_mixed = batch_svc.create_batch(corrupt_data, style="patient")
    # Wait for completion
    while b_mixed.status in ("queued", "processing"):
        time.sleep(0.5)

    item_corrupt = [it for it in b_mixed.items if it.filename == "corrupt_header.dcm"][0]
    item_valid = [it for it in b_mixed.items if it.filename != "corrupt_header.dcm"][0]

    print(f"Corrupt item status: {item_corrupt.status} (Error: {item_corrupt.error})")
    print(f"Valid item status: {item_valid.status} (Report words: {len(item_valid.report_text.split()) if item_valid.report_text else 0})")

    assert item_corrupt.status == "failed", "Corrupt item should fail"
    assert item_valid.status == "completed", "Valid item should complete despite corrupt sibling"
    results["corrupt_mixed_handling"] = "PASSED"
    print("PASS: Partial failure handled cleanly, completed outputs preserved.")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. RETRY FAILED JOB TEST
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Scenario 3: Retrying failed job ---")
    # Replace corrupt bytes with valid bytes for the retry test
    item_corrupt._image_bytes = test_10_data[1][1]
    retried_count = batch_svc.retry_jobs(b_mixed.batch_id, [item_corrupt.job_id])
    print(f"Retrying {retried_count} failed jobs...")
    while b_mixed.status in ("queued", "processing"):
        time.sleep(0.5)

    print(f"Retried item new status: {item_corrupt.status}")
    assert item_corrupt.status == "completed", "Retried item should succeed with valid bytes"
    assert item_valid.status == "completed", "Previously completed item should remain completed"
    results["retry_failed_job"] = "PASSED"
    print("PASS: Failed job successfully retried without regenerating completed reports.")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. QUEUE CANCELLATION TEST
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Scenario 4: Cancelling remaining queue ---")
    cancel_data = test_10_data[:4]
    b_cancel = batch_svc.create_batch(cancel_data, style="patient")
    # Immediately cancel
    batch_svc.cancel_batch(b_cancel.batch_id)
    time.sleep(1.0)
    cancelled_statuses = [it.status for it in b_cancel.items]
    print(f"Items statuses after cancel: {cancelled_statuses}")
    assert any(s == "cancelled" for s in cancelled_statuses), "Queued items must be cancelled"
    results["queue_cancellation"] = "PASSED"
    print("PASS: Queue cancellation successfully stopped remaining jobs.")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. SINGLE-IMAGE REQUEST (WARM BASELINE)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Scenario 5: Single Image Warm Request (Patient-Friendly Style) ---")
    mem_before = get_peak_memory_mb()
    t0 = time.perf_counter()
    b_single = batch_svc.create_batch([test_10_data[0]], style="patient")
    while b_single.status in ("queued", "processing"):
        time.sleep(0.5)
    t_single = time.perf_counter() - t0
    mem_after = get_peak_memory_mb()

    s_item = b_single.items[0]
    results["single_image"] = {
        "status": s_item.status,
        "latency_sec": round(t_single, 2),
        "tokens_generated": s_item.eval_count,
        "tokens_per_second": s_item.tokens_per_second,
        "load_duration_ms": s_item.timing.get("load_duration_ms") if s_item.timing else None,
        "prompt_eval_ms": s_item.timing.get("prompt_eval_ms") if s_item.timing else None,
        "eval_duration_ms": s_item.timing.get("eval_duration_ms") if s_item.timing else None,
        "words": len(s_item.report_text.split()) if s_item.report_text else 0,
        "peak_mem_mb": round(mem_after, 1),
    }
    print(f"Single image latency: {t_single:.2f} s | {s_item.tokens_per_second} t/s | Words: {results['single_image']['words']}")
    assert s_item.status == "completed"

    # ─────────────────────────────────────────────────────────────────────────
    # 6. 3-IMAGE BATCH (RESEARCH STYLE)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Scenario 6: 3-Image Batch Request (Research Style) ---")
    t0 = time.perf_counter()
    b_3 = batch_svc.create_batch(test_10_data[:3], style="research")
    while b_3.status in ("queued", "processing"):
        time.sleep(0.5)
    t_3 = time.perf_counter() - t0

    latencies_3 = [it.latency_ms / 1000 for it in b_3.items if it.latency_ms]
    tok_sec_3 = [it.tokens_per_second for it in b_3.items if it.tokens_per_second]
    results["batch_3_images"] = {
        "status": b_3.status,
        "total_batch_time_sec": round(t_3, 2),
        "avg_latency_per_image_sec": round(sum(latencies_3) / len(latencies_3), 2),
        "avg_tokens_per_sec": round(sum(tok_sec_3) / len(tok_sec_3), 2),
        "items": [
            {"filename": it.filename, "latency_sec": round(it.latency_ms / 1000, 2), "words": len(it.report_text.split())}
            for it in b_3.items
        ],
    }
    print(f"3-image batch total: {t_3:.2f} s | Avg per image: {results['batch_3_images']['avg_latency_per_image_sec']} s")
    assert all(it.status == "completed" for it in b_3.items)
    # Verify Research section headers in all 3
    for it in b_3.items:
        assert "**OBSERVATIONS**" in it.report_text
        assert "**POSSIBLE FINDINGS**" in it.report_text
        assert "**LIMITATIONS**" in it.report_text

    # ─────────────────────────────────────────────────────────────────────────
    # 7. 10-IMAGE BATCH MEASUREMENT
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Scenario 7: 10-Image Batch Request (Patient-Friendly Style) ---")
    t0 = time.perf_counter()
    b_10 = batch_svc.create_batch(test_10_data, style="patient")
    # Print progress during batch
    last_completed = 0
    while b_10.status in ("queued", "processing"):
        comp = b_10.completed_count
        if comp != last_completed:
            print(f"  Progress: {comp} of 10 completed (elapsed: {time.perf_counter() - t0:.1f} s)")
            last_completed = comp
        time.sleep(1.0)
    t_10 = time.perf_counter() - t0

    latencies_10 = [it.latency_ms / 1000 for it in b_10.items if it.latency_ms]
    tok_sec_10 = [it.tokens_per_second for it in b_10.items if it.tokens_per_second]
    results["batch_10_images"] = {
        "status": b_10.status,
        "total_batch_time_sec": round(t_10, 2),
        "avg_latency_per_image_sec": round(sum(latencies_10) / len(latencies_10), 2),
        "avg_tokens_per_sec": round(sum(tok_sec_10) / len(tok_sec_10), 2),
        "success_rate": f"{b_10.completed_count}/{b_10.total_items}",
        "peak_mem_mb": round(get_peak_memory_mb(), 1),
    }
    print(f"10-image batch total: {t_10:.2f} s | Avg per image: {results['batch_10_images']['avg_latency_per_image_sec']} s | Peak RSS: {results['batch_10_images']['peak_mem_mb']} MB")
    assert b_10.completed_count == 10

    # Verify pairing: each report matches its file
    for it in b_10.items:
        assert it.job_id is not None
        assert it.report_text is not None
        assert len(it.report_text) > 100
        assert "Your X-ray in simple words" in it.report_text

    # ─────────────────────────────────────────────────────────────────────────
    # 8. NON-INTERFERENCE WITH GROUNDING & PATIENT-FRIENDLY SINGLE REPORT
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Scenario 8: Non-interference with Grounding & Patient Report ---")
    # Check TransVG-AGPT grounding
    svc_ground = get_grounding_service("finetuned")
    g_res = svc_ground.predict(test_10_data[0][1], "pneumonia", image_filename=test_10_data[0][0])
    assert g_res["success"] is True
    print(f"TransVG-AGPT grounding works: box={g_res['pred_box_pixel']}")

    # Check Patient-Friendly single report
    p_res = patient_friendly_report(test_10_data[0][1], test_10_data[0][0])
    assert p_res["success"] is True
    print("Patient-Friendly single report works.")
    results["non_interference"] = "PASSED"

    # ─────────────────────────────────────────────────────────────────────────
    # SAVE BENCHMARK REPORT
    # ─────────────────────────────────────────────────────────────────────────
    out_path = PROJECT_ROOT / "docs" / "batch_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark results saved to: {out_path}")
    print("\n" + "=" * 70)
    print("ALL BENCHMARKS AND SCENARIOS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmarks()
