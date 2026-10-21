#!/usr/bin/env python3
"""
scripts/profile_hybrid_pipeline_latency.py
==========================================
Granular latency profiling across all 8 sub-components of the Hybrid Reporting Pipeline.
Measures cold-start and warm inference times across multiple test samples.
Outputs: artifacts/medgemma_finetuning/task10_latency_profile.csv
"""

import base64
import json
import logging
import re
import sys
import time
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.grounding_service import get_grounding_service
from src.report_service import extract_crop, _array_to_base64_jpeg
from src.consistency_service import (
    evaluate_cross_report_consistency,
    extract_regional_assessment_state,
    check_whole_image_internal_contradiction,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HybridProfiler")

OLLAMA_URL = "http://localhost:11434/api/generate"
VISION_MODEL = "medgemma:4b"

def ollama_generate(prompt: str, images_b64: List[str] = None, num_predict: int = 150) -> Tuple[str, float]:
    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": num_predict}
    }
    if images_b64:
        payload["images"] = images_b64
    data = json.dumps(payload).encode()
    req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode())
    return body.get("response", "").strip(), time.perf_counter() - t0

def run_profiling():
    out_dir = PROJECT_ROOT / "artifacts/medgemma_finetuning"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load 5 sample cases from Task 8 manifest
    with open(PROJECT_ROOT / "artifacts/report_quality/task8_cohort_manifest.json") as f:
        cases = json.load(f)[:5]

    # Pre-load grounding service to measure cold-start vs warm
    t0_gs = time.perf_counter()
    gs = get_grounding_service()
    cold_grounding_init = time.perf_counter() - t0_gs

    # Load prompts
    v4_prompt_path = PROJECT_ROOT / "artifacts/report_quality/prompt_variants/prompt_v4_structured.txt"
    v5_ext_path = PROJECT_ROOT / "artifacts/report_quality/prompt_variants/prompt_v5_streamlined_extraction.txt"
    v6_rev_path = PROJECT_ROOT / "artifacts/report_quality/prompt_variants/prompt_v6_grounding_review.txt"

    prompt_v4 = open(v4_prompt_path).read().strip() if v4_prompt_path.exists() else "Examine chest radiograph. Generate structured report."
    prompt_v5 = open(v5_ext_path).read().strip() if v5_ext_path.exists() else "Extract support devices into flat JSON."
    prompt_v6_template = open(v6_rev_path).read().strip() if v6_rev_path.exists() else "Review {query_phrase} in crop."

    records = []

    for i, case in enumerate(cases):
        is_cold = (i == 0)
        img_path = Path(case["image_path"])
        query = "pneumothorax" if "pneumothorax" in case["gold_affirmed_labels"] else "cardiomegaly"

        logger.info(f"Profiling Case {i+1}/5: {case['safe_case_id']} ({'COLD' if is_cold else 'WARM'})...")

        # Load image bytes
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        # Step 0: Image base64 encoding (initial load)
        t0 = time.perf_counter()
        with Image.open(BytesIO(img_bytes)) as pil_img:
            img_rgb = pil_img.convert("RGB")
            scale = min(1.0, 1024 / max(img_rgb.size))
            if scale < 1.0:
                img_rgb = img_rgb.resize((int(img_rgb.width * scale), int(img_rgb.height * scale)), Image.Resampling.LANCZOS)
            buf = BytesIO()
            img_rgb.save(buf, format="JPEG", quality=90)
            full_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        time_img_prep = time.perf_counter() - t0

        # Step 1: Whole-Image Report Generation (v4)
        v4_text, time_whole_image = ollama_generate(prompt_v4, [full_b64], num_predict=150)

        # Step 2: Dedicated Device Extraction (v5)
        v5_json_text, time_device_ext = ollama_generate(prompt_v5, [full_b64], num_predict=120)

        # Step 3: TransVG Grounding
        t0 = time.perf_counter()
        g_res = gs.predict(img_bytes, case["safe_case_id"] + ".jpg", query)
        time_grounding = time.perf_counter() - t0
        box = g_res.get("pred_box_pixel")

        # Step 4: Regional Image Preparation (Crop + Base64)
        t0 = time.perf_counter()
        crop_arr = extract_crop(img_bytes, case["safe_case_id"] + ".jpg", box, margin_px=40) if box else None
        crop_b64 = _array_to_base64_jpeg(crop_arr, max_side=512) if crop_arr is not None else None
        time_crop_prep = time.perf_counter() - t0

        # Step 5: Regional Review Inference (v6)
        v6_prompt = prompt_v6_template.replace("{query_phrase}", query)
        imgs_v6 = [full_b64, crop_b64] if crop_b64 else [full_b64]
        v6_text, time_reg_vlm = ollama_generate(v6_prompt, imgs_v6, num_predict=130)

        # Step 6: Cross-Report Consistency Checking
        t0 = time.perf_counter()
        reg_stance = extract_regional_assessment_state(v6_text)
        int_conflict, _ = check_whole_image_internal_contradiction(report_text=v4_text)
        cross_res = evaluate_cross_report_consistency(
            query=query,
            regional_report_text=v6_text,
            whole_image_report_text=v4_text,
        )
        time_consistency = time.perf_counter() - t0

        # Step 7: Report Synthesis and Parsing
        t0 = time.perf_counter()
        m_json = re.search(r"(\{.*\})", v5_json_text, re.DOTALL)
        parsed_json = {}
        if m_json:
            try:
                parsed_json = json.loads(m_json.group(1))
            except Exception:
                pass
        device_det = parsed_json.get("device_details", "No hardware reported")
        has_conflict = cross_res.has_conflict or int_conflict
        synth_report = (
            f"OBSERVATIONS & RECONCILED FINDINGS:\n"
            f"- Whole-Image Survey: {v4_text[:120]}...\n"
            f"- Dedicated Hardware Submodule: {device_det}\n"
            f"- Targeted Regional Grounding ({query}): {reg_stance.upper()}\n"
            f"CLINICAL TRIAGE: {'⚠️ Conflict Flagged' if has_conflict else 'Reconciled'}"
        )
        time_synthesis = time.perf_counter() - t0

        # Step 8: Total End-to-End Time
        total_e2e = (
            time_img_prep + time_whole_image + time_device_ext +
            time_grounding + time_crop_prep + time_reg_vlm +
            time_consistency + time_synthesis
        )

        records.append({
            "case_id": case["safe_case_id"],
            "run_mode": "cold_start" if is_cold else "warm_inference",
            "image_prep_s": round(time_img_prep, 4),
            "whole_image_vlm_s": round(time_whole_image, 4),
            "device_extraction_s": round(time_device_ext, 4),
            "transvg_grounding_s": round(time_grounding, 4),
            "crop_preparation_s": round(time_crop_prep, 4),
            "regional_review_vlm_s": round(time_reg_vlm, 4),
            "consistency_check_s": round(time_consistency, 4),
            "synthesis_and_parsing_s": round(time_synthesis, 4),
            "total_e2e_s": round(total_e2e, 4),
        })

    # Summary table
    df_raw = pd.DataFrame(records)
    
    # Calculate Component Summary across warm runs
    df_warm = df_raw[df_raw["run_mode"] == "warm_inference"]
    df_cold = df_raw[df_raw["run_mode"] == "cold_start"]

    summary_rows = []
    components = [
        ("1. Whole-Image VLM (Prompt v4)", "whole_image_vlm_s"),
        ("2. Dedicated Device Extraction (v5)", "device_extraction_s"),
        ("3. TransVG Grounding Localization", "transvg_grounding_s"),
        ("4. Regional Image Preparation (Crop)", "crop_preparation_s"),
        ("5. Regional Review VLM (Prompt v6)", "regional_review_vlm_s"),
        ("6. Cross-Report Consistency Service", "consistency_check_s"),
        ("7. Report Synthesis and Parsing", "synthesis_and_parsing_s"),
        ("Total End-to-End Pipeline", "total_e2e_s"),
    ]

    for label, col in components:
        summary_rows.append({
            "component": label,
            "cold_start_s": round(df_cold[col].iloc[0], 4) if len(df_cold) > 0 else 0.0,
            "warm_median_s": round(float(df_warm[col].median()), 4),
            "warm_p95_s": round(float(np.percentile(df_warm[col], 95)), 4),
            "warm_mean_s": round(float(df_warm[col].mean()), 4),
            "warm_min_s": round(float(df_warm[col].min()), 4),
            "warm_max_s": round(float(df_warm[col].max()), 4),
            "pct_of_total_warm": round(float(df_warm[col].median()) / float(df_warm["total_e2e_s"].median()) * 100, 2),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / "task10_latency_profile.csv", index=False)
    logger.info(f"Wrote {out_dir / 'task10_latency_profile.csv'}")
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    run_profiling()
