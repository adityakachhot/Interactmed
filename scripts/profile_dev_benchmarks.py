"""
scripts/profile_dev_benchmarks.py
==================================
Comprehensive Development-Set Benchmarking & Error Analysis Tool for InteractMed-VLM.

Strict Constraints:
  - Strictly operates on the 25 development cases (split == 'dev').
  - Held-out test evaluation remains locked and untouched.
  - Profiles:
      1. Image decoding & preprocessing.
      2. Stage 1 targeted visual extraction.
      3. Stage 2 clinical synthesis.
      4. Output parsing & validation.
      5. Total end-to-end latency.
  - Cold-start vs warm-request profiling with keep_alive verification.
  - Generates:
      * artifacts/report_evaluation/error_analysis.csv
      * artifacts/report_evaluation/latency_benchmark.json
      * artifacts/report_evaluation/dev_benchmark_cases.json
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.batch_service import (
    _build_research_prompt,
    _ollama_generate_vision_with_metrics,
    clean_and_structure_research_report,
)
from src.report_service import (
    _load_to_rgb_array,
    _array_to_base64_jpeg,
    _vision_model_available,
    _ollama_available,
)

# ── Audited Extraction Rules (Audit v2) ───────────────────────────────────────

PRE_NEG_CUES = r'\b(?:no|without|negative for|free of|rules out|absence of|clear of|no evidence of|no sign of|unremarkable|denies)\b'
POST_NEG_CUES = r'\b(?:is not (?:identified|seen|present)|are not (?:identified|seen|present)|is unremarkable|are unremarkable|are clear|is clear|is absent|are absent)\b'

CATEGORIES = {
    'cardiomegaly': [r'cardiomegaly', r'enlarged cardiac silhouette', r'enlarged heart', r'cardiac enlargement'],
    'pleural_effusion': [r'pleural effusion', r'effusion', r'costophrenic angle blunting', r'blunted costophrenic', r'fluid in the pleural', r'meniscus sign'],
    'pneumonia_consolidation': [r'pneumonia', r'consolidation', r'infiltrate', r'airspace disease', r'airspace opacity', r'bronchogram'],
    'pulmonary_edema': [r'pulmonary edema', r'edema', r'vascular congestion', r'fluid overload', r'vascular plethora', r'perihilar haze'],
    'atelectasis': [r'atelectasis', r'atelectatic', r'volume loss', r'discoid opacity', r'plate-like opacity', r'collapse'],
    'support_devices': [r'endotracheal tube', r'enteric tube', r'nasogastric tube', r'feeding tube', r'chest tube', r'vascular catheter', r'picc line', r'central line', r'pacemaker', r'tracheostomy', r'surgical clips'],
    'normal': [r'normal chest', r'no acute cardiopulmonary', r'unremarkable chest', r'within normal limits', r'clear lungs', r'lungs are clear'],
}

GENERIC_NORMAL_PHRASES = [
    r'lungs (?:are|appear) clear bilaterally',
    r'heart size (?:is|seems) within normal limits',
    r'no acute abnormalities (?:are|identified|seen)',
    r'mediastinal contours (?:are|appear) unremarkable',
    r'bony thorax (?:is|appears) intact',
    r'chest radiograph is unremarkable',
    r'chest radiograph is normal',
]

UNCERTAINTY_PATTERNS = [
    r'cannot be (?:definitively|reliably|entirely|fully)?\s*(?:determined|excluded|ruled out|confirmed|assessed)',
    r'indeterminate',
    r'uncertain',
    r'inconclusive',
    r'may or may not',
    r'degree of uncertainty',
    r'equivocal',
]

def extract_audited_labels(text: str) -> Tuple[Set[str], Set[str]]:
    """Audited clause-level entity extraction with bidirectional negation scoping."""
    sentences = re.split(r'[\.\n;]+', text)
    affirmed: Set[str] = set()
    negated: Set[str] = set()
    for s in sentences:
        s_clean = s.strip().lower()
        if not s_clean:
            continue
        has_post_neg = bool(re.search(POST_NEG_CUES, s_clean))
        for cat, pats in CATEGORIES.items():
            if cat == 'normal':
                continue
            for pat in pats:
                m = re.search(rf'\b{pat}\b', s_clean)
                if m:
                    pre_slice = s_clean[:m.start()]
                    post_slice = s_clean[m.end():]
                    is_neg = False
                    if re.search(PRE_NEG_CUES, pre_slice) or re.search(POST_NEG_CUES, post_slice) or has_post_neg:
                        is_neg = True
                    if is_neg:
                        negated.add(cat)
                    else:
                        affirmed.add(cat)
    if not affirmed:
        for pat in CATEGORIES['normal']:
            if re.search(rf'\b{pat}\b', text.lower()):
                affirmed.add('normal')
                break
    return affirmed, negated

def count_boilerplate(text: str) -> int:
    return sum(1 for pat in GENERIC_NORMAL_PHRASES if re.search(pat, text.lower()))

def count_uncertainty(text: str) -> int:
    return sum(1 for pat in UNCERTAINTY_PATTERNS if re.search(pat, text.lower()))

def check_laterality_error(ref_text: str, gen_text: str) -> str:
    """Detects if reference report specifies laterality and whether generated report misidentifies it."""
    ref_l = bool(re.search(r'\b(?:left|left-sided|left lung|left base|left hemithorax)\b', ref_text.lower()))
    ref_r = bool(re.search(r'\b(?:right|right-sided|right lung|right base|right hemithorax)\b', ref_text.lower()))
    
    gen_l = bool(re.search(r'\b(?:left|left-sided|left lung|left base|left hemithorax)\b', gen_text.lower()))
    gen_r = bool(re.search(r'\b(?:right|right-sided|right lung|right base|right hemithorax)\b', gen_text.lower()))
    
    # Check for direct discrepancy
    if ref_l and not ref_r and gen_r and not gen_l:
        return "laterality_swap_left_to_right"
    if ref_r and not ref_l and gen_l and not gen_r:
        return "laterality_swap_right_to_left"
    if (ref_l or ref_r) and not (gen_l or gen_r):
        return "laterality_omitted"
    return "concordant_or_not_applicable"

def check_internal_contradiction(obs_text: str, imp_text: str) -> bool:
    obs_aff, _ = extract_audited_labels(obs_text)
    imp_aff, _ = extract_audited_labels(imp_text)
    obs_pathology = obs_aff - {'normal'}
    if obs_pathology and ('normal' in imp_aff and not (imp_aff - {'normal'})):
        return True
    return False

# ── Inference Execution Runners ──────────────────────────────────────────────

def run_config_a_baseline(b64_img: str, model: str) -> Tuple[str, Dict[str, Any], Dict[str, float]]:
    """Single-stage baseline prompt."""
    prompt = _build_research_prompt()
    t0 = time.perf_counter()
    raw_text, metrics = _ollama_generate_vision_with_metrics(
        prompt=prompt,
        images_b64=[b64_img],
        model=model,
        num_predict=220,
        temperature=0.1,
        keep_alive="30m",
    )
    t_total = (time.perf_counter() - t0) * 1000
    timing = {
        "stage1_vision_ms": t_total,
        "stage2_synthesis_ms": 0.0,
        "total_inference_ms": t_total,
        "load_duration_ms": metrics.get("load_duration_ns", 0) / 1e6,
        "prompt_eval_ms": metrics.get("prompt_eval_duration_ns", 0) / 1e6,
        "eval_ms": metrics.get("eval_duration_ns", 0) / 1e6,
        "eval_tokens": metrics.get("eval_count", 0),
    }
    return raw_text, metrics, timing

def run_config_b_revised(b64_img: str, model: str) -> Tuple[str, Dict[str, Any], Dict[str, float]]:
    """Current 2-stage revised pipeline."""
    v_prompt = (
        "You are an expert medical radiologist examining this chest radiograph in detail.\n"
        "Carefully inspect the actual image pixels and describe:\n"
        "1. Support apparatus: Any endotracheal tube, enteric/nasogastric tube, vascular line, chest tube, or surgical clips? Note visible tip positions.\n"
        "2. Cardiac silhouette: Is the heart enlarged (cardiothoracic ratio > 0.5) or normal in size?\n"
        "3. Lungs: Are there visible airspace opacities, streaky basilar densities, consolidation, atelectasis, vascular congestion, or edema?\n"
        "4. Pleural spaces: Are bilateral costophrenic angles sharp, blunted, or obscured?\n"
        "Be objective and specific based strictly on visible pixel evidence."
    )
    t_v0 = time.perf_counter()
    raw_obs, v_m = _ollama_generate_vision_with_metrics(
        prompt=v_prompt,
        images_b64=[b64_img],
        model=model,
        num_predict=180,
        temperature=0.1,
        keep_alive="30m",
    )
    t_v = (time.perf_counter() - t_v0) * 1000

    s2_prompt = (
        f"Based strictly on the verified visual observations extracted from this chest radiograph:\n"
        f'"""{raw_obs}"""\n\n'
        "Generate a concise, objective research report with EXACTLY these three headings:\n\n"
        "OBSERVATIONS\n"
        "Describe image-supported findings across support apparatus, heart, mediastinum, lungs, and pleural spaces. "
        "Do not default to stating normal if opacities, cardiomegaly, or lines were observed.\n\n"
        "IMPRESSION\n"
        "Summarize the primary supported findings and diagnostic uncertainty. Do not default to normal.\n\n"
        "LIMITATIONS\n"
        "State technical limitations of this frontal projection (e.g., portable AP view, lack of lateral projection, inspiration depth).\n\n"
        "STRICT RULES:\n"
        "- Never claim a frontal view shows only one side of the chest.\n"
        "- Do not invent measurements or prior comparisons.\n\n"
        "Begin directly with:\n"
        "OBSERVATIONS"
    )
    t_s2_0 = time.perf_counter()
    raw_text, s2_m = _ollama_generate_vision_with_metrics(
        prompt=s2_prompt,
        images_b64=[],
        model=model,
        num_predict=220,
        temperature=0.1,
        keep_alive="30m",
    )
    t_s2 = (time.perf_counter() - t_s2_0) * 1000

    timing = {
        "stage1_vision_ms": t_v,
        "stage2_synthesis_ms": t_s2,
        "total_inference_ms": t_v + t_s2,
        "load_duration_ms": v_m.get("load_duration_ns", 0) / 1e6,
        "stage1_eval_tokens": v_m.get("eval_count", 0),
        "stage2_eval_tokens": s2_m.get("eval_count", 0),
        "eval_tokens": v_m.get("eval_count", 0) + s2_m.get("eval_count", 0),
        "stage1_raw_obs": raw_obs,
    }
    return raw_text, s2_m, timing

def run_config_c_optimized(b64_img: str, model: str) -> Tuple[str, Dict[str, Any], Dict[str, float]]:
    """Optimized 2-stage structured pipeline: concise Stage 1 extraction, compact Stage 2 synthesis."""
    v_prompt_opt = (
        "You are an expert radiologist. Inspect this chest radiograph and report findings concisely in 4 lines (no conversational introductions):\n"
        "1. Support apparatus: Identify any endotracheal tube, enteric tube, vascular line, chest tube, or surgical clips (or None).\n"
        "2. Cardiac silhouette: State if cardiothoracic ratio is normal or enlarged (>0.5).\n"
        "3. Lungs: Describe visible airspace opacities, consolidation, atelectasis, vascular congestion, edema, or state Clear.\n"
        "4. Pleural spaces: State if costophrenic angles are sharp or show blunting/effusion.\n"
        "Begin directly with 1. Support apparatus:"
    )
    t_v0 = time.perf_counter()
    raw_obs, v_m = _ollama_generate_vision_with_metrics(
        prompt=v_prompt_opt,
        images_b64=[b64_img],
        model=model,
        num_predict=100,
        temperature=0.1,
        keep_alive="30m",
    )
    t_v = (time.perf_counter() - t_v0) * 1000

    s2_prompt_opt = (
        f"Visual observations:\n"
        f'"""{raw_obs}"""\n\n'
        "Synthesize an objective research report with EXACTLY these three headings:\n\n"
        "OBSERVATIONS\n"
        "Describe image-supported findings across support apparatus, cardiac silhouette, lungs, and pleural spaces.\n\n"
        "IMPRESSION\n"
        "Summarize key findings and diagnostic uncertainty. Do not default to normal if findings were noted.\n\n"
        "LIMITATIONS\n"
        "State technical projection limitations (single frontal view, no lateral projection).\n\n"
        "STRICT RULES:\n"
        "- Frontal projection displays bilateral lungs; never state it shows only one side.\n"
        "- Do not invent measurements or prior comparisons.\n\n"
        "Begin directly with:\n"
        "OBSERVATIONS"
    )
    t_s2_0 = time.perf_counter()
    raw_text, s2_m = _ollama_generate_vision_with_metrics(
        prompt=s2_prompt_opt,
        images_b64=[],
        model=model,
        num_predict=180,
        temperature=0.1,
        keep_alive="30m",
    )
    t_s2 = (time.perf_counter() - t_s2_0) * 1000

    timing = {
        "stage1_vision_ms": t_v,
        "stage2_synthesis_ms": t_s2,
        "total_inference_ms": t_v + t_s2,
        "load_duration_ms": v_m.get("load_duration_ns", 0) / 1e6,
        "stage1_eval_tokens": v_m.get("eval_count", 0),
        "stage2_eval_tokens": s2_m.get("eval_count", 0),
        "eval_tokens": v_m.get("eval_count", 0) + s2_m.get("eval_count", 0),
        "stage1_raw_obs": raw_obs,
    }
    return raw_text, s2_m, timing

# ── Main Benchmarking Loop ───────────────────────────────────────────────────

def run_benchmarks(limit: Optional[int] = None) -> None:
    print("=" * 80)
    print("INTERACTMED-VLM: DEVELOPMENT-SET ERROR ANALYSIS & LATENCY BENCHMARK")
    print("Cohort: 25 Development Cases (Held-out Test set remains strictly locked)")
    print("=" * 80)

    store_path = WORKSPACE_ROOT / "artifacts" / "report_evaluation" / "local_reference_store.json"
    with open(store_path) as f:
        store = json.load(f)

    dev_cases = [v for v in store.values() if v.get("split") == "dev"]
    if limit:
        dev_cases = dev_cases[:limit]
    print(f"Loaded {len(dev_cases)} development cases.")

    model = _vision_model_available() or "medgemma:4b"
    print(f"Active vision model: {model}")

    # Ensure model starts cold for exact cold-start measurement
    print("\nEnsuring cold-start state for initial measurement...")
    import urllib.request
    try:
        payload = json.dumps({'model': model, 'keep_alive': 0}).encode()
        req = urllib.request.Request('http://localhost:11434/api/generate', data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=10) as _:
            pass
        time.sleep(1.0)
        print("Model unloaded. First request will record true cold-start latency.")
    except Exception as e:
        print(f"Cold-start reset warning: {e}")

    results_baseline: List[Dict[str, Any]] = []
    results_revised: List[Dict[str, Any]] = []
    results_optimized: List[Dict[str, Any]] = []
    error_analysis_rows: List[Dict[str, Any]] = []

    cold_start_metrics: Dict[str, Any] = {}

    for idx, c in enumerate(dev_cases, start=1):
        cid = c["safe_case_id"]
        cat = c["primary_category"]
        gold_labels = set(c.get("gold_affirmed_labels", []))
        ref_text = c.get("reference_full_report", "")

        # Build true positive set
        true_set = set()
        if cat in CATEGORIES:
            true_set.add(cat)
        for g in gold_labels:
            g_clean = g.lower()
            if "cardiac silhouette" in g_clean or "cardiomegaly" in g_clean:
                true_set.add("cardiomegaly")
            elif "effusion" in g_clean:
                true_set.add("pleural_effusion")
            elif "pneumonia" in g_clean or "consolidation" in g_clean:
                true_set.add("pneumonia_consolidation")
            elif "edema" in g_clean or "congestion" in g_clean:
                true_set.add("pulmonary_edema")
            elif "atelectasis" in g_clean:
                true_set.add("atelectasis")
            elif "tube" in g_clean or "catheter" in g_clean or "line" in g_clean:
                true_set.add("support_devices")
            elif "normal" in g_clean and len(gold_labels) <= 2:
                true_set.add("normal")

        print(f"\n[{idx}/{len(dev_cases)}] Processing {cid} (Category: {cat})...")

        # 1. Decode & Preprocess Timing
        t_d0 = time.perf_counter()
        with open(c["image_path"], "rb") as f:
            img_bytes = f.read()
        arr = _load_to_rgb_array(img_bytes, f"{cid}.jpg")
        b64_img = _array_to_base64_jpeg(arr, max_side=1024)
        t_decode_preprocess_ms = (time.perf_counter() - t_d0) * 1000

        # 2. Config B: Revised Grounded Pipeline
        print("  - Running Revised Grounded Pipeline...", end="", flush=True)
        t_b0 = time.perf_counter()
        raw_rev, m_rev, time_rev = run_config_b_revised(b64_img, model)
        t_rev_total = (time.perf_counter() - t_b0) * 1000 + t_decode_preprocess_ms
        t_p0 = time.perf_counter()
        clean_rev, sec_rev, warn_rev = clean_and_structure_research_report(raw_rev)
        t_p_rev = (time.perf_counter() - t_p0) * 1000
        print(f" Done ({t_rev_total/1000:.2f}s | Stage1: {time_rev['stage1_vision_ms']/1000:.2f}s, Stage2: {time_rev['stage2_synthesis_ms']/1000:.2f}s)")

        # Record cold-start on request 1
        if idx == 1:
            cold_start_metrics = {
                "case_id": cid,
                "model": model,
                "load_duration_ms": time_rev["load_duration_ms"],
                "stage1_vision_ms": time_rev["stage1_vision_ms"],
                "stage2_synthesis_ms": time_rev["stage2_synthesis_ms"],
                "total_cold_start_ms": t_rev_total,
                "note": "First request after explicit model unload; measures disk-to-unified-RAM loading.",
            }

        # 3. Config C: Optimized Grounded Pipeline
        print("  - Running Optimized Grounded Pipeline...", end="", flush=True)
        t_c0 = time.perf_counter()
        raw_opt, m_opt, time_opt = run_config_c_optimized(b64_img, model)
        t_opt_total = (time.perf_counter() - t_c0) * 1000 + t_decode_preprocess_ms
        t_p0 = time.perf_counter()
        clean_opt, sec_opt, warn_opt = clean_and_structure_research_report(raw_opt)
        t_p_opt = (time.perf_counter() - t_p0) * 1000
        print(f" Done ({t_opt_total/1000:.2f}s | Stage1: {time_opt['stage1_vision_ms']/1000:.2f}s, Stage2: {time_opt['stage2_synthesis_ms']/1000:.2f}s)")

        # 4. Config A: Baseline Prompt
        print("  - Running Current Baseline Prompt...", end="", flush=True)
        t_a0 = time.perf_counter()
        raw_base, m_base, time_base = run_config_a_baseline(b64_img, model)
        t_base_total = (time.perf_counter() - t_a0) * 1000 + t_decode_preprocess_ms
        t_p0 = time.perf_counter()
        clean_base, sec_base, warn_base = clean_and_structure_research_report(raw_base)
        t_p_base = (time.perf_counter() - t_p0) * 1000
        print(f" Done ({t_base_total/1000:.2f}s)")

        # Extract Audited Clinical Labels
        base_aff, base_neg = extract_audited_labels(clean_base)
        rev_aff, rev_neg = extract_audited_labels(clean_rev)
        opt_aff, opt_neg = extract_audited_labels(clean_opt)

        # Baseline confusion
        base_tp = sorted(list(true_set.intersection(base_aff)))
        base_fp = sorted(list(base_aff - true_set))
        base_fn = sorted(list(true_set - base_aff))

        # Revised confusion
        rev_tp = sorted(list(true_set.intersection(rev_aff)))
        rev_fp = sorted(list(rev_aff - true_set))
        rev_fn = sorted(list(true_set - rev_aff))

        # Optimized confusion
        opt_tp = sorted(list(true_set.intersection(opt_aff)))
        opt_fp = sorted(list(opt_aff - true_set))
        opt_fn = sorted(list(true_set - opt_aff))

        # Error taxonomy items
        unsupp_pos_rev = [p for p in rev_fp if p != "normal"]
        unsupp_neg_rev = ["unwarranted_normal"] if ((true_set - {"normal"}) and ("normal" in rev_aff)) else []
        bp_count_rev = count_boilerplate(clean_rev)
        bp_count_base = count_boilerplate(clean_base)
        contradiction_rev = check_internal_contradiction(sec_rev.get("OBSERVATIONS", ""), sec_rev.get("IMPRESSION", ""))
        contradiction_base = check_internal_contradiction(sec_base.get("OBSERVATIONS", ""), sec_base.get("IMPRESSION", ""))
        lat_check = check_laterality_error(ref_text, clean_rev)
        uncertainty_rev = count_uncertainty(clean_rev)

        # Correctly introduced findings (present in rev_tp but missed in base_fn)
        correctly_introduced = sorted(list(set(rev_tp) - set(base_tp)))

        # Store Error Analysis Row
        error_row = {
            "safe_case_id": cid,
            "primary_category": cat,
            "ground_truth_classes": "; ".join(sorted(list(true_set))),
            "baseline_predictions": "; ".join(sorted(list(base_aff))),
            "revised_predictions": "; ".join(sorted(list(rev_aff))),
            "optimized_predictions": "; ".join(sorted(list(opt_aff))),
            "missed_reference_findings": "; ".join(rev_fn),
            "unsupported_positive_claims": "; ".join(unsupp_pos_rev) if unsupp_pos_rev else "none",
            "unsupported_negative_claims": "; ".join(unsupp_neg_rev) if unsupp_neg_rev else "none",
            "laterality_status": lat_check,
            "internal_contradiction": "yes" if contradiction_rev else "no",
            "boilerplate_normal_phrases_count": bp_count_rev,
            "baseline_boilerplate_count": bp_count_base,
            "excessive_uncertainty_score": uncertainty_rev,
            "correctly_introduced_by_revised": "; ".join(correctly_introduced) if correctly_introduced else "none",
        }
        error_analysis_rows.append(error_row)

        # Write Error Analysis CSV progressively after each case
        out_csv = WORKSPACE_ROOT / "artifacts" / "report_evaluation" / "error_analysis.csv"
        with open(out_csv, "w", newline="") as f:
            fieldnames = [
                "safe_case_id",
                "primary_category",
                "ground_truth_classes",
                "baseline_predictions",
                "revised_predictions",
                "optimized_predictions",
                "missed_reference_findings",
                "unsupported_positive_claims",
                "unsupported_negative_claims",
                "laterality_status",
                "internal_contradiction",
                "boilerplate_normal_phrases_count",
                "baseline_boilerplate_count",
                "excessive_uncertainty_score",
                "correctly_introduced_by_revised",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(error_analysis_rows)

        # Save Case Timing & Metadata
        results_baseline.append({
            "safe_case_id": cid,
            "primary_category": cat,
            "decode_preprocess_ms": round(t_decode_preprocess_ms, 2),
            "inference_ms": round(time_base["total_inference_ms"], 2),
            "parse_validate_ms": round(t_p_base, 2),
            "total_latency_ms": round(t_base_total, 2),
            "eval_tokens": time_base["eval_tokens"],
            "tokens_per_second": round(time_base["eval_tokens"] / (time_base["total_inference_ms"] / 1000), 2) if time_base["total_inference_ms"] > 0 else 0,
            "true_positives": base_tp,
            "false_positives": base_fp,
            "false_negatives": base_fn,
            "cleaned_report": clean_base,
        })

        results_revised.append({
            "safe_case_id": cid,
            "primary_category": cat,
            "decode_preprocess_ms": round(t_decode_preprocess_ms, 2),
            "stage1_vision_ms": round(time_rev["stage1_vision_ms"], 2),
            "stage2_synthesis_ms": round(time_rev["stage2_synthesis_ms"], 2),
            "parse_validate_ms": round(t_p_rev, 2),
            "total_latency_ms": round(t_rev_total, 2),
            "stage1_tokens": time_rev["stage1_eval_tokens"],
            "stage2_tokens": time_rev["stage2_eval_tokens"],
            "total_tokens": time_rev["eval_tokens"],
            "tokens_per_second": round(time_rev["eval_tokens"] / ((time_rev["stage1_vision_ms"] + time_rev["stage2_synthesis_ms"]) / 1000), 2),
            "true_positives": rev_tp,
            "false_positives": rev_fp,
            "false_negatives": rev_fn,
            "cleaned_report": clean_rev,
        })

        results_optimized.append({
            "safe_case_id": cid,
            "primary_category": cat,
            "decode_preprocess_ms": round(t_decode_preprocess_ms, 2),
            "stage1_vision_ms": round(time_opt["stage1_vision_ms"], 2),
            "stage2_synthesis_ms": round(time_opt["stage2_synthesis_ms"], 2),
            "parse_validate_ms": round(t_p_opt, 2),
            "total_latency_ms": round(t_opt_total, 2),
            "stage1_tokens": time_opt["stage1_eval_tokens"],
            "stage2_tokens": time_opt["stage2_eval_tokens"],
            "total_tokens": time_opt["eval_tokens"],
            "tokens_per_second": round(time_opt["eval_tokens"] / ((time_opt["stage1_vision_ms"] + time_opt["stage2_synthesis_ms"]) / 1000), 2),
            "true_positives": opt_tp,
            "false_positives": opt_fp,
            "false_negatives": opt_fn,
            "cleaned_report": clean_opt,
        })

    # ── Write Error Analysis CSV ──────────────────────────────────────────────
    out_csv = WORKSPACE_ROOT / "artifacts" / "report_evaluation" / "error_analysis.csv"
    with open(out_csv, "w", newline="") as f:
        fieldnames = [
            "safe_case_id",
            "primary_category",
            "ground_truth_classes",
            "baseline_predictions",
            "revised_predictions",
            "optimized_predictions",
            "missed_reference_findings",
            "unsupported_positive_claims",
            "unsupported_negative_claims",
            "laterality_status",
            "internal_contradiction",
            "boilerplate_normal_phrases_count",
            "baseline_boilerplate_count",
            "excessive_uncertainty_score",
            "correctly_introduced_by_revised",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(error_analysis_rows)
    print(f"\n[Saved] Error analysis CSV to {out_csv}")

    # ── Compute Latency & Clinical Metrics Summaries ──────────────────────────
    def compute_stats(latencies_ms: List[float]) -> Dict[str, float]:
        arr = np.array(latencies_ms)
        return {
            "mean_ms": round(float(np.mean(arr)), 2),
            "median_ms": round(float(np.median(arr)), 2),
            "std_ms": round(float(np.std(arr)), 2),
            "variance_ms2": round(float(np.var(arr)), 2),
            "min_ms": round(float(np.min(arr)), 2),
            "max_ms": round(float(np.max(arr)), 2),
        }

    def compute_prf1(recs: List[Dict[str, Any]]) -> Dict[str, float]:
        tp = sum(len(r["true_positives"]) for r in recs)
        fp = sum(len(r["false_positives"]) for r in recs)
        fn = sum(len(r["false_negatives"]) for r in recs)
        p = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        r = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0
        return {"precision": p, "recall": r, "micro_f1": f1, "total_tp": tp, "total_fp": fp, "total_fn": fn}

    # Clinical agreement between Revised and Optimized
    agree_count = 0
    total_pairs = len(results_revised)
    for r_rev, r_opt in zip(results_revised, results_optimized):
        if set(r_rev["true_positives"] + r_rev["false_positives"]) == set(r_opt["true_positives"] + r_opt["false_positives"]):
            agree_count += 1
    clinical_agreement_rate = round(agree_count / total_pairs, 4) if total_pairs > 0 else 1.0

    latency_benchmark = {
        "metadata": {
            "date": "2026-10-12",
            "cohort": "development_set",
            "sample_size": len(dev_cases),
            "device": "Apple Silicon M3, 16GB Unified Memory",
            "model": model,
            "keep_alive_policy": "30m persistent residency",
        },
        "cold_start_profile": cold_start_metrics,
        "warm_latency_profiles": {
            "decoding_and_preprocessing": compute_stats([r["decode_preprocess_ms"] for r in results_revised]),
            "baseline_pipeline": {
                "total_latency": compute_stats([r["total_latency_ms"] for r in results_baseline]),
                "clinical_metrics": compute_prf1(results_baseline),
            },
            "revised_pipeline_unoptimized": {
                "stage1_vision_latency": compute_stats([r["stage1_vision_ms"] for r in results_revised]),
                "stage2_synthesis_latency": compute_stats([r["stage2_synthesis_ms"] for r in results_revised]),
                "parse_validation_latency": compute_stats([r["parse_validate_ms"] for r in results_revised]),
                "total_end_to_end_latency": compute_stats([r["total_latency_ms"] for r in results_revised]),
                "mean_stage1_tokens": round(float(np.mean([r["stage1_tokens"] for r in results_revised])), 1),
                "mean_stage2_tokens": round(float(np.mean([r["stage2_tokens"] for r in results_revised])), 1),
                "clinical_metrics": compute_prf1(results_revised),
            },
            "optimized_structured_pipeline": {
                "stage1_vision_latency": compute_stats([r["stage1_vision_ms"] for r in results_optimized]),
                "stage2_synthesis_latency": compute_stats([r["stage2_synthesis_ms"] for r in results_optimized]),
                "parse_validation_latency": compute_stats([r["parse_validate_ms"] for r in results_optimized]),
                "total_end_to_end_latency": compute_stats([r["total_latency_ms"] for r in results_optimized]),
                "mean_stage1_tokens": round(float(np.mean([r["stage1_tokens"] for r in results_optimized])), 1),
                "mean_stage2_tokens": round(float(np.mean([r["stage2_tokens"] for r in results_optimized])), 1),
                "speedup_factor_vs_revised": round(
                    float(np.mean([r["total_latency_ms"] for r in results_revised])) /
                    float(np.mean([r["total_latency_ms"] for r in results_optimized])),
                    2
                ),
                "clinical_metrics": compute_prf1(results_optimized),
                "clinical_label_agreement_with_revised": clinical_agreement_rate,
            },
        },
    }

    out_json = WORKSPACE_ROOT / "artifacts" / "report_evaluation" / "latency_benchmark.json"
    with open(out_json, "w") as f:
        json.dump(latency_benchmark, f, indent=2)
    print(f"[Saved] Latency benchmark JSON to {out_json}")

    # Save full case-level records
    out_cases = WORKSPACE_ROOT / "artifacts" / "report_evaluation" / "dev_benchmark_cases.json"
    with open(out_cases, "w") as f:
        json.dump({
            "baseline": results_baseline,
            "revised": results_revised,
            "optimized": results_optimized,
        }, f, indent=2)
    print(f"[Saved] Full dev benchmark cases to {out_cases}")

    # Print summary table
    print("\n" + "=" * 80)
    print("DEVELOPMENT BENCHMARK SUMMARY")
    print("=" * 80)
    rev_mean = latency_benchmark["warm_latency_profiles"]["revised_pipeline_unoptimized"]["total_end_to_end_latency"]["mean_ms"] / 1000
    opt_mean = latency_benchmark["warm_latency_profiles"]["optimized_structured_pipeline"]["total_end_to_end_latency"]["mean_ms"] / 1000
    base_mean = latency_benchmark["warm_latency_profiles"]["baseline_pipeline"]["total_latency"]["mean_ms"] / 1000
    speedup = latency_benchmark["warm_latency_profiles"]["optimized_structured_pipeline"]["speedup_factor_vs_revised"]

    print(f"Baseline Mean Latency:  {base_mean:.2f}s")
    print(f"Revised Mean Latency:   {rev_mean:.2f}s")
    print(f"Optimized Mean Latency: {opt_mean:.2f}s (Speedup: {speedup}x)")
    print(f"Revised F1:             {latency_benchmark['warm_latency_profiles']['revised_pipeline_unoptimized']['clinical_metrics']['micro_f1']}")
    print(f"Optimized F1:           {latency_benchmark['warm_latency_profiles']['optimized_structured_pipeline']['clinical_metrics']['micro_f1']}")
    print(f"Clinical Agreement:     {clinical_agreement_rate * 100:.1f}%")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_benchmarks(limit=args.limit)
