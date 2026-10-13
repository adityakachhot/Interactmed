"""
scripts/evaluate_prompt_v2_1.py
===============================
Controlled Hybrid Prompt (v2.1) Experiment on InteractMed-VLM Development Set:
Benchmarks:
  1. Optimized Prompt v2 (fast baseline, 4-line extraction)
  2. Structured Prompt v3 (5 discrete anatomical categories)
  3. Hybrid Prompt v2.1 (concise 4-line structure + 4-status differentiation + explicit laterality + synthesis consistency)

Strict Constraints:
  - Cohort: 25 development cases (split == 'dev'). Held-out test set locked.
  - Image Resolution: 1024 px maximum side.
  - Model: medgemma:4b via Ollama (temperature=0.1, keep_alive="30m").
  - Output Directory: artifacts/report_evaluation/prompt_v2_1/
"""

import argparse
import csv
import json
import os
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.batch_service import (
    _ollama_generate_vision_with_metrics,
    clean_and_structure_research_report,
)
from src.report_service import (
    _load_to_rgb_array,
    _array_to_base64_jpeg,
    _vision_model_available,
)

# ── Extraction Taxonomy & Scoring ─────────────────────────────────────────────

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

def extract_audited_labels(text: str) -> Tuple[Set[str], Set[str]]:
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

def check_laterality_status(ref_text: str, gen_text: str) -> str:
    ref_l = bool(re.search(r'\b(?:left|left-sided|left lung|left base|left hemithorax)\b', ref_text.lower()))
    ref_r = bool(re.search(r'\b(?:right|right-sided|right lung|right base|right hemithorax)\b', ref_text.lower()))
    gen_l = bool(re.search(r'\b(?:left|left-sided|left lung|left base|left hemithorax)\b', gen_text.lower()))
    gen_r = bool(re.search(r'\b(?:right|right-sided|right lung|right base|right hemithorax)\b', gen_text.lower()))
    gen_unspecified = bool(re.search(r'\bunspecified laterality\b', gen_text.lower()))

    if ref_l and not ref_r and gen_r and not gen_l:
        return "laterality_swap_left_to_right"
    if ref_r and not ref_l and gen_l and not gen_r:
        return "laterality_swap_right_to_left"
    if (ref_l or ref_r):
        if (ref_l and gen_l) or (ref_r and gen_r):
            return "concordant"
        if gen_unspecified:
            return "unspecified_stated"
        return "laterality_omitted"
    return "concordant_or_not_applicable"

def check_internal_contradiction(obs_text: str, imp_text: str) -> bool:
    obs_aff, _ = extract_audited_labels(obs_text)
    imp_aff, _ = extract_audited_labels(imp_text)
    obs_pathology = obs_aff - {'normal'}
    if obs_pathology and ('normal' in imp_aff and not (imp_aff - {'normal'})):
        return True
    return False

# ── Pipeline Runners ──────────────────────────────────────────────────────────

def run_pipeline_v2(b64_img: str, model: str) -> Tuple[str, Dict[str, Any], Dict[str, float]]:
    """Optimized Prompt v2 Baseline."""
    v_prompt = (
        "You are an expert radiologist. Inspect this chest radiograph and report findings concisely in 4 lines (no conversational introductions):\n"
        "1. Support apparatus: Identify any endotracheal tube, enteric tube, vascular line, chest tube, or surgical clips (or None).\n"
        "2. Cardiac silhouette: State if cardiothoracic ratio is normal or enlarged (>0.5).\n"
        "3. Lungs: Describe visible airspace opacities, consolidation, atelectasis, vascular congestion, edema, or state Clear.\n"
        "4. Pleural spaces: State if costophrenic angles are sharp or show blunting/effusion.\n"
        "Begin directly with 1. Support apparatus:"
    )
    t_v0 = time.perf_counter()
    raw_obs, v_m = _ollama_generate_vision_with_metrics(
        prompt=v_prompt,
        images_b64=[b64_img],
        model=model,
        num_predict=100,
        temperature=0.1,
        keep_alive="30m",
    )
    t_v = (time.perf_counter() - t_v0) * 1000

    s2_prompt = (
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
        prompt=s2_prompt,
        images_b64=[],
        model=model,
        num_predict=180,
        temperature=0.1,
        keep_alive="30m",
    )
    t_s2 = (time.perf_counter() - t_s2_0) * 1000

    timing = {
        "stage1_ms": t_v,
        "stage2_ms": t_s2,
        "total_ms": t_v + t_s2,
        "stage1_tokens": v_m.get("eval_count", 0),
        "stage2_tokens": s2_m.get("eval_count", 0),
        "total_tokens": v_m.get("eval_count", 0) + s2_m.get("eval_count", 0),
        "raw_obs": raw_obs,
    }
    return raw_text, s2_m, timing


def run_pipeline_v3(b64_img: str, model: str) -> Tuple[str, Dict[str, Any], Dict[str, float]]:
    """Structured Prompt v3 (5 Discrete Categories)."""
    v_prompt_v3 = (
        "You are an expert radiologist conducting structured visual feature extraction on this chest radiograph.\n"
        "For each category below, state: Observed (<details and laterality>) | Not identified | Indeterminate | Not assessable.\n\n"
        "1. Support apparatus: Inspect for endotracheal tube, enteric tube, vascular catheter/line, chest tube, pacemaker, or surgical clips.\n"
        "2. Cardiac silhouette: Normal, Enlarged (CTR > 0.5), or Indeterminate.\n"
        "3. Lungs & opacities: Consolidation, focal opacity, atelectasis, congestion, or edema. Specify laterality (right, left, bilateral, or unspecified laterality).\n"
        "4. Pleural spaces: Costophrenic angles sharp, blunted/effusion (specify laterality: right, left, bilateral, or unspecified laterality), or obscured.\n"
        "5. Mediastinum: Normal contour, widened, or indeterminate.\n\n"
        "RULES:\n"
        "- State laterality (right/left/bilateral) only when visually supported; if uncertain, state 'unspecified laterality'.\n"
        "- If equivocal or confidence is low, classify as Indeterminate; DO NOT default to normal.\n"
        "- Concise bullet list, no conversational filler.\n"
        "Begin directly with 1. Support apparatus:"
    )
    t_v0 = time.perf_counter()
    raw_obs, v_m = _ollama_generate_vision_with_metrics(
        prompt=v_prompt_v3,
        images_b64=[b64_img],
        model=model,
        num_predict=130,
        temperature=0.1,
        keep_alive="30m",
    )
    t_v = (time.perf_counter() - t_v0) * 1000

    s2_prompt_v3 = textwrap.dedent(f"""
        Structured visual findings from chest radiograph:
        \"\"\"{raw_obs}\"\"\"

        Synthesize an objective research report with EXACTLY these three headings:

        OBSERVATIONS
        Detail findings across support apparatus, cardiac silhouette, lungs, pleural spaces, and mediastinum. Incorporate supported laterality (right/left/bilateral) or state unspecified laterality when unverified.

        IMPRESSION
        Concisely summarize key findings and diagnostic uncertainty. Every abnormality or device noted in OBSERVATIONS must be explicitly addressed here. Never declare normal or no acute cardiopulmonary disease if abnormalities or indeterminate features are present.

        LIMITATIONS
        State technical projection limitations (single frontal view, no lateral projection).

        STRICT RULES:
        - Never echo prompt instructions, rules, or headings other than the three requested.
        - Never claim a frontal view shows only one side of the chest.
        - Do not invent measurements or prior comparisons.

        Begin directly with:
        OBSERVATIONS
    """).strip()
    t_s2_0 = time.perf_counter()
    raw_text, s2_m = _ollama_generate_vision_with_metrics(
        prompt=s2_prompt_v3,
        images_b64=[],
        model=model,
        num_predict=180,
        temperature=0.1,
        keep_alive="30m",
    )
    t_s2 = (time.perf_counter() - t_s2_0) * 1000

    timing = {
        "stage1_ms": t_v,
        "stage2_ms": t_s2,
        "total_ms": t_v + t_s2,
        "stage1_tokens": v_m.get("eval_count", 0),
        "stage2_tokens": s2_m.get("eval_count", 0),
        "total_tokens": v_m.get("eval_count", 0) + s2_m.get("eval_count", 0),
        "raw_obs": raw_obs,
    }
    return raw_text, s2_m, timing


def run_pipeline_v2_1(b64_img: str, model: str) -> Tuple[str, Dict[str, Any], Dict[str, float]]:
    """
    Hybrid Prompt v2.1:
      - Concise 4-line Stage 1 structure of v2 (fast, ~35-50 tokens).
      - 4-state assessment: Observed | Not identified | Indeterminate | Not assessable.
      - Explicit laterality: right, left, bilateral, or unspecified laterality.
      - Strict synthesis consistency: Clinically relevant abnormal or uncertain findings must be reflected in IMPRESSION.
      - Never defaults lack of findings to proof of normal.
    """
    v_prompt_v2_1 = (
        "You are an expert radiologist. Inspect this chest radiograph and report findings concisely in 4 lines:\n"
        "For each item, classify as: Observed (<details and laterality: right/left/bilateral/unspecified>) | Not identified | Indeterminate | Not assessable.\n"
        "1. Support apparatus: Inspect for endotracheal tube, enteric tube, vascular line, chest tube, or surgical clips.\n"
        "2. Cardiac silhouette: Normal, enlarged (CTR > 0.5), or indeterminate.\n"
        "3. Lungs: Inspect for opacity, consolidation, atelectasis, congestion, or edema (state laterality).\n"
        "4. Pleural spaces: Costophrenic angles sharp, blunted/effusion (state laterality), or obscured.\n"
        "Rule: If equivocal, state Indeterminate; lack of a detected finding is not proof of normal.\n"
        "Begin directly with 1. Support apparatus:"
    )
    t_v0 = time.perf_counter()
    raw_obs, v_m = _ollama_generate_vision_with_metrics(
        prompt=v_prompt_v2_1,
        images_b64=[b64_img],
        model=model,
        num_predict=110,
        temperature=0.1,
        keep_alive="30m",
    )
    t_v = (time.perf_counter() - t_v0) * 1000

    s2_prompt_v2_1 = textwrap.dedent(f"""
        Visual observations from chest radiograph:
        \"\"\"{raw_obs}\"\"\"

        Synthesize an objective research report with EXACTLY these three headings:

        OBSERVATIONS
        Describe image-supported findings across support apparatus, cardiac silhouette, lungs, and pleural spaces, retaining stated laterality (right/left/bilateral or unspecified).

        IMPRESSION
        Summarize primary findings and diagnostic uncertainty. Every clinically relevant abnormal or uncertain finding from Observations must be explicitly addressed here. Do not declare normal or absence of disease if abnormalities or indeterminate features were noted.

        LIMITATIONS
        State technical projection limitations (single frontal view, no lateral projection).

        STRICT RULES:
        - Never echo prompt instructions, rules, or extra headings.
        - Frontal projection displays bilateral lungs; never state it shows only one side.
        - Do not invent measurements or prior comparisons.

        Begin directly with:
        OBSERVATIONS
    """).strip()
    t_s2_0 = time.perf_counter()
    raw_text, s2_m = _ollama_generate_vision_with_metrics(
        prompt=s2_prompt_v2_1,
        images_b64=[],
        model=model,
        num_predict=180,
        temperature=0.1,
        keep_alive="30m",
    )
    t_s2 = (time.perf_counter() - t_s2_0) * 1000

    timing = {
        "stage1_ms": t_v,
        "stage2_ms": t_s2,
        "total_ms": t_v + t_s2,
        "stage1_tokens": v_m.get("eval_count", 0),
        "stage2_tokens": s2_m.get("eval_count", 0),
        "total_tokens": v_m.get("eval_count", 0) + s2_m.get("eval_count", 0),
        "raw_obs": raw_obs,
    }
    return raw_text, s2_m, timing

# ── Head-to-Head Evaluation Execution ─────────────────────────────────────────

def run_evaluation(limit: Optional[int] = None) -> Dict[str, Any]:
    print("=" * 80)
    print("INTERACTMED-VLM: THREE-WAY CONTROLLED PROMPT EXPERIMENT (v2 vs v3 vs v2.1)")
    print("Cohort: 25 Development Cases (Held-Out Test Set Permanently Locked)")
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

    out_dir = WORKSPACE_ROOT / "artifacts" / "report_evaluation" / "prompt_v2_1"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_v2: List[Dict[str, Any]] = []
    results_v3: List[Dict[str, Any]] = []
    results_v2_1: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []

    for idx, c in enumerate(dev_cases, start=1):
        cid = c["safe_case_id"]
        cat = c["primary_category"]
        gold_labels = set(c.get("gold_affirmed_labels", []))
        ref_text = c.get("reference_full_report", "")

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

        print(f"\n[{idx}/{len(dev_cases)}] Processing {cid} ({cat})...")

        t_d0 = time.perf_counter()
        with open(c["image_path"], "rb") as f:
            img_bytes = f.read()
        arr = _load_to_rgb_array(img_bytes, f"{cid}.jpg")
        b64_img = _array_to_base64_jpeg(arr, max_side=1024, quality=85)
        t_decode_ms = (time.perf_counter() - t_d0) * 1000

        # 1. Run v2
        print("  - Running v2...", end="", flush=True)
        t0 = time.perf_counter()
        raw_v2, m_v2, time_v2 = run_pipeline_v2(b64_img, model)
        tot_v2 = (time.perf_counter() - t0) * 1000 + t_decode_ms
        clean_v2, sec_v2, _ = clean_and_structure_research_report(raw_v2)
        print(f" Done ({tot_v2/1000:.2f}s | S1: {time_v2['stage1_ms']/1000:.2f}s, tokens: {time_v2['total_tokens']})")

        # 2. Run v3
        print("  - Running v3...", end="", flush=True)
        t0 = time.perf_counter()
        raw_v3, m_v3, time_v3 = run_pipeline_v3(b64_img, model)
        tot_v3 = (time.perf_counter() - t0) * 1000 + t_decode_ms
        clean_v3, sec_v3, _ = clean_and_structure_research_report(raw_v3)
        print(f" Done ({tot_v3/1000:.2f}s | S1: {time_v3['stage1_ms']/1000:.2f}s, tokens: {time_v3['total_tokens']})")

        # 3. Run v2.1
        print("  - Running v2.1 (Hybrid)...", end="", flush=True)
        t0 = time.perf_counter()
        raw_v2_1, m_v2_1, time_v2_1 = run_pipeline_v2_1(b64_img, model)
        tot_v2_1 = (time.perf_counter() - t0) * 1000 + t_decode_ms
        clean_v2_1, sec_v2_1, _ = clean_and_structure_research_report(raw_v2_1)
        print(f" Done ({tot_v2_1/1000:.2f}s | S1: {time_v2_1['stage1_ms']/1000:.2f}s, tokens: {time_v2_1['total_tokens']})")

        # Extract Audited Labels
        aff_v2, _ = extract_audited_labels(clean_v2)
        aff_v3, _ = extract_audited_labels(clean_v3)
        aff_v2_1, _ = extract_audited_labels(clean_v2_1)

        tp_v2 = sorted(list(true_set.intersection(aff_v2)))
        fp_v2 = sorted(list(aff_v2 - true_set))
        fn_v2 = sorted(list(true_set - aff_v2))

        tp_v3 = sorted(list(true_set.intersection(aff_v3)))
        fp_v3 = sorted(list(aff_v3 - true_set))
        fn_v3 = sorted(list(true_set - aff_v3))

        tp_v2_1 = sorted(list(true_set.intersection(aff_v2_1)))
        fp_v2_1 = sorted(list(aff_v2_1 - true_set))
        fn_v2_1 = sorted(list(true_set - aff_v2_1))

        lat_v2 = check_laterality_status(ref_text, clean_v2)
        lat_v3 = check_laterality_status(ref_text, clean_v3)
        lat_v2_1 = check_laterality_status(ref_text, clean_v2_1)

        contra_v2 = check_internal_contradiction(sec_v2.get("OBSERVATIONS", ""), sec_v2.get("IMPRESSION", ""))
        contra_v3 = check_internal_contradiction(sec_v3.get("OBSERVATIONS", ""), sec_v3.get("IMPRESSION", ""))
        contra_v2_1 = check_internal_contradiction(sec_v2_1.get("OBSERVATIONS", ""), sec_v2_1.get("IMPRESSION", ""))

        unsupp_neg_v2 = ["unwarranted_normal"] if ((true_set - {"normal"}) and ("normal" in aff_v2)) else []
        unsupp_neg_v3 = ["unwarranted_normal"] if ((true_set - {"normal"}) and ("normal" in aff_v3)) else []
        unsupp_neg_v2_1 = ["unwarranted_normal"] if ((true_set - {"normal"}) and ("normal" in aff_v2_1)) else []

        unsupp_pos_v2 = [p for p in fp_v2 if p != "normal"]
        unsupp_pos_v3 = [p for p in fp_v3 if p != "normal"]
        unsupp_pos_v2_1 = [p for p in fp_v2_1 if p != "normal"]

        bp_v2 = count_boilerplate(clean_v2)
        bp_v3 = count_boilerplate(clean_v3)
        bp_v2_1 = count_boilerplate(clean_v2_1)

        row = {
            "safe_case_id": cid,
            "primary_category": cat,
            "ground_truth_classes": "; ".join(sorted(list(true_set))),
            "v2_predictions": "; ".join(sorted(list(aff_v2))),
            "v3_predictions": "; ".join(sorted(list(aff_v3))),
            "v2_1_predictions": "; ".join(sorted(list(aff_v2_1))),
            "v2_missed": "; ".join(fn_v2),
            "v3_missed": "; ".join(fn_v3),
            "v2_1_missed": "; ".join(fn_v2_1),
            "v2_laterality": lat_v2,
            "v3_laterality": lat_v3,
            "v2_1_laterality": lat_v2_1,
            "v2_contradiction": "yes" if contra_v2 else "no",
            "v3_contradiction": "yes" if contra_v3 else "no",
            "v2_1_contradiction": "yes" if contra_v2_1 else "no",
            "v2_unsupported_neg": "; ".join(unsupp_neg_v2) if unsupp_neg_v2 else "none",
            "v3_unsupported_neg": "; ".join(unsupp_neg_v3) if unsupp_neg_v3 else "none",
            "v2_1_unsupported_neg": "; ".join(unsupp_neg_v2_1) if unsupp_neg_v2_1 else "none",
            "v2_latency_sec": round(tot_v2 / 1000, 2),
            "v3_latency_sec": round(tot_v3 / 1000, 2),
            "v2_1_latency_sec": round(tot_v2_1 / 1000, 2),
            "v2_s1_tokens": time_v2["stage1_tokens"],
            "v3_s1_tokens": time_v3["stage1_tokens"],
            "v2_1_s1_tokens": time_v2_1["stage1_tokens"],
        }
        comparison_rows.append(row)

        # Progressively save comparison CSV
        comp_csv = out_dir / "v2_1_comparison.csv"
        with open(comp_csv, "w", newline="") as f:
            fieldnames = list(comparison_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison_rows)

        results_v2.append({
            "safe_case_id": cid, "primary_category": cat, "total_ms": tot_v2,
            "stage1_ms": time_v2["stage1_ms"], "stage2_ms": time_v2["stage2_ms"],
            "stage1_tokens": time_v2["stage1_tokens"], "stage2_tokens": time_v2["stage2_tokens"],
            "total_tokens": time_v2["total_tokens"], "true_positives": tp_v2,
            "false_positives": fp_v2, "false_negatives": fn_v2, "laterality": lat_v2,
            "unsupported_neg": unsupp_neg_v2, "unsupported_pos": unsupp_pos_v2,
            "contradiction": contra_v2, "boilerplate": bp_v2, "cleaned_report": clean_v2,
            "raw_obs": time_v2["raw_obs"],
        })
        results_v3.append({
            "safe_case_id": cid, "primary_category": cat, "total_ms": tot_v3,
            "stage1_ms": time_v3["stage1_ms"], "stage2_ms": time_v3["stage2_ms"],
            "stage1_tokens": time_v3["stage1_tokens"], "stage2_tokens": time_v3["stage2_tokens"],
            "total_tokens": time_v3["total_tokens"], "true_positives": tp_v3,
            "false_positives": fp_v3, "false_negatives": fn_v3, "laterality": lat_v3,
            "unsupported_neg": unsupp_neg_v3, "unsupported_pos": unsupp_pos_v3,
            "contradiction": contra_v3, "boilerplate": bp_v3, "cleaned_report": clean_v3,
            "raw_obs": time_v3["raw_obs"],
        })
        results_v2_1.append({
            "safe_case_id": cid, "primary_category": cat, "total_ms": tot_v2_1,
            "stage1_ms": time_v2_1["stage1_ms"], "stage2_ms": time_v2_1["stage2_ms"],
            "stage1_tokens": time_v2_1["stage1_tokens"], "stage2_tokens": time_v2_1["stage2_tokens"],
            "total_tokens": time_v2_1["total_tokens"], "true_positives": tp_v2_1,
            "false_positives": fp_v2_1, "false_negatives": fn_v2_1, "laterality": lat_v2_1,
            "unsupported_neg": unsupp_neg_v2_1, "unsupported_pos": unsupp_pos_v2_1,
            "contradiction": contra_v2_1, "boilerplate": bp_v2_1, "cleaned_report": clean_v2_1,
            "raw_obs": time_v2_1["raw_obs"],
        })

    def compute_stats(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        lats = [r["total_ms"] for r in recs]
        s1_lats = [r["stage1_ms"] for r in recs]
        s2_lats = [r["stage2_ms"] for r in recs]
        toks = [r["total_tokens"] for r in recs]
        s1_toks = [r["stage1_tokens"] for r in recs]
        s2_toks = [r["stage2_tokens"] for r in recs]

        tp = sum(len(r["true_positives"]) for r in recs)
        fp = sum(len(r["false_positives"]) for r in recs)
        fn = sum(len(r["false_negatives"]) for r in recs)
        p = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        r = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0

        abnormal_recs = [r for r in recs if r["primary_category"] != "normal"]
        unsupp_neg_count = sum(1 for r in abnormal_recs if r["unsupported_neg"])
        unsupp_neg_rate = round(unsupp_neg_count / len(abnormal_recs), 4) if abnormal_recs else 0.0

        unsupp_pos_count = sum(1 for r in recs if r["unsupported_pos"])
        unsupp_pos_rate = round(unsupp_pos_count / len(recs), 4)

        bp_rate = round(sum(1 for r in recs if r["boilerplate"] >= 2) / len(recs), 4)
        contra_count = sum(1 for r in recs if r["contradiction"])
        contra_rate = round(contra_count / len(recs), 4)

        lat_omitted = sum(1 for r in recs if r["laterality"] == "laterality_omitted")
        lat_concordant = sum(1 for r in recs if r["laterality"] == "concordant")
        lat_swapped = sum(1 for r in recs if "laterality_swap" in r["laterality"])

        t_total_sec = sum(lats) / 1000
        total_eval_tokens = sum(toks)
        tok_per_sec = round(total_eval_tokens / t_total_sec, 2) if t_total_sec > 0 else 0.0

        return {
            "sample_size": len(recs),
            "abnormal_cohort_size": len(abnormal_recs),
            "micro_precision": p,
            "micro_recall": r,
            "micro_f1": f1,
            "total_tp": tp,
            "total_fp": fp,
            "total_fn": fn,
            "false_normal_count": unsupp_neg_count,
            "false_normal_rate_abnormal": unsupp_neg_rate,
            "unsupported_positive_rate": unsupp_pos_rate,
            "boilerplate_normal_rate": bp_rate,
            "internal_contradiction_count": contra_count,
            "internal_contradiction_rate": contra_rate,
            "laterality_omissions_count": lat_omitted,
            "laterality_concordant_count": lat_concordant,
            "laterality_swaps_count": lat_swapped,
            "mean_latency_ms": round(float(np.mean(lats)), 2),
            "median_latency_ms": round(float(np.median(lats)), 2),
            "std_latency_ms": round(float(np.std(lats)), 2),
            "mean_stage1_ms": round(float(np.mean(s1_lats)), 2),
            "mean_stage2_ms": round(float(np.mean(s2_lats)), 2),
            "mean_stage1_tokens": round(float(np.mean(s1_toks)), 1),
            "mean_stage2_tokens": round(float(np.mean(s2_toks)), 1),
            "mean_total_tokens": round(float(np.mean(toks)), 1),
            "tokens_per_second": tok_per_sec,
        }

    sum_v2 = compute_stats(results_v2)
    sum_v3 = compute_stats(results_v3)
    sum_v2_1 = compute_stats(results_v2_1)

    summary_data = {
        "evaluation_split": "dev",
        "sample_size": len(dev_cases),
        "pipeline_v2_optimized": sum_v2,
        "pipeline_v3_structured": sum_v3,
        "pipeline_v2_1_hybrid": sum_v2_1,
        "deltas_v2_1_vs_v2": {
            "f1_delta": round(sum_v2_1["micro_f1"] - sum_v2["micro_f1"], 4),
            "precision_delta": round(sum_v2_1["micro_precision"] - sum_v2["micro_precision"], 4),
            "recall_delta": round(sum_v2_1["micro_recall"] - sum_v2["micro_recall"], 4),
            "laterality_omissions_delta": sum_v2_1["laterality_omissions_count"] - sum_v2["laterality_omissions_count"],
            "mean_latency_delta_sec": round((sum_v2_1["mean_latency_ms"] - sum_v2["mean_latency_ms"]) / 1000, 2),
            "stage1_tokens_delta": round(sum_v2_1["mean_stage1_tokens"] - sum_v2["mean_stage1_tokens"], 1),
        },
        "deltas_v2_1_vs_v3": {
            "f1_delta": round(sum_v2_1["micro_f1"] - sum_v3["micro_f1"], 4),
            "precision_delta": round(sum_v2_1["micro_precision"] - sum_v3["micro_precision"], 4),
            "recall_delta": round(sum_v2_1["micro_recall"] - sum_v3["micro_recall"], 4),
            "laterality_omissions_delta": sum_v2_1["laterality_omissions_count"] - sum_v3["laterality_omissions_count"],
            "mean_latency_delta_sec": round((sum_v2_1["mean_latency_ms"] - sum_v3["mean_latency_ms"]) / 1000, 2),
            "stage1_tokens_delta": round(sum_v2_1["mean_stage1_tokens"] - sum_v3["mean_stage1_tokens"], 1),
        },
    }

    sum_json = out_dir / "v2_1_summary.json"
    with open(sum_json, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[Saved] Summary JSON to {sum_json}")

    case_json = out_dir / "v2_1_case_results.json"
    with open(case_json, "w") as f:
        json.dump({
            "pipeline_v2": results_v2,
            "pipeline_v3": results_v3,
            "pipeline_v2_1": results_v2_1,
        }, f, indent=2)
    print(f"[Saved] Case Results JSON to {case_json}")

    print("\n" + "=" * 90)
    print("PHASE D: THREE-WAY DEVELOPMENT EVALUATION SUMMARY (N=25)")
    print("=" * 90)
    print(f"{'Metric':<34} | {'v2 (Opt)':<14} | {'v3 (Struct)':<14} | {'v2.1 (Hybrid)':<14} | {'Delta(v2.1-v2)':<10}")
    print("-" * 96)
    print(f"{'Clinical Micro-F1':<34} | {sum_v2['micro_f1']:<14.4f} | {sum_v3['micro_f1']:<14.4f} | {sum_v2_1['micro_f1']:<14.4f} | {summary_data['deltas_v2_1_vs_v2']['f1_delta']:+<10.4f}")
    print(f"{'Clinical Micro-Precision':<34} | {sum_v2['micro_precision']:<14.4f} | {sum_v3['micro_precision']:<14.4f} | {sum_v2_1['micro_precision']:<14.4f} | {summary_data['deltas_v2_1_vs_v2']['precision_delta']:+<10.4f}")
    print(f"{'Clinical Micro-Recall':<34} | {sum_v2['micro_recall']:<14.4f} | {sum_v3['micro_recall']:<14.4f} | {sum_v2_1['micro_recall']:<14.4f} | {summary_data['deltas_v2_1_vs_v2']['recall_delta']:+<10.4f}")
    print(f"{'False-Normal Rate (Abnormal, N=19)':<34} | {sum_v2['false_normal_rate_abnormal']:<14.4f} | {sum_v3['false_normal_rate_abnormal']:<14.4f} | {sum_v2_1['false_normal_rate_abnormal']:<14.4f} | {sum_v2_1['false_normal_rate_abnormal']-sum_v2['false_normal_rate_abnormal']:+<10.4f}")
    print(f"{'Laterality Omissions (Count)':<34} | {sum_v2['laterality_omissions_count']:<14} | {sum_v3['laterality_omissions_count']:<14} | {sum_v2_1['laterality_omissions_count']:<14} | {summary_data['deltas_v2_1_vs_v2']['laterality_omissions_delta']:+<10}")
    print(f"{'Internal Contradictions (Count)':<34} | {sum_v2['internal_contradiction_count']:<14} | {sum_v3['internal_contradiction_count']:<14} | {sum_v2_1['internal_contradiction_count']:<14} | {sum_v2_1['internal_contradiction_count']-sum_v2['internal_contradiction_count']:+<10}")
    print(f"{'Mean End-to-End Latency (sec)':<34} | {sum_v2['mean_latency_ms']/1000:<14.2f} | {sum_v3['mean_latency_ms']/1000:<14.2f} | {sum_v2_1['mean_latency_ms']/1000:<14.2f} | {summary_data['deltas_v2_1_vs_v2']['mean_latency_delta_sec']:+<10.2f}")
    print(f"{'Median Latency (sec)':<34} | {sum_v2['median_latency_ms']/1000:<14.2f} | {sum_v3['median_latency_ms']/1000:<14.2f} | {sum_v2_1['median_latency_ms']/1000:<14.2f} | {(sum_v2_1['median_latency_ms']-sum_v2['median_latency_ms'])/1000:+<10.2f}")
    print(f"{'Mean Stage 1 Tokens':<34} | {sum_v2['mean_stage1_tokens']:<14.1f} | {sum_v3['mean_stage1_tokens']:<14.1f} | {sum_v2_1['mean_stage1_tokens']:<14.1f} | {summary_data['deltas_v2_1_vs_v2']['stage1_tokens_delta']:+<10.1f}")
    print(f"{'Throughput (tokens/sec)':<34} | {sum_v2['tokens_per_second']:<14.2f} | {sum_v3['tokens_per_second']:<14.2f} | {sum_v2_1['tokens_per_second']:<14.2f} | {sum_v2_1['tokens_per_second']-sum_v2['tokens_per_second']:+<10.2f}")
    print("=" * 90)

    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_evaluation(limit=args.limit)
