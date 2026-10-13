"""
scripts/evaluate_expanded_reports.py
====================================
Expanded Report Generation Evaluation for InteractMed-VLM:
Compares the Single-Stage Baseline vs the Production Prompt v3 Structured pipeline
on verified patient-stratified cases from the expanded cohort.

Strict Controls:
  - Hardware: Apple Silicon M3, 16 GB Unified Memory.
  - Model: medgemma:4b via Ollama (temperature=0.1, keep_alive="30m").
  - Resolution: 1024 px maximum side (LANCZOS, JPEG Q85).
  - Preprocessing: Identical aspect-ratio preserving array-to-jpeg.
  - Cohort: Loaded from artifacts/report_evaluation/expanded_cohort/expanded_reference_store.json.
  - Primary target partition: expanded_dev (preserves expanded_test untouched).
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

# ── Extraction Taxonomy & Audited Scoring ─────────────────────────────────────

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

def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculates Wilson score interval for binomial proportions."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996  # 95%
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return round(max(0.0, float(centre - half)), 4), round(min(1.0, float(centre + half)), 4)


# ── Pipeline Runners ──────────────────────────────────────────────────────────

def run_baseline_pipeline(b64_img: str, model: str) -> Tuple[str, Dict[str, Any], Dict[str, float]]:
    """Original single-stage free-form prompt."""
    prompt = (
        "You are an expert medical radiologist. Examine this chest radiograph carefully and generate an objective research report.\n"
        "Describe visible findings in the lungs, heart, and pleural spaces, followed by an impression.\n"
        "Begin directly with OBSERVATIONS:"
    )
    t0 = time.perf_counter()
    raw_text, metrics = _ollama_generate_vision_with_metrics(
        prompt=prompt,
        images_b64=[b64_img],
        model=model,
        num_predict=220,
        temperature=0.1,
        keep_alive="30m",
    )
    t_tot = (time.perf_counter() - t0) * 1000
    timing = {
        "stage1_ms": t_tot,
        "stage2_ms": 0.0,
        "total_ms": t_tot,
        "stage1_tokens": metrics.get("eval_count", 0),
        "stage2_tokens": 0,
        "total_tokens": metrics.get("eval_count", 0),
        "raw_obs": "",
    }
    return raw_text, metrics, timing


def run_prompt_v3_structured(b64_img: str, model: str) -> Tuple[str, Dict[str, Any], Dict[str, float]]:
    """Production 2-stage prompt (5 categories, 4 statuses, explicit laterality)."""
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


# ── Expanded Evaluation Engine ────────────────────────────────────────────────

# ── Expanded Evaluation Engine ────────────────────────────────────────────────

def run_expanded_evaluation(
    split: str = "expanded_dev",
    limit: Optional[int] = None,
    output_dir: Optional[str] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    print("=" * 80)
    print(f"INTERACTMED-VLM: EXPANDED COHORT EVALUATION (Partition: {split})")
    print("=" * 80)

    store_path = WORKSPACE_ROOT / "artifacts" / "report_evaluation" / "expanded_cohort" / "expanded_reference_store.json"
    if not os.path.exists(store_path):
        raise FileNotFoundError(f"Expanded reference store not found at {store_path}. Run audit_expanded_report_data.py first.")

    with open(store_path) as f:
        store = json.load(f)

    cohort_cases = [v for v in store.values() if v.get("split") == split]
    if limit:
        cohort_cases = cohort_cases[:limit]
    print(f"Loaded {len(cohort_cases)} cases from partition '{split}'.")

    model = _vision_model_available() or "medgemma:4b"
    print(f"Active vision model: {model}")

    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = WORKSPACE_ROOT / "artifacts" / "report_evaluation" / "expanded_cohort_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target artifact directory: {out_dir}")

    results_base: List[Dict[str, Any]] = []
    results_v3: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    evaluated_cids: Set[str] = set()

    case_json = out_dir / "expanded_case_results.json"
    comp_csv = out_dir / "expanded_comparison.csv"

    # Resume capability: load existing results if available
    if resume and case_json.exists() and comp_csv.exists():
        try:
            with open(case_json) as f:
                saved_cases = json.load(f)
            base_saved = saved_cases.get("baseline_single_stage", [])
            v3_saved = saved_cases.get("production_prompt_v3", [])
            base_map = {r["safe_case_id"]: r for r in base_saved}
            v3_map = {r["safe_case_id"]: r for r in v3_saved}

            with open(comp_csv, newline="") as f:
                reader = csv.DictReader(f)
                saved_comp = list(reader)

            for row in saved_comp:
                cid = row["safe_case_id"]
                if cid in base_map and cid in v3_map:
                    comparison_rows.append(row)
                    results_base.append(base_map[cid])
                    results_v3.append(v3_map[cid])
                    evaluated_cids.add(cid)
            print(f"[Resume] Loaded {len(evaluated_cids)} previously completed cases from {out_dir}.")
        except Exception as e:
            print(f"[Resume Warning] Could not load existing cases: {e}. Starting fresh.")
            results_base = []
            results_v3 = []
            comparison_rows = []
            evaluated_cids = set()

    for idx, c in enumerate(cohort_cases, start=1):
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

        if cid in evaluated_cids:
            print(f"[{idx}/{len(cohort_cases)}] {cid} ({cat}) already evaluated. Skipping inference.")
            continue

        print(f"\n[{idx}/{len(cohort_cases)}] Processing {cid} ({cat})...")

        t_d0 = time.perf_counter()
        with open(c["image_path"], "rb") as f:
            img_bytes = f.read()
        arr = _load_to_rgb_array(img_bytes, f"{cid}.jpg")
        b64_img = _array_to_base64_jpeg(arr, max_side=1024, quality=85)
        t_decode_ms = (time.perf_counter() - t_d0) * 1000

        # 1. Baseline Pipeline
        print("  - Running Baseline (Single-Stage)...", end="", flush=True)
        t0 = time.perf_counter()
        raw_b, m_b, time_b = run_baseline_pipeline(b64_img, model)
        tot_b = (time.perf_counter() - t0) * 1000 + t_decode_ms
        clean_b, sec_b, _ = clean_and_structure_research_report(raw_b)
        print(f" Done ({tot_b/1000:.2f}s | tokens: {time_b['total_tokens']})")

        # 2. Production Prompt v3
        print("  - Running Production Prompt v3 (Structured)...", end="", flush=True)
        t0 = time.perf_counter()
        raw_v3, m_v3, time_v3 = run_prompt_v3_structured(b64_img, model)
        tot_v3 = (time.perf_counter() - t0) * 1000 + t_decode_ms
        clean_v3, sec_v3, _ = clean_and_structure_research_report(raw_v3)
        print(f" Done ({tot_v3/1000:.2f}s | S1: {time_v3['stage1_ms']/1000:.2f}s, tokens: {time_v3['total_tokens']})")

        # Extract Audited Labels
        aff_b, _ = extract_audited_labels(clean_b)
        aff_v3, _ = extract_audited_labels(clean_v3)

        tp_b = sorted(list(true_set.intersection(aff_b)))
        fp_b = sorted(list(aff_b - true_set))
        fn_b = sorted(list(true_set - aff_b))

        tp_v3 = sorted(list(true_set.intersection(aff_v3)))
        fp_v3 = sorted(list(aff_v3 - true_set))
        fn_v3 = sorted(list(true_set - aff_v3))

        lat_b = check_laterality_status(ref_text, clean_b)
        lat_v3 = check_laterality_status(ref_text, clean_v3)

        contra_b = check_internal_contradiction(sec_b.get("OBSERVATIONS", ""), sec_b.get("IMPRESSION", ""))
        contra_v3 = check_internal_contradiction(sec_v3.get("OBSERVATIONS", ""), sec_v3.get("IMPRESSION", ""))

        unsupp_neg_b = ["unwarranted_normal"] if ((true_set - {"normal"}) and ("normal" in aff_b)) else []
        unsupp_neg_v3 = ["unwarranted_normal"] if ((true_set - {"normal"}) and ("normal" in aff_v3)) else []

        unsupp_pos_b = [p for p in fp_b if p != "normal"]
        unsupp_pos_v3 = [p for p in fp_v3 if p != "normal"]

        bp_b = count_boilerplate(clean_b)
        bp_v3 = count_boilerplate(clean_v3)

        row = {
            "safe_case_id": cid,
            "partition": split,
            "primary_category": cat,
            "ground_truth_classes": "; ".join(sorted(list(true_set))),
            "baseline_predictions": "; ".join(sorted(list(aff_b))),
            "prompt_v3_predictions": "; ".join(sorted(list(aff_v3))),
            "baseline_missed": "; ".join(fn_b),
            "prompt_v3_missed": "; ".join(fn_v3),
            "baseline_laterality": lat_b,
            "prompt_v3_laterality": lat_v3,
            "baseline_unsupported_neg": "; ".join(unsupp_neg_b) if unsupp_neg_b else "none",
            "prompt_v3_unsupported_neg": "; ".join(unsupp_neg_v3) if unsupp_neg_v3 else "none",
            "baseline_latency_sec": round(tot_b / 1000, 2),
            "prompt_v3_latency_sec": round(tot_v3 / 1000, 2),
            "baseline_tokens": time_b["total_tokens"],
            "prompt_v3_tokens": time_v3["total_tokens"],
        }
        comparison_rows.append(row)
        evaluated_cids.add(cid)

        results_base.append({
            "safe_case_id": cid, "primary_category": cat, "total_ms": tot_b,
            "total_tokens": time_b["total_tokens"], "true_positives": tp_b,
            "false_positives": fp_b, "false_negatives": fn_b, "laterality": lat_b,
            "unsupported_neg": unsupp_neg_b, "unsupported_pos": unsupp_pos_b,
            "contradiction": contra_b, "boilerplate": bp_b, "cleaned_report": clean_b,
        })
        results_v3.append({
            "safe_case_id": cid, "primary_category": cat, "total_ms": tot_v3,
            "total_tokens": time_v3["total_tokens"], "true_positives": tp_v3,
            "false_positives": fp_v3, "false_negatives": fn_v3, "laterality": lat_v3,
            "unsupported_neg": unsupp_neg_v3, "unsupported_pos": unsupp_pos_v3,
            "contradiction": contra_v3, "boilerplate": bp_v3, "cleaned_report": clean_v3,
        })

        # Progressively save comparison CSV & case results JSON after each case
        with open(comp_csv, "w", newline="") as f:
            fieldnames = list(comparison_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison_rows)

        with open(case_json, "w") as f:
            json.dump({
                "baseline_single_stage": results_base,
                "production_prompt_v3": results_v3,
            }, f, indent=2)

    # Summary Statistics
    def compute_stats(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not recs:
            return {}
        lats = [r["total_ms"] for r in recs]
        toks = [r["total_tokens"] for r in recs]

        tp = sum(len(r["true_positives"]) for r in recs)
        fp = sum(len(r["false_positives"]) for r in recs)
        fn = sum(len(r["false_negatives"]) for r in recs)

        p = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        r = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0

        p_ci = compute_wilson_ci(tp, tp + fp)
        r_ci = compute_wilson_ci(tp, tp + fn)

        abnormal_recs = [r for r in recs if r["primary_category"] != "normal"]
        unsupp_neg_count = sum(1 for r in abnormal_recs if r["unsupported_neg"])
        unsupp_neg_rate = round(unsupp_neg_count / len(abnormal_recs), 4) if abnormal_recs else 0.0

        unsupp_pos_count = sum(1 for r in recs if r["unsupported_pos"])
        unsupp_pos_rate = round(unsupp_pos_count / len(recs), 4)

        bp_count = sum(1 for r in recs if r["boilerplate"] >= 2)
        bp_rate = round(bp_count / len(recs), 4)

        contra_count = sum(1 for r in recs if r["contradiction"])
        contra_rate = round(contra_count / len(recs), 4)

        lat_omitted = sum(1 for r in recs if r["laterality"] == "laterality_omitted")
        lat_concordant = sum(1 for r in recs if r["laterality"] == "concordant")
        lat_swapped = sum(1 for r in recs if "laterality_swap" in r["laterality"])

        t_total_sec = sum(lats) / 1000
        total_eval_tokens = sum(toks)
        tok_per_sec = round(total_eval_tokens / t_total_sec, 2) if t_total_sec > 0 else 0.0

        # Category-level breakdown
        cat_stats = {}
        for cat in CATEGORIES:
            cat_recs = [r for r in recs if r["primary_category"] == cat]
            c_tp = sum(1 for r in cat_recs if cat in r["true_positives"])
            c_fn = sum(1 for r in cat_recs if cat in r["false_negatives"])
            c_fp = sum(1 for r in recs if cat in r["false_positives"])
            c_p = round(c_tp / (c_tp + c_fp), 4) if (c_tp + c_fp) > 0 else 0.0
            c_r = round(c_tp / (c_tp + c_fn), 4) if (c_tp + c_fn) > 0 else 0.0
            c_f1 = round(2 * c_p * c_r / (c_p + c_r), 4) if (c_p + c_r) > 0 else 0.0
            cat_stats[cat] = {
                "cohort_cases": len(cat_recs),
                "true_positives": c_tp,
                "false_positives": c_fp,
                "false_negatives": c_fn,
                "precision": c_p,
                "recall": c_r,
                "f1": c_f1,
            }

        return {
            "sample_size": len(recs),
            "abnormal_cohort_size": len(abnormal_recs),
            "micro_precision": p,
            "micro_precision_95_ci": list(p_ci),
            "micro_recall": r,
            "micro_recall_95_ci": list(r_ci),
            "micro_f1": f1,
            "total_tp": tp,
            "total_fp": fp,
            "total_fn": fn,
            "false_normal_count": unsupp_neg_count,
            "false_normal_rate_abnormal": unsupp_neg_rate,
            "unsupported_positive_rate": unsupp_pos_rate,
            "boilerplate_normal_count": bp_count,
            "boilerplate_normal_rate": bp_rate,
            "internal_contradiction_count": contra_count,
            "internal_contradiction_rate": contra_rate,
            "laterality_omissions_count": lat_omitted,
            "laterality_concordant_count": lat_concordant,
            "laterality_swaps_count": lat_swapped,
            "mean_latency_ms": round(float(np.mean(lats)), 2),
            "median_latency_ms": round(float(np.median(lats)), 2),
            "std_latency_ms": round(float(np.std(lats)), 2),
            "mean_total_tokens": round(float(np.mean(toks)), 1),
            "tokens_per_second": tok_per_sec,
            "category_level_metrics": cat_stats,
        }

    sum_base = compute_stats(results_base)
    sum_v3 = compute_stats(results_v3)

    summary_data = {
        "evaluation_partition": split,
        "sample_size": len(results_base),
        "target_cohort_size": len(cohort_cases),
        "baseline_single_stage": sum_base,
        "production_prompt_v3": sum_v3,
        "deltas_v3_vs_baseline": {
            "f1_delta": round(sum_v3["micro_f1"] - sum_base["micro_f1"], 4) if sum_v3 and sum_base else 0.0,
            "precision_delta": round(sum_v3["micro_precision"] - sum_base["micro_precision"], 4) if sum_v3 and sum_base else 0.0,
            "recall_delta": round(sum_v3["micro_recall"] - sum_base["micro_recall"], 4) if sum_v3 and sum_base else 0.0,
            "boilerplate_delta": round(sum_v3["boilerplate_normal_rate"] - sum_base["boilerplate_normal_rate"], 4) if sum_v3 and sum_base else 0.0,
            "laterality_omissions_delta": sum_v3.get("laterality_omissions_count", 0) - sum_base.get("laterality_omissions_count", 0) if sum_v3 and sum_base else 0,
            "mean_latency_delta_sec": round((sum_v3["mean_latency_ms"] - sum_base["mean_latency_ms"]) / 1000, 2) if sum_v3 and sum_base else 0.0,
        },
    }

    sum_json = out_dir / "expanded_summary.json"
    with open(sum_json, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n[Saved] Expanded Summary JSON to {sum_json}")
    print(f"[Saved] Expanded Case Results JSON to {case_json}")
    print(f"[Saved] Expanded Comparison CSV to {comp_csv}")

    print("\n" + "=" * 90)
    print(f"EXPANDED COHORT EVALUATION RESULTS (N={len(results_base)} from '{split}')")
    print("=" * 90)
    print(f"{'Metric':<34} | {'Baseline (1-Stage)':<20} | {'Prompt v3 (2-Stage)':<20} | {'Delta':<12}")
    print("-" * 92)
    print(f"{'Clinical Micro-F1':<34} | {sum_base['micro_f1']:<20.4f} | {sum_v3['micro_f1']:<20.4f} | {summary_data['deltas_v3_vs_baseline']['f1_delta']:+<12.4f}")
    print(f"{'Clinical Precision (95% CI)':<34} | {sum_base['micro_precision']:<6.4f} {str(sum_base['micro_precision_95_ci']):<13} | {sum_v3['micro_precision']:<6.4f} {str(sum_v3['micro_precision_95_ci']):<13} | {summary_data['deltas_v3_vs_baseline']['precision_delta']:+<12.4f}")
    print(f"{'Clinical Recall (95% CI)':<34} | {sum_base['micro_recall']:<6.4f} {str(sum_base['micro_recall_95_ci']):<13} | {sum_v3['micro_recall']:<6.4f} {str(sum_v3['micro_recall_95_ci']):<13} | {summary_data['deltas_v3_vs_baseline']['recall_delta']:+<12.4f}")
    print(f"{'Boilerplate Normal Rate':<34} | {sum_base['boilerplate_normal_rate']:<20.4f} | {sum_v3['boilerplate_normal_rate']:<20.4f} | {summary_data['deltas_v3_vs_baseline']['boilerplate_delta']:+<12.4f}")
    print(f"{'False-Normal Rate (Abnormal)':<34} | {sum_base['false_normal_rate_abnormal']:<20.4f} | {sum_v3['false_normal_rate_abnormal']:<20.4f} | {sum_v3['false_normal_rate_abnormal']-sum_base['false_normal_rate_abnormal']:+<12.4f}")
    print(f"{'Laterality Omissions (Count)':<34} | {sum_base['laterality_omissions_count']:<20} | {sum_v3['laterality_omissions_count']:<20} | {summary_data['deltas_v3_vs_baseline']['laterality_omissions_delta']:+<12}")
    print(f"{'Concordant Laterality (Count)':<34} | {sum_base['laterality_concordant_count']:<20} | {sum_v3['laterality_concordant_count']:<20} | {sum_v3['laterality_concordant_count']-sum_base['laterality_concordant_count']:+<12}")
    print(f"{'Internal Contradictions (Count)':<34} | {sum_base['internal_contradiction_count']:<20} | {sum_v3['internal_contradiction_count']:<20} | {sum_v3['internal_contradiction_count']-sum_base['internal_contradiction_count']:+<12}")
    print(f"{'Mean End-to-End Latency (sec)':<34} | {sum_base['mean_latency_ms']/1000:<20.2f} | {sum_v3['mean_latency_ms']/1000:<20.2f} | {summary_data['deltas_v3_vs_baseline']['mean_latency_delta_sec']:+<12.2f}")
    print(f"{'Median Latency (sec)':<34} | {sum_base['median_latency_ms']/1000:<20.2f} | {sum_v3['median_latency_ms']/1000:<20.2f} | {(sum_v3['median_latency_ms']-sum_base['median_latency_ms'])/1000:+<12.2f}")
    print(f"{'Throughput (tokens/sec)':<34} | {sum_base['tokens_per_second']:<20.2f} | {sum_v3['tokens_per_second']:<20.2f} | {sum_v3['tokens_per_second']-sum_base['tokens_per_second']:+<12.2f}")
    print("=" * 90)

    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="expanded_dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=str(WORKSPACE_ROOT / "artifacts" / "report_evaluation" / "expanded_cohort_full"))
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from existing results")
    args = parser.parse_args()
    run_expanded_evaluation(
        split=args.split,
        limit=args.limit,
        output_dir=args.output_dir,
        resume=not args.no_resume,
    )

