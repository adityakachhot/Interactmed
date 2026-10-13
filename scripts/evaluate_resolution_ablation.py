"""
scripts/evaluate_resolution_ablation.py
=======================================
Phase C Controlled Resolution Experiment for InteractMed-VLM:
Compares 1024 px vs 1536 px maximum-side image inputs on the 25 development cases.

Strict Constraints:
  - Fixed Prompt: Selected Prompt v3 structured pipeline.
  - Fixed Model: Local medgemma:4b via Ollama.
  - Fixed Generation Settings: temperature=0.1, num_predict=(130, 180), keep_alive="30m".
  - Cohort: 25 development cases (split == 'dev'). Held-out test set locked.
  - Outputs saved to:
      artifacts/report_evaluation/resolution_ablation/
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

# ── Extraction Rules ──────────────────────────────────────────────────────────

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

# ── Pipeline Runner ───────────────────────────────────────────────────────────

def run_prompt_v3_with_image(b64_img: str, model: str) -> Tuple[str, Dict[str, Any], Dict[str, float]]:
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
        "stage1_vision_ms": t_v,
        "stage1_prompt_eval_ms": v_m.get("prompt_eval_duration_ns", 0) / 1e6,
        "stage2_synthesis_ms": t_s2,
        "total_inference_ms": t_v + t_s2,
        "stage1_tokens": v_m.get("eval_count", 0),
        "stage2_tokens": s2_m.get("eval_count", 0),
        "total_tokens": v_m.get("eval_count", 0) + s2_m.get("eval_count", 0),
        "stage1_raw_obs": raw_obs,
    }
    return raw_text, s2_m, timing

# ── Resolution Experiment Execution ──────────────────────────────────────────

def run_resolution_experiment(limit: Optional[int] = None) -> Dict[str, Any]:
    print("=" * 80)
    print("INTERACTMED-VLM: PHASE C CONTROLLED RESOLUTION EXPERIMENT (1024px vs 1536px)")
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

    out_dir = WORKSPACE_ROOT / "artifacts" / "report_evaluation" / "resolution_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_1024: List[Dict[str, Any]] = []
    results_1536: List[Dict[str, Any]] = []
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

        with open(c["image_path"], "rb") as f:
            img_bytes = f.read()
        arr = _load_to_rgb_array(img_bytes, f"{cid}.jpg")

        # Encode 1024px and 1536px
        t0 = time.perf_counter()
        b64_1024 = _array_to_base64_jpeg(arr, max_side=1024, quality=85)
        t_prep_1024 = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        b64_1536 = _array_to_base64_jpeg(arr, max_side=1536, quality=85)
        t_prep_1536 = (time.perf_counter() - t0) * 1000

        # Run 1024px
        print(f"  - Running 1024px input (payload {len(b64_1024)//1024} KB)...", end="", flush=True)
        t0 = time.perf_counter()
        raw_1024, m_1024, time_1024 = run_prompt_v3_with_image(b64_1024, model)
        tot_1024 = (time.perf_counter() - t0) * 1000 + t_prep_1024
        clean_1024, sec_1024, warn_1024 = clean_and_structure_research_report(raw_1024)
        print(f" Done ({tot_1024/1000:.2f}s | S1: {time_1024['stage1_vision_ms']/1000:.2f}s, prompt_eval: {time_1024['stage1_prompt_eval_ms']:.0f}ms)")

        # Run 1536px
        print(f"  - Running 1536px input (payload {len(b64_1536)//1024} KB)...", end="", flush=True)
        t0 = time.perf_counter()
        raw_1536, m_1536, time_1536 = run_prompt_v3_with_image(b64_1536, model)
        tot_1536 = (time.perf_counter() - t0) * 1000 + t_prep_1536
        clean_1536, sec_1536, warn_1536 = clean_and_structure_research_report(raw_1536)
        print(f" Done ({tot_1536/1000:.2f}s | S1: {time_1536['stage1_vision_ms']/1000:.2f}s, prompt_eval: {time_1536['stage1_prompt_eval_ms']:.0f}ms)")

        # Extract Audited Labels
        aff_1024, _ = extract_audited_labels(clean_1024)
        aff_1536, _ = extract_audited_labels(clean_1536)

        tp_1024 = sorted(list(true_set.intersection(aff_1024)))
        fp_1024 = sorted(list(aff_1024 - true_set))
        fn_1024 = sorted(list(true_set - aff_1024))

        tp_1536 = sorted(list(true_set.intersection(aff_1536)))
        fp_1536 = sorted(list(aff_1536 - true_set))
        fn_1536 = sorted(list(true_set - aff_1536))

        lat_1024 = check_laterality_status(ref_text, clean_1024)
        lat_1536 = check_laterality_status(ref_text, clean_1536)

        unsupp_neg_1024 = ["unwarranted_normal"] if ((true_set - {"normal"}) and ("normal" in aff_1024)) else []
        unsupp_neg_1536 = ["unwarranted_normal"] if ((true_set - {"normal"}) and ("normal" in aff_1536)) else []

        row = {
            "safe_case_id": cid,
            "primary_category": cat,
            "ground_truth_classes": "; ".join(sorted(list(true_set))),
            "preds_1024px": "; ".join(sorted(list(aff_1024))),
            "preds_1536px": "; ".join(sorted(list(aff_1536))),
            "missed_1024px": "; ".join(fn_1024),
            "missed_1536px": "; ".join(fn_1536),
            "unsupported_neg_1024px": "; ".join(unsupp_neg_1024) if unsupp_neg_1024 else "none",
            "unsupported_neg_1536px": "; ".join(unsupp_neg_1536) if unsupp_neg_1536 else "none",
            "laterality_1024px": lat_1024,
            "laterality_1536px": lat_1536,
            "payload_kb_1024px": len(b64_1024) // 1024,
            "payload_kb_1536px": len(b64_1536) // 1024,
            "latency_1024px_sec": round(tot_1024 / 1000, 2),
            "latency_1536px_sec": round(tot_1536 / 1000, 2),
            "prompt_eval_1024px_ms": round(time_1024["stage1_prompt_eval_ms"], 1),
            "prompt_eval_1536px_ms": round(time_1536["stage1_prompt_eval_ms"], 1),
        }
        comparison_rows.append(row)

        # Progressively save comparison CSV
        comp_csv = out_dir / "resolution_comparison.csv"
        with open(comp_csv, "w", newline="") as f:
            fieldnames = list(comparison_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison_rows)

        results_1024.append({
            "safe_case_id": cid,
            "primary_category": cat,
            "prep_ms": round(t_prep_1024, 2),
            "stage1_ms": round(time_1024["stage1_vision_ms"], 2),
            "prompt_eval_ms": round(time_1024["stage1_prompt_eval_ms"], 2),
            "stage2_ms": round(time_1024["stage2_synthesis_ms"], 2),
            "total_ms": round(tot_1024, 2),
            "total_tokens": time_1024["total_tokens"],
            "true_positives": tp_1024,
            "false_positives": fp_1024,
            "false_negatives": fn_1024,
            "laterality": lat_1024,
            "unsupported_negative": unsupp_neg_1024,
            "cleaned_report": clean_1024,
        })

        results_1536.append({
            "safe_case_id": cid,
            "primary_category": cat,
            "prep_ms": round(t_prep_1536, 2),
            "stage1_ms": round(time_1536["stage1_vision_ms"], 2),
            "prompt_eval_ms": round(time_1536["stage1_prompt_eval_ms"], 2),
            "stage2_ms": round(time_1536["stage2_synthesis_ms"], 2),
            "total_ms": round(tot_1536, 2),
            "total_tokens": time_1536["total_tokens"],
            "true_positives": tp_1536,
            "false_positives": fp_1536,
            "false_negatives": fn_1536,
            "laterality": lat_1536,
            "unsupported_negative": unsupp_neg_1536,
            "cleaned_report": clean_1536,
        })

    # Summary Statistics
    def compute_stats(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        lats = [r["total_ms"] for r in recs]
        s1_lats = [r["stage1_ms"] for r in recs]
        pe_lats = [r["prompt_eval_ms"] for r in recs]
        toks = [r["total_tokens"] for r in recs]

        tp = sum(len(r["true_positives"]) for r in recs)
        fp = sum(len(r["false_positives"]) for r in recs)
        fn = sum(len(r["false_negatives"]) for r in recs)
        p = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        r = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0

        abnormal_recs = [r for r in recs if r["primary_category"] != "normal"]
        unsupp_neg_count = sum(1 for r in abnormal_recs if r["unsupported_negative"])
        unsupp_neg_rate = round(unsupp_neg_count / len(abnormal_recs), 4) if abnormal_recs else 0.0

        lat_omitted = sum(1 for r in recs if r["laterality"] == "laterality_omitted")
        lat_concordant = sum(1 for r in recs if r["laterality"] == "concordant")

        return {
            "sample_size": len(recs),
            "micro_precision": p,
            "micro_recall": r,
            "micro_f1": f1,
            "total_tp": tp,
            "total_fp": fp,
            "total_fn": fn,
            "false_normal_rate_abnormal": unsupp_neg_rate,
            "laterality_omissions_count": lat_omitted,
            "laterality_concordant_count": lat_concordant,
            "mean_latency_ms": round(float(np.mean(lats)), 2),
            "median_latency_ms": round(float(np.median(lats)), 2),
            "std_latency_ms": round(float(np.std(lats)), 2),
            "mean_prompt_eval_ms": round(float(np.mean(pe_lats)), 2),
            "mean_stage1_ms": round(float(np.mean(s1_lats)), 2),
            "mean_total_tokens": round(float(np.mean(toks)), 1),
        }

    sum_1024 = compute_stats(results_1024)
    sum_1536 = compute_stats(results_1536)

    # Label agreement rate
    agree = sum(1 for r1, r2 in zip(results_1024, results_1536) if set(r1["true_positives"] + r1["false_positives"]) == set(r2["true_positives"] + r2["false_positives"]))
    agree_rate = round(agree / len(dev_cases), 4)

    summary_data = {
        "evaluation_split": "dev",
        "sample_size": len(dev_cases),
        "fixed_pipeline": "prompt_v3_structured",
        "resolution_1024px": sum_1024,
        "resolution_1536px": sum_1536,
        "label_agreement_rate": agree_rate,
        "deltas_1536_vs_1024": {
            "f1_delta": round(sum_1536["micro_f1"] - sum_1024["micro_f1"], 4),
            "precision_delta": round(sum_1536["micro_precision"] - sum_1024["micro_precision"], 4),
            "recall_delta": round(sum_1536["micro_recall"] - sum_1024["micro_recall"], 4),
            "false_normal_delta": round(sum_1536["false_normal_rate_abnormal"] - sum_1024["false_normal_rate_abnormal"], 4),
            "mean_latency_delta_sec": round((sum_1536["mean_latency_ms"] - sum_1024["mean_latency_ms"]) / 1000, 2),
            "prompt_eval_delta_ms": round(sum_1536["mean_prompt_eval_ms"] - sum_1024["mean_prompt_eval_ms"], 2),
        },
    }

    sum_json = out_dir / "resolution_summary.json"
    with open(sum_json, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[Saved] Resolution Summary JSON to {sum_json}")

    case_json = out_dir / "resolution_cases.json"
    with open(case_json, "w") as f:
        json.dump({
            "resolution_1024px": results_1024,
            "resolution_1536px": results_1536,
        }, f, indent=2)
    print(f"[Saved] Resolution Cases JSON to {case_json}")

    # Display Summary Table
    print("\n" + "=" * 80)
    print("PHASE C: RESOLUTION ABLATION EXPERIMENT SUMMARY (1024px vs 1536px)")
    print("=" * 80)
    print(f"{'Metric':<36} | {'1024 px':<16} | {'1536 px':<16} | {'Delta':<10}")
    print("-" * 88)
    print(f"{'Clinical Micro-F1':<36} | {sum_1024['micro_f1']:<16.4f} | {sum_1536['micro_f1']:<16.4f} | {summary_data['deltas_1536_vs_1024']['f1_delta']:+<10.4f}")
    print(f"{'Clinical Micro-Precision':<36} | {sum_1024['micro_precision']:<16.4f} | {sum_1536['micro_precision']:<16.4f} | {summary_data['deltas_1536_vs_1024']['precision_delta']:+<10.4f}")
    print(f"{'Clinical Micro-Recall':<36} | {sum_1024['micro_recall']:<16.4f} | {sum_1536['micro_recall']:<16.4f} | {summary_data['deltas_1536_vs_1024']['recall_delta']:+<10.4f}")
    print(f"{'False-Normal Rate (Abnormal, N=19)':<36} | {sum_1024['false_normal_rate_abnormal']:<16.4f} | {sum_1536['false_normal_rate_abnormal']:<16.4f} | {summary_data['deltas_1536_vs_1024']['false_normal_delta']:+<10.4f}")
    print(f"{'Laterality Omissions (Count)':<36} | {sum_1024['laterality_omissions_count']:<16} | {sum_1536['laterality_omissions_count']:<16} | {sum_1536['laterality_omissions_count'] - sum_1024['laterality_omissions_count']:+<10}")
    print(f"{'Mean Latency (seconds)':<36} | {sum_1024['mean_latency_ms']/1000:<16.2f} | {sum_1536['mean_latency_ms']/1000:<16.2f} | {summary_data['deltas_1536_vs_1024']['mean_latency_delta_sec']:+<10.2f}")
    print(f"{'Prompt Eval Latency (ms)':<36} | {sum_1024['mean_prompt_eval_ms']:<16.1f} | {sum_1536['mean_prompt_eval_ms']:<16.1f} | {summary_data['deltas_1536_vs_1024']['prompt_eval_delta_ms']:+<10.1f}")
    print(f"{'Label Agreement Rate':<36} | {'—':<16} | {'—':<16} | {agree_rate*100:.1f}%")
    print("=" * 80)

    return summary_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_resolution_experiment(limit=args.limit)
