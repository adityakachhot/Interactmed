"""
scripts/evaluate_report_quality.py
==================================
Senior Medical VLM Researcher Evaluation Tool for InteractMed-VLM.

Evaluates and compares Whole-Chest-X-Ray research report generation:
  - Current prompt vs Revised prompt pipeline.
  - Development cohort (25 cases) and Held-out evaluation cohort (25 cases).
  - Quantitative Clinical Metrics:
      * Finding-label agreement (Precision, Recall, F1 across 7 CheXpert/MIMIC categories).
      * Factual grounding rate.
  - Quantitative Textual Metrics:
      * ROUGE-1, ROUGE-2, ROUGE-L, and BLEU-2.
  - Qualitative Error Taxonomy:
      * Unsupported positive claims (hallucinations).
      * Incorrect / unsupported negative claims (unwarranted normal assertions).
      * Repeated generic normal boilerplate rate.
      * Missed important reference findings.
      * Internal contradictions between sections.
      * Latency and generation throughput (tokens/sec).

Usage:
  /Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/evaluate_report_quality.py --split dev --mode both
  /Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/evaluate_report_quality.py --split held_out_test --mode both
"""

from __future__ import annotations

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
from PIL import Image

# Ensure workspace root is in sys.path
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
    _ollama_available,
    _vision_model_available,
)

# Optional NLP metric libraries
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False


# ── Clinical Entity & Pathology Extraction Rules ─────────────────────────────

PATHOLOGY_CLASSES = [
    "normal",
    "cardiomegaly",
    "pleural_effusion",
    "pneumonia_consolidation",
    "pulmonary_edema",
    "atelectasis",
    "support_devices",
]

CLINICAL_PATTERNS: Dict[str, List[str]] = {
    "cardiomegaly": [
        r"cardiomegaly",
        r"enlarged cardiac silhouette",
        r"enlarged heart",
        r"cardiac enlargement",
        r"cardiothoracic ratio\s*(?:is\s*)?(?:increased|greater than 0\.5|elevated)",
    ],
    "pleural_effusion": [
        r"pleural effusion",
        r"effusion",
        r"blunting of the (?:left|right|bilateral) costophrenic",
        r"blunted costophrenic",
        r"fluid in the pleural",
        r"meniscus sign",
    ],
    "pneumonia_consolidation": [
        r"pneumonia",
        r"consolidation",
        r"infiltrate",
        r"airspace disease",
        r"airspace opacity",
        r"bronchogram",
    ],
    "pulmonary_edema": [
        r"pulmonary edema",
        r"edema",
        r"vascular congestion",
        r"fluid overload",
        r"vascular plethora",
        r"perihilar haze",
        r"cephalization",
        r"kerley",
    ],
    "atelectasis": [
        r"atelectasis",
        r"atelectatic",
        r"volume loss",
        r"discoid opacity",
        r"plate-like opacity",
        r"collapse",
    ],
    "support_devices": [
        r"endotracheal tube",
        r"enteric tube",
        r"nasogastric tube",
        r"feeding tube",
        r"chest tube",
        r"vascular catheter",
        r"picc",
        r"pacemaker",
        r"lead",
        r"tracheostomy",
        r"hardware",
    ],
    "normal": [
        r"normal chest radiograph",
        r"normal heart size and clear lungs",
        r"no acute cardiopulmonary",
        r"no acute findings",
        r"unremarkable chest",
        r"within normal limits",
        r"no acute abnormality",
        r"lungs are clear bilaterally",
    ],
}

GENERIC_NORMAL_PHRASES = [
    r"lungs (?:are|appear) clear bilaterally",
    r"heart size (?:is|seems) within normal limits",
    r"no acute abnormalities (?:are|identified|seen)",
    r"mediastinal contours (?:are|appear) unremarkable",
    r"bony thorax (?:is|appears) intact",
    r"chest radiograph is unremarkable",
    r"chest radiograph is normal",
]


def extract_clinical_labels(text: str) -> Tuple[Set[str], Set[str]]:
    """
    Extracts affirmed and negated clinical pathology classes using regex rules
    with standard clinical negation scopes.
    Returns: (affirmed_classes, negated_classes)
    """
    text_lower = text.lower()
    neg_prefix = r'(?:no|without|negative for|free of|rules out|absence of|clear of|no evidence of|no sign of|unremarkable)\s+'

    affirmed: Set[str] = set()
    negated: Set[str] = set()

    for cls_name, patterns in CLINICAL_PATTERNS.items():
        if cls_name == "normal":
            continue
        for pat in patterns:
            # Check negation
            neg_pat = re.compile(rf'{neg_prefix}(?:[\w\s]{{0,35}})?{pat}')
            pos_pat = re.compile(rf'\b{pat}\b')
            if neg_pat.search(text_lower):
                negated.add(cls_name)
            elif pos_pat.search(text_lower):
                affirmed.add(cls_name)

    # Determine normal class
    if not affirmed:
        for pat in CLINICAL_PATTERNS["normal"]:
            if re.search(rf'\b{pat}\b', text_lower):
                affirmed.add("normal")
                break

    return affirmed, negated


def count_boilerplate_normal_phrases(text: str) -> int:
    """Counts occurrence of stereotypical canned normal phrases."""
    text_lower = text.lower()
    count = 0
    for pat in GENERIC_NORMAL_PHRASES:
        if re.search(pat, text_lower):
            count += 1
    return count


def detect_internal_contradiction(obs_text: str, imp_text: str) -> bool:
    """
    Detects if OBSERVATIONS describes an active pathology while
    IMPRESSION declares the scan normal without caveats.
    """
    obs_aff, _ = extract_clinical_labels(obs_text)
    imp_aff, _ = extract_clinical_labels(imp_text)

    # Contradiction: observations noticed pathology, but impression claimed normal
    obs_pathology = obs_aff - {"normal"}
    if obs_pathology and ("normal" in imp_aff and not (imp_aff - {"normal"})):
        return True
    return False


# ── Textual Similarity Metrics ────────────────────────────────────────────────

def compute_textual_metrics(generated: str, reference: str) -> Dict[str, float]:
    """Computes ROUGE-L and BLEU-2 textual similarity metrics."""
    res = {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0, "bleu_2": 0.0}

    gen_clean = generated.strip()
    ref_clean = reference.strip()
    if not gen_clean or not ref_clean:
        return res

    if ROUGE_AVAILABLE:
        try:
            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            scores = scorer.score(ref_clean, gen_clean)
            res["rouge_1"] = round(scores["rouge1"].fmeasure, 4)
            res["rouge_2"] = round(scores["rouge2"].fmeasure, 4)
            res["rouge_l"] = round(scores["rougeL"].fmeasure, 4)
        except Exception:
            pass

    if BLEU_AVAILABLE:
        try:
            smooth = SmoothingFunction().method1
            gen_tokens = gen_clean.lower().split()
            ref_tokens = [ref_clean.lower().split()]
            res["bleu_2"] = round(sentence_bleu(ref_tokens, gen_tokens, weights=(0.5, 0.5), smoothing_function=smooth), 4)
        except Exception:
            pass

    return res


# ── Prompt Implementations ────────────────────────────────────────────────────

def run_current_prompt(b64_img: str, model: str) -> Tuple[str, Dict[str, Any]]:
    """Runs the baseline current prompt (single-stage naive prompt)."""
    prompt = _build_research_prompt()
    raw_text, metrics = _ollama_generate_vision_with_metrics(
        prompt=prompt,
        images_b64=[b64_img],
        model=model,
        num_predict=220,
        temperature=0.1,
    )
    return raw_text, metrics


def run_revised_prompt(b64_img: str, model: str) -> Tuple[str, Dict[str, Any]]:
    """
    Runs the revised 2-stage grounded research report pipeline:
      Stage 1: Focused visual feature query on the 1024px radiograph.
      Stage 2: Structured 3-section research report synthesis (OBSERVATIONS, IMPRESSION, LIMITATIONS).
    """
    # Stage 1: Focused interrogative visual inspection
    v_prompt = (
        "You are an expert medical radiologist examining this chest radiograph in detail.\n"
        "Carefully inspect the actual image pixels and describe:\n"
        "1. Support apparatus: Any endotracheal tube, enteric/nasogastric tube, vascular line, chest tube, or surgical clips? Note visible tip positions.\n"
        "2. Cardiac silhouette: Is the heart enlarged (cardiothoracic ratio > 0.5) or normal in size?\n"
        "3. Lungs: Are there visible airspace opacities, streaky basilar densities, consolidation, atelectasis, vascular congestion, or edema?\n"
        "4. Pleural spaces: Are bilateral costophrenic angles sharp, blunted, or obscured?\n"
        "Be objective and specific based strictly on visible pixel evidence."
    )
    raw_obs, v_metrics = _ollama_generate_vision_with_metrics(
        prompt=v_prompt,
        images_b64=[b64_img],
        model=model,
        num_predict=180,
        temperature=0.1,
    )

    # Stage 2: Structured research synthesis
    s2_prompt = (
        f"Based strictly on the verified visual observations extracted from this chest radiograph:\n"
        f"\"\"\"{raw_obs}\"\"\"\n\n"
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

    raw_text, s2_metrics = _ollama_generate_vision_with_metrics(
        prompt=s2_prompt,
        images_b64=[],
        model=model,
        num_predict=220,
        temperature=0.1,
    )

    total_metrics = {
        "total_duration_ns": v_metrics.get("total_duration_ns", 0) + s2_metrics.get("total_duration_ns", 0),
        "load_duration_ns": v_metrics.get("load_duration_ns", 0),
        "prompt_eval_count": v_metrics.get("prompt_eval_count", 0) + s2_metrics.get("prompt_eval_count", 0),
        "prompt_eval_duration_ns": v_metrics.get("prompt_eval_duration_ns", 0) + s2_metrics.get("prompt_eval_duration_ns", 0),
        "eval_count": v_metrics.get("eval_count", 0) + s2_metrics.get("eval_count", 0),
        "eval_duration_ns": v_metrics.get("eval_duration_ns", 0) + s2_metrics.get("eval_duration_ns", 0),
        "done_reason": s2_metrics.get("done_reason", "completed"),
    }
    return raw_text, total_metrics


# ── Evaluation Engine ─────────────────────────────────────────────────────────

def evaluate_case(
    case_record: Dict[str, Any],
    raw_output: str,
    metrics_dict: Dict[str, Any],
    prompt_name: str,
) -> Dict[str, Any]:
    """Evaluates a single generated report against reference and gold labels."""
    safe_case_id = case_record["safe_case_id"]
    primary_category = case_record["primary_category"]
    gold_labels = set(case_record.get("gold_affirmed_labels", []))
    ref_text = case_record.get("reference_full_report", "")

    # Clean & structure sections
    cleaned_text, sections, warnings = clean_and_structure_research_report(raw_output)

    obs_text = sections.get("OBSERVATIONS", "")
    imp_text = sections.get("IMPRESSION", sections.get("ASSESSMENT AND UNCERTAINTY", ""))

    # Clinical entity extraction
    gen_affirmed, gen_negated = extract_clinical_labels(cleaned_text)
    ref_affirmed, ref_negated = extract_clinical_labels(ref_text)

    # Reference truth set
    true_set = set()
    if primary_category in PATHOLOGY_CLASSES:
        true_set.add(primary_category)
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
        elif "tube" in g_clean or "catheter" in g_clean:
            true_set.add("support_devices")
        elif "normal" in g_clean and len(gold_labels) <= 2:
            true_set.add("normal")

    pred_set = gen_affirmed

    # Calculate confusion elements across classes
    tp_list = sorted(list(true_set.intersection(pred_set)))
    fp_list = sorted(list(pred_set - true_set))
    fn_list = sorted(list(true_set - pred_set))

    # Error taxonomy
    unsupported_positives = [p for p in fp_list if p != "normal"]
    unsupported_negatives = []
    # If ground truth had real pathology but report asserted normal
    if (true_set - {"normal"}) and ("normal" in pred_set):
        unsupported_negatives.append("unwarranted_normal_assertion")

    boilerplate_count = count_boilerplate_normal_phrases(cleaned_text)
    contradiction = detect_internal_contradiction(obs_text, imp_text)

    # Textual similarity
    text_metrics = compute_textual_metrics(cleaned_text, ref_text)

    # Timing & throughput
    total_sec = metrics_dict.get("total_duration_ns", 0) / 1e9
    eval_count = metrics_dict.get("eval_count", 0)
    tok_per_sec = round(eval_count / total_sec, 2) if total_sec > 0 else 0.0

    return {
        "safe_case_id": safe_case_id,
        "split": case_record["split"],
        "primary_category": primary_category,
        "view_position": case_record["view_position"],
        "prompt_version": prompt_name,
        "generated_raw": raw_output,
        "cleaned_report": cleaned_text,
        "sections": sections,
        "validation_warnings": warnings,
        "gold_affirmed_labels": list(gold_labels),
        "ground_truth_classes": sorted(list(true_set)),
        "predicted_classes": sorted(list(pred_set)),
        "true_positives": tp_list,
        "false_positives": fp_list,
        "false_negatives": fn_list,
        "unsupported_positives": unsupported_positives,
        "unsupported_negatives": unsupported_negatives,
        "boilerplate_normal_count": boilerplate_count,
        "internal_contradiction": contradiction,
        "rouge_l": text_metrics["rouge_l"],
        "bleu_2": text_metrics["bleu_2"],
        "latency_sec": round(total_sec, 2),
        "eval_count": eval_count,
        "tokens_per_second": tok_per_sec,
    }


def aggregate_evaluation_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes macro and micro summary statistics across evaluation cases."""
    if not results:
        return {}

    total_cases = len(results)
    tp_total = sum(len(r["true_positives"]) for r in results)
    fp_total = sum(len(r["false_positives"]) for r in results)
    fn_total = sum(len(r["false_negatives"]) for r in results)

    precision = round(tp_total / (tp_total + fp_total), 4) if (tp_total + fp_total) > 0 else 0.0
    recall = round(tp_total / (tp_total + fn_total), 4) if (tp_total + fn_total) > 0 else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0

    unsupported_pos_cases = sum(1 for r in results if len(r["unsupported_positives"]) > 0)
    unsupported_neg_cases = sum(1 for r in results if len(r["unsupported_negatives"]) > 0)
    boilerplate_cases = sum(1 for r in results if r["boilerplate_normal_count"] >= 2)
    contradiction_cases = sum(1 for r in results if r["internal_contradiction"])

    avg_rouge_l = round(float(np.mean([r["rouge_l"] for r in results])), 4)
    avg_bleu_2 = round(float(np.mean([r["bleu_2"] for r in results])), 4)
    avg_latency = round(float(np.mean([r["latency_sec"] for r in results])), 2)
    avg_tok_sec = round(float(np.mean([r["tokens_per_second"] for r in results])), 2)

    return {
        "sample_size": total_cases,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "unsupported_positive_rate": round(unsupported_pos_cases / total_cases, 4),
        "unsupported_negative_rate": round(unsupported_neg_cases / total_cases, 4),
        "boilerplate_normal_rate": round(boilerplate_cases / total_cases, 4),
        "internal_contradiction_rate": round(contradiction_cases / total_cases, 4),
        "mean_rouge_l": avg_rouge_l,
        "mean_bleu_2": avg_bleu_2,
        "mean_latency_sec": avg_latency,
        "mean_tokens_per_second": avg_tok_sec,
    }


def run_full_evaluation(
    split: str = "held_out_test",
    mode: str = "both",
    limit: Optional[int] = None,
) -> None:
    print("=" * 80)
    print(f"INTERACTMED-VLM REPORT QUALITY EVALUATION (Split: {split}, Mode: {mode})")
    print("=" * 80)

    store_path = WORKSPACE_ROOT / "artifacts" / "report_evaluation" / "local_reference_store.json"
    if not store_path.exists():
        raise FileNotFoundError(f"Missing local reference store at {store_path}. Run dataset setup first.")

    with open(store_path, "r") as f:
        store: Dict[str, Any] = json.load(f)

    # Filter split
    if split == "all":
        case_items = list(store.values())
    else:
        case_items = [v for v in store.values() if v["split"] == split]

    if limit:
        case_items = case_items[:limit]

    print(f"Loaded {len(case_items)} cases for evaluation.")

    model_name = _vision_model_available() or "medgemma:4b"
    print(f"Active vision model: {model_name}")

    current_results: List[Dict[str, Any]] = []
    revised_results: List[Dict[str, Any]] = []

    # 1. Run Baseline Current Prompt
    if mode in ("current", "both"):
        print("\n>>> Running CURRENT BASELINE PROMPT evaluation...")
        for i, c in enumerate(case_items, start=1):
            cid = c["safe_case_id"]
            img_path = c["image_path"]
            print(f"  [{i}/{len(case_items)}] Processing {cid} ({c['primary_category']})...", end="", flush=True)

            with open(img_path, "rb") as f:
                img_bytes = f.read()
            arr = _load_to_rgb_array(img_bytes, f"{cid}.jpg")
            b64 = _array_to_base64_jpeg(arr, max_side=1024)

            t0 = time.time()
            raw_text, metrics = run_current_prompt(b64, model_name)
            t_el = time.time() - t0
            print(f" Done ({t_el:.2f}s)")

            rec = evaluate_case(c, raw_text, metrics, prompt_name="current_baseline")
            current_results.append(rec)

        # Save current results
        curr_out = WORKSPACE_ROOT / "artifacts" / "report_evaluation" / "current_prompt_results.json"
        with open(curr_out, "w") as f:
            json.dump(current_results, f, indent=2)
        print(f"Saved current prompt results to {curr_out}")

    # 2. Run Revised Grounded Pipeline
    if mode in ("revised", "both"):
        print("\n>>> Running REVISED GROUNDED PIPELINE evaluation...")
        for i, c in enumerate(case_items, start=1):
            cid = c["safe_case_id"]
            img_path = c["image_path"]
            print(f"  [{i}/{len(case_items)}] Processing {cid} ({c['primary_category']})...", end="", flush=True)

            with open(img_path, "rb") as f:
                img_bytes = f.read()
            arr = _load_to_rgb_array(img_bytes, f"{cid}.jpg")
            b64 = _array_to_base64_jpeg(arr, max_side=1024)

            t0 = time.time()
            raw_text, metrics = run_revised_prompt(b64, model_name)
            t_el = time.time() - t0
            print(f" Done ({t_el:.2f}s)")

            rec = evaluate_case(c, raw_text, metrics, prompt_name="revised_grounded")
            revised_results.append(rec)

        # Save revised results
        rev_out = WORKSPACE_ROOT / "artifacts" / "report_evaluation" / "revised_prompt_results.json"
        with open(rev_out, "w") as f:
            json.dump(revised_results, f, indent=2)
        print(f"Saved revised prompt results to {rev_out}")

    # 3. Build Comparison Tables
    if mode == "both" and current_results and revised_results:
        print("\n" + "=" * 80)
        print("GENERATING COMPARISON TABLES & SUMMARY STATISTICS")
        print("=" * 80)

        curr_agg = aggregate_evaluation_results(current_results)
        rev_agg = aggregate_evaluation_results(revised_results)

        summary_data = {
            "evaluation_split": split,
            "sample_size": len(case_items),
            "current_prompt_baseline": curr_agg,
            "revised_prompt": rev_agg,
            "delta_improvements": {
                "f1_delta": round(rev_agg["micro_f1"] - curr_agg["micro_f1"], 4),
                "recall_delta": round(rev_agg["micro_recall"] - curr_agg["micro_recall"], 4),
                "unsupported_negative_delta": round(rev_agg["unsupported_negative_rate"] - curr_agg["unsupported_negative_rate"], 4),
                "boilerplate_normal_delta": round(rev_agg["boilerplate_normal_rate"] - curr_agg["boilerplate_normal_rate"], 4),
                "rouge_l_delta": round(rev_agg["mean_rouge_l"] - curr_agg["mean_rouge_l"], 4),
            },
        }

        sum_out = WORKSPACE_ROOT / "artifacts" / "report_evaluation" / "comparison_summary.json"
        with open(sum_out, "w") as f:
            json.dump(summary_data, f, indent=2)
        print(f"Saved comparison summary to {sum_out}")

        # Per-case CSV comparison
        comp_csv = WORKSPACE_ROOT / "artifacts" / "report_evaluation" / "comparison.csv"
        with open(comp_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Safe Case ID",
                "Split",
                "Primary Category",
                "Ground Truth Classes",
                "Current Preds",
                "Revised Preds",
                "Current F1",
                "Revised F1",
                "Current Boilerplate",
                "Revised Boilerplate",
                "Current Latency (s)",
                "Revised Latency (s)",
            ])
            for c_rec, r_rec in zip(current_results, revised_results):
                c_p, c_r, c_f1 = (
                    c_rec["true_positives"],
                    c_rec["false_negatives"],
                    round(2 * len(c_rec["true_positives"]) / (2 * len(c_rec["true_positives"]) + len(c_rec["false_positives"]) + len(c_rec["false_negatives"])), 2)
                    if (2 * len(c_rec["true_positives"]) + len(c_rec["false_positives"]) + len(c_rec["false_negatives"])) > 0
                    else 0.0,
                )
                r_p, r_r, r_f1 = (
                    r_rec["true_positives"],
                    r_rec["false_negatives"],
                    round(2 * len(r_rec["true_positives"]) / (2 * len(r_rec["true_positives"]) + len(r_rec["false_positives"]) + len(r_rec["false_negatives"])), 2)
                    if (2 * len(r_rec["true_positives"]) + len(r_rec["false_positives"]) + len(r_rec["false_negatives"])) > 0
                    else 0.0,
                )
                writer.writerow([
                    c_rec["safe_case_id"],
                    c_rec["split"],
                    c_rec["primary_category"],
                    ", ".join(c_rec["ground_truth_classes"]),
                    ", ".join(c_rec["predicted_classes"]),
                    ", ".join(r_rec["predicted_classes"]),
                    c_f1,
                    r_f1,
                    c_rec["boilerplate_normal_count"],
                    r_rec["boilerplate_normal_count"],
                    c_rec["latency_sec"],
                    r_rec["latency_sec"],
                ])
        print(f"Saved per-case comparison table to {comp_csv}")

        # Display Summary Table
        print("\n" + "=" * 80)
        print(f"SUMMARY RESULTS (Evaluation Cohort: {split.upper()}, N={len(case_items)})")
        print("=" * 80)
        print(f"{'Metric':<35} | {'Current Baseline':<18} | {'Revised Prompt':<18} | {'Delta':<10}")
        print("-" * 88)
        print(f"{'Clinical Label Micro-F1':<35} | {curr_agg['micro_f1']:<18.4f} | {rev_agg['micro_f1']:<18.4f} | {summary_data['delta_improvements']['f1_delta']:+<10.4f}")
        print(f"{'Clinical Label Recall':<35} | {curr_agg['micro_recall']:<18.4f} | {rev_agg['micro_recall']:<18.4f} | {summary_data['delta_improvements']['recall_delta']:+<10.4f}")
        print(f"{'Boilerplate Normal Rate':<35} | {curr_agg['boilerplate_normal_rate']:<18.4f} | {rev_agg['boilerplate_normal_rate']:<18.4f} | {summary_data['delta_improvements']['boilerplate_normal_delta']:+<10.4f}")
        print(f"{'Unsupported Negative Rate':<35} | {curr_agg['unsupported_negative_rate']:<18.4f} | {rev_agg['unsupported_negative_rate']:<18.4f} | {summary_data['delta_improvements']['unsupported_negative_delta']:+<10.4f}")
        print(f"{'ROUGE-L Score':<35} | {curr_agg['mean_rouge_l']:<18.4f} | {rev_agg['mean_rouge_l']:<18.4f} | {summary_data['delta_improvements']['rouge_l_delta']:+<10.4f}")
        print(f"{'BLEU-2 Score':<35} | {curr_agg['mean_bleu_2']:<18.4f} | {rev_agg['mean_bleu_2']:<18.4f} | {round(rev_agg['mean_bleu_2'] - curr_agg['mean_bleu_2'], 4):+<10.4f}")
        print(f"{'Mean Latency (seconds)':<35} | {curr_agg['mean_latency_sec']:<18.2f} | {rev_agg['mean_latency_sec']:<18.2f} | {round(rev_agg['mean_latency_sec'] - curr_agg['mean_latency_sec'], 2):+<10.2f}")
        print(f"{'Throughput (tokens/sec)':<35} | {curr_agg['mean_tokens_per_second']:<18.2f} | {rev_agg['mean_tokens_per_second']:<18.2f} | {round(rev_agg['mean_tokens_per_second'] - curr_agg['mean_tokens_per_second'], 2):+<10.2f}")
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate InteractMed-VLM whole-image reports.")
    parser.add_argument("--split", type=str, default="held_out_test", choices=["dev", "held_out_test", "all"])
    parser.add_argument("--mode", type=str, default="both", choices=["current", "revised", "both"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_full_evaluation(split=args.split, mode=args.mode, limit=args.limit)
    sys.exit(0)
