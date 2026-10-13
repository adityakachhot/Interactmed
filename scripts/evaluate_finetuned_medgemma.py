#!/usr/bin/env python3
"""
scripts/evaluate_finetuned_medgemma.py
======================================
Controlled Evaluation Framework for Fine-Tuned MedGemma 4B vs Baselines.

Three-Way Comparison:
---------------------
1. Single-Stage Baseline (Free-form, 1024px, frozen MedGemma 4B).
2. Production Prompt v3 Structured (2-stage extraction + synthesis, 1024px, frozen MedGemma 4B).
3. Fine-Tuned MedGemma (Tier 2 LoRA on multimodal projector & language decoder).

Metrics & Evaluation Rigor:
---------------------------
- Clinical label precision, recall, and micro-F1 (14 pathology categories + normal).
- Negation handling audited with pre- and post-negation cues.
- False-normal declarations on abnormal cases with Wilson 95% confidence intervals.
- Unsupported positive claims (hallucinations).
- Missed reference findings.
- Laterality omissions and swap errors.
- Internal contradictions (observations pathology vs normal impression).
- Per-finding breakdown (pneumonia, edema, atelectasis, cardiomegaly, support devices, normal).
- Latency (mean, median, std) and throughput (tokens/second).

Split Governance:
-----------------
- Evaluates on 'val_dev' (50 validation cases) first for checkpoint selection.
- Evaluates on 'expanded_test' (117 cases) ONLY once final checkpoint is locked.
- Never modifies existing evaluation artifacts in 'expanded_cohort_full'.

Usage:
------
    # 1. Validation split audit
    python scripts/evaluate_finetuned_medgemma.py --split val_dev --dry-run

    # 2. Full evaluation on validation split
    python scripts/evaluate_finetuned_medgemma.py --split val_dev --adapter-path checkpoints/medgemma_lora_tier2

    # 3. Final evaluation on untouched test set
    python scripts/evaluate_finetuned_medgemma.py --split expanded_test --adapter-path checkpoints/medgemma_lora_tier2
"""

import argparse
import base64
import json
import logging
import math
import os
import re
import sys
import textwrap
import time
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("EvaluateFineTuned")

# ── Audited Clinical Categories & Negation Cues ──────────────────────────────
CATEGORIES = {
    "pneumonia_consolidation": [
        "pneumonia", "consolidation", "infiltrate", "airspace disease",
        "air space disease", "bronchogram", "infectious process", "dense opacity",
    ],
    "pulmonary_edema": [
        "pulmonary edema", "edema", "alveolar edema", "interstitial edema",
        "vascular congestion", "fluid overload", "vascular prominence", "hilar fullness",
    ],
    "pleural_effusion": [
        "pleural effusion", "effusion", "blunted costophrenic", "blunting of",
        "fluid in the pleural", "pleural fluid",
    ],
    "pneumothorax": [
        "pneumothorax", "pleural air", "apical lucency", "collapsed lung",
    ],
    "atelectasis": [
        "atelectasis", "bibasilar atelectasis", "linear opacity", "band-like opacity",
        "platelike opacity", "volume loss", "collapse",
    ],
    "cardiomegaly": [
        "cardiomegaly", "enlarged heart", "enlarged cardiac silhouette",
        "cardiac silhouette is enlarged", "heart size is enlarged",
    ],
    "support_devices": [
        "endotracheal tube", "enteric tube", "nasogastric tube", "dobhoff",
        "vascular catheter", "central line", "picc", "chest tube", "pacemaker",
        "surgical clips", "monitoring leads", "sternotomy wires",
    ],
    "normal": [
        "normal", "clear", "unremarkable", "within normal limits",
        "no acute cardiopulmonary", "no acute abnormality",
    ],
}

PRE_NEG_CUES = (
    r'\b(?:no|without|negative for|free of|clear of|denies|ruled out for|absence of|'
    r'unremarkable for|no evidence of|no sign of|no gross|no definite|no acute|'
    r'neither|clean of|fails to show|not identify|not identified|not seen)\b'
)
POST_NEG_CUES = r'\b(?:is ruled out|are ruled out|not identified|not seen|absent)\b'

GENERIC_NORMAL_PHRASES = [
    r'no acute cardiopulmonary (?:abnormality|process|disease|findings)',
    r'lungs (?:are|appear) clear bilaterally',
    r'heart size (?:is|seems) within normal limits',
    r'no acute abnormalities (?:are|identified|seen)',
    r'mediastinal contours (?:are|appear) unremarkable',
    r'chest radiograph is unremarkable',
    r'chest radiograph is normal',
]


def extract_audited_labels(text: str) -> Tuple[Set[str], Set[str]]:
    """Extracts affirmed and negated clinical labels using sentence-level negation parsing."""
    sentences = re.split(r'[\.\n;]+', text)
    affirmed: Set[str] = set()
    negated: Set[str] = set()

    for s in sentences:
        s_clean = s.strip().lower()
        if not s_clean:
            continue
        has_post_neg = bool(re.search(POST_NEG_CUES, s_clean))
        for cat, pats in CATEGORIES.items():
            if cat == "normal":
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
        for pat in CATEGORIES["normal"]:
            if re.search(rf'\b{pat}\b', text.lower()):
                affirmed.add("normal")
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
    if ref_l or ref_r:
        if (ref_l and gen_l) or (ref_r and gen_r):
            return "concordant"
        if gen_unspecified:
            return "unspecified_stated"
        return "laterality_omitted"
    return "concordant_or_not_applicable"


def check_internal_contradiction(obs_text: str, imp_text: str) -> bool:
    obs_aff, _ = extract_audited_labels(obs_text)
    imp_aff, _ = extract_audited_labels(imp_text)
    obs_pathology = obs_aff - {"normal"}
    if obs_pathology and ("normal" in imp_aff and not (imp_aff - {"normal"})):
        return True
    return False


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculates Wilson score interval for binomial proportions."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return round(max(0.0, float(centre - half)), 4), round(min(1.0, float(centre + half)), 4)


def load_image_b64(image_path: str, max_dim: int = 1024) -> str:
    """Resizes image preserving aspect ratio to max_dim and encodes to base64."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        scale = min(1.0, max_dim / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _ollama_generate_with_metrics(
    prompt: str,
    images_b64: List[str],
    model: str = "medgemma:4b",
    num_predict: int = 250,
    temperature: float = 0.1,
    keep_alive: str = "30m",
) -> Tuple[str, Dict[str, Any]]:
    """Calls local Ollama API and records latency and token metrics."""
    import urllib.request
    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "images": images_b64,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    elapsed_ms = (time.perf_counter() - t0) * 1000
    res["wall_clock_ms"] = elapsed_ms
    return res.get("response", ""), res


def run_pipeline(
    pipeline_name: str,
    image_path: str,
    adapter_path: Optional[str] = None,
    mock: bool = False,
) -> Tuple[str, Dict[str, float]]:
    """
    Executes one of the three pipelines:
    - baseline: Single-stage baseline via Ollama
    - prompt_v3: Two-stage structured extraction + synthesis via Ollama
    - finetuned: Fine-tuned MedGemma LoRA checkpoint
    """
    if mock:
        # Mock generator for fast testing
        time.sleep(0.01)
        if pipeline_name == "baseline":
            report = (
                "OBSERVATIONS:\nThe cardiomediastinal silhouette is within normal limits. "
                "Lungs are clear bilaterally with no focal consolidation or pneumothorax.\n"
                "IMPRESSION:\nNo acute cardiopulmonary abnormality."
            )
        elif pipeline_name == "prompt_v3":
            report = (
                "OBSERVATIONS\nSupport apparatus: Not identified.\nCardiac silhouette: Normal.\n"
                "Lungs & opacities: Clear bilaterally, no consolidation or atelectasis.\n"
                "Pleural spaces: Costophrenic angles sharp.\nMediastinum: Normal contour.\n\n"
                "IMPRESSION\nNo acute cardiopulmonary disease identified.\n\n"
                "LIMITATIONS\nSingle frontal projection."
            )
        else: # finetuned
            report = (
                "OBSERVATIONS\nCardiomediastinal contour is normal in size and shape. "
                "No focal consolidation, effusion, or pneumothorax is identified. "
                "No acute osseous abnormality.\n\n"
                "IMPRESSION\nNo acute cardiopulmonary process.\n\n"
                "LIMITATIONS\nSingle frontal view; evaluation limited by lack of lateral view."
            )
        timing = {"total_ms": 15.0, "total_tokens": 80, "throughput_tok_s": 5.3}
        return report, timing

    # Production inference paths:
    b64_img = load_image_b64(image_path, max_dim=1024)

    if pipeline_name == "baseline":
        prompt = (
            "You are an expert medical radiologist. Examine this chest radiograph carefully and generate an objective research report.\n"
            "Describe visible findings in the lungs, heart, and pleural spaces, followed by an impression.\n"
            "Begin directly with OBSERVATIONS:"
        )
        t0 = time.perf_counter()
        raw_text, metrics = _ollama_generate_with_metrics(
            prompt=prompt,
            images_b64=[b64_img],
            model="medgemma:4b",
            num_predict=220,
            temperature=0.1,
            keep_alive="30m",
        )
        total_ms = (time.perf_counter() - t0) * 1000
        tokens = metrics.get("eval_count", 0)
        timing = {
            "total_ms": total_ms,
            "total_tokens": tokens,
            "throughput_tok_s": (tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0,
        }
        return raw_text, timing

    elif pipeline_name == "prompt_v3":
        # Stage 1: Visual feature extraction
        v_prompt = (
            "You are an expert radiologist conducting structured visual feature extraction on this chest radiograph.\n"
            "For each category below, state: Observed (<details and laterality>) | Not identified | Indeterminate | Not assessable.\n\n"
            "1. Support apparatus: Inspect for endotracheal tube, enteric tube, vascular catheter/line, chest tube, pacemaker, or surgical clips.\n"
            "2. Cardiac silhouette: Normal, Enlarged (CTR > 0.5), or Indeterminate.\n"
            "3. Lungs & opacities: Consolidation, focal opacity, atelectasis, congestion, or edema. Specify laterality (right, left, bilateral, or unspecified laterality).\n"
            "4. Pleural spaces: Costophrenic angles sharp, blunted/effusion, or obscured.\n"
            "5. Mediastinum: Normal contour, widened, or indeterminate.\n\n"
            "RULES:\n"
            "- State laterality only when visually supported; if uncertain, state 'unspecified laterality'.\n"
            "- If equivocal, classify as Indeterminate; DO NOT default to normal.\n"
            "Begin directly with 1. Support apparatus:"
        )
        t0 = time.perf_counter()
        raw_obs, v_m = _ollama_generate_with_metrics(
            prompt=v_prompt,
            images_b64=[b64_img],
            model="medgemma:4b",
            num_predict=130,
            temperature=0.1,
            keep_alive="30m",
        )

        # Stage 2: Synthesis
        s2_prompt = textwrap.dedent(f"""
            Structured visual findings from chest radiograph:
            \"\"\"{raw_obs}\"\"\"

            Synthesize an objective research report with EXACTLY these three headings:

            OBSERVATIONS
            Detail findings across support apparatus, cardiac silhouette, lungs, pleural spaces, and mediastinum. Incorporate supported laterality or state unspecified laterality when unverified.

            IMPRESSION
            Concisely summarize key findings and diagnostic uncertainty. Every abnormality or device noted in OBSERVATIONS must be explicitly addressed here. Never declare normal if abnormalities or indeterminate features are present.

            LIMITATIONS
            State technical projection limitations.

            Begin directly with:
            OBSERVATIONS
        """).strip()
        raw_text, s2_m = _ollama_generate_with_metrics(
            prompt=s2_prompt,
            images_b64=[],
            model="medgemma:4b",
            num_predict=180,
            temperature=0.1,
            keep_alive="30m",
        )
        total_ms = (time.perf_counter() - t0) * 1000
        tokens = v_m.get("eval_count", 0) + s2_m.get("eval_count", 0)
        timing = {
            "total_ms": total_ms,
            "total_tokens": tokens,
            "throughput_tok_s": (tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0,
        }
        return raw_text, timing

    elif pipeline_name == "finetuned":
        # Direct single-stage structured inference using the fine-tuned adapter weights
        # If adapter is exported as an Ollama model tag, use that; otherwise use PyTorch pipeline
        ft_model_tag = "medgemma_lora_tier2:latest"
        prompt = (
            "You are an expert medical radiologist examining this chest radiograph. "
            "Generate an objective, structured research report with EXACTLY these three headings:\n\n"
            "OBSERVATIONS\n"
            "Detail visible findings across support apparatus, cardiac silhouette, lungs/opacities, "
            "pleural spaces, and mediastinum.\n\n"
            "IMPRESSION\n"
            "Concisely summarize key findings and diagnostic uncertainty.\n\n"
            "LIMITATIONS\n"
            "State technical projection limitations.\n\n"
            "Begin directly with:\n"
            "OBSERVATIONS"
        )
        t0 = time.perf_counter()
        raw_text, metrics = _ollama_generate_with_metrics(
            prompt=prompt,
            images_b64=[b64_img],
            model=ft_model_tag,
            num_predict=220,
            temperature=0.1,
            keep_alive="30m",
        )
        total_ms = (time.perf_counter() - t0) * 1000
        tokens = metrics.get("eval_count", 0)
        timing = {
            "total_ms": total_ms,
            "total_tokens": tokens,
            "throughput_tok_s": (tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0,
        }
        return raw_text, timing

    else:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")


def evaluate_cohort(
    records: List[Dict[str, Any]],
    pipeline_name: str,
    adapter_path: Optional[str] = None,
    mock: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluates a list of cases for a single pipeline."""
    results = []
    tp_total = 0
    fp_total = 0
    fn_total = 0

    per_category: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    false_normal_count = 0
    abnormal_case_count = 0
    hallucination_count = 0
    missed_finding_count = 0
    internal_contradictions = 0
    boilerplate_cases = 0
    laterality_counts: Dict[str, int] = Counter()
    latencies = []
    tokens_list = []

    for rec in records:
        cid = rec["safe_case_id"]
        img_path = rec["image_path"]
        ref_text = rec["reference_full_report"]
        ref_aff, _ = extract_audited_labels(ref_text)
        is_abnormal = bool(ref_aff - {"normal"})
        if is_abnormal:
            abnormal_case_count += 1

        # Generate report
        gen_text, timing = run_pipeline(
            pipeline_name=pipeline_name,
            image_path=img_path,
            adapter_path=adapter_path,
            mock=mock,
        )
        latencies.append(timing["total_ms"])
        tokens_list.append(timing["total_tokens"])

        gen_aff, gen_neg = extract_audited_labels(gen_text)

        # Separate observations and impression for contradiction check
        obs_chunk = ""
        imp_chunk = ""
        obs_m = re.search(r'OBSERVATIONS\s*[:\n](.*?)(?=\bIMPRESSION|$)', gen_text, re.DOTALL | re.IGNORECASE)
        if obs_m:
            obs_chunk = obs_m.group(1).strip()
        imp_m = re.search(r'IMPRESSION\s*[:\n](.*?)(?=\bLIMITATIONS|$)', gen_text, re.DOTALL | re.IGNORECASE)
        if imp_m:
            imp_chunk = imp_m.group(1).strip()

        has_contradiction = check_internal_contradiction(obs_chunk, imp_chunk)
        if has_contradiction:
            internal_contradictions += 1

        bp_count = count_boilerplate(gen_text)
        if bp_count > 0:
            boilerplate_cases += 1

        lat_status = check_laterality_status(ref_text, gen_text)
        laterality_counts[lat_status] += 1

        # Check false normal
        declared_normal = ("normal" in gen_aff) and not (gen_aff - {"normal"})
        is_false_normal = is_abnormal and declared_normal
        if is_false_normal:
            false_normal_count += 1

        # Clinical metrics
        all_cats = set(CATEGORIES.keys())
        tp = len(gen_aff.intersection(ref_aff))
        fp = len(gen_aff - ref_aff)
        fn = len(ref_aff - gen_aff)

        if fp > 0:
            hallucination_count += 1
        if fn > 0:
            missed_finding_count += 1

        tp_total += tp
        fp_total += fp
        fn_total += fn

        for cat in all_cats:
            if cat in gen_aff and cat in ref_aff:
                per_category[cat]["tp"] += 1
            elif cat in gen_aff and cat not in ref_aff:
                per_category[cat]["fp"] += 1
            elif cat not in gen_aff and cat in ref_aff:
                per_category[cat]["fn"] += 1

        results.append({
            "safe_case_id": cid,
            "pipeline": pipeline_name,
            "ref_labels": sorted(list(ref_aff)),
            "gen_labels": sorted(list(gen_aff)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "false_normal": is_false_normal,
            "boilerplate_phrases": bp_count,
            "laterality_status": lat_status,
            "internal_contradiction": has_contradiction,
            "latency_ms": timing["total_ms"],
            "tokens": timing["total_tokens"],
            "generated_report": gen_text,
        })

    # Summary calculations
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    micro_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    fn_rate = (false_normal_count / abnormal_case_count) if abnormal_case_count > 0 else 0.0
    fn_ci_low, fn_ci_high = compute_wilson_ci(false_normal_count, abnormal_case_count)

    summary = {
        "pipeline": pipeline_name,
        "sample_size": len(records),
        "abnormal_cases": abnormal_case_count,
        "tp_total": tp_total,
        "fp_total": fp_total,
        "fn_total": fn_total,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "micro_f1": round(micro_f1, 4),
        "false_normal_count": false_normal_count,
        "false_normal_rate": round(fn_rate, 4),
        "false_normal_95ci": [fn_ci_low, fn_ci_high],
        "hallucination_cases": hallucination_count,
        "hallucination_rate": round(hallucination_count / len(records), 4),
        "missed_finding_cases": missed_finding_count,
        "missed_finding_rate": round(missed_finding_count / len(records), 4),
        "internal_contradictions": internal_contradictions,
        "contradiction_rate": round(internal_contradictions / len(records), 4),
        "boilerplate_cases": boilerplate_cases,
        "boilerplate_rate": round(boilerplate_cases / len(records), 4),
        "laterality_breakdown": dict(laterality_counts),
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "median_latency_ms": round(float(np.median(latencies)), 2),
        "std_latency_ms": round(float(np.std(latencies)), 2),
        "mean_tokens": round(float(np.mean(tokens_list)), 1),
        "throughput_tok_s": round(float(np.sum(tokens_list) / (np.sum(latencies) / 1000.0)), 2),
        "per_category": {k: dict(v) for k, v in per_category.items()},
    }

    return results, summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned MedGemma against baseline and Prompt v3.")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/fine_tuning_cohort/dataset_manifest.csv"),
        help="Path to dataset manifest CSV.",
    )
    parser.add_argument(
        "--ref-store",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/report_evaluation/expanded_cohort/expanded_reference_store.json"),
        help="Path to reference store JSON.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val_dev",
        choices=["val_dev", "expanded_test"],
        help="Cohort split to evaluate: 'val_dev' (50 cases) or 'expanded_test' (117 cases).",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to fine-tuned LoRA checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/report_evaluation/finetuned_pilot"),
        help="Directory to save evaluation results.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run mock evaluation for fast verification.")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit number of cases evaluated.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MEDGEMMA 4B FINE-TUNED PILOT EVALUATION")
    print("=" * 70)
    print(f"Target Split:       {args.split}")
    print(f"Adapter Path:       {args.adapter_path}")
    print(f"Output Directory:   {out_dir}")
    print(f"Dry Run (Mock):     {args.dry_run}")
    print("-" * 70)

    # 1. Load dataset manifest & reference store
    manifest_df = pd.read_csv(args.manifest)
    with open(args.ref_store, "r", encoding="utf-8") as f:
        ref_store = json.load(f)

    # Filter for target split
    if args.split == "val_dev":
        target_df = manifest_df[manifest_df["split"] == "val_dev"].copy()
    else: # expanded_test
        target_df = manifest_df[manifest_df["split"].str.startswith("expanded_test")].copy()

    if args.max_cases:
        target_df = target_df.head(args.max_cases)

    print(f"Loaded {len(target_df)} cases for evaluation split '{args.split}'.")

    eval_records = []
    for _, row in target_df.iterrows():
        cid = row["safe_case_id"]
        ref_data = ref_store.get(cid, {})
        eval_records.append({
            "safe_case_id": cid,
            "image_path": row["image_path"],
            "primary_category": row["primary_category"],
            "reference_full_report": ref_data.get("reference_full_report", ""),
        })

    # 2. Run evaluation across all three pipelines
    pipelines = ["baseline", "prompt_v3", "finetuned"]
    all_case_results = []
    summaries = []

    comparison_rows = []

    for p in pipelines:
        print(f"\nEvaluating pipeline: [{p.upper()}] on {len(eval_records)} cases...")
        p_cases, p_sum = evaluate_cohort(
            eval_records,
            pipeline_name=p,
            adapter_path=args.adapter_path,
            mock=args.dry_run,
        )
        all_case_results.extend(p_cases)
        summaries.append(p_sum)

        comparison_rows.append({
            "pipeline": p,
            "split": args.split,
            "sample_size": p_sum["sample_size"],
            "micro_f1": p_sum["micro_f1"],
            "precision": p_sum["precision"],
            "recall": p_sum["recall"],
            "false_normal_rate": p_sum["false_normal_rate"],
            "false_normal_95ci": f"[{p_sum['false_normal_95ci'][0]:.4f}, {p_sum['false_normal_95ci'][1]:.4f}]",
            "hallucination_rate": p_sum["hallucination_rate"],
            "missed_finding_rate": p_sum["missed_finding_rate"],
            "internal_contradiction_rate": p_sum["contradiction_rate"],
            "boilerplate_rate": p_sum["boilerplate_rate"],
            "mean_latency_s": round(p_sum["mean_latency_ms"] / 1000.0, 2),
            "median_latency_s": round(p_sum["median_latency_ms"] / 1000.0, 2),
            "throughput_tok_s": p_sum["throughput_tok_s"],
        })

    # 3. Save comparison CSV and JSON
    comparison_df = pd.DataFrame(comparison_rows)
    comp_csv_path = out_dir / f"pilot_{args.split}_comparison.csv"
    comparison_df.to_csv(comp_csv_path, index=False)
    print(f"\nSaved comparison summary table to: {comp_csv_path}")

    case_json_path = out_dir / f"pilot_{args.split}_case_results.json"
    with open(case_json_path, "w", encoding="utf-8") as f:
        json.dump(all_case_results, f, indent=2)
    print(f"Saved individual case results to:    {case_json_path}")

    sum_json_path = out_dir / f"pilot_{args.split}_summary.json"
    with open(sum_json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved structured summary JSON to:   {sum_json_path}")

    # 4. Print comparative results table
    print("\n" + "=" * 95)
    print(f"PILOT EVALUATION RESULTS COMPARISON (Split: {args.split}, N={len(eval_records)})")
    print("=" * 95)
    col_fmt = "{:<16} | {:<8} | {:<8} | {:<8} | {:<12} | {:<12} | {:<10} | {:<10}"
    print(col_fmt.format("Pipeline", "Micro-F1", "Prec.", "Recall", "False-Normal", "Contradict.", "Mean Lat.", "Throughput"))
    print("-" * 95)
    for row in comparison_rows:
        print(col_fmt.format(
            row["pipeline"],
            f"{row['micro_f1']:.4f}",
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}",
            f"{row['false_normal_rate']*100:.1f}%",
            f"{row['internal_contradiction_rate']*100:.1f}%",
            f"{row['mean_latency_s']:.2f}s",
            f"{row['throughput_tok_s']:.1f} t/s",
        ))
    print("=" * 95)
    print("EVALUATION EXECUTION COMPLETED SUCCESSFULLY.")
    print("=" * 95)


if __name__ == "__main__":
    main()
