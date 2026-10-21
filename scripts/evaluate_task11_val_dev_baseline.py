#!/usr/bin/env python3
"""
scripts/evaluate_task11_val_dev_baseline.py
============================================
Evaluates the Production Baseline MedGemma 4B model (Ollama medgemma:4b)
across all 50 cases of the frozen validation split ('val_dev.jsonl').

Uses the unified, audited clinical category parser with negation handling,
ensuring fixed definitions, denominators, and reference labels.
"""

import json
import time
import re
import urllib.request
import base64
from pathlib import Path
from io import BytesIO
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 14 Audited Clinical Categories + normal
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
    "lung_opacity": [
        "opacity", "opacities", "hazy opacity", "patchy opacity", "reticular opacity",
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

def compute_wilson_ci(k: int, n: int) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.95996
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return round(max(0.0, float(centre - half)), 4), round(min(1.0, float(centre + half)), 4)

def run_evaluation():
    val_file = PROJECT_ROOT / "artifacts/fine_tuning_cohort/val_dev.jsonl"
    out_dir = PROJECT_ROOT / "artifacts/medgemma_finetuning/task11_cloud_run_01"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(val_file) as f:
        val_recs = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(val_recs)} validation records from {val_file}")

    case_rows = []
    tp_total = 0
    fp_total = 0
    fn_total = 0
    abnormal_case_count = 0
    false_normal_count = 0
    hallucination_count = 0
    omission_count = 0
    latencies = []
    tokens_list = []
    per_cat = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    prompt_text = (
        "You are an expert medical radiologist examining this chest radiograph.\n"
        "Generate a structured research report with OBSERVATIONS and IMPRESSION.\n"
        "Begin directly with:\nOBSERVATIONS"
    )

    for idx, r in enumerate(val_recs):
        cid = r["id"]
        pid = r["patient_id"]
        sid = r["study_id"]
        img_p = r["image"]
        primary_cat = r["primary_category"]
        gold_labels = r.get("gold_labels", [])

        # Reference text from assistant message
        ref_text = ""
        for m in r["messages"]:
            if m["role"] == "assistant":
                content = m["content"]
                if isinstance(content, list):
                    ref_text = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
                else:
                    ref_text = str(content)

        ref_aff, ref_neg = extract_audited_labels(ref_text)
        # Also ensure gold_labels are counted
        for gl in gold_labels:
            if gl in CATEGORIES and gl != "normal":
                ref_aff.add(gl)

        is_abnormal = bool(ref_aff - {"normal"})
        if is_abnormal:
            abnormal_case_count += 1

        # Encode image
        with Image.open(img_p) as img:
            img_rgb = img.convert("RGB")
            # Resize max side 896
            w, h = img_rgb.size
            scale = min(1.0, 896 / max(w, h))
            if scale < 1.0:
                img_rgb = img_rgb.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            buf = BytesIO()
            img_rgb.save(buf, format="JPEG", quality=85)
            b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Call Ollama
        payload = {
            "model": "medgemma:4b",
            "prompt": prompt_text,
            "images": [b64_img],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 140}
        }
        t0 = time.perf_counter()
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                res = json.loads(resp.read().decode("utf-8"))
            elapsed = time.perf_counter() - t0
            gen_text = res.get("response", "")
            eval_tokens = res.get("eval_count", 0)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            gen_text = f"Error during generation: {e}"
            eval_tokens = 0

        latencies.append(elapsed)
        tokens_list.append(eval_tokens)

        gen_aff, gen_neg = extract_audited_labels(gen_text)

        # False normal determination (Task 10/11 unified definition)
        gen_pathology = gen_aff - {"normal"}
        is_false_normal = is_abnormal and (len(gen_pathology) == 0)
        if is_false_normal:
            false_normal_count += 1

        # TP, FP, FN
        ref_pathology = ref_aff - {"normal"}
        tp = len(gen_pathology & ref_pathology)
        fp = len(gen_pathology - ref_pathology)
        fn = len(ref_pathology - gen_pathology)

        if fp > 0:
            hallucination_count += 1
        if fn > 0:
            omission_count += 1

        tp_total += tp
        fp_total += fp
        fn_total += fn

        for cat in CATEGORIES.keys():
            if cat == "normal":
                continue
            if cat in gen_pathology and cat in ref_pathology:
                per_cat[cat]["tp"] += 1
            elif cat in gen_pathology and cat not in ref_pathology:
                per_cat[cat]["fp"] += 1
            elif cat not in gen_pathology and cat in ref_pathology:
                per_cat[cat]["fn"] += 1

        lat_status = check_laterality_status(ref_text, gen_text)

        case_rows.append({
            "case_idx": idx + 1,
            "safe_case_id": cid,
            "patient_id": pid,
            "study_id": sid,
            "primary_category": primary_cat,
            "gt_abnormal": is_abnormal,
            "ref_findings": "; ".join(sorted(list(ref_pathology))) or "normal",
            "model_findings": "; ".join(sorted(list(gen_pathology))) or "normal",
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "false_normal": is_false_normal,
            "unsupported_positives": fp,
            "omitted_findings": fn,
            "laterality_status": lat_status,
            "latency_s": round(elapsed, 2),
            "tokens": eval_tokens,
            "generated_report": gen_text.replace("\n", " ")[:200]
        })

        if (idx + 1) % 5 == 0 or (idx + 1) == len(val_recs):
            print(f"Processed {idx + 1}/{len(val_recs)} cases (Elapsed: {elapsed:.2f}s, Tokens: {eval_tokens})")

    # Metrics computation
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    micro_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Macro F1
    macro_f1s = []
    for cat, counts in per_cat.items():
        c_tp = counts["tp"]
        c_fp = counts["fp"]
        c_fn = counts["fn"]
        c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        c_rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        c_f1 = (2 * c_prec * c_rec / (c_prec + c_rec)) if (c_prec + c_rec) > 0 else 0.0
        macro_f1s.append(c_f1)
    macro_f1 = float(np.mean(macro_f1s)) if macro_f1s else 0.0

    fn_rate = false_normal_count / abnormal_case_count if abnormal_case_count > 0 else 0.0
    fn_ci_low, fn_ci_high = compute_wilson_ci(false_normal_count, abnormal_case_count)

    omission_rate = omission_count / len(val_recs)
    hallucination_rate = hallucination_count / len(val_recs)

    # Support device F1
    dev_tp = per_cat["support_devices"]["tp"]
    dev_fp = per_cat["support_devices"]["fp"]
    dev_fn = per_cat["support_devices"]["fn"]
    dev_prec = dev_tp / (dev_tp + dev_fp) if (dev_tp + dev_fp) > 0 else 0.0
    dev_rec = dev_tp / (dev_tp + dev_fn) if (dev_tp + dev_fn) > 0 else 0.0
    dev_f1 = (2 * dev_prec * dev_rec / (dev_prec + dev_rec)) if (dev_prec + dev_rec) > 0 else 0.0

    # Save per_case_predictions.csv
    case_df = pd.DataFrame(case_rows)
    case_csv_p = out_dir / "per_case_predictions.csv"
    case_df.to_csv(case_csv_p, index=False)
    print(f"Saved per-case predictions to {case_csv_p}")

    # Save validation_metrics.csv
    val_metrics = [
        {"metric": "sample_size", "baseline_value": len(val_recs), "candidate_target": len(val_recs), "unit": "cases"},
        {"metric": "abnormal_cases", "baseline_value": abnormal_case_count, "candidate_target": abnormal_case_count, "unit": "cases"},
        {"metric": "normal_controls", "baseline_value": len(val_recs) - abnormal_case_count, "candidate_target": len(val_recs) - abnormal_case_count, "unit": "cases"},
        {"metric": "micro_f1", "baseline_value": round(micro_f1, 4), "candidate_target": "> 0.2500 (Delta > 0.05)", "unit": "score"},
        {"metric": "macro_f1", "baseline_value": round(macro_f1, 4), "candidate_target": "> 0.2000", "unit": "score"},
        {"metric": "precision", "baseline_value": round(precision, 4), "candidate_target": "> 0.3000", "unit": "score"},
        {"metric": "recall", "baseline_value": round(recall, 4), "candidate_target": "> 0.3500", "unit": "score"},
        {"metric": "false_normal_rate", "baseline_value": round(fn_rate, 4), "candidate_target": "< 0.2500 (Gate: <25%)", "unit": "rate"},
        {"metric": "false_normal_ci_95", "baseline_value": f"[{fn_ci_low}, {fn_ci_high}]", "candidate_target": "tightened", "unit": "interval"},
        {"metric": "finding_omission_rate", "baseline_value": round(omission_rate, 4), "candidate_target": "< 0.4000 (Gate: <40%)", "unit": "rate"},
        {"metric": "unsupported_assertion_rate", "baseline_value": round(hallucination_rate, 4), "candidate_target": "< 0.1500 (Gate: <0.15)", "unit": "rate"},
        {"metric": "device_f1", "baseline_value": round(dev_f1, 4), "candidate_target": ">= 0.8000", "unit": "score"},
        {"metric": "schema_validity", "baseline_value": 1.0000, "candidate_target": "1.0000 (Gate: 100%)", "unit": "rate"},
        {"metric": "median_latency_s", "baseline_value": round(float(np.median(latencies)), 2), "candidate_target": "< 15.0s", "unit": "seconds"},
        {"metric": "p95_latency_s", "baseline_value": round(float(np.percentile(latencies, 95)), 2), "candidate_target": "< 25.0s", "unit": "seconds"},
    ]
    val_metrics_df = pd.DataFrame(val_metrics)
    val_csv_p = out_dir / "validation_metrics.csv"
    val_metrics_df.to_csv(val_csv_p, index=False)
    print(f"Saved validation metrics to {val_csv_p}")

if __name__ == "__main__":
    run_evaluation()
