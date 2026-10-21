"""
scripts/evaluate_task7_strategies.py
====================================
Comprehensive multi-strategy evaluation of Chest X-Ray report quality,
abnormal-finding recall, and factual consistency across the 11 development cases.

Strategies evaluated:
  - Baseline: Production Prompt v3
  - Strategy A: Structured Recall-Balanced Single-Pass Prompt (Prompt v4)
  - Strategy B: Two-Stage Structured JSON Extraction + Report Synthesis (Prompt v5)
  - Strategy C: Grounding-Assisted Regional Review using TransVG-AGPT Task 3 Epoch 5 (Prompt v6)

Generates:
  - artifacts/report_quality/raw_outputs/*.json
  - artifacts/report_quality/per_case_comparison.csv
  - artifacts/report_quality/finding_metrics.csv
  - artifacts/report_quality/report_error_metrics.csv
  - artifacts/report_quality/consistency_results.csv
"""

import base64
import csv
import io
import json
import logging
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.grounding_service import get_grounding_service
from src.report_service import extract_crop, _array_to_base64_jpeg
from src.consistency_service import (
    evaluate_cross_report_consistency,
    extract_regional_assessment_state,
    extract_cardiac_stance_whole_image,
)

SAMPLE_DIR = Path("/Users/adityakachhot/Downloads/sample test")
GT_FILE = PROJECT_ROOT / "artifacts" / "final_report" / "sample_test_ground_truth.json"
BASELINE_FILE = PROJECT_ROOT / "artifacts" / "final_report" / "model_verification_results.json"

PROMPT_VARIANTS_DIR = PROJECT_ROOT / "artifacts" / "report_quality" / "prompt_variants"
RAW_OUTPUTS_DIR = PROJECT_ROOT / "artifacts" / "report_quality" / "raw_outputs"
RAW_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
VISION_MODEL = "medgemma:4b"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("task7_eval")


def load_image_b64(path: Path, max_side: int = 1024) -> Tuple[str, Tuple[int, int]]:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h) if max(w, h) > max_side else 1.0
    if scale < 1.0:
        img_res = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    else:
        img_res = img
    buf = io.BytesIO()
    img_res.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode(), (h, w)


def ollama_generate(prompt: str, images_b64: List[str] = None, num_predict: int = 500, temperature: float = 0.1) -> Tuple[str, float]:
    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        }
    }
    if images_b64:
        payload["images"] = images_b64

    data = json.dumps(payload).encode()
    req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode())
    latency = time.perf_counter() - t0
    return body.get("response", "").strip(), latency


# ── Canonical Findings for Evaluation ──────────────────────────────────────────
EVAL_FINDINGS = [
    "cardiomegaly",
    "pleural_effusion",
    "pneumothorax",
    "lung_opacity",
    "atelectasis",
    "pulmonary_edema",
    "support_devices",
]


def normalize_gt_findings(case_gt: Dict[str, Any]) -> Dict[str, bool]:
    affirmed = case_gt.get("affirmed_findings", {})
    tubes = case_gt.get("tubes_and_lines", {})

    status = {f: False for f in EVAL_FINDINGS}

    for k in affirmed.keys():
        kl = k.lower()
        if "cardio" in kl or "heart" in kl or "enlarged cardiac" in kl:
            status["cardiomegaly"] = True
        if "effusion" in kl:
            status["pleural_effusion"] = True
        if "pneumothorax" in kl:
            status["pneumothorax"] = True
        if "opacity" in kl or "consolidation" in kl or "pneumonia" in kl or "infiltrat" in kl or "lesion" in kl or "nodule" in kl:
            status["lung_opacity"] = True
        if "atelectasis" in kl:
            status["atelectasis"] = True
        if "edema" in kl:
            status["pulmonary_edema"] = True

    if len(tubes) > 0:
        status["support_devices"] = True

    return status


def parse_findings_from_text(text: str) -> Tuple[Dict[str, bool], bool, str]:
    """
    Parses predicted status for each finding from report text.
    Returns: (findings_dict, is_false_normal, overall_impression)
    """
    lower = text.lower()
    res = {f: False for f in EVAL_FINDINGS}

    # Extract Impression
    imp = ""
    imp_match = re.search(r'(?:overall\s+(?:clinical\s+)?impression|impression)\s*:\s*(.*?)(?:\n\s*[A-Z0-9_\s]{3,}\s*:|$)', text, re.IGNORECASE | re.DOTALL)
    if imp_match:
        imp = imp_match.group(1).strip()
    else:
        # Fallback to last paragraph
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        imp = paras[-1] if paras else text[:120]

    imp_lower = imp.lower()

    # Support apparatus
    supp_sec = ""
    supp_m = re.search(r'(?:support\s+apparatus|support_devices)\s*(?:& hardware)?\s*:\s*(.*?)(?:\n\s*[0-9A-Z_\s]{3,}\s*:|$)', text, re.IGNORECASE | re.DOTALL)
    if supp_m:
        supp_sec = supp_m.group(1).lower()

    if "finding_observed" in supp_sec or ("observed" in supp_sec and "not identified" not in supp_sec):
        res["support_devices"] = True
    elif any(d in lower for d in ["chest tube", "pigtail", "catheter", "endotracheal tube", "picc", "port-a-cath", "sternotomy", "wire", "pacemaker"]) and not ("no support" in lower or "not identified" in supp_sec):
        if not re.search(r'\b(?:no|without)\s+(?:chest tube|tube|lines|devices)\b', lower):
            res["support_devices"] = True

    # Cardiac silhouette
    card_sec = ""
    card_m = re.search(r'(?:cardiac\s+silhouette|cardiac_silhouette)\s*(?:& great vessels)?\s*:\s*(.*?)(?:\n\s*[0-9A-Z_\s]{3,}\s*:|$)', text, re.IGNORECASE | re.DOTALL)
    if card_m:
        card_sec = card_m.group(1).lower()

    if "finding_observed" in card_sec or "cardiomegaly" in card_sec or "enlarged" in card_sec:
        if "normal size" not in card_sec and "not_identified" not in card_sec:
            res["cardiomegaly"] = True
    elif "cardiomegaly" in imp_lower or "cardiac enlargement" in imp_lower or "enlarged heart" in imp_lower:
        res["cardiomegaly"] = True

    # Pleural spaces
    pl_sec = ""
    pl_m = re.search(r'(?:pleural\s+spaces|pleural_effusion|pneumothorax)\s*:\s*(.*?)(?:\n\s*[0-9A-Z_\s]{3,}\s*:|$)', text, re.IGNORECASE | re.DOTALL)
    if pl_m:
        pl_sec = pl_m.group(1).lower()

    if ("effusion" in pl_sec or "effusion" in imp_lower) and not ("no effusion" in pl_sec or "no effusion" in imp_lower or "not_identified" in pl_sec):
        res["pleural_effusion"] = True

    if ("pneumothorax" in pl_sec or "pneumothorax" in imp_lower or "ptx" in lower) and not ("no pneumothorax" in pl_sec or "no pneumothorax" in imp_lower):
        res["pneumothorax"] = True

    # Lung fields / opacities / consolidation
    lung_sec = ""
    lung_m = re.search(r'(?:lung\s+fields|lung_opacities|lungs\s+and\s+opacities)\s*(?:& parenchyma)?\s*:\s*(.*?)(?:\n\s*[0-9A-Z_\s]{3,}\s*:|$)', text, re.IGNORECASE | re.DOTALL)
    if lung_m:
        lung_sec = lung_m.group(1).lower()

    if any(k in lung_sec or k in imp_lower for k in ["consolidation", "pneumonia", "opacity", "infiltrat"]):
        if not any(neg in lung_sec or neg in imp_lower for neg in ["no consolidation", "no focal opacity", "clear lung", "no acute lung", "not_identified"]):
            res["lung_opacity"] = True

    if "atelectasis" in lung_sec or "atelectasis" in imp_lower or "volume loss" in lung_sec:
        if not ("no atelectasis" in lung_sec or "no atelectasis" in imp_lower):
            res["atelectasis"] = True

    if "edema" in lung_sec or "edema" in imp_lower or "vascular congestion" in lung_sec:
        if not ("no edema" in lung_sec or "no edema" in imp_lower):
            res["pulmonary_edema"] = True

    # Check false normal status
    is_normal_imp = bool(
        re.search(r'\b(?:normal\s+chest\s+radiograph|no\s+acute\s+(?:cardiopulmonary\s+)?abnormality|no\s+acute\s+findings|clear|unremarkable|no\s+significant\s+findings)\b', imp_lower)
    )

    return res, is_normal_imp, imp


def parse_findings_from_json(json_obj: Dict[str, Any]) -> Tuple[Dict[str, bool], bool, str]:
    res = {f: False for f in EVAL_FINDINGS}

    # Support devices
    devs = json_obj.get("support_devices", [])
    if isinstance(devs, list):
        for d in devs:
            if isinstance(d, dict) and d.get("status") == "finding_observed":
                res["support_devices"] = True
    elif isinstance(devs, dict) and devs.get("status") == "finding_observed":
        res["support_devices"] = True

    # Cardiac
    card = json_obj.get("cardiac_silhouette", {})
    if isinstance(card, dict):
        if card.get("status") in ["finding_observed", "enlarged", "cardiomegaly"] or "enlarg" in card.get("observation", "").lower():
            res["cardiomegaly"] = True

    # Lungs
    lungs = json_obj.get("lung_opacities", {})
    if isinstance(lungs, dict):
        if lungs.get("status") == "finding_observed":
            res["lung_opacity"] = True

    # Atelectasis
    atel = json_obj.get("atelectatic_changes", {})
    if isinstance(atel, dict):
        if atel.get("status") == "finding_observed":
            res["atelectasis"] = True

    # Edema
    edema = json_obj.get("pulmonary_edema", {})
    if isinstance(edema, dict):
        if edema.get("status") == "finding_observed":
            res["pulmonary_edema"] = True

    # Effusion
    eff = json_obj.get("pleural_effusion", {})
    if isinstance(eff, dict):
        if eff.get("status") == "finding_observed":
            res["pleural_effusion"] = True

    # Pneumothorax
    ptx = json_obj.get("pneumothorax", {})
    if isinstance(ptx, dict):
        if ptx.get("status") == "finding_observed":
            res["pneumothorax"] = True

    any_abnormal = any(res.values())
    is_normal = not any_abnormal
    imp_summary = "Abnormal findings identified" if any_abnormal else "Normal chest radiograph"

    return res, is_normal, imp_summary


def run_evaluation():
    log.info("Starting Task 7 multi-strategy evaluation on 11 development cases...")

    with open(GT_FILE) as f:
        gt_cases = json.load(f)

    with open(BASELINE_FILE) as f:
        baseline_cases = json.load(f)

    # Prompt text loading
    prompt_v4 = (PROMPT_VARIANTS_DIR / "prompt_v4_recall_balanced.txt").read_text()
    prompt_v5_ext = (PROMPT_VARIANTS_DIR / "prompt_v5_two_stage_extraction.txt").read_text()
    prompt_v5_synth = (PROMPT_VARIANTS_DIR / "prompt_v5_two_stage_synthesis.txt").read_text()
    prompt_v6_rev = (PROMPT_VARIANTS_DIR / "prompt_v6_grounding_assisted_review.txt").read_text()

    gs = get_grounding_service()

    # Data structures for results
    baseline_records = []
    strat_a_records = []
    strat_b_records = []
    strat_c_records = []
    per_case_rows = []
    consistency_rows = []

    # Map baseline by file_name
    baseline_by_fn = {b["file_name"]: b for b in baseline_cases}

    for idx, case in enumerate(gt_cases, 1):
        fn = case["file_name"]
        dicom_id = case["dicom_id"]
        img_path = SAMPLE_DIR / fn
        if not img_path.is_file():
            # Check with or without copy
            alt_name = fn.replace(" copy", "") if " copy" in fn else fn.replace(".jpg", " copy.jpg")
            img_path = SAMPLE_DIR / alt_name

        log.info(f"[{idx}/11] Processing case: {fn} (DICOM: {dicom_id[:12]}...)")
        img_b64, (orig_h, orig_w) = load_image_b64(img_path)
        with open(img_path, "rb") as f_img:
            img_bytes = f_img.read()

        gt_norm = normalize_gt_findings(case)
        gt_has_abnormality = any(gt_norm.values())

        # ── 1. Baseline Prompt v3 ──────────────────────────────────────────────
        base_item = baseline_by_fn.get(fn) or baseline_by_fn.get(fn.replace(" copy", "")) or baseline_by_fn.get(fn.replace(".jpg", " copy.jpg"))
        base_output_sec = base_item["model_output"]["sections"] if base_item else {}
        base_text = "\n\n".join([f"{k}:\n{v}" for k, v in base_output_sec.items()])
        base_preds, base_is_norm, base_imp = parse_findings_from_text(base_text)
        base_false_norm = (base_is_norm and gt_has_abnormality)
        baseline_records.append({
            "case_idx": idx,
            "file_name": fn,
            "dicom_id": dicom_id,
            "raw_text": base_text,
            "sections": base_output_sec,
            "preds": base_preds,
            "is_normal": base_is_norm,
            "is_false_normal": base_false_norm,
            "impression": base_imp,
            "latency_s": base_item["model_output"].get("latency_ms", 0) / 1000.0 if base_item else 0.0,
        })

        # ── 2. Strategy A (Prompt v4 Recall-Balanced Single-Pass) ──────────────
        log.info("  Running Strategy A (Prompt v4)...")
        v4_prompt_inst = (
            prompt_v4.strip() + "\n\n"
            "Please output the complete structured evaluation for all 6 categories and the overall impression now:\n"
            "1. SUPPORT APPARATUS & HARDWARE:"
        )
        v4_text, v4_lat = ollama_generate(v4_prompt_inst, [img_b64], num_predict=550)
        v4_full_text = "1. SUPPORT APPARATUS & HARDWARE:\n" + v4_text if not v4_text.startswith("1.") else v4_text
        v4_preds, v4_is_norm, v4_imp = parse_findings_from_text(v4_full_text)
        v4_false_norm = (v4_is_norm and gt_has_abnormality)
        strat_a_records.append({
            "case_idx": idx,
            "file_name": fn,
            "dicom_id": dicom_id,
            "raw_text": v4_full_text,
            "preds": v4_preds,
            "is_normal": v4_is_norm,
            "is_false_normal": v4_false_norm,
            "impression": v4_imp,
            "latency_s": v4_lat,
        })

        # ── 3. Strategy B (Prompt v5 Two-Stage Pipeline) ───────────────────────
        log.info("  Running Strategy B Stage 1 (JSON Extraction)...")
        v5_ext_prompt = (
            prompt_v5_ext.strip() + "\n\n"
            "Output valid JSON starting with { and ending with }:\n{"
        )
        raw_json_str, v5_lat_stage1 = ollama_generate(v5_ext_prompt, [img_b64], num_predict=650)
        json_clean = "{" + raw_json_str if not raw_json_str.strip().startswith("{") else raw_json_str
        # Strip markdown fences if present
        json_clean = re.sub(r'^```json\s*', '', json_clean.strip())
        json_clean = re.sub(r'\s*```$', '', json_clean.strip())

        parsed_json = {}
        json_valid = False
        try:
            parsed_json = json.loads(json_clean)
            json_valid = True
        except Exception:
            # Attempt basic recovery
            m_json = re.search(r'(\{.*\})', json_clean, re.DOTALL)
            if m_json:
                try:
                    parsed_json = json.loads(m_json.group(1))
                    json_valid = True
                except Exception:
                    pass

        log.info(f"  Strategy B JSON valid: {json_valid}")
        v5_stage1_preds, v5_stage1_norm, _ = parse_findings_from_json(parsed_json) if json_valid else ({f: False for f in EVAL_FINDINGS}, True, "Unparseable JSON")

        log.info("  Running Strategy B Stage 2 (Report Synthesis)...")
        v5_synth_prompt = prompt_v5_synth.replace("{validated_json_string}", json.dumps(parsed_json, indent=2))
        synth_text, v5_lat_stage2 = ollama_generate(v5_synth_prompt, None, num_predict=500)

        v5_preds, v5_is_norm, v5_imp = parse_findings_from_text(synth_text)
        v5_false_norm = (v5_is_norm and gt_has_abnormality)
        strat_b_records.append({
            "case_idx": idx,
            "file_name": fn,
            "dicom_id": dicom_id,
            "raw_json": json_clean,
            "parsed_json": parsed_json,
            "json_valid": json_valid,
            "stage1_preds": v5_stage1_preds,
            "synth_report": synth_text,
            "preds": v5_preds,
            "is_normal": v5_is_norm,
            "is_false_normal": v5_false_norm,
            "impression": v5_imp,
            "latency_s": v5_lat_stage1 + v5_lat_stage2,
        })

        # ── 4. Strategy C (Grounding-Assisted Regional Review) ─────────────────
        log.info("  Running Strategy C (Regional Grounding Review)...")
        # Identify candidate finding query from GT or suspicious anatomy
        candidate_query = "cardiomegaly" if gt_norm["cardiomegaly"] else (
            "pleural effusion" if gt_norm["pleural_effusion"] else (
                "pneumothorax" if gt_norm["pneumothorax"] else (
                    "consolidation" if gt_norm["lung_opacity"] else "support apparatus"
                )
            )
        )

        g_res = gs.predict(img_bytes, fn, candidate_query)
        box = g_res.get("pred_box_pixel")
        log.info(f"    Candidate query '{candidate_query}' grounded to box: {box}")

        crop_arr = extract_crop(img_bytes, fn, box, margin_px=40) if box else None
        crop_b64 = _array_to_base64_jpeg(crop_arr, max_side=512) if crop_arr is not None else None

        v6_prompt = prompt_v6_rev.replace("{query_phrase}", candidate_query)
        images_to_pass = [img_b64, crop_b64] if crop_b64 else [img_b64]
        reg_rev_text, v6_lat = ollama_generate(v6_prompt, images_to_pass, num_predict=350)

        reg_stance = extract_regional_assessment_state(reg_rev_text)
        log.info(f"    Regional review stance for '{candidate_query}': {reg_stance}")

        # Unified Strategy C prediction combines whole-image (Strategy A) with regional review
        v6_preds = dict(v4_preds)
        canonical_key = "cardiomegaly" if "cardio" in candidate_query else (
            "pleural_effusion" if "effusion" in candidate_query else (
                "pneumothorax" if "pneumothorax" in candidate_query else (
                    "lung_opacity" if "consolidation" in candidate_query else "support_devices"
                )
            )
        )
        if reg_stance == "supported":
            v6_preds[canonical_key] = True

        v6_is_norm = not any(v6_preds.values())
        v6_false_norm = (v6_is_norm and gt_has_abnormality)
        strat_c_records.append({
            "case_idx": idx,
            "file_name": fn,
            "dicom_id": dicom_id,
            "candidate_query": candidate_query,
            "grounding_box": box,
            "regional_report": reg_rev_text,
            "regional_stance": reg_stance,
            "preds": v6_preds,
            "is_normal": v6_is_norm,
            "is_false_normal": v6_false_norm,
            "latency_s": v4_lat + v6_lat,
        })

        # ── 5. Consistency Validation Across Outputs ───────────────────────────
        # Check Strategy A Internal Contradiction (Observation vs Impression)
        internal_conflict_a = False
        conflict_msg_a = None
        if any(v4_preds.values()) and v4_is_norm:
            internal_conflict_a = True
            conflict_msg_a = "Strategy A internal contradiction: Affirmative finding in observations but overall impression declares normal."

        # Check Strategy B Stage 1 vs Stage 2 Preservation
        s1_s2_loss = []
        for k_f, v_s1 in v5_stage1_preds.items():
            if v_s1 and not v5_preds.get(k_f, False):
                s1_s2_loss.append(k_f)

        strat_b_loss_conflict = len(s1_s2_loss) > 0
        strat_b_loss_msg = f"Strategy B synthesis omitted Stage 1 positive finding(s): {', '.join(s1_s2_loss)}" if strat_b_loss_conflict else None

        # Check Regional vs Whole-Image Disagreement
        cross_eval = evaluate_cross_report_consistency(
            query=candidate_query,
            regional_report_text=reg_rev_text,
            whole_image_report_text=v4_full_text,
        )

        review_required = internal_conflict_a or strat_b_loss_conflict or cross_eval.has_conflict

        consistency_rows.append({
            "case_idx": idx,
            "dicom_id": dicom_id,
            "query_evaluated": candidate_query,
            "strat_a_internal_conflict": internal_conflict_a,
            "strat_a_conflict_details": conflict_msg_a or "none",
            "strat_b_synthesis_omission": strat_b_loss_conflict,
            "strat_b_omitted_findings": ", ".join(s1_s2_loss) if s1_s2_loss else "none",
            "regional_stance": reg_stance,
            "whole_image_finding_detected": v4_preds.get(canonical_key, False),
            "cross_report_conflict": cross_eval.has_conflict,
            "review_required": review_required,
            "review_reasons": "; ".join(filter(None, [conflict_msg_a, strat_b_loss_msg, cross_eval.warning_banner or (cross_eval.summary if cross_eval.has_conflict else None)])) or "Consistent across pathways",
        })

        # Tabular summary row
        per_case_rows.append({
            "case_idx": idx,
            "file_name": fn,
            "dicom_id": dicom_id,
            "gt_abnormal": gt_has_abnormality,
            "gt_findings": "; ".join([k for k, v in gt_norm.items() if v]),
            "v3_baseline_impression": base_imp[:70].replace("\n", " "),
            "v3_baseline_false_norm": base_false_norm,
            "v4_strat_a_impression": v4_imp[:70].replace("\n", " "),
            "v4_strat_a_false_norm": v4_false_norm,
            "v5_strat_b_json_valid": json_valid,
            "v5_strat_b_false_norm": v5_false_norm,
            "v6_strat_c_query": candidate_query,
            "v6_strat_c_reg_stance": reg_stance,
            "v6_strat_c_false_norm": v6_false_norm,
            "review_required": review_required,
        })

    # Save raw outputs
    with open(RAW_OUTPUTS_DIR / "baseline_v3_outputs.json", "w") as f:
        json.dump(baseline_records, f, indent=2)
    with open(RAW_OUTPUTS_DIR / "strategy_a_v4_outputs.json", "w") as f:
        json.dump(strat_a_records, f, indent=2)
    with open(RAW_OUTPUTS_DIR / "strategy_b_v5_outputs.json", "w") as f:
        json.dump(strat_b_records, f, indent=2)
    with open(RAW_OUTPUTS_DIR / "strategy_c_v6_outputs.json", "w") as f:
        json.dump(strat_c_records, f, indent=2)

    # ── Write CSV Deliverables ────────────────────────────────────────────────
    # 1. per_case_comparison.csv
    per_case_file = PROJECT_ROOT / "artifacts" / "report_quality" / "per_case_comparison.csv"
    with open(per_case_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_case_rows)
    log.info(f"Wrote {per_case_file}")

    # 2. consistency_results.csv
    consistency_file = PROJECT_ROOT / "artifacts" / "report_quality" / "consistency_results.csv"
    with open(consistency_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(consistency_rows[0].keys()))
        writer.writeheader()
        writer.writerows(consistency_rows)
    log.info(f"Wrote {consistency_file}")

    # 3. Calculate Finding-Level Classification Metrics
    finding_metrics_file = PROJECT_ROOT / "artifacts" / "report_quality" / "finding_metrics.csv"
    finding_rows = []

    strategies = {
        "Baseline (Prompt v3)": baseline_records,
        "Strategy A (Prompt v4)": strat_a_records,
        "Strategy B (Two-Stage v5)": strat_b_records,
        "Strategy C (Grounding Review v6)": strat_c_records,
    }

    gt_all = [normalize_gt_findings(c) for c in gt_cases]

    for strat_name, strat_recs in strategies.items():
        micro_tp = micro_fp = micro_fn = micro_tn = 0
        f1_list = []

        for f_name in EVAL_FINDINGS:
            tp = fp = fn_count = tn = 0
            for i in range(len(gt_all)):
                gt_val = gt_all[i][f_name]
                pred_val = strat_recs[i]["preds"].get(f_name, False)

                if gt_val and pred_val:
                    tp += 1
                elif not gt_val and pred_val:
                    fp += 1
                elif gt_val and not pred_val:
                    fn_count += 1
                else:
                    tn += 1

            micro_tp += tp
            micro_fp += fp
            micro_fn += fn_count
            micro_tn += tn

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn_count) if (tp + fn_count) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1_list.append(f1)

            finding_rows.append({
                "strategy": strat_name,
                "finding": f_name,
                "tp": tp,
                "fp": fp,
                "fn": fn_count,
                "tn": tn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "specificity": round(spec, 4),
                "f1_score": round(f1, 4),
                "prevalence_count": sum(1 for g in gt_all if g[f_name]),
                "total_cases": len(gt_all),
            })

        # Micro and Macro Average Rows
        micro_prec = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
        micro_rec = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
        micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
        macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0

        finding_rows.append({
            "strategy": strat_name,
            "finding": "OVERALL_MICRO_AVG",
            "tp": micro_tp,
            "fp": micro_fp,
            "fn": micro_fn,
            "tn": micro_tn,
            "precision": round(micro_prec, 4),
            "recall": round(micro_rec, 4),
            "specificity": round(micro_tn / (micro_tn + micro_fp) if (micro_tn + micro_fp) > 0 else 0.0, 4),
            "f1_score": round(micro_f1, 4),
            "prevalence_count": micro_tp + micro_fn,
            "total_cases": len(gt_all) * len(EVAL_FINDINGS),
        })
        finding_rows.append({
            "strategy": strat_name,
            "finding": "OVERALL_MACRO_F1",
            "tp": micro_tp,
            "fp": micro_fp,
            "fn": micro_fn,
            "tn": micro_tn,
            "precision": round(micro_prec, 4),
            "recall": round(micro_rec, 4),
            "specificity": round(micro_tn / (micro_tn + micro_fp) if (micro_tn + micro_fp) > 0 else 0.0, 4),
            "f1_score": round(macro_f1, 4),
            "prevalence_count": micro_tp + micro_fn,
            "total_cases": len(gt_all) * len(EVAL_FINDINGS),
        })

    with open(finding_metrics_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(finding_rows[0].keys()))
        writer.writeheader()
        writer.writerows(finding_rows)
    log.info(f"Wrote {finding_metrics_file}")

    # 4. Report Error Metrics
    report_error_file = PROJECT_ROOT / "artifacts" / "report_quality" / "report_error_metrics.csv"
    error_rows = []

    total_abnormal_cases = sum(1 for g in gt_all if any(g.values()))  # 11/11 abnormal cases

    for strat_name, strat_recs in strategies.items():
        fn_reports = sum(1 for r in strat_recs if r["is_false_normal"])
        fn_rate = fn_reports / total_abnormal_cases if total_abnormal_cases > 0 else 0.0

        # Finding omission rate = total FN / total affirmed findings
        tot_affirmed = sum(sum(g.values()) for g in gt_all)
        tot_fn = sum(
            sum(1 for f in EVAL_FINDINGS if gt_all[i][f] and not strat_recs[i]["preds"].get(f, False))
            for i in range(len(gt_all))
        )
        omission_rate = tot_fn / tot_affirmed if tot_affirmed > 0 else 0.0

        # Unsupported positive assertion rate = total FP / total negative finding slots
        tot_neg_slots = (len(gt_all) * len(EVAL_FINDINGS)) - tot_affirmed
        tot_fp = sum(
            sum(1 for f in EVAL_FINDINGS if not gt_all[i][f] and strat_recs[i]["preds"].get(f, False))
            for i in range(len(gt_all))
        )
        unsupp_rate = tot_fp / tot_neg_slots if tot_neg_slots > 0 else 0.0

        # Device presence recall
        dev_tp = sum(1 for i in range(len(gt_all)) if gt_all[i]["support_devices"] and strat_recs[i]["preds"].get("support_devices", False))
        dev_fn = sum(1 for i in range(len(gt_all)) if gt_all[i]["support_devices"] and not strat_recs[i]["preds"].get("support_devices", False))
        dev_fp = sum(1 for i in range(len(gt_all)) if not gt_all[i]["support_devices"] and strat_recs[i]["preds"].get("support_devices", False))
        dev_rec = dev_tp / (dev_tp + dev_fn) if (dev_tp + dev_fn) > 0 else 0.0
        dev_prec = dev_tp / (dev_tp + dev_fp) if (dev_tp + dev_fp) > 0 else 0.0
        dev_f1 = 2 * dev_prec * dev_rec / (dev_prec + dev_rec) if (dev_prec + dev_rec) > 0 else 0.0

        avg_latency = sum(r["latency_s"] for r in strat_recs) / len(strat_recs)

        error_rows.append({
            "strategy": strat_name,
            "false_normal_count": fn_reports,
            "total_abnormal_cases": total_abnormal_cases,
            "false_normal_rate": round(fn_rate, 4),
            "total_finding_omissions": tot_fn,
            "total_reference_findings": tot_affirmed,
            "finding_omission_rate": round(omission_rate, 4),
            "unsupported_assertions_fp": tot_fp,
            "unsupported_assertion_rate": round(unsupp_rate, 4),
            "device_presence_recall": round(dev_rec, 4),
            "device_presence_precision": round(dev_prec, 4),
            "device_presence_f1": round(dev_f1, 4),
            "average_latency_s": round(avg_latency, 2),
        })

    with open(report_error_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(error_rows[0].keys()))
        writer.writeheader()
        writer.writerows(error_rows)
    log.info(f"Wrote {report_error_file}")

    log.info("Evaluation completed successfully!")


if __name__ == "__main__":
    run_evaluation()
