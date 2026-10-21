# Task 19B: Technical Audit of Inference Pipeline Data Isolation, Evidence Fidelity, and Leakage Safeguards

**Auditor:** Independent Technical & Medical Imaging AI Audit  
**Date:** October 2026  
**Target Repository:** InteractMed-VLM (`src/evidence_schema.py`, `src/report_service.py`, `src/batch_service.py`, `backend/main.py`)

---

## 1. Executive Summary

This audit examined whether reference dataset annotations, ground-truth bounding boxes, or training/validation labels leak into the runtime inference pipeline of InteractMed-VLM.

### Key Audit Findings:
1. **Zero Ground-Truth Contamination in Production Services:**  
   Inspection of `src/report_service.py`, `src/grounding_service.py`, `src/consistency_service.py`, and `backend/main.py` confirms that **no offline annotation files, MS-CXR labels, or reference ground-truth stores are accessed during inference**. The inference pipeline operates strictly on input image bytes and prompt queries.
2. **Offline Evaluation Script Defect (`scripts/evaluate_task19a_reports.py`):**  
   While production services are clean, the offline script `evaluate_task19a_reports.py` bypassed live MedGemma inference and injected ground-truth boxes as `pred_box_norm`. Consequently, the report ablation scores in Task 19A were synthetic benchmarks rather than live evaluations.
3. **Evidence Modality Separation:**  
   `src/evidence_schema.py` strictly decouples visual bounding boxes from finding presence probabilities, and treats secondary classifier alerts ($	au \ge 0.62$) as review signals (`presence_status="uncertain"`, `source="classifier_alert"`), never promoting them to established diagnoses.
4. **Batch Isolation:**  
   Batch execution in `src/batch_service.py` preserves per-item state isolation with distinct `job_id`, `structured_findings`, and `completeness_flags`. Zero state bleeding between batch items was detected.

---

## 2. Detailed Audit by Subsystem

### 2.1 Whole-Image Inference (`src/report_service.py`)
- **Inspection:** Checked `generate_whole_image_report()` and prompt templates.
- **Finding:** The service constructs prompts based solely on `clinical_indication` and `relevant_history` passed in the HTTP request. It does not perform retrieval against MS-CXR or Chest ImaGenome reference databases.
- **Audit Verdict:** **VERIFIED — Zero Leakage.**

### 2.2 Finding Extraction & Anatomical Normalization (`src/evidence_schema.py`)
- **Inspection:** Audited `extract_canonical_findings_from_report()` and `normalize_anatomical_location()`.
- **Finding:** Findings are extracted strictly from narrative text using deterministic regular expressions. Anatomical locations are mapped to Chest ImaGenome's 36 controlled entities. Missing findings remain unmentioned; they are **never silently marked as absent**.
- **Audit Verdict:** **VERIFIED — Rule-based and deterministic.**

### 2.3 Secondary Classifier Alert Handling
- **Inspection:** Audited `reconcile_evidence_with_grounding_and_classifier()`.
- **Finding:** When a classifier score exceeds the threshold ($	au \ge 0.62$), it is added with `presence_status="uncertain"`, `uncertainty="high"`, and `review_required=True`. The alert is explicitly logged in `audit_log["classifier_alerts_added"]` and is **never** rewritten as a confirmed diagnosis (`presence_status="present"`).
- **Audit Verdict:** **VERIFIED — Clinical safety principle upheld.**

### 2.4 TransVG Visual Grounding Geometry vs. Presence
- **Inspection:** Audited `src/grounding_service.py` and `reconcile_evidence_with_grounding_and_classifier()`.
- **Finding:** Predicted bounding boxes populate `supporting_region` as spatial coordinates (`[cx, cy, w, h]`). TransVG coordinates do not alter or calibrate the finding presence probability. If a region cannot be localized, `grounding_status` is flagged as `"unsupported"` or `"failed"`.
- **Audit Verdict:** **VERIFIED — Bounding box is spatial geometry, not diagnostic truth.**

### 2.5 Batch Processing Isolation (`src/batch_service.py`)
- **Inspection:** Audited `BatchService.create_batch()` and `_process_single_item()`.
- **Finding:** Each image is assigned a unique `job_id` (`f"job_{uuid.uuid4().hex[:8]}"`). Structured findings and completeness flags are stored per `BatchItem` instance. Prompt variables and caching layers are instantiated per image. Regression test `test_batch_service_per_image_isolation` verifies that distinct findings across batch items do not bleed into neighboring items.
- **Audit Verdict:** **VERIFIED — Strict per-image isolation.**

### 2.6 Patient-Friendly Translation Fidelity
- **Inspection:** Audited `translate_regional_report_to_patient_friendly()` in `src/report_service.py` and `check_report_completeness_and_fidelity()`.
- **Finding:** Patient translations strictly mirror reconciled source reports. The validation check flags hallucinations of support apparatus or claims of "completely clear lungs" when abnormalities exist.
- **Audit Verdict:** **VERIFIED — Reconciled source fidelity enforced.**

---

## 3. Summary of Identified Defects in Task 19A Artifacts

| Component | Code Location | Nature of Defect | Severity | Proposed Remediation |
| :--- | :--- | :--- | :--- | :--- |
| **Offline Report Evaluation Script** | `scripts/evaluate_task19a_reports.py:120-195` | MedGemma generation was mocked with hardcoded templates; TransVG was mocked with ground-truth boxes; Latency (+120ms) was synthetic. | **HIGH (Documentation / Evaluation Integrity)** | Clearly mark Task 19A report ablation metrics as *Synthetic Benchmark / Simulated Pipeline* in all reports. Do not claim live 425-512 ms end-to-end latency. |
| **Overlap Audit Table** | `artifacts/dataset_reporting/task19a_dataset_overlap_audit.md:18` | Unique patient and image counts were reported as 148, 134, 642, 423 instead of actual 99, 89, 444, 306. | **MEDIUM (Reporting Error)** | Reconcile with actual unique counts computed directly from `task19a_selected_images.csv`. |
| **Overlapping Patient Identification** | `artifacts/dataset_reporting/task19a_final_report.md:58` | Narrative stated patient `p10625523` was pruned due to Task 14 overlap; actual pruned patient was `p15486932`. | **LOW (Typographical / Attribution)** | Correct patient ID to `p15486932` in audit documentation. |
| **images_to_avoid Description** | `artifacts/dataset_reporting/task19a_final_report.md:44` | Claimed `images_to_avoid.csv` flagged "mostly lateral/oblique views or technical defects". | **MEDIUM (Dataset Interpretation)** | Clarify that `images_to_avoid.csv` defines the 500 gold-benchmark patient split to prevent silver data leakage; 72.8% of images are standard frontal projections. |
