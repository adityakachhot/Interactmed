# Task 16: Batch Failure Audit and Root-Cause Investigation

**Repository:** InteractMed-VLM  
**Evaluation Date:** October 2026  
**Target File Analyzed:** `batch_b000715d_summary.csv` (10 completed batch jobs) + 11 Chest Radiograph Images  
**Evaluator:** Senior ML / Vision-Language Systems / CV / Reliability Engineer  

---

## 1. Executive Summary

This audit investigates the reproducible root causes of the widespread quality defects, false-normal claims, token mangling, mid-sentence truncations, and triage priority inversions observed in the InteractMed-VLM batch reporting and research-triage pipeline.

All 11 uploaded chest X-rays were audited at the raw-byte and pixel level, matched to clinical reference ground truth in `artifacts/final_report/sample_test_ground_truth.json`, and traced through the backend pipeline from HTTP upload to CSV summary export.

---

## 2. Confirmed Defects and Reproducible Root Causes

### Defect 1: Token Mangling / Merged Heading Artifact (`What is uncertaThis...`)
* **Observed Symptom:** In jobs `job_553f7091` and `job_b194ec8b`, the section heading `What is uncertain` was rendered as:
  ```text
  What is uncertaThis frontal view displays both the left and right sides of the chest...
  ```
  Immediately followed by a redundant repetition of `**What is uncertain**`.
* **Investigation & Exact Mechanism:**
  In `src/report_service.py` within `clean_and_structure_patient_report()`, the regex cleaner was designed to normalize phrasing such as *"In this frontal view..."* or *"On this frontal view..."*. The pattern used was:
  ```python
  one_side_patterns = [
      r'(?:as\s+|on\s+|in\s+)?(?:this\s+)?(?:single\s+)?(?:frontal\s+view\s+.*?)(?=(?:depth|subtle|cannot|it\s+doesn\'t))',
      ...
  ]
  ```
  The substring `in\s+` was unanchored by word boundary `\b`. When processing the header string `"What is uncertain this frontal view..."`, the letters `"in "` at the tail of the word `"uncertain "` matched `in\s+`! The replacement string `"This frontal view..."` substituted starting from the `"in "` of `"uncertain"`, leaving `"What is uncerta"` directly prepended to `"This frontal view..."`.
* **Fix Implemented:**
  Added strict word boundary anchors: `\b(?:as\s+|on\s+|in\s+)?`. Added regression unit tests in `tests/test_task16_batch_regression.py`.

---

### Defect 2: Abrupt Mid-Sentence Truncation
* **Observed Symptom:**
  - `job_6df25184`: Truncated abruptly in the uncertainty section: `What is uncertain: The X-ray is a frontal view, which shows both sides of the chest. However, it is a single two` (no terminal punctuation or conclusion).
  - `job_2b47aad8`: Truncated in the final section: `What happens next: Please consult your ordering physician for qualified clinical evaluation` (missing period and cut off before disclaimer).
* **Investigation & Exact Mechanism:**
  In `src/report_service.py`, Stage 2 (synthesis and structuring) called Ollama with a rigid token cap of `num_predict=280`. When MedGemma produced detailed observations across all 5 anatomical zones in Stage 1, the 4-section patient synthesis routinely hit the 280-token ceiling mid-sentence. Crucially, the pipeline lacked any terminal sentence validator, silently accepting truncated text as a successful completion.
* **Fix Implemented:**
  1. Expanded Stage 2 synthesis token limit from `num_predict=280` to `num_predict=380` in `src/report_service.py`.
  2. Implemented `validate_generated_report()` with deterministic sentence integrity checks, detecting dangling conjunctions, incomplete phrases (e.g., `"single two"`, `"qualified clinical evaluation"` without period), and setting status `incomplete_report` with `review_required=true`.

---

### Defect 3: Batch Execution Bypassed the Auxiliary Classifier and Consistency Service
* **Observed Symptom:**
  Batch reports exhibited an 80% false-normal rate (8 of 10 cases declared "normal heart size, clear lungs", despite reference ground truth containing apical pneumothorax, large pleural effusions, severe cardiomegaly, and sepsis-associated pneumonia).
* **Investigation & Exact Mechanism:**
  In the single-image interactive workflow, InteractMed-VLM integrated the TorchXRayVision DenseNet-121 classifier (`predict()`) and cross-report consistency checks (`evaluate_cross_report_consistency()`). However, `src/batch_service.py` called MedGemma 4B alone via `generate_whole_image_report()`. Because batch processing ran MedGemma in isolation:
  - The auxiliary classifier never ran.
  - The consistency service was never invoked.
  - No cross-model discordance warnings were raised.
  - Triage only saw MedGemma's raw text, defaulting to `RULE_NO_PRIORITY_FINDINGS_FLAGGED`.
* **Fix Implemented:**
  Updated `src/batch_service.py` to directly execute TorchXRayVision DenseNet-121 auxiliary inference (`cls_svc.predict()`), cross-report consistency checks against whole-image sections, and multi-evidence research triage on every batch item.

---

### Defect 4: Triage Review Priority Inversion
* **Observed Symptom:**
  The baseline batch summary exhibited an inverted review priority distribution:
  - `no_priority_finding_flagged` (8 jobs): assigned `standard_review`.
  - `abnormality_flagged` (2 jobs with endotracheal tubes and chest tubes): assigned `routine`.
  In clinical / workflow semantics, "routine" is lower priority than "standard review", causing abnormal scans with life-support apparatus to receive lower review urgency than unremarkable scans.
* **Investigation & Exact Mechanism:**
  In `configs/triage_rules.yaml` and `src/triage_service.py`, rule `RULE_ABNORMAL_SUPPORT_APPARATUS_PRESENT` had `human_review_priority: routine`, while `RULE_NO_PRIORITY_FINDINGS_FLAGGED` had `human_review_priority: standard_review`. Furthermore, triage lacked states for multi-evidence conflicts, processing errors, or report validity defects.
* **Fix Implemented:**
  1. Refactored research triage into the 8 required evidence-aware states (`processing_error`, `incomplete_or_malformed_report`, `conflicting_ai_assessments`, `potential_abnormality_signal`, `support_device_signal`, `insufficient_or_indeterminate_evidence`, `unsupported_query`, `no_configured_signal`).
  2. Fixed review priority mappings: support apparatus and abnormality signals map to `routine` / `priority_review` with `requires_human_review: true`.
  3. Incomplete, malformed, or conflicting AI assessments map to `urgent` review.
  4. Added prominent disclaimers in UI and API exports stating that review priorities are experimental research flags, not validated clinical urgency recommendations.

---

## 3. Potential Causes Investigated But Not Confirmed

| Potential Cause | Investigation Conducted | Finding |
|---|---|---|
| **Cache Collision / Shared Job State** | Traced image SHA-256 and byte hashes across all 10 jobs. Checked report cache key generation in `report_service.py`. | **Refuted.** Each job had unique SHA-256 byte hashes (`91f56b...`, `e99239...`, `ce9f8b...`). Ollama requests were dispatched sequentially with distinct payloads. Similarities stemmed from model prompt bias, not caching. |
| **Worker Race Conditions / Concurrency Corruptions** | Audited `ThreadPoolExecutor` and threading locks in `src/batch_service.py`. | **Refuted.** Batch service strictly uses `max_workers=1` on Apple Silicon and guards mutable batch state with `threading.Lock()`. |
| **Silent Exception Catching Returning Normal Template** | Inspected all `try/except` blocks in `src/batch_service.py` and `src/report_service.py`. | **Refuted.** Parser exceptions raised explicit error dictionaries. The normal texts were emitted directly by MedGemma 4B when prompted with whole radiographs without regional crops. |

---

## 4. Exact Code Paths Changed

1. `src/report_service.py`:
   - Line 745: Added `\b` boundary to `one_side_patterns`.
   - Line 1101: Implemented `validate_generated_report()` with structural section checks, truncation detection, and style-scoped raw header leak checks.
   - Line 1395: Increased Stage 2 synthesis token limit from 280 to 380.
   - Line 1558: Integrated deterministic validation into whole-image report return schema.
2. `src/triage_service.py`:
   - Line 56: Updated `TriageEvaluationResult` dataclass to include `triage_state`, `matched_rule_ids`, `review_reasons`, `evidence_sources`.
   - Line 390: Implemented `evaluate_multi_evidence()` supporting the 8 research triage states.
3. `src/consistency_service.py`:
   - Line 369: Enabled auxiliary classifier discordance evaluation when whole-image report exists without regional phrase grounding.
4. `src/batch_service.py`:
   - Line 84: Added `report_validity`, `review_required`, `classifier_alerts`, `classifier_scores`, `consistency_result` to `BatchItem`.
   - Line 293: Added 4 new audit columns to `export_summary(format="csv")`.
   - Line 425: Integrated DenseNet-121 aux inference, cross-report consistency checks, and multi-evidence triage into worker pipeline.
5. `backend/main.py`:
   - Line 250: Updated `BatchItemResponse` Pydantic model with validation and classifier evidence fields.
6. `frontend/src/api.ts` & `frontend/src/App.tsx`:
   - Added TypeScript types and created `BatchItemStatusBadge` distinguishing `Completed — report validated`, `Completed — review required`, `Incomplete report`, `Report parsing failed`, and `Processing failed`.
   - Added classifier review alerts with explicit research disclaimer.
7. `tests/test_task16_batch_regression.py`:
   - Added 14 unit tests covering regex word boundaries, report validation, multi-evidence triage states, and batch CSV export integrity.
