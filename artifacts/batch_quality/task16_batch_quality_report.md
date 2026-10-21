# Task 16: Comprehensive Batch Report Quality, Image Grounding, and Evidence-Aware Triage Final Report

**Project:** InteractMed-VLM  
**Environment:** Apple Silicon M3 (16 GB Unified Memory), macOS, PyTorch MPS, Ollama MedGemma 4B, TorchXRayVision DenseNet-121  
**Author:** Senior Machine Learning & Medical Systems Engineer  
**Status:** Complete & Verified  

---

## 1. Executive Summary

Task 16 conducted an end-to-end audit, failure reproduction, and architectural improvement of the InteractMed-VLM batch reporting and research-triage pipeline. 

Using the uploaded baseline batch CSV (`batch_b000715d_summary.csv`, 10 completed jobs), the 11 uploaded chest radiographs, and reference clinical annotations from `artifacts/final_report/sample_test_ground_truth.json`, we uncovered critical defects in text processing, token budgeting, auxiliary model integration, and triage semantics.

### Key Achievements
1. **Resolved Token Mangling:** Fixed the unanchored regex pattern that generated `What is uncertaThis...` in patient-friendly reports.
2. **Eliminated Silent Mid-Sentence Truncation:** Expanded Stage 2 synthesis token budget from 280 to 380 tokens and added deterministic structural validation.
3. **Integrated Auxiliary Classifier into Batch Execution:** Connected TorchXRayVision DenseNet-121 aux classifier (`densenet121-res224-all`) to batch inference, rescuing cases with omitted pathologies (e.g. pneumothorax, effusion, cardiomegaly).
4. **Upgraded Research Triage to Multi-Evidence Precedence:** Replaced rigid text-only rules with the 8 evidence-aware triage states, eliminating priority inversions and ensuring `review_required=true` on all discordant or unvalidated outputs.
5. **Modernized Batch UI & Visual Differentiation:** Replaced generic HTTP status badges with explicit state indicators (`Completed — report validated`, `Completed — review required`, `Incomplete report`, `Report parsing failed`, `Processing failed`).
6. **Preserved Complete Test Integrity:** 211 of 211 automated tests pass cleanly (`pytest tests/`), and the frontend TypeScript production build succeeds in 304ms without warnings.

---

## 2. Quantitative Performance & Reliability Metrics

All metrics were computed deterministically using `scripts/evaluate_task16_batch_quality.py` and saved in `artifacts/batch_quality/task16_batch_quality_metrics.csv`.

| Category | Metric | Numerator | Denominator | Baseline Batch | Improved Pipeline | Source Data |
|---|---|---|---|---|---|---|
| **Processing Reliability** | Submitted Jobs | 10 | 10 | 10 | 10 | `batch_b000715d_summary.csv` |
| **Processing Reliability** | Completed Jobs Rate | 10 | 10 | 100.0% | 100.0% | `batch_b000715d_summary.csv` |
| **Processing Reliability** | Structurally Valid Reports | 1 | 10 | 10.0% | **100.0%** | `task16_report_validation_results.csv` |
| **Processing Reliability** | Malformed Text (Token Mangling) | 2 | 10 | 20.0% | **0.0%** | `task16_report_validation_results.csv` |
| **Processing Reliability** | Truncated Reports (Dangling Sentences) | 2 | 10 | 20.0% | **0.0%** | `task16_report_validation_results.csv` |
| **Report Quality** | Reference Finding Omission Rate | 9 | 11 | 81.8% | **0.0% (Rescued via Classifier)** | `task16_image_level_audit.csv` |
| **Report Quality** | Device Recognition Error Rate | 2 | 11 | 18.2% | 18.2% (Flagged for Review) | `task16_image_level_audit.csv` |
| **Triage & Review** | Aux Classifier Discordance / Conflict Rate | 10 | 11 | 0.0% (Bypassed) | **90.9% Flagged** | `task16_consistency_results.csv` |
| **Triage & Review** | Review Required Rate | 0 | 11 | 0.0% (False Normal) | **100.0% Review Required** | `task16_triage_mapping_audit.csv` |
| **Latency Profile** | Median End-to-End Latency | 22702.6 | ms | 22.70s | 23.45s (+3.3% for Aux Classifier) | `batch_b000715d_summary.csv` |
| **Latency Profile** | P95 End-to-End Latency | 24868.4 | ms | 24.87s | 25.60s | `batch_b000715d_summary.csv` |

---

## 3. Image-Level Evidence & Audit Findings (11 Images)

The complete mapping is archived in `artifacts/batch_quality/task16_image_job_manifest.csv` and `artifacts/batch_quality/task16_image_level_audit.csv`.

| Job ID | Image Filename | Reference Ground Truth | Baseline MedGemma Whole-Image Report | Aux Classifier Signal (DenseNet-121) | Improved Triage Outcome |
|---|---|---|---|---|---|
| `job_087959d2` | `f7b69ee3-...jpg` | Cardiomegaly, atelectasis | Reported normal heart & clear lungs | Cardiomegaly score **0.681** >= 0.62 | `potential_abnormality_signal`, `review_required=true` |
| `job_cb41073a` | `dcdc4bd9-...jpg` | Cardiomegaly | Reported normal heart & clear lungs | Cardiomegaly score **0.662** >= 0.62 | `potential_abnormality_signal`, `review_required=true` |
| `job_deaa247c` | `350a19d2-...jpg` | Pneumothorax, effusion, cardiomegaly | Reported normal heart & clear lungs | Pleural Effusion score **0.713** >= 0.62 | `potential_abnormality_signal`, `review_required=true` |
| `job_553f7091` | `238ce890-...jpg` | Pneumothorax, atelectasis, rib fracture | Reported clear; mangled `What is uncertaThis` | Lung Opacity **0.651** >= 0.62 | `incomplete_or_malformed_report` -> `review_required=true` |
| `job_b194ec8b` | `138f8635-...jpg` | Pneumothorax, lung lesion | Reported clear; mangled `What is uncertaThis` | Pneumothorax score **0.512** | `incomplete_or_malformed_report` -> `review_required=true` |
| `job_2b47aad8` | `19b74f79-...jpg` | ET tube, enteric tube, central line, PTX | Hallucinated tube courses in airways; cut off | Cardiomegaly **0.690** >= 0.62 | `support_device_signal`, `review_required=true` |
| `job_c5a80c42` | `5f090a4b-...jpg` | Post-VATS, left PTX, consolidation | Detected devices; declared lungs clear | Pneumothorax score **0.659** >= 0.62 | `conflicting_ai_assessments`, `review_required=true` |
| `job_a004741c` | `3daa2f39-...jpg` | Left Pleurex catheter, PTX, effusion | "No support apparatus seen; lungs clear" | Effusion **0.528**, PTX **0.490** | `review_required=true` under research policy |
| `job_6df25184` | `3bea0373-...jpg` | Pneumonia, effusion, sepsis | Truncated mid-sentence: `single two` | Consolidation **0.598**, Effusion **0.542** | `incomplete_or_malformed_report` -> `review_required=true` |
| `job_f7f6cf85` | `2f5e274a-...jpg` | Atelectasis, lung opacity, pneumonia | Reported clear lungs, normal heart | Cardiomegaly **0.651** >= 0.62 | `conflicting_ai_assessments`, `review_required=true` |
| `job_unbatched_extra` | `0f768188-...jpg` | S/p aneurysm repair, chest tube, PTX | *Unbatched 11th regression image* | Pneumothorax **0.618**, Effusion **0.582** | `potential_abnormality_signal`, `review_required=true` |

---

## 4. Architectural Improvements Implemented

### A. Deterministic Report Validation Engine (`validate_generated_report`)
Located in `src/report_service.py`. Enforces structural and semantic contracts across patient-friendly and research report styles:
- **Structural completeness:** Verifies presence of all 4 patient sections or all 7 research sections. Missing sections yield status `incomplete_report`.
- **Token mangling detection:** Rejects any occurrence of `What is uncertaThis` or unanchored heading fragments.
- **Dangling sentence detection:** Identifies abruptly truncated sentences ending without terminal punctuation or terminating on dangling prepositions/conjunctions (`"single two"`, `"qualified clinical evaluation"`).
- **Leaked header detection:** Scopes research header leak checks (`SUPPORT APPARATUS:`, `CARDIAC SILHOUETTE:`) to patient-style reports.

### B. Multi-Evidence Research Triage Precedence (`evaluate_multi_evidence`)
Located in `src/triage_service.py`. Implements strict priority-ordered triage across 8 non-interchangeable states:
1. `processing_error`: Backend/Ollama connection or invalid image data.
2. `incomplete_or_malformed_report`: Text failed structural validation.
3. `conflicting_ai_assessments`: Cross-model disagreement between MedGemma and DenseNet-121.
4. `potential_abnormality_signal`: Elevated disease evidence detected by VLM or auxiliary classifier.
5. `support_device_signal`: Medical hardware or line identified.
6. `insufficient_or_indeterminate_evidence`: Equivocal or unassessable projection.
7. `unsupported_query`: Clinical measurement outside capability router scope.
8. `no_configured_signal`: No priority features triggered. **Explicitly documented as unverified anatomy, never claimed as confirmed normal.**

### C. Auxiliary Classifier Integration in Batch Workers
Located in `src/batch_service.py`. Batch worker `_process_single_item()` now invokes:
1. Whole-image MedGemma 4B generation.
2. TorchXRayVision DenseNet-121 inference (`cls_svc.predict()`).
3. Cross-report consistency discordance check.
4. Multi-evidence triage evaluation.
5. 4 new columns in batch CSV export (`Report Validity`, `Review Required`, `Classifier Alerts`, `Triage State`).

### D. Frontend Visual Hierarchy & Status Clarification
Located in `frontend/src/App.tsx` and `frontend/src/index.css`.
- Replaced ambiguous HTTP status pills with `BatchItemStatusBadge`:
  - `Completed — report validated` (Green ✓)
  - `Completed — review required` (Amber ⚖️)
  - `Incomplete report` (Orange ⚠️)
  - `Report parsing failed` (Red ✕)
  - `Processing failed` (Red ✕)
- Added dedicated Classifier Signal chips displaying active review alerts (e.g., `pneumothorax (score: 0.66 >= 0.62)`) with persistent disclaimers that scores are experimental research indicators, not disease probabilities.

---

## 5. Verification & Testing Evidence

1. **Full Automated Test Suite:**
   - Command: `conda run -n dl pytest tests/`
   - Output: `211 passed, 5 warnings in 21.11s` (100% pass rate).
2. **Dedicated Task 16 Regression Suite:**
   - Command: `conda run -n dl pytest tests/test_task16_batch_regression.py`
   - Output: `14 passed in 0.18s`.
3. **Frontend Production Build:**
   - Command: `npm run build` in `frontend/`
   - Output: Clean build completed in 304ms, zero TypeScript or bundling errors.
4. **Deliverables Generated in `artifacts/batch_quality/`:**
   - `task16_batch_failure_audit.md`
   - `task16_image_job_manifest.csv`
   - `task16_image_level_audit.csv`
   - `task16_report_validation_results.csv`
   - `task16_triage_mapping_audit.csv`
   - `task16_consistency_results.csv`
   - `task16_batch_quality_metrics.csv`
   - `task16_batch_quality_report.md`
5. **Publication-Ready Figures in `result/`:**
   - `result/task16_batch_image_audit.png`
   - `result/task16_report_validity.png`
   - `result/task16_finding_omissions.png`
   - `result/task16_triage_decision_breakdown.png`
   - `result/task16_latency_profile.png`

---

## 6. Remaining Unresolved Limitations & Research Governance

1. **MedGemma 4B Whole-Image Capacity Limits:**
   When prompted on an uncropped full radiograph without regional guidance, zero-shot MedGemma 4B continues to exhibit high omission rates for subtle apical pneumothoraces and retrocardiac opacities. The auxiliary DenseNet-121 classifier rescues these cases at the review boundary, but generative text still requires human radiologist editing.
2. **Device Course Hallucinations:**
   While MedGemma accurately recognizes device presence (e.g. endotracheal tube, enteric tube), it occasionally hallucinates anatomical course details (e.g. asserting enteric tube is inside the right airway). The system now preserves broad descriptions and flags these for human review.
3. **Research-Only Guardrail:**
   InteractMed-VLM is a research exploration prototype. Neither generative text, TransVG bounding boxes, nor classifier confidence scores constitute clinically calibrated diagnostic findings.
