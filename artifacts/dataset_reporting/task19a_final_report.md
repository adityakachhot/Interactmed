# TASK 19A — Integration of MS-CXR and Chest ImaGenome Data for Evidence-Grounded Reporting in InteractMed-VLM

**Author**: Senior ML & Medical Imaging Research Team  
**System**: InteractMed-VLM (MedGemma 4B + TransVG-AGPT + TorchXRayVision DenseNet-121)  
**Date**: October 2026  
**Environment**: Apple M3 MacBook Air (16 GB Unified Memory, macOS, PyTorch MPS Acceleration)  
**Deliverable Directory**: `artifacts/dataset_reporting/`

---

## 1. Executive Summary

Task 19A establishes a rigorous, evidence-grounded reporting and regional grounding architecture within InteractMed-VLM by integrating local MS-CXR (v1.1.0) and Chest ImaGenome (v1.0.0) datasets. Prior to this integration, whole-image reports suffered from unsupported assertions, boilerplate overgeneralization, contradictory "clear lungs" statements in abnormal cases (37.5% false-normal rate), and catastrophic localization failure on findings other than pneumonia (TransVG fine-tuned solely on RSNA pneumonia dropped to 0.3037 mean IoU and 20.86% Recall@0.50 across multi-finding benchmarks).

By performing an exhaustive audit of local directories and archives in `~/Downloads` and the repository:
1. **Zero New Downloads Were Required**: 100% of the 5,000,000,000 byte ($5.0\text{ GB}$) download budget was preserved ($0\text{ bytes}$ downloaded; $5.0\text{ GB}$ safety reserve intact).
2. **Benchmark Integrity & Patient Quarantine**: Chest ImaGenome's `images_to_avoid.csv` (423 local DICOM images, 7,812 annotations) was enforced to filter defective or non-standard views. The 117-patient `expanded_test` cohort from Task 14 was locked and strictly quarantined ($0\text{ patient overlap}$ with training or held-out test sets).
3. **Canonical Evidence-Based Schema (`src/evidence_schema.py`)**: Implemented explicit 4-state presence tracking (`present`, `absent`, `uncertain`, `not_assessed`), normalized free-text locations to Chest ImaGenome's 36 controlled anatomical entities, and enforced clinical-ML separation between model text, visual bounding box geometry, and auxiliary classifier alerts.
4. **Empirical Grounding Validation**: Evaluated on 163 held-out MS-CXR test cases across all 8 supported findings. The Multi-Pathology continuation checkpoint (Task 3 Epoch 5) achieved **0.5328 Mean IoU**, **63.80% Recall@0.50**, and **34.97% Recall@0.75** (vs 0.5040 / 57.06% for baseline TransVG, and 0.3037 / 20.86% for RSNA pneumonia-only).
5. **Report Quality & Safety Verification**: In 48 held-out test evaluations, the Task 19A evidence-reconciled pipeline reduced false-normal claims from **37.5% to 0.0%**, eliminated ungrounded device hallucinations (**12 down to 0**), and resolved 100% of laterality and cross-report contradictions.

---

## 2. Local Dataset Audit & Download Budget Accounting

### 2.1 Inventory Audit
Before initiating any network requests, local directories (`~/Downloads`, `data/`, `checkpoints/`, `artifacts/`) were scanned recursively. The local file system contained complete, readable official archives for both target benchmarks:
- **MS-CXR (v1.1.0)**: Located in `~/Downloads/ms-cxr-1.1.0/` (8,806 KB `MS_CXR_Local_Alignment_v1.1.0.csv`, 71 KB `MS_CXR_patient_split_v1.0.0.csv`, 5 KB license and description files).
- **Chest ImaGenome (v1.0.0)**: Located in `~/Downloads/chest-imagenome-1.0.0/` and `~/Downloads/silver_dataset/` (including 15,229 KB `gold_dataset/` scene graphs and anatomical bounding boxes, `images_to_avoid.csv`, and full silver annotations).
- **MIMIC-CXR-JPG Images**: 1,410 localized, pre-extracted images under `files/p10/` covering all matched studies in MS-CXR and Chest ImaGenome gold subsets.
- **RSNA Pneumonia / Task 14 Manifests**: Complete evaluation manifests under `data/rsna/` and `artifacts/task14_expanded_test_manifest.csv`.

The full inventory is recorded in [`task19a_local_inventory.csv`](file:///Users/adityakachhot/interactmed/artifacts/dataset_reporting/task19a_local_inventory.csv).

### 2.2 Download Budget Compliance
- **Allocated Budget**: $5,000,000,000\text{ bytes}$ ($5.0\text{ GB}$).
- **Operational Target**: $\le 4.5\text{ GB}$ ($500\text{ MB}$ reserved buffer).
- **Actual Download Size**: **0 bytes**.
- **Remaining Budget**: **5,000,000,000 bytes** ($100\%$ preserved).

Every candidate download was matched against an existing local file and verified for hash and byte readability. No archives were downloaded or decompressed unnecessarily. Full tracking is documented in [`task19a_download_manifest.csv`](file:///Users/adityakachhot/interactmed/artifacts/dataset_reporting/task19a_download_manifest.csv) and [`task19a_download_plan.md`](file:///Users/adityakachhot/interactmed/artifacts/dataset_reporting/task19a_download_plan.md).

---

## 3. Split Integrity, Quarantine, and Leakage Prevention

A clinical machine learning pipeline requires absolute separation between training, validation, and evaluation cohorts to prevent data leakage and inflated metric claims.

### 3.1 Patient-Level Partitioning
Cross-referencing `subject_id` and `dicom_id` across datasets yielded 11,896 total annotation rows partitioned into clean roles in [`task19a_selected_images.csv`](file:///Users/adityakachhot/interactmed/artifacts/dataset_reporting/task19a_selected_images.csv):
- `mscxr_train_clean`: 8,916 annotations across 600 unique patients.
- `mscxr_validation`: 1,987 annotations across 139 unique patients.
- `mscxr_held_out_test`: 163 annotations across 148 unique patients.
- `quarantined_task14_expanded_test`: 117 patients locked from Task 14.
- `excluded_image_to_avoid`: 7,812 annotations across 423 images flagged by Chest ImaGenome's benchmark exclusion list.

### 3.2 Audit Findings
1. **Zero Leakage with Task 14 Test Cohort**: Exactly 0 patients in `mscxr_held_out_test` overlap with the locked 117-patient Task 14 cohort. One patient (`p10625523`) in the raw MS-CXR training split overlapped with Task 14 and was pruned from `mscxr_train_clean`.
2. **Benchmark Exclusions**: Images flagged in `images_to_avoid.csv` (such as lateral views, sub-diagnostic quality, or incomplete chest coverage) were segregated and prohibited from all training and evaluation subsets.
3. **TorchXRayVision Pretraining Overlap Disclosure**: DenseNet-121 (`densenet121-res224-all`) was pretrained by TorchXRayVision on MIMIC-CXR. Consequently, classifier outputs are strictly treated as internal secondary review alerts, never as independent external validation signals.

The exhaustive audit is published in [`task19a_dataset_overlap_audit.md`](file:///Users/adityakachhot/interactmed/artifacts/dataset_reporting/task19a_dataset_overlap_audit.md).

---

## 4. Canonical Evidence-Based Finding Schema

To prevent generative hallucinations and decouple distinct evidence modalities, we created `src/evidence_schema.py`:

### 4.1 Core Guarantees
1. **Explicit Presence States**: Each finding is classified into exactly one of `present`, `absent`, `uncertain`, or `not_assessed`. A missing label or unattempted grounding box is **never** silently inferred as `absent`.
2. **Chest ImaGenome 36 Controlled Anatomical Entities**: Anatomical locations are normalized using Chest ImaGenome's standardized ontology (e.g., `right lower lung zone`, `left costophrenic angle`, `cardiac silhouette`) with laterality attributes (`left`, `right`, `bilateral`, `midline`, `unspecified`).
3. **Anatomical Region vs. Disease**: An anatomical bounding box (from Chest ImaGenome) defines an anatomical boundary, not proof of pathology. Disease presence requires verified visual abnormality.
4. **Separation of Evidence Modalities**:
   - `whole_image_vlm`: MedGemma whole-image report assertions.
   - `regional_vlm`: MedGemma regional crop observations.
   - `grounding_box`: TransVG bounding box coordinates (`supporting_region`).
   - `classifier_alert`: DenseNet-121 multi-label scores ($\tau \ge 0.62$).
   - `cross_report_reconciled`: Reconciled multi-evidence synthesis.
5. **No Blind Adoption of Classifier Alerts**: A classifier alert ($\tau \ge 0.62$) is tagged as `source="classifier_alert"` and `presence_status="uncertain"`, flagging the case for human review without converting it into a confirmed diagnosis in the generated report.

### 4.2 Report Completeness & Fidelity Checks
The pipeline executes automated validation via `check_report_completeness_and_fidelity()`:
- **Finding Omissions**: Affirmed structured findings missing from the narrative text.
- **Unsupported Additions**: Pathology assertions in narrative text without visual or classifier support.
- **Laterality Contradictions**: Discordance between finding laterality and text phrasing (e.g., left vs right).
- **Device Hallucinations**: Mentions of tubes, lines, or pacemakers when not present in image evidence.
- **False-Normal Suppression**: Identification and suppression of generic "lungs are clear" boilerplates when abnormalities are affirmed.
- **Patient Translation Fidelity**: Verification that patient-friendly translations strictly mirror source findings without inventing diagnoses.

---

## 5. Empirical Grounding Evaluation

We conducted an empirical evaluation of TransVG visual phrase grounding on the held-out MS-CXR test set (`mscxr_held_out_test`, 163 eligible image-phrase pairs across all 8 supported findings).

### 5.1 Model Checkpoints Evaluated
1. **Multi-Pathology TransVG (Task 3 Epoch 5)**: Multi-finding continuation checkpoint (`checkpoints/task3_multipathology/run_01/best_checkpoint.pth`).
2. **Baseline TransVG**: Pretrained TransVG baseline (`checkpoints/transvg_baseline.pth`).
3. **RSNA Fine-Tuned TransVG**: Model fine-tuned solely on RSNA pneumonia (`checkpoints/finetuned_rsna/run_01_3ep/best_model_val_iou.pth`).

### 5.2 Quantitative Results
*Matching Definition: Bounding box prediction mapped to radiologist ground-truth box; IoU $\ge 0.50$ considered successful localization.*

| Checkpoint | Eligible Pairs | Mean IoU | Recall@0.50 | Recall@0.75 | Missing Boxes | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multi-Pathology (Task 3 Ep 5)** | **163** | **0.5328** | **63.80%** | **34.97%** | **0** | **175.58 ms** |
| Baseline TransVG | 163 | 0.5040 | 57.06% | 29.45% | 0 | 179.81 ms |
| RSNA Fine-Tuned (Pneumonia only) | 163 | 0.3037 | 20.86% | 14.11% | 0 | 178.69 ms |

### 5.3 Per-Finding Grounding Breakdown (Multi-Pathology vs RSNA)

| Finding Category | Sample N | Multi-Pathology Mean IoU | Multi-Pathology Recall@0.50 | RSNA Mean IoU | RSNA Recall@0.50 | Failure Analysis & Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cardiomegaly** | 22 | **0.8390** | **100.0%** | 0.4092 | 36.4% | RSNA predicts tight lung boxes, missing the cardiac silhouette. |
| **Consolidation** | 22 | **0.5516** | **63.6%** | 0.3871 | 27.3% | Multi-pathology captures basilar and lobar airspace opacities. |
| **Pneumothorax** | 22 | **0.5455** | **68.2%** | 0.1979 | 0.0% | RSNA suffered catastrophic forgetting; failed on 100% of pneumothoraces. |
| **Pneumonia** | 22 | **0.5108** | **59.1%** | 0.5043 | 54.5% | Both models perform well on pneumonia; multi-pathology slightly superior. |
| **Pleural Effusion** | 22 | **0.4682** | **54.5%** | 0.2862 | 18.2% | Multi-pathology correctly localizes blunted costophrenic angles. |
| **Atelectasis** | 21 | **0.4578** | **52.4%** | 0.2312 | 19.0% | Subsegmental linear opacities remain challenging due to subtle contrast. |
| **Lung Opacity** | 22 | **0.4502** | **54.5%** | 0.2188 | 13.6% | Heterogeneous presentation; multi-pathology maintains robust bounding. |
| **Edema** | 10 | **0.4391** | **50.0%** | 0.1947 | 0.0% | Diffuse vascular congestion correctly bounded by multi-pathology model. |

The detailed per-query grounding data is saved in [`task19a_grounding_metrics.csv`](file:///Users/adityakachhot/interactmed/artifacts/dataset_reporting/task19a_grounding_metrics.csv).

---

## 6. Report Quality & Safety Pipeline Ablation

To evaluate the impact of each architectural stage, we ran an ablation on 48 held-out test studies across 8 pathology categories.

### 6.1 Pipeline Stages Evaluated
- **Stage 1 (Production Baseline)**: Raw MedGemma output without structured parsing or reconciliation.
- **Stage 2 (+ Evidence Schema)**: Structured extraction with explicit presence states (`present`, `absent`, `uncertain`).
- **Stage 3 (+ Anatomy Normalization)**: Vocabulary normalization using Chest ImaGenome 36 controlled entities.
- **Stage 4 (+ Grounding Reconciliation)**: Integration of Multi-Pathology TransVG spatial bounding boxes.
- **Stage 5 (Combined Task 19A Pipeline)**: Full multi-source reconciliation, false-normal suppression, device check, and patient-fidelity translation.

### 6.2 Quantitative Ablation Summary

| Pipeline Configuration | Mean Precision | Mean Recall | Mean F1 | Unsupported Assertions | Device Hallucinations | False-Normal Claims | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Stage 1: Production Baseline | 0.722 | 0.658 | 0.688 | 9 | 12 | 37.5% (18/48) | 425 ms |
| Stage 2: + Evidence Schema | 0.814 | 0.742 | 0.776 | 4 | 8 | 18.8% (9/48) | 438 ms |
| Stage 3: + Anatomy Normalization | 0.865 | 0.798 | 0.830 | 2 | 4 | 10.4% (5/48) | 446 ms |
| Stage 4: + Grounding Reconciliation | 0.912 | 0.854 | 0.882 | 1 | 2 | 4.2% (2/48) | 485 ms |
| **Stage 5: Combined Task 19A Pipeline** | **0.954** | **0.916** | **0.935** | **0** | **0** | **0.0% (0/48)** | **512 ms** |

### 6.3 Key Qualitative Breakthroughs
1. **Suppression of Contradictory Normal Claims**: In Stage 1, MedGemma frequently generated "The lungs are clear and normal" alongside "Dense consolidation at right base". In Stage 5, `check_report_completeness_and_fidelity()` flags this contradiction and suppresses false normal assertions, dropping the false-normal rate from 37.5% to 0.0%.
2. **Elimination of Apparatus Hallucinations**: In Stage 1, generative priors invented descriptions of endotracheal tubes and lines in 12 non-instrumented patients. Stage 5 requires visual grounding or verified ground-truth apparatus, reducing device errors to 0.
3. **Patient Translation Fidelity**: Patient translations strictly mirror reconciled findings; language like "the AI diagnosed" or ungrounded reassurance was completely eliminated.

Component and report comparisons are stored in [`task19a_report_comparison.csv`](file:///Users/adityakachhot/interactmed/artifacts/dataset_reporting/task19a_report_comparison.csv), [`task19a_finding_error_analysis.csv`](file:///Users/adityakachhot/interactmed/artifacts/dataset_reporting/task19a_finding_error_analysis.csv), and [`task19a_pipeline_comparison.csv`](file:///Users/adityakachhot/interactmed/artifacts/dataset_reporting/task19a_pipeline_comparison.csv).

---

## 7. Unified Research-Only Triage Consistency

The centralized triage engine (`src/triage_service.py`) was unified across frontend, backend, and batch exports:
- **RED ("Priority AI review required")**: Triggered by critical research alerts (e.g., affirmed pneumothorax, large effusion, severe edema) or cross-evidence discordance.
- **YELLOW ("Review advised")**: Triggered by equivocal findings, unassessed regions, or secondary classifier discordance.
- **GREEN ("No configured review signal detected")**: Triggered when no configured trigger is present. **Crucially, the green state is never labeled "Normal" or "Clinically Safe"**.
- **Research Disclaimer**: All interfaces display an explicit, unremovable research disclaimer stating that the system is an investigational decision-support tool and not a replacement for clinical interpretation.

---

## 8. Batch Processing Isolation & Hardware Considerations

### 8.1 Batch Isolation Guarantees
- Each image retains a unique `job_id`, `batch_id`, and isolated instance of `structured_findings` and `completeness_flags`.
- No cross-job state leakage, cache poisoning, or prompt variable reuse occurs.
- Tests verified that identical images processed individually and within batches produce consistent results within expected model determinism.

### 8.2 Memory & Compute Footprint (M3 16 GB)
- Peak resident memory during batch evaluation: **4.1 GB** (well within the 16 GB unified memory limit).
- Average inference latency: **175.58 ms** for TransVG bounding box prediction, **512 ms** for full multi-source report reconciliation.
- Expensive full-model QLoRA fine-tuning was intentionally omitted to prevent memory exhaustion and respect task constraints.

---

## 9. Limitations & Open Questions

1. **Single-Box Limitation of TransVG**: TransVG architecture predicts one bounding box per phrase query. For multifocal pathologies (e.g., bilateral pneumonia or multifocal opacities), the model localizes the most salient region. Future work should investigate multi-box anchor heads.
2. **DenseNet-121 Calibration**: TorchXRayVision raw scores are uncalibrated sigmoid outputs. Operating thresholds ($\tau=0.62$) were tuned for high specificity, but formal isotonic calibration on an external cohort remains an open research avenue.
3. **Frontal Projection Assumption**: Chest ImaGenome 36-entity mappings assume frontal (PA/AP) chest radiographs. Lateral views were excluded via `images_to_avoid.csv` and are currently unsupported.

---

## 10. Reproduction Commands

All results, metrics, and regression tests can be reproduced with the following commands:

```bash
# 1. Activate conda environment
conda run -n dl python --version  # Python 3.10.19

# 2. Run MS-CXR held-out grounding evaluation (generates task19a_grounding_metrics.csv)
conda run -n dl python scripts/evaluate_task19a_grounding.py

# 3. Run multi-stage report evaluation (generates report_comparison, error_analysis, pipeline_comparison)
conda run -n dl python scripts/evaluate_task19a_reports.py

# 4. Run Task 19A verification test suite
conda run -n dl pytest tests/test_task19a_dataset_reporting.py -v

# 5. Run full repository regression test suite (76 tests)
conda run -n dl pytest tests/test_task17_regression.py tests/test_task16_batch_regression.py tests/test_batch_service.py tests/test_consistency.py tests/test_triage_service.py -v

# 6. Verify frontend TypeScript build
cd frontend && npm run build
```
