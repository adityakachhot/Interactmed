# TASK 19B — Independent Audit of Task 19A Dataset Integrity, Benchmark Exclusions, Grounding Scores, and Report Metrics

**Author:** Senior ML & Medical Imaging Research Auditor  
**Date:** October 2026  
**System:** InteractMed-VLM  
**Audit Artifacts Directory:** `artifacts/dataset_reporting/task19b_audit/`

---

## 1. Audit Scope and Objectives

TASK 19B provides an independent, objective audit of the Task 19A implementation and deliverables. The audit evaluated:
1. The construction and patient counts of the MS-CXR held-out test split (`mscxr_held_out_test`).
2. The interpretation, projection breakdown, and filtering mechanics of Chest ImaGenome's `images_to_avoid.csv`.
3. The validity, methodology, and reproducibility of the 5-stage report-quality ablation metrics.
4. The accuracy and scope of reported latency figures.
5. Patient-level cohort overlap, Task 14 test set quarantine, and pretraining disclosures.
6. Runtime inference isolation and absence of ground-truth leakage.

---

## 2. Summary of Claim Classifications

Each major claim made in Task 19A has been audited and classified according to the specified 5-tier rubric:

| Task 19A Claim | Section | Audit Classification | Key Auditor Finding |
| :--- | :---: | :---: | :--- |
| **Zero New Downloads & Budget Compliance** | Sec 2 | **VERIFIED** | 0 bytes downloaded; 100% of 5.0 GB budget preserved. Local files in `~/Downloads` and `data/` were reused. |
| **MS-CXR Held-Out Test Construction** | Sec 3 | **PARTIALLY VERIFIED** | 163 rows originate 100% from official test split. But actual unique patients is 99 (not 148), and 53 official test rows were omitted due to missing local images. |
| **Zero Overlap with Task 14 Quarantined Test** | Sec 3 | **VERIFIED** | Exactly 0 patients in `mscxr_held_out_test` overlap with Task 14's locked 117-patient cohort. |
| **Train Overlap Pruning Attribution** | Sec 3 | **INCORRECTLY DESCRIBED** | Task 19A pruned patient `p15486932` (row 924) into `quarantined_task14_test`, but the report narrative misstated the patient as `p10625523`. |
| **Chest ImaGenome images_to_avoid Purpose** | Sec 4 | **INCORRECTLY DESCRIBED** | Claimed images were excluded for "lateral/oblique views or technical defects". In reality, `images_to_avoid.csv` isolates the 500 gold-benchmark patients; 72.8% of images are frontal (AP/PA). |
| **Chest ImaGenome Exclusion Counts** | Sec 4 | **INCORRECTLY DESCRIBED** | Claimed 423 unique excluded images PLUS 117 Task 14 images (540 total). In reality, the 117 Task 14 images are a subset of the 423 local matches, leaving 306 in `excluded_image_to_avoid`. |
| **TransVG Grounding Metrics (Multi-Pathology)** | Sec 5 | **VERIFIED** | Multi-Pathology continuation checkpoint achieves Mean IoU 0.5328, Recall@0.50 63.80%, Recall@0.75 34.97% across 163 held-out test cases. Exact match in `task19a_grounding_metrics.csv`. |
| **TransVG Grounding Metrics (RSNA 3ep)** | Sec 5 | **INCORRECTLY DESCRIBED** | Report table stated Recall@0.75 as 14.11% and latency as 178.69 ms, but artifact CSV records Recall@0.75 as 0.0184 (1.84%) and latency as 183.27 ms. |
| **Report Quality Ablation Metrics** | Sec 6 | **NOT REPRODUCED (SIMULATED)** | `scripts/evaluate_task19a_reports.py` did not execute MedGemma or TransVG live. It evaluated regexes against hardcoded template sentences and ground-truth boxes. |
| **End-to-End Latency (425–512 ms)** | Sec 6 | **INCORRECTLY DESCRIBED** | Latencies were synthetic formulas (`+120.0 ms` to `+132.0 ms`). Live MedGemma generation alone takes 4.2–7.8 seconds on Apple M3. |
| **Runtime Data Isolation & Triage** | Sec 7 | **VERIFIED** | Services do not access reference annotations during inference. Classifier alerts remain review triggers. Batch items are isolated. Research disclaimers are preserved. |

---

## 3. Section 1 Audit: MS-CXR Split Reconciliation

### 3.1 The 163-Pair, 148-Patient Discrepancy
- **Official MS-CXR v1.1.0 Test Split:**
  - 216 bounding box rows.
  - 176 unique `(dicom_id, label_text)` pairs.
  - 155 unique DICOM images / studies.
  - 128 unique patients (`subject_id`).
- **Task 19A Selected Test Set (`mscxr_held_out_test`):**
  - 163 bounding box rows.
  - 130 unique `(dicom_id, phrase)` pairs.
  - 116 unique DICOM images / studies.
  - 99 unique patients (`subject_id`).
- **Audit Findings:**
  1. **Source Integrity:** All 163 rows in `mscxr_held_out_test` originate strictly from rows where `split == 'test'` in `MS_CXR_Local_Alignment_v1.1.0.csv`. Zero rows were drawn from `train` or `val`.
  2. **Why 53 rows were dropped:** 53 rows (across 39 unique DICOM images and 29 patients) were dropped because those image files were not present in the local `files/p10/` directory when `data/processed/local_grounding_verified.csv` was built.
  3. **Patient Count Misreporting:** The author of Task 19A reported 148 unique patients, 148 unique studies, and 148 unique images in `task19a_dataset_overlap_audit.md` (Table in Section 2). The true count in `task19a_selected_images.csv` is **99 unique patients and 116 unique images**.
  4. **Patient Overlap:** Patient-level isolation is strictly maintained. Zero patients overlap between `mscxr_held_out_test` and `mscxr_train_clean` or `quarantined_task14_test`.

### 3.2 Corrected Split Inventory Table

| Partition Role | Annotation Rows | Unique DICOMs | Unique Studies | Unique Patients | True Audit Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `mscxr_held_out_test` | 163 | 116 | 116 | 99 | Filtered official test subset (local images available) |
| `mscxr_validation` | 145 | 104 | 104 | 89 | Filtered official validation subset |
| `mscxr_train_clean` | 733 | 531 | 531 | 444 | Filtered clean train subset (Task 14 overlap removed) |
| `quarantined_task14_test` | 3,043 | 117 | 117 | 117 | Locked Task 14 benchmark (116 Chest ImaGenome + 1 MS-CXR) |
| `excluded_image_to_avoid` | 7,812 | 306 | 306 | 298 | Non-Task 14 images matching `images_to_avoid.csv` |

---

## 4. Section 2 Audit: Chest ImaGenome Exclusions

### 4.1 Mischaracterization of `images_to_avoid.csv`
Task 19A claimed:
> *"Official exclusion metadata from Chest ImaGenome (images_to_avoid.csv, 3,821 images) is strictly enforced. Lateral projections, distorted crops, and technically inadequate radiographs are excluded from grounding evaluations."*

**Auditor Verification:**
1. Official file `silver_dataset/splits/images_to_avoid.csv` contains 3,821 images from 2,502 studies across 500 patients.
2. View breakdown:
   - **AP (Anteroposterior Frontal):** 2,059 (53.9%)
   - **PA (Posteroanterior Frontal):** 722 (18.9%)
   - **LATERAL:** 807 (21.1%)
   - **LL (Lateral Left):** 233 (6.1%)
   - **Total Frontal:** 2,781 (72.8%)
3. **Official Purpose:** In the Chest ImaGenome benchmark design, `images_to_avoid.csv` enumerates all images belonging to the **500 gold-standard patients**. Its purpose is to prevent leakage into silver-dataset training pipelines. It is **not** an image quality or lateral-only filter.
4. **Local Match Count:** 423 unique DICOM images in `local_grounding_verified.csv` match `images_to_avoid.csv`. Of these, 296 are AP and 114 are PA (96.9% frontal). Only 13 are LATERAL.
5. **Double-Counting:** In `task19a_selected_images.csv`, 117 of these images were assigned to `quarantined_task14_test`. The remaining 306 were assigned to `excluded_image_to_avoid`. Task 19A's narrative double-counted the 117 images as separate entities.

### 4.2 Anatomical Entities Audit
- In Chest ImaGenome `gold_dataset/`, 29 anatomical entities have radiologist-annotated bounding box coordinates.
- 7 entities in `src/evidence_schema.py` (e.g., `lungs bilaterally`, `lower lung zones`, `mediastinum`) represent composite anatomical groupings or scene-graph entities without standalone bounding boxes in the gold coordinates file.
- `evidence_schema.py` correctly uses these 36 entities for **free-text anatomical normalization**, but narrative documentation should clarify that only 29 possess direct radiologist bounding box ground truth.

---

## 5. Section 3 & 4 Audit: Report-Quality & Latency Metrics

### 5.1 Analysis of `scripts/evaluate_task19a_reports.py`
A thorough line-by-line inspection of `evaluate_task19a_reports.py` revealed a major methodological shortcut:

```python
# Lines 120-141:
if gt_is_pneumonia:
    baseline_text = "Focal airspace consolidation in the right lower lung. Lungs otherwise clear. Support lines in standard anatomical position."
elif gt_is_effusion:
    baseline_text = "Blunting of the left costophrenic angle consistent with pleural effusion. Heart size normal."
...
# Line 189:
reconciled_findings, audit = reconcile_evidence_with_grounding_and_classifier(
    whole_image_findings=f_comb,
    grounding_result={"success": True, "query": phrase, "pred_box_norm": [float(row["norm_x1"]), ...], "assessment_state": "Supported"},
    classifier_scores={gt_category: 0.75} if gt_category in EVAL_FINDINGS else None,
)
```

**Key Findings:**
1. **Mocked Generative VLM:** MedGemma 4B was not invoked. Instead, 6 hardcoded template sentences were assigned based on ground-truth category flags.
2. **Mocked Visual Grounding:** TransVG was not invoked. The radiologist ground-truth box `[row["norm_x1"], ...]` was passed directly into `reconcile_evidence_with_grounding_and_classifier()`.
3. **Synthetic Latencies:** Latencies were produced by adding constant offsets:
   - `lat_base = (perf_counter() - t0) * 1000.0 + 120.0`
   - `lat_comb = (perf_counter() - t0) * 1000.0 + 132.0`
4. **Conclusion on Report Metrics:** The reported metrics (Precision 0.954, Recall 0.916, F1 0.935, 0 unsupported assertions, 0.0% false normals, 425–512 ms latency) reflect the behavior of the rule-based schema on synthetic templates, **not real model reports on chest radiographs**.
5. **Real End-to-End Latency on Apple M3:**
   - MedGemma 4B text generation: **4,200–7,800 ms**
   - TransVG grounding inference: **175.58 ms**
   - TorchXRayVision DenseNet-121: **52.3 ms**
   - Pre/post-processing: **20.0 ms**
   - **True End-to-End Latency:** **4,500–8,100 ms** (not 425–512 ms).

---

## 6. Section 5 Audit: Evidence Fidelity and Runtime Isolation

Despite the offline report script defect, the core codebase and runtime services were verified to adhere strictly to safety guidelines:

1. **Classifier Alerts as Review Triggers:** `src/evidence_schema.py` (lines 409–420) adds secondary classifier alerts as `presence_status="uncertain"`, `uncertainty="high"`, and `review_required=True`. Alerts are never promoted to established diagnoses.
2. **Grounding Geometry vs. Presence:** TransVG bounding boxes populate `supporting_region` and do not alter presence probabilities.
3. **Absence of Ground-Truth Leakage in Services:** Neither `src/report_service.py` nor `src/batch_service.py` nor `backend/main.py` query reference manifests or ground-truth stores at runtime.
4. **Batch Isolation:** Each `BatchItem` maintains independent state. Regression tests verify zero cross-job leakage.
5. **Triage Safety:** Canonical triage strictly enforces research disclaimers and prevents green states from displaying as "Normal".

---

## 7. Deliverables Generated

All required Task 19B deliverables are saved under [`artifacts/dataset_reporting/task19b_audit/`](file:///Users/adityakachhot/interactmed/artifacts/dataset_reporting/task19b_audit/):
1. [`task19b_split_reconciliation.csv`](file:///Users/adityakachhot/interactmed/artifacts/dataset_reporting/task19b_audit/task19b_split_reconciliation.csv) — Exact counts for official MS-CXR, LGV, and Task 19A partitions.
2. [`task19b_exclusion_audit.csv`](file:///Users/adityakachhot/interactmed/artifacts/dataset_reporting/task19b_audit/task19b_exclusion_audit.csv) — Full projection breakdown and purpose audit of `images_to_avoid.csv`.
3. [`task19b_metric_reproduction.csv`](file:///Users/adityakachhot/interactmed/artifacts/dataset_reporting/task19b_audit/task19b_metric_reproduction.csv) — Reproduction and discrepancy accounting for grounding and report ablation.
4. [`task19b_cohort_overlap.csv`](file:///Users/adityakachhot/interactmed/artifacts/dataset_reporting/task19b_audit/task19b_cohort_overlap.csv) — Complete patient overlap matrix and Task 14 isolation proof.
5. [`task19b_latency_breakdown.csv`](file:///Users/adityakachhot/interactmed/artifacts/dataset_reporting/task19b_audit/task19b_latency_breakdown.csv) — Granular component timing vs synthetic claimed latencies.
6. [`task19b_inference_leakage_audit.md`](file:///Users/adityakachhot/interactmed/artifacts/dataset_reporting/task19b_audit/task19b_inference_leakage_audit.md) — Runtime data isolation and evidence fidelity verification.
7. [`task19b_final_audit.md`](file:///Users/adityakachhot/interactmed/artifacts/dataset_reporting/task19b_audit/task19b_final_audit.md) — Comprehensive audit synthesis report.
