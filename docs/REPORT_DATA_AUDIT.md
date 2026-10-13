# InteractMed-VLM: Report Data & Linkage Audit

**Audit Date:** October 12, 2026  
**Auditor:** Senior Medical Vision-Language Model Researcher  
**Application:** InteractMed-VLM Whole-Image Report Generation  
**Model Backbone:** MedGemma 4B via local Ollama inference (`keep_alive: "30m"`)  
**Evaluation Cohort:** 50 Matched Frontal Radiographs (25 Development, 25 Held-Out Evaluation)

---

## 1. Executive Summary & Purpose

The Batch Reports module of InteractMed-VLM accepts multiple whole chest radiographs and generates an independent, structured research report for each image. Preliminary evaluations noted a persistent failure mode: generated reports across visibly distinct pathologies collapsed into repetitive, generic normal boilerplate ("lungs clear", "heart size within normal limits", "no acute abnormalities"), failing to describe cardiomegaly, parenchymal opacities, pleural effusions, and support apparatus.

This audit establishes a **ground-truth-linked, leakage-free evaluation foundation**:
1. Verifies that all inference inputs receive genuine, uncorrupted, un-cached full radiographs.
2. Discovers and catalogs local multimodal datasets across Chest-ImaGenome, MS-CXR, and MIMIC-CXR.
3. Constructs an auditable mapping between local radiographs and board-certified radiologist reference reports.
4. Generates a patient-stratified cohort of 50 verified cases (`artifacts/report_evaluation/image_report_manifest.csv`) with zero patient leakage between development and test subsets.
5. Protects Protected Health Information (PHI) by establishing safe hashed identifiers and storing clinical text in a local store.

---

## 2. Dataset Discovery & Schema Inventory

| Dataset Asset | Local File / Directory Path | Schema / Format | Total Holdings | Locally Evaluated Subset |
| :--- | :--- | :--- | :--- | :--- |
| **MIMIC-CXR-JPG Image Store** | `/Users/adityakachhot/Downloads/data/mimic-cxr/files` | `p{prefix}/p{subj}/s{study}/{dicom}.jpg` | 1,174 `.jpg` | 1,174 local files verified |
| **Chest-ImaGenome Gold Sentences** | `/Users/adityakachhot/Downloads/data/chest-imagenome/gold_dataset/gold_all_sentences_500pts_1000studies.txt` | TSV: `patient_id`, `study_id`, `image_id`, `sent_loc`, `section`, `sentence` | 10,769 sentence rows (1,000 studies) | **417 studies** with verified local `.jpg` images |
| **Chest-ImaGenome Gold Object & Attributes** | `/Users/adityakachhot/Downloads/data/chest-imagenome/gold_dataset/gold_object_attribute_with_coordinates.txt` | TSV: `patient_id`, `study_id`, `image_id`, `bbox`, `label_name`, `context`, `coord_original` | 25,989 bounding box / attribute records | 12,936 local attribute annotations |
| **Chest-ImaGenome View Records** | `/Users/adityakachhot/Downloads/chest-imagenome-dataset-1.0.0/utils/cxr-record-list_view.csv` | CSV: `subject_id`, `study_id`, `dicom_id`, `path`, `ViewPosition` | 243,334 records | 417 local gold images classified (264 AP, 153 PA) |
| **MS-CXR Local Alignment** | `/Users/adityakachhot/Downloads/data/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.csv` | CSV: `dicom_id`, `category_name`, `label_text`, `x,y,w,h`, `image_width,image_height` | 1,448 annotations (1,047 DICOMs) | 764 images present locally |
| **Unified Master Grounding Manifest** | `data/processed/unified_grounding_master.csv` | CSV: `dicom_id`, `subject_id`, `study_id`, `bbox`, `phrase`, `split`, `source` | 17,135 rows | 13,992 local rows |

---

## 3. Trusted Image-to-Report Linkage Architecture

### 3.1 Resolving Identity & Discarding Upload Filenames
Uploaded filenames in web applications can be ephemeral (e.g., `blob_1`, `cxr_upload.png`, or random client UUIDs). To guarantee ground truth integrity:
* Linkage is established strictly via deterministic image hashes (`SHA-256`) and original dataset identifiers (`dicom_id`, `study_id`, `subject_id`).
* Every test case in the evaluation manifest is matched to its definitive MIMIC-CXR DICOM metadata.

### 3.2 Unit of Comparison: Study-Level vs Image-Level
* **Radiologist Reference Reports:** Written at the study level (`study_id`). A diagnostic study may consist of a single portable AP radiograph or a two-view PA and Lateral examination.
* **Model Inference Target:** Full-image single-view frontal chest radiograph (AP or PA).
* **Audit Rule:** All 417 candidate images were checked against `cxr-record-list_view.csv`. Exactly 100% of the candidate images evaluated are frontal projections (264 AP, 153 PA; zero lateral radiographs).
* **Context Accounting:** When evaluating a frontal radiograph against a two-view study report, findings visible only on the lateral projection (e.g., retrocardiac opacity, small posterior pleural effusion) are not penalized as model hallucinations if absent, but evaluated with appropriate uncertainty.

### 3.3 Role of Chest-ImaGenome Gold Bounding Boxes
* Chest-ImaGenome gold bounding boxes provide **anatomical localization ground truth** (e.g., bounding boxes for cardiac silhouette, costophrenic angles, tracheobronchial tree).
* Gold bounding boxes are **not complete radiology reports**. They serve to verify anatomical coverage and spatial validity.
* The reference radiology prose is drawn directly from `gold_all_sentences_500pts_1000studies.txt`, which preserves the actual clinical reports written by attending radiologists and fellows at Beth Israel Deaconess Medical Center.

---

## 4. Cohort Selection & Patient-Grouped Splitting

To enable reproducible prompt experimentation without evaluation contamination, a balanced cohort of **50 matched cases** was constructed from the 417 verified local candidates:

* **Development Subset (`split = "dev"`):** 25 cases reserved for prompt ablation, systematic inspection, and error diagnosis.
* **Held-Out Test Subset (`split = "held_out_test"`):** 25 cases held strictly untouched until final prompt comparison.
* **Patient Disjointness:** **Zero patient overlap** exists between dev and test (`len(patient_dev ∩ patient_test) == 0`). No individual patient contributed images to both splits.

### Category Stratification:
Both subsets are identically balanced across seven primary clinical categories:

| Clinical Category | Development Cases | Held-Out Test Cases | Total Cases | Primary Radiographic Feature |
| :--- | :---: | :---: | :---: | :--- |
| **Normal / Clear** | 6 | 6 | 12 | Unremarkable lungs, normal cardiac contour, sharp sulci |
| **Pneumonia / Consolidation** | 4 | 4 | 8 | Focal consolidation, airspace opacity, air bronchograms |
| **Pulmonary Edema** | 4 | 4 | 8 | Vascular congestion, perihilar haziness, cephalization |
| **Cardiomegaly** | 3 | 3 | 6 | Cardiothoracic ratio > 0.5, enlarged cardiac silhouette |
| **Pleural Effusion** | 3 | 3 | 6 | Blunting of costophrenic angles, meniscus sign |
| **Atelectasis** | 3 | 3 | 6 | Linear/patchy volume loss, plate-like discoid opacities |
| **Support Devices** | 2 | 2 | 4 | Endotracheal tubes, nasogastric/enteric tubes, lines |
| **Total** | **25** | **25** | **50** | **100% frontal projections with full reference reports** |

---

## 5. Safe Identifiers & PHI Protection

To prevent accidental leakage of health data, the evaluation manifest adheres to strict de-identification rules:

1. **Manifest File (`artifacts/report_evaluation/image_report_manifest.csv`):**
   * Uses safe case IDs: `CASE_DEV_001` through `CASE_DEV_025` and `CASE_TEST_001` through `CASE_TEST_025`.
   * Replaces direct database identifiers with 16-character SHA-256 hashes (`study_uid_hash`, `image_uid_hash`).
   * Omits raw patient identifiers, medical record numbers, dates, and un-sanitized prose.
2. **Local Protected Store (`artifacts/report_evaluation/local_reference_store.json`):**
   * Stored locally on the user's filesystem for evaluation scripts.
   * Maps safe case IDs to local file paths and reference radiologist report sections.
   * Kept strictly local; never exported or transmitted to external networks.
