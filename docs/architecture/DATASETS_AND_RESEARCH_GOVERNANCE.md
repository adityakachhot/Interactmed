# InteractMed-VLM: Datasets, Cohorts, and Research Governance

> **Authoritative Dataset Inventory, Partitioning & Governance Reference**  
> **Project:** InteractMed-VLM  
> **Document:** `docs/architecture/DATASETS_AND_RESEARCH_GOVERNANCE.md`  
> **Status:** Audited against Filesystem Holdings, Manifests, and Integrity Scripts

---

## 1. Master Dataset Inventory

InteractMed-VLM incorporates multiple clinical radiography datasets, each providing a specific tier of supervision. The table below outlines all datasets verified in the repository:

| Dataset Name | Source Repository & Filesystem Location | Supervision Modality | Total Available Records | Locally Available Images | Bounding Box Convention | Role in InteractMed-VLM | Data Access & Licensing Constraints |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RSNA Pneumonia Detection Challenge 2018** | `data/raw/rsna/` | Binary opacity label + localized bounding boxes | 26,684 DICOMs (6,012 positive cases) | 6,012 positive DICOMs in `stage_2_train_images` | `[x, y, w, h]` (Absolute pixels, 1024x1024 native) | Visual grounding fine-tuning (Train: 4,210; Val: 901; Held-Out Test: 901) | Kaggle Competition License; open for non-commercial academic research. |
| **Chest-ImaGenome (v1.0.0 Gold)** | Local metadata store / PhysioNet | Gold anatomical bounding boxes + clinical relations & sentences | 25,989 bounding boxes, 11,885 sentences across 1,000 studies | **417 unique frontal images** (264 AP, 153 PA) verified on disk | `coord_original`: `[x1, y1, x2, y2]`; `coord224`: `[x1, y1, x2, y2]` | Reference report generation and expanded evaluation cohort construction | PhysioNet Credentialed Health Data License; requires ethics training (CITI) and DUA. |
| **MS-CXR (v1.1.0 Local Alignment)** | Local metadata store / PhysioNet | Phrase-level bounding boxes across 8 pathology classes | 1,448 annotation records across 1,047 unique DICOMs | **764 unique images** present locally (73.0% coverage) | `[x, y, w, h]` (Absolute pixels, native image resolution) | Pretraining source for TransVG-AGPT baseline; grounding benchmark | PhysioNet Credentialed Access License (MIMIC-CXR derivative). |
| **MIMIC-CXR-JPG (Local Store)** | `files/p{prefix}/p{subject_id}/s{study_id}/` | Full-radiograph images (JPEG) + DICOM metadata | 377,110 studies in full MIMIC | **1,174 local images** across 10 subject prefixes (`p10`–`p19`) | N/A (Image Repository) | Underlying image files for local Chest-ImaGenome and MS-CXR evaluations | PhysioNet Credentialed Data Use Agreement; strictly prohibited from public distribution. |
| **Unified Grounding Manifest** | [`manifests/unified_manifest.csv`](file:///Users/adityakachhot/interactmed/manifests/unified_manifest.csv) | Harmonized phrase-to-bounding-box mappings | 17,135 harmonized records | **1,174 unique images** (13,992 present records; 81.7%) | Standardized: `[bbox_x, bbox_y, bbox_w, bbox_h]` | Cross-dataset phrase grounding alignment and audit | Derived from PhysioNet holdings; governed by MIMIC DUA. |
| **MIMIC-CXR-VQA** | Local metadata store | Clinical Question-Answer tuples linked to MIMIC images | 87,360+ QA pairs across train, val, and test | Linked to local MIMIC image store | N/A (QA Pairs) | Secondary evaluation candidate for multimodal conversational validation | Research use only under PhysioNet DUA. |

---

## 2. Cross-Dataset Overlap & Set Arithmetic

A critical risk in medical machine learning is **unintentional data leakage** between pretraining corpora, fine-tuning datasets, and evaluation splits. 

To quantify overlap between the local benchmark holdings, programmatic set arithmetic was executed across verified unique DICOM identifiers ([`scripts/audit_overlap.py`](file:///Users/adityakachhot/interactmed/scripts/audit_overlap.py)):

```
MS-CXR Unique DICOM IDs:                     1,047
Chest-ImaGenome Gold Unique DICOM IDs:       1,000
───────────────────────────────────────────────────
Overlap (MS-CXR ∩ Chest-ImaGenome Gold):         9 DICOMs
Total Unique Benchmark Images:               2,038 DICOMs
Total Locally Available MIMIC Images:        1,174 DICOMs
───────────────────────────────────────────────────
Locally Available MS-CXR Images:               764 DICOMs (72.97%)
Locally Available Chest-ImaGenome Gold:        417 DICOMs (41.70%)
Locally Available in BOTH Benchmarks:            7 DICOMs
```

### Research Insight on Benchmark Complementarity
The minimal overlap (**only 9 DICOMs out of 2,038**) demonstrates that MS-CXR and Chest-ImaGenome Gold evaluate complementary, non-redundant capabilities:
1. **Chest-ImaGenome Gold:** Evaluates **anatomical region localization** (e.g., *cardiac silhouette*, *right middle lung zone*, *left costophrenic angle*, *trachea*).
2. **MS-CXR:** Evaluates **pathology phrase grounding** (e.g., *dense consolidation in right lower lobe*, *moderate bilateral pleural effusion*, *severe cardiomegaly*).

---

## 3. Cohort Architecture & Patient-Disjoint Partitions

To ensure uncompromising evaluation integrity, the repository strictly isolates distinct research cohorts. **These cohorts serve distinct purposes and must never be merged or conflated**:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              COHORT ISOLATION ARCHITECTURE                             │
├──────────────────────────┬─────────────────────────────┬───────────────────────────────┤
│ Historical Benchmark     │ Expanded Evaluation Cohort  │ MedGemma Fine-Tuning Cohort   │
│ (Quarantined)            │ (Report Evaluation)         │ (LoRA Adaptation)             │
├──────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ • 50 Cases (50 Patients) │ • 367 Frontal Images        │ • 250 Cases (from Dev Cohort) │
│ • 25 Dev / 25 Test       │ • 365 Unique Patients       │ • 200 Train / 50 Val          │
│ • Unit tests & baseline  │ • 250 Dev / 117 Locked Test │ • 117 Locked Test Held-Out    │
│ • STRICTLY ISOLATED      │ • Full Benchmark Comparison │ • 1 Image Per Study (1:1)     │
└──────────────────────────┴─────────────────────────────┴───────────────────────────────┘
```

---

### 3.1 Cohort 1: Quarantined Historical Benchmark ($N=50$)
- **Composition:** 50 cases from 50 unique patients (25 development cases, 25 exploratory test cases).
- **Metadata Store:** [`artifacts/report_evaluation/local_reference_store.json`](file:///Users/adityakachhot/interactmed/artifacts/report_evaluation/local_reference_store.json).
- **Governance:** Strictly quarantined. It was used during initial v0.1.0 prototyping and early prompt exploration (e.g. Prompt v2.1 vs Baseline). **Zero patients from this historical cohort are permitted in the expanded evaluation cohort.**

---

### 3.2 Cohort 2: Expanded Evaluation Cohort ($N=367$)
Formed by intersecting verified local MIMIC-CXR images with Chest-ImaGenome Gold clinical sentences, excluding all historical patients:
- **Total Eligible Frontal Radiographs:** **367 images across 365 unique patients** (229 AP, 138 PA; 100% verified readable).
- **Patient-Grouped Partitions:**
  - **`expanded_dev` (Development Split):** **250 images across 248 patients** (154 AP, 96 PA). Used for active development, prompt engineering, baseline comparison, and resolution ablation. Category distribution: 94 Pneumonia, 71 Normal, 57 Edema, 10 Devices, 9 Atelectasis, 9 Cardiomegaly.
  - **`expanded_test` (Locked Test Split):** **117 images across 117 patients** (75 AP, 42 PA). Category distribution: 44 Pneumonia, 33 Normal, 26 Edema, 6 Cardiomegaly, 4 Atelectasis, 4 Devices.

#### Strict Locked Test Set Policy
> [!IMPORTANT]
> **GOVERNANCE POLICY FOR `expanded_test` ($N=117$):**  
> 1. The 117-case test partition is **permanently sealed and quarantined**.  
> 2. No prompt tuning, few-shot exemplar selection, hyperparameter adjustment, or checkpoint selection is permitted on `expanded_test`.  
> 3. All model development must be conducted exclusively on `expanded_dev`.  
> 4. `expanded_test` will be unsealed only for the final, one-shot milestone evaluation of the completed platform.

---

### 3.3 Cohort 3: MedGemma 4B Fine-Tuning Cohort ($N=250 + 117$)
Derived from the development cohort to train and validate LoRA adapter weights ([`artifacts/fine_tuning_cohort/`](file:///Users/adityakachhot/interactmed/artifacts/fine_tuning_cohort/)):
- **`train_dev`:** **200 cases across 199 patients** (124 AP, 76 PA). Formatted as JSONL pairs with system prompt, image path, and target radiologist report.
- **`val_dev`:** **50 cases across 49 patients** (30 AP, 20 PA). Dedicated to validation loss tracking and early stopping during LoRA training.
- **Held-Out Test Set:** Corresponds exactly to the locked `expanded_test` (117 cases across 117 patients).

---

## 4. Multi-View vs. Single-View Integrity Audit

A well-documented source of error in multimodal medical imaging research is **multi-view report conflation**:
- Radiologists write **study-level reports**, which interpret all radiographs acquired during a single examination session (e.g. both PA and Lateral views).
- If an automated pipeline assigns the full study-level report as ground truth to a single frontal radiograph, the model is penalized for failing to describe findings that were only visible on the lateral projection, or conversely, the model is trained to hallucinate findings not visible on the frontal view.

### Verification of 1-to-1 Study-to-Image Mapping
An exhaustive programmatic audit of the fine-tuning cohort ([`scripts/audit_expanded_report_data.py`](file:///Users/adityakachhot/interactmed/scripts/audit_expanded_report_data.py)) verified:
1. **Single-View Restriction:** In `train_dev`, `val_dev`, and `expanded_test`, **every single study ID maps to exactly 1 radiograph image** ($\text{MaxImagesPerStudy} = 1$).
2. **Exclusion of Multi-Image Studies:** All studies with multiple projections were eliminated from the training cohort during manifest construction, ensuring that every training report corresponds strictly and exclusively to the single visual input provided.
3. **Frontal Projection Standardization:** All 367 images in the expanded cohort are verified frontal projections (AP or PA); lateral views were filtered out using Chest-ImaGenome's `cxr-record-list_view.csv`.

---

## 5. Patient Leakage & Disjointness Proof

To provide mathematical certainty against optimistic performance bias from intra-patient correlation, set intersections were verified across all partitions:

$$\operatorname{Patients}(\text{train\_dev}) \cap \operatorname{Patients}(\text{val\_dev}) = \emptyset \quad (\text{0 overlapping patients})$$
$$\operatorname{Patients}(\text{train\_dev}) \cap \operatorname{Patients}(\text{expanded\_test}) = \emptyset \quad (\text{0 overlapping patients})$$
$$\operatorname{Patients}(\text{val\_dev}) \cap \operatorname{Patients}(\text{expanded\_test}) = \emptyset \quad (\text{0 overlapping patients})$$
$$\operatorname{Patients}(\text{expanded\_cohort}) \cap \operatorname{Patients}(\text{historical\_cohort}) = \emptyset \quad (\text{0 overlapping patients})$$

Furthermore, in `expanded_dev`, 246 patients have exactly 1 study, and only 2 patients have 2 studies ($<1\%$ clustering effect), confirming that observations are statistically independent.

---

## 6. Patient Privacy, De-Identification & Ethics Governance

### 6.1 Compliance with PhysioNet Credentialed Data Use Agreements
The underlying imaging data and clinical sentences originate from the MIMIC-CXR database, collected by the Beth Israel Deaconess Medical Center (BIDMC) and distributed via PhysioNet. All handling within InteractMed-VLM complies with strict ethical guidelines:
1. **No Re-Identification:** Attempting to link de-identified records to individual patients or identities is strictly prohibited.
2. **Local Confinement:** No medical images or text records are uploaded to third-party cloud translation or commercial LLM services (e.g., OpenAI ChatGPT, Anthropic Claude, or public cloud endpoints).
3. **No Protected Health Information (PHI):** All facial features, lead markers containing patient names, and burned-in DICOM text headers were stripped during MIMIC-CXR-JPG creation.
4. **Synthetic Documentation Examples:** All examples in documentation, diagrams, test assertions, and API schemas use synthetic identifiers (e.g. `p_anon_101`, `s_anon_501`) and anonymized SHA hashes. Real patient IDs are never displayed.
