# TASK 13: Pretrained Model Provenance and Dataset Overlap Audit

**Date**: October 17, 2026  
**Auditor**: Antigravity Senior ML & Clinical AI Systems Engineer  
**Component Audited**: TorchXRayVision Auxiliary Pretrained Classifier (`densenet121-res224-all`)  
**Context**: InteractMed-VLM Local Development Cohort Evaluation  

---

## 1. Objective and Scope of Audit

In Task 12, an auxiliary pretrained DenseNet-121 classifier from the open-source TorchXRayVision library was evaluated on InteractMed-VLM's 50-case development cohort (`artifacts/fine_tuning_cohort/val_dev.jsonl`). The model demonstrated exceptional sensitivity, yielding **100% recall** across reference pathologies at $\tau = 0.62$ and eliminating all case-level false-normal outputs.

The objective of this audit is to:
1. Inspect the exact model weights, training architecture, and pretraining datasets used by TorchXRayVision.
2. Cross-reference the pretraining sources against the image origins of InteractMed-VLM's development partition.
3. Assess the possibility and degree of pretraining data leakage or patient/study overlap.
4. Document the scientific and clinical limitations that this potential overlap imposes on validation findings.

---

## 2. Pretrained Model Provenance

### 2.1 Model Artifact Specification
- **Library Version**: `torchxrayvision==1.5.5`
- **Model Class**: `torchxrayvision.models.DenseNet(weights="densenet121-res224-all")`
- **Checkpoint File**: `nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt`
- **File Size**: 27,241,123 bytes (~27 MB)
- **Local Download Cache**: `~/.torchxrayvision/models_data/`
- **Pretraining Reference**: Joseph Paul Cohen, Joseph D. Viviano, Paul Bertin, Paul Morrison, Pengzhen Huang, Matthew Hirsch, Haisam Hassouna, Florian Rohner, Arash Rafiei, Tim F. C. Hermans, et al. *"TorchXRayVision: A library of chest X-ray datasets and models"*, Medical Imaging with Deep Learning (MIDL) 2022 / arXiv:2002.02497.

### 2.2 Datasets Comprising Pretraining Cohort
According to TorchXRayVision repository documentation (`model_urls['all']`), the model was trained jointly across the following major multi-center radiography datasets:
1. **NIH ChestX-ray14**: 112,120 frontal views from 30,805 unique patients (National Institutes of Health Clinical Center).
2. **PadChest**: 160,000+ views from 67,000 patients (Hospital San Juan, Alicante, Spain).
3. **CheXpert**: 224,316 chest radiographs from 65,240 patients (Stanford Hospital).
4. **MIMIC-CXR (`mimic_ch`)**: 377,110 chest radiographs from 65,379 unique patients (Beth Israel Deaconess Medical Center, MIT / PhysioNet), labeled using the CheXpert labeler.
5. **Google Health CXR**: Curated clinical chest radiographs.
6. **OpenI (Indiana University)**: 7,470 chest radiographs from 3,996 patients.
7. **RSNA Pneumonia Detection Challenge (Kaggle)**: Stage 2 pneumonia bounding-box dataset derived from NIH CXR.

---

## 3. Dataset Provenance of InteractMed-VLM Development Cohort

### 3.1 Origin of `val_dev.jsonl`
InteractMed-VLM's 50-case development cohort (`val_dev.jsonl`) was assembled during Task 8/10 from the Chest ImaGenome dataset. Chest ImaGenome is an anatomical and pathological scene-graph annotation project built directly on top of the **MIMIC-CXR (v2.0.0)** imaging corpus.

Inspection of the raw file paths and metadata in `val_dev.jsonl`:
- **Image Directory Structure**: `/Users/adityakachhot/Downloads/data/mimic-cxr/files/p14/p14548229/s53645928/...jpg`
- **Patient Identifiers**: Prefix `p10...`, `p14...`, `p17...`, `p19...` (e.g., `p14548229`, `p17377807`, `p14455452`, `p13975291`).
- **Study Identifiers**: MIMIC-CXR accession numbers (e.g., `s53645928`, `s50075778`, `s56063789`).

---

## 4. Assessment of Overlap Risk

### 4.1 Structural Overlap
Because TorchXRayVision's `densenet121-res224-all` model explicitly incorporates **`mimic_ch` (MIMIC-CXR)** as one of its primary training sources, and InteractMed-VLM's `val_dev` cases are also derived from MIMIC-CXR, **the auxiliary model and the evaluation cohort originate from the exact same underlying hospital imaging repository (Beth Israel Deaconess Medical Center)**.

### 4.2 Split Manifest Governance
- **Local MedGemma/TransVG Partitioning**: InteractMed-VLM maintains strict patient-level separation between its internal training, validation (`val_dev`, 50 cases), and locked test (`expanded_test_locked`, 117 cases) partitions. No patient in `val_dev` exists in MedGemma's local training data.
- **External TorchXRayVision Partitioning**: The open-source TorchXRayVision weights package distributes only the compiled model weights (`.pt`). It does **not bundle the patient ID inclusion/exclusion manifests** used during its original multi-dataset joint pretraining run.

### 4.3 Definitive Conclusion on Overlap
> [!WARNING]
> **Data Independence Cannot Be Claimed**:
> Because the TorchXRayVision pretraining package does not publish exact patient-level split manifests for the joint multi-dataset weights, **overlap between the classifier's pretraining training images and our `val_dev` images cannot be conclusively ruled out**.
> 
> Patient-level separation within InteractMed-VLM does not guarantee that TorchXRayVision's authors excluded these exact MIMIC-CXR patients during their 2020 pretraining run.

---

## 5. Scientific and Clinical Implications

1. **In-Distribution vs. Generalization Bias**:
   The near-perfect sensitivity (100% recall at $\tau=0.62$, median positive score 0.65 vs median negative score 0.62) observed on `val_dev` may partially stem from in-distribution exposure to MIMIC-CXR acquisition parameters (image contrast, view angles, DICOM bit-depth scaling, and institution-specific detector artifacts).
2. **Review Alert Behavior on Genuinely Unseen Data**:
   On external, hospital-independent datasets (e.g., European patients from PadChest or community hospital data), the classifier's score distributions may shift, potentially degrading sensitivity or altering the optimal threshold.
3. **Integrity of the Locked Test Partition**:
   Because `expanded_test_locked` is also drawn from MIMIC-CXR, testing the TorchXRayVision model on `expanded_test_locked` will test generalization within the MIMIC-CXR domain, but will remain subject to the same potential pretraining overlap limitation.
4. **Clinical Readiness Disclaimer**:
   The auxiliary classifier's performance on `val_dev` provides valuable proof-of-concept for multi-model secondary review, but **must never be cited as proof of clinical efficacy or diagnostic readiness** on uncurated patient populations.
