# TASK 14: Pretraining Overlap Limitation and Research Governance Audit

**Date**: October 17, 2026  
**Auditor**: Antigravity Senior ML & Clinical AI Systems Engineer  
**Scope**: Verification of Dataset Independence, Third-Party Weights Governance, and Held-Out Test Eligibility  

---

## 1. Local Dataset Disjointness Verification

InteractMed-VLM enforces strict patient-level separation across all internal evaluation partitions. We audited the patient identifiers across all three partitions:

| Partition | Total Cases | Unique Patients | Images per Patient | Overlap with `train_dev` | Overlap with `val_dev` | Overlap with `expanded_test` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`train_dev` (Training Pilot)** | 200 | 199 | 1.01 | — | **0 (0.0%)** | **0 (0.0%)** |
| **`val_dev` (Development Validation)** | 50 | 49 | 1.02 | **0 (0.0%)** | — | **0 (0.0%)** |
| **`expanded_test` (Held-Out Benchmark)** | **117** | **117** | **1.00** | **0 (0.0%)** | **0 (0.0%)** | — |

**Local Independence Finding**:
Within InteractMed-VLM's data governance boundary, **zero patient leakage exists**. All 117 cases in `expanded_test` belong to 117 distinct patients who have never appeared in `train_dev` or `val_dev`. Every single case maps to exactly 1 frontal radiograph image.

---

## 2. Third-Party Pretrained Model Overlap Limitation

While internal dataset disjointness is mathematically verified, **external pretraining overlap presents a critical scientific limitation**:

1. **Pretraining Source of DenseNet-121**:
   - Checkpoint: `nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt`
   - Library: TorchXRayVision (Cohen et al., MIDL 2022)
   - Pretrained Datasets: NIH, PadChest, CheXpert, Google Health, OpenI, Kaggle, and **MIMIC-CXR (`mimic_ch`)**.
2. **Data Origin of `expanded_test`**:
   - `expanded_test` was constructed from the Chest ImaGenome dataset, which is an annotation layer built directly on **MIMIC-CXR (v2.0.0)**.
3. **Absence of Third-Party Split Manifests**:
   - The open-source TorchXRayVision distribution provides only compiled `.pt` weights. It does **not provide the exact subject-level inclusion/exclusion manifests** used during its multi-institutional pretraining run.
   - Consequently, it is impossible to verify whether any of the 117 patients in `expanded_test` were included in TorchXRayVision's training corpus.

> [!WARNING]
> **Definitive Methodological Ruling**:
> Because third-party pretraining overlap cannot be ruled out, the evaluation on `expanded_test` constitutes a **project-held-out final benchmark**, NOT an independent external clinical validation.
> 
> High sensitivity or F1 on `expanded_test` demonstrates that the hybrid architecture successfully integrates multiple model modalities without regressions, but **must not be claimed as proof of generalizability to external health systems**.

---

## 3. Pre-Declared Reporting Constraints for Held-Out Evaluation

When executing the held-out evaluation on `expanded_test`:
1. **Terminology**: The evaluation must be explicitly designated as a **"Project-Held-Out Benchmark"**, never "External Validation".
2. **Disentangled Metrics**: Classifier review alerts must be reported separately from generated narrative report assertions.
3. **No Retuning**: Model weights, prompt templates, category mappings, and threshold parameters remain strictly frozen.
