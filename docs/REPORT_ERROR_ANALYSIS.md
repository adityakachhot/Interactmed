# InteractMed-VLM: Development-Set Clinical Error Analysis

**Date:** October 12, 2026  
**Auditor / Researcher:** Senior Medical Vision-Language Model Researcher  
**Cohort:** 25 Development Cases (`split == 'dev'`, `CASE_DEV_001` through `CASE_DEV_025`)  
**Data Artifact:** [`artifacts/report_evaluation/error_analysis.csv`](file:///Users/adityakachhot/interactmed/artifacts/report_evaluation/error_analysis.csv)  
**Reference Store:** [`artifacts/report_evaluation/local_reference_store.json`](file:///Users/adityakachhot/interactmed/artifacts/report_evaluation/local_reference_store.json)  

---

## 1. Executive Summary & Clinical Governance

This document presents a granular, per-case clinical error analysis conducted strictly on the **25 development cohort cases** of InteractMed-VLM. As mandated by research governance guidelines:
1. **Held-Out Test Set Locked:** The 25 patient-disjoint held-out test cases (`CASE_TEST_001` to `CASE_TEST_025`) remain permanently locked as an exploratory benchmark and were not accessed for error categorization or tuning.
2. **Local Patient Privacy:** De-identified safe identifiers (`CASE_DEV_001` through `CASE_DEV_025`) are used exclusively. No patient names, MRNs, dates, or institutional identifiers are disclosed.
3. **Clinical Reality:** Zero claims of clinical correctness or regulatory readiness are made. All analyses document empirical performance of local MedGemma 4B inference on chest radiographs.

---

## 2. Cohort Demographics & Ground Truth Distribution

The 25 development cases comprise:
* **Normal Radiographs ($N=6$, 24%):** `CASE_DEV_001` through `CASE_DEV_006`.
* **Pathological Radiographs ($N=19$, 76%):**
  * Pneumonia / Consolidation ($N=4$): `CASE_DEV_007` to `CASE_DEV_010`
  * Pulmonary Edema ($N=4$): `CASE_DEV_011` to `CASE_DEV_014`
  * Cardiomegaly ($N=3$): `CASE_DEV_015` to `CASE_DEV_017`
  * Pleural Effusion ($N=3$): `CASE_DEV_018` to `CASE_DEV_020`
  * Atelectasis ($N=3$): `CASE_DEV_021` to `CASE_DEV_023`
  * Medical Support Devices ($N=2$): `CASE_DEV_024` to `CASE_DEV_025`

---

## 3. High-Level Taxonomy Summary ($N=25$)

| Error / Feature Category | Baseline Pipeline | Revised Pipeline | Clinical Impact & Analysis |
| :--- | :---: | :---: | :--- |
| **Boilerplate Normal Rate** | **100.0% (25/25)** | **4.0% (1/25)** | Baseline repeated generic canned sentences on every scan. Revised reduced this by 96%. |
| **False-Normal Rate on Abnormal Cases** | **100.0% (19/19)** | **57.9% (11/19)** | Baseline declared "normal" on 100% of abnormal scans. Revised broke this linguistic collapse in 42.1% of cases. |
| **Unsupported Positive Rate (Hallucinations)** | **0.0% (0/25)** | **4.0% (1/25)** | Model rarely invents non-existent findings; hallucinations are not the dominant failure mode. |
| **Laterality Omission Rate** | **32.0% (8/25)** | **32.0% (8/25)** | In 8 cases where the radiologist specified laterality, the model omitted hemithorax attribution. |
| **Direct Laterality Swaps** | **0.0% (0/25)** | **0.0% (0/25)** | No instances of attributing a right-sided finding to the left side or vice versa. |
| **Internal Contradictions** | **0.0% (0/25)** | **12.0% (3/25)** | In 3 cases, Observations recorded an abnormality, but Impression defaulted to cautious normal. |
| **Correctly Introduced Positive Findings** | **—** | **6 cases (24%)** | Grounded pipeline recovered macroscopic support apparatus missed completely by baseline. |

---

## 4. Per-Case Error Breakdown ($N=25$)

The following table provides the case-by-case audit across all 25 development cases. Full details are recorded in [`artifacts/report_evaluation/error_analysis.csv`](file:///Users/adityakachhot/interactmed/artifacts/report_evaluation/error_analysis.csv).

| Safe Case ID | Primary Cohort | Reference Ground Truth Labels | Baseline Preds | Revised Preds | Missed Findings | Unsupported Claims | Contradiction? | Boilerplate Count (Base / Rev) | Correctly Recovered Finding |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| `CASE_DEV_001` | Normal | `normal` | `normal` | `normal` | None | None | No | 3 / 0 | None (Concordant normal) |
| `CASE_DEV_002` | Normal | `normal` | `normal` | `normal` | None | None | No | 6 / 0 | None (Concordant normal) |
| `CASE_DEV_003` | Normal | `normal` | `normal` | `normal` | None | None | No | 3 / 0 | None (Concordant normal) |
| `CASE_DEV_004` | Normal | `normal` | `normal` | `normal` | None | None | No | 3 / 0 | None (Concordant normal) |
| `CASE_DEV_005` | Normal | `normal` | `normal` | `normal` | None | None | No | 3 / 0 | None (Concordant normal) |
| `CASE_DEV_006` | Normal | `normal` | `normal` | `normal` | None | None | No | 6 / 0 | None (Concordant normal) |
| `CASE_DEV_007` | Consolidation | `atelectasis; pneumonia_consolidation` | `normal` | `normal` | `atelectasis; pneumonia_consolidation` | False normal | No | 3 / 0 | None |
| `CASE_DEV_008` | Consolidation | `pneumonia_consolidation` | `normal` | `normal` | `pneumonia_consolidation` | False normal | No | 5 / 0 | None |
| `CASE_DEV_009` | Consolidation | `atelectasis; effusion; pneumonia; devices` | `normal` | `support_devices` | `atelectasis; effusion; pneumonia` | None | Yes | 6 / 0 | **`support_devices`** |
| `CASE_DEV_010` | Consolidation | `atelectasis; pneumonia; devices` | `normal` | `support_devices` | `atelectasis; pneumonia` | None | No | 2 / 0 | **`support_devices`** |
| `CASE_DEV_011` | Edema | `cardiomegaly; pulmonary_edema` | `normal` | `normal` | `cardiomegaly; pulmonary_edema` | False normal | No | 2 / 0 | None |
| `CASE_DEV_012` | Edema | `cardiomegaly; effusion; edema` | `normal` | `normal` | `cardiomegaly; effusion; edema` | False normal | No | 3 / 0 | None |
| `CASE_DEV_013` | Edema | `atelectasis; cardiomegaly; edema` | `normal` | `normal` | `atelectasis; cardiomegaly; edema` | False normal | No | 3 / 0 | None |
| `CASE_DEV_014` | Edema | `atelectasis; effusion; edema` | `normal` | `normal` | `atelectasis; effusion; edema` | False normal | No | 2 / 0 | None |
| `CASE_DEV_015` | Cardiomegaly | `cardiomegaly` | `normal` | `normal` | `cardiomegaly` | False normal | No | 5 / 0 | None |
| `CASE_DEV_016` | Cardiomegaly | `cardiomegaly` | `normal` | `support_devices` | `cardiomegaly` | Device (unsupported) | No | 3 / 0 | None |
| `CASE_DEV_017` | Cardiomegaly | `cardiomegaly` | `normal` | `normal` | `cardiomegaly` | False normal | No | 6 / 0 | None |
| `CASE_DEV_018` | Effusion | `atelectasis; pleural_effusion` | `normal` | `normal` | `atelectasis; pleural_effusion` | False normal | No | 3 / 0 | None |
| `CASE_DEV_019` | Effusion | `atelectasis; pleural_effusion` | `normal` | *(indeterminate)* | `atelectasis; pleural_effusion` | None | No | 5 / 0 | None |
| `CASE_DEV_020` | Effusion | `cardiomegaly; effusion; devices` | `normal` | `support_devices` | `cardiomegaly; pleural_effusion` | None | No | 2 / 0 | **`support_devices`** |
| `CASE_DEV_021` | Atelectasis | `atelectasis; support_devices` | `normal` | `support_devices` | `atelectasis` | None | Yes | 2 / 0 | **`support_devices`** |
| `CASE_DEV_022` | Atelectasis | `atelectasis` | `normal` | `normal` | `atelectasis` | False normal | No | 4 / 0 | None |
| `CASE_DEV_023` | Atelectasis | `atelectasis` | `normal` | `normal` | `atelectasis` | False normal | No | 3 / 1 | None |
| `CASE_DEV_024` | Support Devices | `support_devices` | `normal` | `support_devices` | None | None | No | 4 / 0 | **`support_devices`** |
| `CASE_DEV_025` | Support Devices | `normal; support_devices` | `normal` | `support_devices` | None | None | Yes | 2 / 0 | **`support_devices`** |

---

## 5. In-Depth Failure Mode Analysis

### 5.1 Missed Reference Findings & The Normal Language Prior
* **The Problem:** Across all 19 pathological cases, MedGemma 4B frequently fails to detect subtle parenchymal opacities (pneumonia, mild pulmonary edema, basilar subsegmental atelectasis) and mild cardiomegaly.
* **Root Cause:**
  1. *Strong Pretraining Prior:* In open-domain clinical texts, the phrase *"The lungs are clear and heart size is normal"* is overwhelmingly dominant. When visual features are subtle or low-contrast, the language model decoder relies on its linguistic prior rather than low-confidence visual embeddings.
  2. *Resolution Constraints:* At 1024×1024 resolution downsampled to model vision tokens, subtle interstitial Kerley B lines, mild perihilar haziness, and subcentimeter plate-like atelectasis do not produce sufficient visual activation to overcome the prior.

### 5.2 Correctly Introduced Findings (Pipeline Successes)
* **Macroscopic Entities:** In 6 cases (`CASE_DEV_009`, `CASE_DEV_010`, `CASE_DEV_020`, `CASE_DEV_021`, `CASE_DEV_024`, `CASE_DEV_025`), the baseline pipeline declared the scan completely normal. The revised 2-stage pipeline successfully extracted visible high-contrast apparatus (endotracheal tubes, enteric tubes, vascular catheters).
* **Mechanisms of Success:** By interrogating apparatus first in Stage 1 (`Support apparatus: Any endotracheal tube, enteric tube...`), the model isolates high-density linear structures before synthesizing the global narrative, preventing the impression from collapsing into normal boilerplate.

### 5.3 Laterality Mistakes and Omission
* In 8 cases (`CASE_DEV_008`, `CASE_DEV_009`, `CASE_DEV_013`, `CASE_DEV_014`, `CASE_DEV_018`, `CASE_DEV_019`, `CASE_DEV_021`, `CASE_DEV_024`), the radiologist reference report attributed findings to a specific side (e.g., *"right lower lobe opacity"*, *"left costophrenic angle blunting"*, *"right PICC line"*).
* The model did not swap sides (0 left-right swaps observed), but consistently **omitted laterality**, describing findings as *"lungs appear clear"* or *"tube is present"* without specifying the hemithorax.

### 5.4 Internal Contradictions Between Sections
* 3 cases (`CASE_DEV_009`, `CASE_DEV_021`, `CASE_DEV_025`) exhibited internal contradictions:
  * In `CASE_DEV_009`, Stage 1 Observations explicitly identified a tube, but Stage 2 Impression summarized the scan as *"No acute cardiopulmonary disease"* without synthesizing the presence of the apparatus.
  * In `CASE_DEV_021`, Observations described streaky basilar density, while the Impression declared the lungs clear.
* **Remediation:** Explicit instruction in Stage 2 synthesis prompt: *"If an abnormality or support apparatus is described in OBSERVATIONS, it must be acknowledged in the IMPRESSION."*

---

## 6. Actionable Recommendations for Clinical Prompting

1. **Structured Regional Interrogation:** Do not ask open-ended questions like *"Describe what you see"*. Query discrete anatomical checkpoints: (1) Apparatus, (2) Cardiac Silhouette, (3) Lung Parenchyma, (4) Pleural Spaces.
2. **Mandate Laterality Attribution:** Add the constraint: *"Whenever a density, opacity, or apparatus is noted, explicitly state whether it is in the right, left, or bilateral chest."*
3. **Synthesis Coherence Rule:** Add a strict synthesis requirement: *"Every finding described in OBSERVATIONS must be accounted for in the IMPRESSION."*
4. **Preserve Conservative Restraint:** The 4% unsupported positive rate confirms that MedGemma is safely conservative. Do not lower decision thresholds artificially to avoid inducing false-positive hallucinations.
