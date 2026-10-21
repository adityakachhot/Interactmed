# Task 7 — MedGemma Report Quality, Abnormal-Finding Recall & Failure Analysis

**Author:** Senior Medical Vision-Language Model Researcher & ML Evaluation Engineer  
**Date:** October 2026  
**Status:** Comprehensive Baseline Error Audit & Ground-Truth Verification  
**Evaluation Set:** 11 Development Cases (`/Downloads/sample test`)  
**Reference Data:** Chest ImaGenome Gold/Silver Dataset + MIMIC-CXR Verified Reports (`/Downloads/files`)  

---

## 1. Executive Summary & Audit Mandate

An audit of 11 chest radiographs evaluated under **Production Prompt v3** (`medgemma:4b` via Ollama) revealed a high rate of clinical under-calling (false negatives), device omission, and device-type confusion:
- **Severe False-Normal Rate:** In **6 out of 11 cases** (Cases 2, 4, 5, 6, 7, 8), the model rendered an impression of *"Normal chest radiograph"* despite reference-confirmed abnormalities such as dense lobar pneumonia, pneumothorax, and bilateral pleural effusions.
- **Cardiomegaly Under-Calling:** In **3 out of 3 cardiomegaly cases** (Cases 6, 10, 11), cardiac silhouette was graded *"Normal"*.
- **Support Apparatus Confusion & Omission:** In **3 cases** (Cases 1, 3, 9), support apparatus was flagged, but specific device types were misclassified (e.g., chest tubes misidentified as endotracheal or enteric tubes) or co-occurring lines were missed (Swan-Ganz, PICC, Port-A-Cath). In **3 cases** with visible hardware (Cases 2, 4, 8), the model stated *"Not identified on available image"*.
- **Root Cause:** Prompt v3 was engineered during Task 5 to eliminate generic boilerplate hallucinations (achieving 0% boilerplate). However, its strict formatting and conservative penalization inadvertently induced an **extreme conservative bias**, driving the vision-language backbone to declare sections "Clear" or "Normal" whenever features deviated from high-contrast, unequivocal presentations.

---

## 2. Ground-Truth Verification & Cohort Governance

Every case was cross-verified against the local MIMIC-CXR directory (`/Users/adityakachhot/Downloads/files`) and the Chest ImaGenome knowledge graph (`chest-imagenome-dataset-1.0.0`):

| Case # | Image File Name | DICOM UID | Patient ID | Study ID | View | Age/Sex | Reference Pathologies | Support Apparatus / Hardware |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **1** | `0f768188-ace0476b...copy.jpg` | `0f768188-ace0476b-e815f596-4bcca8f6-f6614985` | `p10014729` | `s52149569` | AP | 20-30 F | Small apical PTX, mild interstitial edema (L > R), L retrocardiac atelectasis, surgical mediastinal opacity | L chest tube, ET tube (2.5 cm above carina), NG tube, Swan-Ganz catheter |
| **2** | `138f8635-f0427bf9....jpg` | `138f8635-f0427bf9-8fd0a81f-7004c876-f913b5fe` | `p10039272` | `s56948355` | AP | 80-90 M | New small L apical-lateral PTX, stable R base granuloma/nodule | L basilar chest tube |
| **3** | `19b74f79-7ad317b1...copy.jpg` | `19b74f79-7ad317b1-887496c3-2645db29-48245a51` | `p10003400` | `s53341253` | AP | 50-60 F | Moderate cardiomegaly, moderate L pleural effusion, basilar atelectasis, millimetric R PTX | R chest tube |
| **4** | `238ce890-bbec475f...copy.jpg` | `238ce890-bbec475f-c82ec6bb-7080bd2f-c4b471f1` | `p10026255` | `s55384601` | AP | 50-60 M | Small R apical PTX, R lower rib fractures, worsening bibasilar contusion/opacities/aspiration | R pigtail pleural catheter |
| **5** | `2f5e274a-749f1c06....jpg` | `2f5e274a-749f1c06-1c8be147-4fabf575-e333e808` | `p10012261` | `s58270659` | AP | 50-60 M | Severe L lower lobe consolidation/pneumonia, L basilar volume loss/atelectasis | *None* |
| **6** | `350a19d2-7329f60d....jpg` | `350a19d2-7329f60d-17416b9a-d6eca016-187e6876` | `p10043139` | `s51632240` | AP | 70-80 M | Cardiomegaly / pericardial effusion, bilateral pleural effusions (L > R), probable R apical PTX, bibasilar opacities | Median sternotomy wires, mediastinal surgical clips |
| **7** | `3bea0373-0d10dd77....jpg` | `3bea0373-0d10dd77-1cac5b90-651be924-d343b184` | `p10002428` | `s55758034` | AP | 60-70 F | Persistent RML & RLL consolidation/pneumonia, layering L pleural effusion | *None* |
| **8** | `3daa2f39-89cf3e9d....jpg` | `3daa2f39-89cf3e9d-6be38a60-2cd19002-964fed61` | `p10039272` | `s50589184` | AP | 80-90 M | Small L pleural effusion, very small L apical PTX | L chest tube / indwelling pleural catheter |
| **9** | `5f090a4b-dc8a796d....jpg` | `5f090a4b-dc8a796d-726db5a0-62f19c9f-58cccd6f` | `p10014729` | `s52359357` | AP | 20-30 F | Residual small L apical PTX, L base consolidation, R base atelectasis (improved) | L chest tubes, R PICC line |
| **10** | `dcdc4bd9-4301b111....jpg` | `dcdc4bd9-4301b111-2a65a814-ee8e7bc5-7f0b9a5a` | `p10003400` | `s56466802` | AP | 50-60 F | Moderate cardiomegaly; **Lungs clear** | R Port-A-Cath (tip in RA), abdominal pigtail catheter (RUQ) |
| **11** | `f7b69ee3-db7f264c....jpg` | `f7b69ee3-db7f264c-fca7d1c7-1d372fc0-02b35a47` | `p10025139` | `s50196457` | AP | 50-60 F | Moderate cardiomegaly, mild bibasilar atelectasis, tortuous aorta; **No acute consolidation** | Proximal left humerus orthopedic hardware |

### Research Governance Notice:
- **Zero Historical Overlap:** None of the 11 DICOM IDs exist in `local_grounding_verified.csv`, `expanded_dev` (250 cases), or `expanded_test` (117 cases).
- **Cohort Classification:** Because these 11 cases have been manually audited and inspected, they are designated strictly as a **development error-analysis set**. They must **not** be presented as an untouched, independent test cohort.

---

## 3. Detailed Baseline Failure Taxonomy

### Failure Mode 1: False-Normal Cascade (Cases 2, 4, 5, 6, 7, 8)
- **Manifestation:** Prompt v3 outputs standard negative phrases:
  - `LUNGS AND OPACITIES: Clear` or `No focal opacity, no consolidation, no atelectasis, no congestion, no edema.`
  - `PLEURAL SPACES: Costophrenic angles sharp.`
  - `OVERALL IMPRESSION: Normal chest radiograph.`
- **Underlying Mechanism:** Prompt v3 gives the model negative reinforcement against ungrounded hallucinations. MedGemma 4B opts for safety by outputting the canonical negative template unless pixel evidence is overwhelming.

### Failure Mode 2: Device Type Misclassification & Hallucination (Cases 1, 3, 9)
- **Manifestation:** In Case 9, the model detected apparatus, but output: `Endotracheal tube, enteric tube, vascular catheter, chest tube`. The actual patient had only a PICC and chest tube (no ET tube or enteric tube). In Case 3, the chest tube was mislabeled as ET and enteric tubes.
- **Underlying Mechanism:** When the model notices high-density radiopaque line features in the thorax, it groups them into common ICU apparatus templates without tracing anatomical origin or termination.

### Failure Mode 3: Subtle Cardiomegaly Under-Calling (Cases 6, 10, 11)
- **Manifestation:** All 3 cases feature radiologist-confirmed moderate cardiomegaly, but Prompt v3 classified `CARDIAC SILHOUETTE: Normal`.
- **Underlying Mechanism:** Portable AP projections cause magnification of the cardiac silhouette. In standard training, radiologist dictations often discount minor cardiothoracic ratio enlargement on poor inspiration AP films. MedGemma inherits this tendency and marks non-extreme cardiac shadows as normal.

---

## 4. Remediation Architecture

To address these failure modes without model retraining, three targeted strategies are developed and tested:
1. **Strategy A (Prompt v4 — Recall-Balanced Single-Pass):** Introduces four mutually exclusive categorical status tokens (`finding_observed`, `not_identified`, `indeterminate`, `not_assessable`) with explicit mandatory checklist prompts.
2. **Strategy B (Two-Stage Decoupled Pipeline):**
   - *Stage 1:* Vision-only structured JSON visual feature extraction.
   - *Stage 2:* Clinical report synthesis from validated JSON with strict hallucination barriers.
3. **Strategy C (Grounding-Assisted Regional Review):** Uses TransVG-AGPT Task 3 Epoch 5 to propose spatial regions for ambiguous parenchymal findings, passing both full radiograph and clamped regional context crops to MedGemma.
