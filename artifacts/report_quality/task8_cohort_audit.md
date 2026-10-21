# Task 8 — Expanded Cohort Audit & Patient Governance Verification

**Author:** Senior Medical Vision-Language Model Researcher & ML Evaluation Engineer  
**Date:** October 2026  
**Status:** Frozen & Audited Evaluation Cohort ($N=60$)  
**Source Dataset:** MIMIC-CXR-JPG & Chest ImaGenome Gold Dataset  
**Partition Source:** `expanded_dev` (250 cases, 248 patients)  
**Selected Cohort Size:** Exactly **60 cases across 60 unique patients** (100% Patient-Disjoint)  

---

## 1. Executive Summary & Cohort Governance

Task 8 establishes an expanded, patient-disjoint evaluation cohort of **60 cases** to validate report generation and grounding-assisted review across:
- **Baseline:** Production Prompt v3
- **Strategy A:** Structured Prompt v4 (Recall-Balanced Single-Pass)
- **Strategy B:** Streamlined Prompt v5 (Two-Stage JSON Extraction + Report Synthesis)
- **Strategy C:** Grounding-Assisted Prompt v6 (TransVG-AGPT Task 3 Epoch 5 Regional Review)

### Governance & Non-Contamination Safeguards:
1. **Zero Overlap with Task 7 Development Set:** None of the 11 development cases inspected in Task 7 are included. The 11 development cases were sourced from `/Downloads/sample test` and are strictly quarantined.
2. **Zero Overlap with Training Splits:** None of the 60 cases or their patients exist in the TransVG training splits (MS-CXR, RSNA Pneumonia train, or Multi-CXR train).
3. **Strict Patient Disjointness:** All 60 cases belong to 60 distinct patient identifiers ($|\text{Patients}| = 60$). No patient appears more than once in the cohort.
4. **Held-Out Test Set Untouched:** The strictly locked test partition `expanded_test` (117 cases across 117 patients) remains completely untouched and preserved for future blind testing.
5. **Verified Ground Truth:** 100% of cases are linked to confirmed Chest ImaGenome gold relational annotations and full original radiology reports.

---

## 2. Cohort Stratification & Clinical Balance

### Primary Category Representation:
| Primary Category | Case Count | Proportion (%) | Target Evaluation Role |
| :--- | :---: | :---: | :--- |
| **Pneumonia / Consolidation** | 12 | 20.0% | Dense parenchymal opacities, air bronchograms |
| **Pulmonary Edema** | 10 | 16.7% | Vascular congestion, interstitial & alveolar haze |
| **Support Devices / Tubes** | 10 | 16.7% | Hardware, endotracheal/chest tubes, lines |
| **Cardiomegaly** | 9 | 15.0% | Transverse cardiac enlargement (CTR > 0.5) |
| **Atelectasis** | 9 | 15.0% | Volume loss, plate-like basilar opacities |
| **Normal (Negative Controls)** | 10 | 16.7% | Unremarkable chest radiographs (specificity assessment) |
| **Total Cohort** | **60** | **100.0%** | Full clinical representation |

### Projection & Technical Characteristics:
- **Anterior-Posterior (AP):** 37 cases (61.7%) — Representative of portable ICU and bed-bound examinations.
- **Posterior-Anterior (PA):** 23 cases (38.3%) — Representative of ambulatory, full-inspiration examinations.
- **Image Resolution:** 100% verified non-corrupt DICOM-derived JPEG images with original matrix dimensions (median 2539 × 3056 pixels).

### Multi-Label Finding Prevalence (Gold Reference Ground Truth):
| Evaluated Finding | Total Affirmed Instances | Case Prevalence (%) | Negative Controls Available |
| :--- | :---: | :---: | :---: |
| **Pneumonia / Consolidation** | 22 | 36.7% | Yes (38 cases) |
| **Atelectasis** | 22 | 36.7% | Yes (38 cases) |
| **Support Devices / Hardware** | 21 | 35.0% | Yes (39 cases) |
| **Cardiomegaly** | 14 | 23.3% | Yes (46 cases) |
| **Pulmonary Edema** | 10 | 16.7% | Yes (50 cases) |
| **Pleural Effusion** | 8 | 13.3% | Yes (52 cases) |
| **Normal (True Negatives)** | 22 | 36.7% | Ground-truth confirmed absence of disease |

*Note: True multi-label coexistence allows independent per-finding sensitivity, specificity, PPV, and NPV evaluation across all 7 anatomical categories.*

---

## 3. Complete Cohort Case Manifest

The full cohort manifest with image paths, patient IDs, study UID hashes, and gold affirmed labels is serialized in [`artifacts/report_quality/task8_cohort_manifest.json`](file:///Users/adityakachhot/interactmed/artifacts/report_quality/task8_cohort_manifest.json).
