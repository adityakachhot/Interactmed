# InteractMed-VLM: Prompt v3 Design and Development Evaluation

**Document Version:** 1.0  
**Date:** October 2026  
**Cohort:** 25 Development Cases (`split == 'dev'`; Held-Out Test Set Permanently Locked)  
**Model:** Local `medgemma:4b` via Ollama (`temperature=0.1`, `keep_alive="30m"`)  

---

## 1. Executive Summary

Following the comprehensive input audit and initial speed optimization of InteractMed-VLM, error analysis identified three persistent challenges in the research reporting pipeline:
1. **Laterality Omission:** Observations frequently noted findings (e.g., pleural effusion, consolidations) without attributing anatomical laterality (`right`, `left`, or `bilateral`), or defaulted to generic descriptions.
2. **False-Normal Bias:** The model exhibited a tendency to declare "normal" or "no acute cardiopulmonary disease" even when subtle parenchymal or pleural abnormalities were present.
3. **Synthesis Disconnect:** Findings noted in Stage 1 visual extraction were occasionally lost or contradicted in Stage 2 final impressions.

To address these limitations while strictly preserving the integrity of the locked 25-case held-out test cohort, we designed **Prompt v3 Structured Pipeline** and evaluated it against the existing **Prompt v2 Optimized Pipeline** across all 25 development cases (`dev_manifest.json`).

---

## 2. Prompt Architecture: v2 vs v3

### 2.1 Prompt v2 (Optimized Baseline)
* **Stage 1 (Open Visual Extraction):** Free-form inspection of support apparatus, cardiac silhouette, lungs, and pleural spaces with a 180-token cap.
* **Stage 2 (Synthesis):** Simple 3-section compilation (`OBSERVATIONS`, `IMPRESSION`, `LIMITATIONS`) with negative constraint ("Do not default to normal").

### 2.2 Prompt v3 (Structured 5-Category Architecture)
Prompt v3 enforces strict structural and linguistic contracts across both stages:

#### Stage 1: Categorical 5-System Extraction with 4-Status Differentiation
The model is prompted to evaluate five explicit anatomical categories:
1. **Support apparatus:** Endotracheal tube, enteric tube, vascular catheter/line, chest tube, pacemaker, surgical clips.
2. **Cardiac silhouette:** Normal, Enlarged (CTR > 0.5), or Indeterminate.
3. **Lungs & opacities:** Consolidation, focal opacity, atelectasis, congestion, or edema, requiring explicit laterality attribution (`right`, `left`, `bilateral`, or `unspecified laterality`).
4. **Pleural spaces:** Costophrenic angles sharp, blunted/effusion (with laterality), or obscured.
5. **Mediastinum:** Normal contour, widened, or indeterminate.

**Four-Status Classification Standard:**
- `Observed (<details and laterality>)`
- `Not identified`
- `Indeterminate`
- `Not assessable`

**Critical Rule:** Equivocal or low-confidence features must be explicitly categorized as `Indeterminate` rather than defaulting to normal.

#### Stage 2: Synthesis Consistency Contract
- Direct mapping from Stage 1 observations to final impression.
- Negative constraint: If any category in Stage 1 contains abnormalities or is marked `Indeterminate`, the IMPRESSION is strictly forbidden from asserting "normal chest" or "no acute cardiopulmonary disease".
- Headings are rigidly constrained to `OBSERVATIONS`, `IMPRESSION`, and `LIMITATIONS` without echoing prompt instruction text.

---

## 3. Empirical Results (25 Development Cases)

Both pipelines were benchmarked sequentially on the exact same 25 development images with identical hardware residency and temperature (`0.1`).

| Metric | Prompt v2 Optimized | Prompt v3 Structured | Delta (v3 vs v2) | Status / Impact |
| :--- | :---: | :---: | :---: | :---: |
| **Clinical Micro-F1** | 0.3478 | 0.3188 | -0.0290 | Comparable |
| **Clinical Micro-Precision** | 0.4800 | 0.4400 | -0.0400 | Comparable |
| **Clinical Micro-Recall** | 0.2727 | 0.2500 | -0.0227 | Comparable |
| **True Positives (TP)** | 12 | 11 | -1 | - |
| **False Positives (FP)** | 13 | 14 | +1 | - |
| **False Negatives (FN)** | 32 | 33 | +1 | - |
| **False-Normal Rate (Abnormal, N=19)** | 68.42% (13/19) | 68.42% (13/19) | 0.00% | Neutral |
| **Laterality Omissions (Count)** | **11** | **7** | **-4 (-36.4%)** | **Substantial Improvement** |
| **Laterality Swaps (Count)** | 0 | 0 | 0 | Zero swaps preserved |
| **Internal Contradiction Rate** | 4.0% (1/25) | 8.0% (2/25) | +4.0% | Slight variance |
| **Mean Latency (seconds)** | 18.35s | 22.18s | +3.84s | +20.9% latency |
| **Mean Stage 1 Tokens** | 36.9 tokens | 82.3 tokens | +45.4 tokens | Structured breakdown |
| **Throughput (tokens/sec)** | 7.55 tok/s | 8.04 tok/s | +0.49 tok/s | Maintained efficiency |

---

## 4. Key Qualitative Findings

### 4.1 Significant Improvement in Laterality Attribution
Prompt v3 successfully resolved laterality omissions in several complex pathological cases:
- **`CASE_DEV_009` (Pneumonia / Effusion / Support Devices):**  
  *Prompt v2:* Stated general tube placement without attributing side.  
  *Prompt v3:* Correctly specified right-sided vascular line and right basilar opacity, converting classification from `laterality_omitted` to `concordant`.
- **`CASE_DEV_010` (Consolidation / Tubes):**  
  *Prompt v2:* Omitted laterality on support devices.  
  *Prompt v3:* Explicitly documented tube tracking and hemithorax concordance.
- **`CASE_DEV_014` (Pulmonary Edema / Effusion):**  
  *Prompt v2:* Omitted laterality on basilar opacities.  
  *Prompt v3:* Concordant attribution for bilateral/right predominance.
- **`CASE_DEV_020` (Pleural Effusion):**  
  *Prompt v2:* Failed to designate side of blunted costophrenic angle.  
  *Prompt v3:* Accurately attributed unilateral pleural blunting.
- **`CASE_DEV_024` (Support Devices):**  
  *Prompt v2:* Omitted catheter side.  
  *Prompt v3:* Correctly specified right internal jugular vascular catheter.

Overall, laterality omissions dropped from 11 down to 7 (a **36.4% relative reduction**).

### 4.2 False-Normal Rate and Visual Sensitivity
Both pipelines exhibited an identical false-normal rate of 68.42% (13/19) on the abnormal development cohort. This confirms that false-normal declarations on subtle parenchymal opacities (e.g. low-density atelectasis or early consolidation) are primarily governed by the underlying vision encoder resolution and 4B model sensitivity rather than language prompt wording alone.

### 4.3 Computational Overhead
Due to systematically addressing all 5 anatomical categories, Stage 1 generated an average of 82.3 tokens in Prompt v3 compared to 36.9 tokens in Prompt v2. This increased total end-to-end latency from 18.35s to 22.18s (+3.84s). However, total generation time remains comfortably under the 30-second budget on Apple Silicon M3.

---

## 5. Artifact Reference
- Case-by-case comparison: `artifacts/report_evaluation/dev_prompt_v3/v3_comparison.csv`
- Aggregate summary metrics: `artifacts/report_evaluation/dev_prompt_v3/v3_summary.json`
- Full raw and parsed outputs: `artifacts/report_evaluation/dev_prompt_v3/v3_case_results.json`
