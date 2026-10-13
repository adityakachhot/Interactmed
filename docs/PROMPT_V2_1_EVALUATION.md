# InteractMed-VLM: Three-Way Prompt Evaluation (v2 vs v3 vs v2.1)

**Evaluation Date:** October 2026  
**Cohort:** 25 Development Cases (`split == 'dev'`; Held-Out Test Cohort Permanently Locked)  
**Vision Model:** `medgemma:4b` via local Ollama inference (`keep_alive="30m"`, `temperature=0.1`)  
**Image Resolution:** 1024 px maximum side (LANCZOS, JPEG Q85)  
**Primary Artifact:** [`artifacts/report_evaluation/prompt_v2_1/v2_1_summary.json`](file:///Users/adityakachhot/interactmed/artifacts/report_evaluation/prompt_v2_1/v2_1_summary.json)  
**Per-Case Comparison:** [`artifacts/report_evaluation/prompt_v2_1/v2_1_comparison.csv`](file:///Users/adityakachhot/interactmed/artifacts/report_evaluation/prompt_v2_1/v2_1_comparison.csv)  

---

## 1. Executive Summary & Purpose

To balance reporting speed against diagnostic and laterality fidelity, we conducted a head-to-head empirical comparison across three whole-image prompt architectures:
1. **Prompt v2 Optimized (Baseline):** Fast 4-line visual extraction followed by a 3-section synthesis contract.
2. **Prompt v3 Structured:** Comprehensive 5-category extraction (Apparatus, Cardiac, Lungs, Pleural, Mediastinum) with 4-status differentiation, explicit laterality attribution, and negative synthesis constraints.
3. **Prompt v2.1 Hybrid:** An experimental hybrid combining the concise 4-line Stage 1 structure of v2 with v3's 4-status assessment (`Observed` | `Not identified` | `Indeterminate` | `Not assessable`), explicit laterality mandate, and strict synthesis consistency.

All experiments were executed within an identical execution environment on Apple Silicon M3 (16 GB unified RAM). The 25-case held-out test cohort was kept strictly locked and untouched.

---

## 2. Methodology & Experimental Controls

- **Hardware & Software Platform:** Apple Silicon M3, macOS, Ollama local inference server.
- **Model Residency:** Continuous warm residency with `keep_alive="30m"` to eliminate cold model reload overhead.
- **Image Inputs:** 1024 px maximum side.
- **Decoding & Extraction Standards:**
  - Clause-level entity extraction with bidirectional negation scoping (`extract_audited_labels`).
  - Strict laterality classification (`concordant`, `laterality_omitted`, `unspecified_stated`, or laterality swaps).
  - Internal contradiction detection between Stage 1 observations and Stage 2 impressions.
- **Cohorts Evaluated:** 25 development cases (19 abnormal, 6 normal) stratified across 7 pathology categories.

---

## 3. Measured Results (Head-to-Head Comparison)

| Metric | Prompt v2 Optimized | Prompt v3 Structured | Prompt v2.1 Hybrid | Delta (v2.1 vs v2) | Delta (v2.1 vs v3) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clinical Micro-F1** | **0.3478** | **0.3478** | **0.2857** | -0.0621 (-17.9%) | -0.0621 (-17.9%) |
| **Clinical Micro-Precision** | **0.4800** | **0.4800** | 0.4737 | -0.0063 | -0.0063 |
| **Clinical Micro-Recall** | **0.2727** | **0.2727** | 0.2045 | -0.0682 (-25.0%) | -0.0682 (-25.0%) |
| **True Positives (TP)** | 12 | 12 | 9 | -3 | -3 |
| **False Positives (FP)** | 13 | 13 | 10 | -3 | -3 |
| **False Negatives (FN)** | 32 | 32 | 35 | +3 | +3 |
| **False-Normal Rate (Abnormal, N=19)** | 68.42% (13/19) | 68.42% (13/19) | **57.89% (11/19)** | -10.53% | -10.53% |
| **Laterality Omissions (Count)** | 11 | **7** | 11 | 0 | +4 regressions |
| **Concordant Laterality (Count)** | 1 | **5** | 1 | 0 | -4 regressions |
| **Laterality Swaps (Count)** | 0 | 0 | 0 | 0 | 0 |
| **Internal Contradictions (Count)** | 0 | 0 | 0 | 0 | 0 |
| **Mean End-to-End Latency** | 20.55s | 24.90s | **20.01s** | **-0.54s** | **-4.89s (-19.6%)** |
| **Median End-to-End Latency** | 20.54s | 24.44s | 20.56s | +0.02s | -3.88s |
| **Latency Std Dev** | 3.35s | 4.80s | **2.89s** | -0.46s | -1.91s |
| **Mean Stage 1 Tokens** | 37.7 tokens | 86.4 tokens | 41.0 tokens | +3.3 tokens | -45.4 tokens |
| **Throughput (tokens/sec)** | 6.72 tok/s | 7.36 tok/s | 5.86 tok/s | -0.86 tok/s | -1.50 tok/s |

---

## 4. Deep-Dive Error Analysis

### 4.1 Prompt v2.1 Behavioral Dynamics: Why Micro-F1 Dropped
1. **Recall Suppression on Normal Cases:**  
   Prompt v2.1 incorporated the instruction: *"Rule: If equivocal, state Indeterminate; lack of a detected finding is not proof of normal. Do not declare normal or absence of disease if abnormalities or indeterminate features were noted."*  
   On normal cases (`CASE_DEV_001`, `CASE_DEV_004`, `CASE_DEV_006`), the model adhered so strictly to avoiding definitive declarations of "normal" that Stage 2 described the absence of acute findings while noting single-view limitations, without uttering the specific keyword *"normal chest"* or *"no acute cardiopulmonary disease"*. Consequently, our clause extractor classified these as unasserted findings (3 fewer True Positives for normal, dropping recall from 0.2727 to 0.2045).
2. **False-Normal Reduction on Abnormal Cases:**  
   Concurrently, that same negative caution helped Prompt v2.1 reduce unwarranted normal declarations on abnormal cases from 68.42% (13/19) down to 57.89% (11/19), successfully avoiding false-normal assertions on `CASE_DEV_011` and `CASE_DEV_013`.

### 4.2 Why Prompt v3 Outperformed v2.1 on Laterality
- **Prompt v3:** Dedicates explicit, standalone bullet points for Lungs and Pleural Spaces with the explicit requirement: `Specify laterality (right, left, bilateral, or unspecified laterality)`. This provided the 4B parameter model with sufficient prompt attention to emit laterality tokens, yielding **5 concordant laterality cases** and reducing omissions to **7**.
- **Prompt v2.1:** In condensing the extraction prompt into 4 concise lines, the model tended to output brief phrases (e.g. `Lungs: Clear`, `Pleural spaces: Sharp`) without attributing laterality to unilateral abnormalities, reverting laterality omissions back to **11**.

---

## 5. Architectural Decision & Chosen Default

### Decision: **Retain Prompt v3 Structured at 1024 px as Production Default**

| Criteria | Prompt v2 | Prompt v3 (Chosen) | Prompt v2.1 | Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Laterality Adherence** | 11 omissions | **7 omissions** | 11 omissions | Prompt v3 achieves superior anatomical laterality (36.4% reduction in omissions). |
| **Diagnostic Fidelity (F1)** | 0.3478 | **0.3478** | 0.2857 | Prompt v3 matches v2 while avoiding v2.1's recall suppression. |
| **Inference Budget** | 20.55s | 24.90s | 20.01s | All configurations remain comfortably under the 30s SLA on Apple Silicon M3. |
| **Internal Contradictions** | 0 | 0 | 0 | Zero contradictions across all variants. |

**Verdict:** The 4.89s latency difference between v3 and v2.1 is an acceptable trade-off for the substantial gain in anatomical laterality attribution and overall diagnostic recall. Prompt v3 Structured is retained as the default pipeline in [`src/batch_service.py`](file:///Users/adityakachhot/interactmed/src/batch_service.py).

---

## 6. Reproduction Commands

To replicate this three-way evaluation exactly:

```bash
# Activate the environment
source /Users/adityakachhot/miniforge3/bin/activate dl

# Verify Ollama is running and medgemma:4b is resident
curl -s http://localhost:11434/api/tags | grep "medgemma:4b"

# Run the three-way benchmark across the 25 development cases
python /Users/adityakachhot/interactmed/scripts/evaluate_prompt_v2_1.py

# Run the test suite
pytest tests/test_batch_service.py tests/test_backend_api.py -v

# Run the frontend production build
cd frontend && npm run build
```

---

## 7. Limitations & Clinical Research Disclaimer

- **Research Artifact Only:** All reports generated by InteractMed-VLM are automated AI drafts for technical research purposes and are not approved diagnostic medical devices.
- **Model Resolution Limits:** MedGemma 4B exhibits persistent false-normal rates (~58–68%) on subtle, low-density parenchymal consolidations, which are constrained by vision transformer capacity rather than prompt wording.
- **Statistical Power:** Sample size is $N=25$ development cases. Differences between prompts reflect technical qualitative behaviors rather than clinical statistically significant efficacy.
