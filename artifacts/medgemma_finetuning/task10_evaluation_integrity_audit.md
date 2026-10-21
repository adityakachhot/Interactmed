# TASK 10 — Evaluation Integrity Reconciliation, Hybrid Failure Audit, and Cloud LoRA Preparation

**InteractMed-VLM: Medical Vision-Language Model Research, Evaluation, and Pipeline Optimization**  
*Audit Run: `artifacts/medgemma_finetuning/task10_evaluation_integrity_audit.md`*  
*Timestamp: October 17, 2026 | Environment: Apple Silicon MPS & Ollama (`medgemma:4b`)*  
*Target Architecture: Google MedGemma 4B (`google/medgemma-4b-it`) | Grounding Backbone: TransVG-AGPT Task 3 Epoch 5*

---

## Executive Summary & Core Reconciliations

Task 10 conducted an exhaustive audit of the evaluation methodology, data governance, and hybrid pipeline failures observed in Tasks 8 and 9. Furthermore, it prepared a strictly reproducible, cloud-ready execution workflow for NVIDIA A100 GPUs.

### Key Audit Reconciliations:
1. **Reconciliation of Task 8 Dataset Governance**:
   - In Task 8, the 60 evaluation cases were described as "patient-disjoint". Our cross-cohort audit confirms that while the 60 cases are **mutually disjoint among themselves** (60 distinct patients) and strictly disjoint from Task 7's 11 cases and TransVG training splits, they are **NOT disjoint from the prepared MedGemma fine-tuning dataset** (`train_dev` and `val_dev`).
   - Specifically, out of Task 8's 60 cases, **51 patients are in `train_dev`** and **9 patients are in `val_dev`**!
   - Consequently, Task 8 data must be classified strictly as **development evidence**. Evaluating a fine-tuned MedGemma model on Task 8 cases would constitute **severe train-test data leakage (85% overlap with training)**.
   - The **only genuinely independent, unexposed evaluation cohort** is `expanded_test` ($N=117$ across 117 unique patients), which has 0 overlap with train, val, Task 7, or Task 8. In accordance with strict governance policies, `expanded_test` remains **permanently locked, quarantined, and unseen**.
2. **Reconciliation of the Hybrid False-Normal Paradox (36 False Normals vs. 2 Consistency Alerts)**:
   - In Task 9, the hybrid pipeline yielded 36 false-normal cases out of 58 abnormal cases (62.07%), yet the consistency engine only flagged 2 cases.
   - **Mechanism Revealed**:
     - The consistency service was programmed to trigger a conflict banner *only* when the regional visual review explicitly returned **`supported`** on an abnormality that whole-image inference omitted.
     - Out of 60 cases, regional review returned `supported` in exactly **2 cases** (Case 29 pneumothorax and Case 35 pleural effusion). In both cases, the alert banner triggered correctly (`review_required: True`).
     - In the remaining 36 false-normal cases, MedGemma's regional review returned **`unknown` (inconclusive / unverified)** in 32 cases and **`not_supported`** in 4 cases.
     - Because both whole-image and regional review failed to affirm disease, the consistency service classified them as `compatible` (no active contradiction), permitting **36 abnormal patients to proceed without clinical review** (`review_required: False`).
   - This uncovers a critical clinical triage gap: **silent double-omissions bypass the consistency firewall**.
3. **Paired Statistical Comparison (Hybrid vs. Baseline)**:
   - The point estimate for Micro-F1 improved from **0.2162** (Baseline Prompt v3) to **0.2381** (Integrated Hybrid Pipeline), representing a $+0.0216$ gain.
   - However, paired 10,000-sample bootstrap analysis reveals a **95% Confidence Interval of $[-0.0819, +0.1189]$** with an empirical $p$-value of $0.3278$. The improvement is **not statistically significant** on $N=60$ cases.
   - The primary genuine advantage of the Hybrid Pipeline lies in **support hardware identification** (Device-F1: **0.8372** vs 0.5185 in Prompt v4), achieved by isolating apparatus extraction into a specialized flat schema submodule.
4. **Empirical Latency Profile of the 8 Pipeline Sub-components**:
   - Total warm end-to-end latency is **29.92 seconds** (median) and **30.32 seconds** (P95).
   - **95.5% of total runtime is spent in three LLM generation passes**: Whole-Image VLM (12.74s, 42.6%), Regional Review VLM (10.02s, 33.5%), and Device Extraction (5.82s, 19.4%).
   - PyTorch TransVG grounding on MPS takes only **0.47 seconds** (1.56%), image cropping takes **0.054 seconds** (0.18%), and consistency checking takes **0.0015 seconds** (<0.01%).
5. **Readiness for Cloud Fine-Tuning**:
   - All cloud assets have been frozen: [`configs/medgemma_cloud_finetune.yaml`](file:///Users/adityakachhot/interactmed/configs/medgemma_cloud_finetune.yaml), [`scripts/check_cloud_environment.py`](file:///Users/adityakachhot/interactmed/scripts/check_cloud_environment.py), and [`notebooks/medgemma_finetune_colab.ipynb`](file:///Users/adityakachhot/interactmed/notebooks/medgemma_finetune_colab.ipynb).
   - In adherence to research integrity, **we do not claim MedGemma has been fine-tuned**. Production weights remain intact and protected.

---

## Phase 1 — Dataset Governance & Non-Contamination Audit

### Pairwise Patient-Level and Study-Level Overlap Matrix

A comprehensive programmatic audit of all patient IDs, study UIDs, and DICOM image hashes was executed across all five project cohorts:

```
[MIMIC-CXR / Chest ImaGenome Verified Corpus]
  ├── expanded_dev (250 cases / 248 unique patients)
  │     ├── train_dev (200 cases / 199 unique patients)  <── Overlap with Task 8: 51 patients
  │     └── val_dev   (50 cases / 49 unique patients)    <── Overlap with Task 8: 9 patients
  ├── task7_dev (11 cases / 8 unique patients)           <── 0 overlap with train, val, task 8, or test
  └── expanded_test (117 cases / 117 unique patients)    <── 100% QUARANTINED & LOCKED (0 overlap)
```

### Table 1: Complete Cross-Cohort Overlap & Governance Matrix
| Cohort Name | Total Patients | Total Studies | Unique Images | Train Overlap | Val Overlap | Task 7 Overlap | Task 8 Overlap | Locked Test Overlap | Governance Role & Exposure History |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`train_dev`** | 199 | 200 | 200 | 199 | 0 | 0 | **51** | **0** | Supervised fine-tuning training set; 51 patients exposed to Task 8 benchmark |
| **`val_dev`** | 49 | 50 | 50 | 0 | 49 | 0 | **9** | **0** | Validation loss monitoring & early stopping; 9 patients exposed to Task 8 benchmark |
| **Task 7 Dev** | 8 | 11 | 11 | 0 | 0 | 8 | 0 | **0** | Initial development pilot for Prompts v3–v6; disjoint from expanded cohort |
| **Task 8 Dev** | 60 | 60 | 60 | **51** | **9** | 0 | 60 | **0** | Expanded 4-strategy benchmark; drawn from `expanded_dev` (**NOT independent**) |
| **`expanded_test`**| **117** | **117** | **117** | **0** | **0** | **0** | **0** | **117** | **Permanently locked & quarantined benchmark**; zero exposure to training or prompts |

### Governance Conclusion & Policy:
1. **Reconciliation**: Task 8's 60 cases were drawn from `expanded_dev` and contain 51 patients from `train_dev` and 9 patients from `val_dev`. Therefore, Task 8 is **development evidence**. Evaluating a candidate fine-tuned on `train_dev` against Task 8 would result in severe train-test contamination.
2. **Quarantine Policy**: `expanded_test` ($N=117$) is the **sole unpolluted test set**. It must not be opened for prompt engineering, debugging, or hyperparameter selection, and will only be evaluated once a final candidate checkpoint is produced.

The frozen matrix is saved on disk at [`artifacts/medgemma_finetuning/task10_cohort_overlap_matrix.csv`](file:///Users/adityakachhot/interactmed/artifacts/medgemma_finetuning/task10_cohort_overlap_matrix.csv).

---

## Phase 1 — Hybrid False-Normal and Consistency Service Audit

### The 36/58 False-Normal vs. 2 Alert Paradox
In Task 9, the Hybrid Reporting Pipeline achieved a high Device-F1 (0.8372) and overall Micro-F1 (0.2381), but generated **36 false-normal impressions out of 58 abnormal cases (62.07%)**, while the cross-report consistency service only triggered **2 review alerts**.

### Case-Level Safety Category Distribution ($N=60$):
An exhaustive case-by-case audit was conducted across all 60 cases (saved at [`artifacts/medgemma_finetuning/task10_hybrid_case_audit.csv`](file:///Users/adityakachhot/interactmed/artifacts/medgemma_finetuning/task10_hybrid_case_audit.csv)):

| Case Safety Classification | Case Count | % of Cohort | Description of Clinical Behavior | Permits Proceed Without Review? |
| :--- | :---: | :---: | :--- | :---: |
| **Silent False-Normal (Silent Omission)** | **36** | **60.0%** | Whole image normal, device none, regional review `unknown`/`not_supported`. Neither affirms disease. | **YES (VULNERABILITY)** |
| **Partial Capture (Missed Co-occurring)** | **15** | **25.0%** | Device was captured (e.g. NG tube), but co-occurring disease (effusion/edema) omitted. | **YES (Partial Miss)** |
| **Abnormal with Complete Capture** | **5** | **8.3%** | Solitary abnormality was hardware; captured completely and safely. | **YES (Clinically Sound)** |
| **Safety Intercepted (Conflict Banner)** | **2** | **3.3%** | Regional review affirmed pathology (`supported`) where whole-image was negative. | **NO (Review Blocked)** |
| **True Normal Controls** | **2** | **3.3%** | Reference normal studies correctly declared free of disease. | **YES (Accurate Normal)** |

### Table 2: Representative Cases from the Hybrid Case Audit
| Case Index & Safe ID | Patient ID | Reference Abnormal | Ground Truth Findings | Whole-Image Impression | Regional Stance | Reported Hybrid Findings | Missed Findings | Review Required? | Permits Proceed? | Safety Classification |
| :--- | :---: | :---: | :--- | :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **Case 1 (`DEV_027`)** | `p16954495` | True | cardiomegaly | Normal | `unknown` | None | cardiomegaly | False | **Yes** | **Silent False-Normal** |
| **Case 2 (`DEV_040`)** | `p16191519` | True | cardiomegaly; pneumothorax | Normal | `unknown` | None | cardiomegaly; pneumothorax | False | **Yes** | **Silent False-Normal** |
| **Case 11 (`DEV_029`)** | `p16132343` | True | atelectasis; support_devices | Normal | `unknown` | support_devices | atelectasis | False | **Yes** | **Partial Capture** |
| **Case 21 (`DEV_035`)** | `p11844144` | True | support_devices | Normal | `unknown` | support_devices | None | False | **Yes** | **Complete Capture** |
| **Case 29 (`DEV_005`)** | `p10527186` | True | pneumothorax; edema; opacity | Normal | **`supported`** | pneumothorax | edema; opacity | **True** | **NO** | **Safety Intercepted** |
| **Case 35 (`DEV_030`)** | `p14180570` | True | effusion; edema; opacity; devices | Normal | **`supported`** | effusion; devices | edema; opacity | **True** | **NO** | **Safety Intercepted** |
| **Case 55 (`DEV_013`)** | `p15649581` | False | None (normal) | Normal | `not_supported` | None | None | False | **Yes** | **True Normal Control** |

### Mechanism of the Safety Gap:
1. **The Consistency Rule**:
   $$\text{Conflict} \iff (\text{Regional Stance} == \text{"supported"}) \land (\text{Whole-Image Stance} \in \{\text{"normal"}, \text{"unmentioned"}\})$$
2. **The Failure Mode**:
   When MedGemma 4B encounters subtle parenchymal disease (e.g. mild cardiomegaly, faint bibasilar atelectasis, small apical pneumothorax):
   - Whole-Image Prompt v4 defaults to `"No acute cardiopulmonary abnormality"`.
   - Regional Review Prompt v6 outputs an inconclusive text stance (`"unknown"` in 32 cases).
   - Because the regional stance is NOT `"supported"`, the consistency engine sees no explicit contradiction and classifies the status as `"compatible"`.
   - Result: `review_required` remains `False`, allowing a false-negative report to bypass clinician review.
3. **Recommended Clinical Remediation**:
   Introduce an **Uncertainty Triage Trigger**: Whenever a targeted regional review query returns `unknown` or `indeterminate` on an image where clinical suspicion or high-risk indication was provided, the system must set `review_required: True` with an amber advisory banner:
   `"⚠️ ADVISORY: Targeted regional review for <query> was inconclusive. Manual radiologist verification advised."`

---

## Phase 1 — Strategy Comparison & Trade-Off Audit

### Table 3: Comprehensive Multi-Metric Trade-Off Matrix ($N=60$)
| Evaluation Strategy | Micro-F1 | Macro-F1 | Bootstrap 95% CI | Micro-Recall | False-Normal Rate | Finding Omission | Device F1 | Schema Validity | Median Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Prompt v3)** | 0.2162 | **0.1813** | [0.1348 – 0.2995] | 0.1389 | **5.17% (3/58)** | 86.11% | 0.7647 | 100.0% | **15.33 s** |
| **Whole-Image (Prompt v4)** | 0.2059 | 0.0741 | [0.1354 – 0.2727] | 0.1458 | 100.0% (58/58) | 85.42% | 0.5185 | 100.0% | 17.34 s |
| **Specialized Device (Prompt v5)**| 0.2169 | 0.1196 | [0.1370 – 0.2953] | 0.1250 | 62.07% (36/58) | 87.50% | **0.8372** | 100.0% | 31.59 s |
| **Grounding Review (Prompt v6)** | 0.2233 | 0.0961 | [0.1532 – 0.2956] | **0.1597** | **0.00% (0/58)** | **84.03%** | 0.5185 | 100.0% | 44.07 s |
| **Integrated Hybrid Pipeline** | **0.2381** | 0.1416 | **[0.1557 – 0.3147]** | 0.1389 | 62.07% (36/58)* | 86.11% | **0.8372** | 100.0% | 73.92 s |

### Paired Comparison Analysis (Hybrid vs. Baseline):
- $\Delta \text{Micro-F1} = +0.0216$
- **10,000-Sample Paired Bootstrap 95% CI**: $[-0.0819, +0.1189]$
- **Empirical Paired $p$-value**: $0.3278$ ($p > 0.05$, statistically non-significant).
- **Core Clinical Trade-Off**:
  - The Hybrid Pipeline eliminates Prompt v4's disastrous hardware false-positive flood (improving Device-F1 from 0.5185 to 0.8372) and intercepts severe contradictions with TransVG regional crops.
  - However, it increases median latency to 73.92s (Task 8 batch profile) / 29.92s (Task 10 optimized warm profile) and does not bridge the underlying omission gap for subtle parenchymal consolidations without fine-tuning.

---

## Phase 2 — Hybrid Pipeline Latency Profile

A dedicated benchmarking script ([`scripts/profile_hybrid_pipeline_latency.py`](file:///Users/adityakachhot/interactmed/scripts/profile_hybrid_pipeline_latency.py)) measured cold-start and warm inference times across all 8 sub-components of the hybrid pipeline:

### Table 4: Component-by-Component Latency Breakdown
| Pipeline Component | Cold Start (s) | Warm Median (s) | Warm P95 (s) | Warm Mean (s) | Warm Min (s) | Warm Max (s) | % of Warm Pipeline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Whole-Image VLM (Prompt v4)** | 16.3179 | **12.7435** | 13.9652 | 13.0796 | 12.6586 | 14.1726 | **42.60%** |
| **2. Dedicated Device Extraction (v5)**| 6.0487 | **5.8152** | 5.9218 | 5.7112 | 5.2813 | 5.9334 | **19.44%** |
| **3. TransVG Grounding Localization** | 7.9762* | **0.4672** | 0.5123 | 0.4190 | 0.2244 | 0.5169 | **1.56%** |
| **4. Regional Crop Preparation** | 0.0482 | **0.0539** | 0.0737 | 0.0525 | 0.0256 | 0.0766 | **0.18%** |
| **5. Regional Review VLM (Prompt v6)** | 8.6638 | **10.0172** | 11.6974 | 9.9071 | 7.7737 | 11.8202 | **33.48%** |
| **6. Cross-Report Consistency Service**| 0.0050 | **0.0015** | 0.0031 | 0.0018 | 0.0006 | 0.0034 | **0.01%** |
| **7. Report Synthesis and Parsing** | 0.0001 | **0.0000** | 0.0001 | 0.0000 | 0.0000 | 0.0001 | **0.00%** |
| **Total End-to-End Pipeline** | **39.1321** | **29.9158** | **30.3178** | **29.2683** | **26.8823** | **30.3590** | **100.00%** |

*\*Note*: Cold-start TransVG grounding latency includes initial checkpoint loading into Apple Silicon MPS.

### Profiling Observations & Recommended Optimizations:
1. **LLM Dominance (95.5%)**: Three separate Ollama generation passes consume 28.58s of the 29.92s warm total.
2. **Ultra-Fast Grounding (1.56%)**: TransVG-AGPT on Apple Silicon MPS processes the 640×640 multi-pathology tensor in just **0.47 seconds**.
3. **Negligible Consistency Overhead (0.01%)**: Regex extraction and logic evaluation execute in **1.5 milliseconds**.
4. **Recommended Production Optimizations**:
   - **Unified Prompting**: Consolidate whole-image survey and flat device extraction into a single multi-turn forward pass to shave ~5.8 seconds per study.
   - **Conditional Grounding**: Run TransVG and regional VLM review *only* when whole-image findings are equivocal or a specific clinical indication is requested, cutting normal study latency from 29.9s down to 12.7s.
   - **Ollama Model Keep-Alive**: Ensure `keep_alive="30m"` to prevent model eviction between studies.

Saved on disk at [`artifacts/medgemma_finetuning/task10_latency_profile.csv`](file:///Users/adityakachhot/interactmed/artifacts/medgemma_finetuning/task10_latency_profile.csv).

---

## Phase 3 — Cloud Fine-Tuning Preparation & Verification

All necessary cloud training artifacts have been generated, frozen, and verified:
1. **Cloud Configuration**: [`configs/medgemma_cloud_finetune.yaml`](file:///Users/adityakachhot/interactmed/configs/medgemma_cloud_finetune.yaml)
   - Base model: `google/medgemma-4b-it` (Gemma 3 multimodal, 4.3B params).
   - Precision: 4-bit NormalFloat (NF4) in `bfloat16` compute dtype.
   - Target Modules: Language attention projections (`q, k, v, o`), language MLP (`gate, up, down`), and multimodal projector (`linear_1, linear_2`).
   - SigLIP Vision Tower: **Strictly frozen** (27,243,648 parameters).
   - Training: 3 epochs, effective batch size 16 (batch size 1, grad accumulation 16), learning rate $1\times 10^{-4}$ with cosine decay.
   - Checkpointing: Evaluates `eval_loss` per epoch, saves best model, applies `EarlyStoppingCallback(early_stopping_patience=2)`.
2. **Environment Verification Script**: [`scripts/check_cloud_environment.py`](file:///Users/adityakachhot/interactmed/scripts/check_cloud_environment.py)
   - Performs automated assertions before training begins: CUDA availability, compute capability $\ge 8.0$, dedicated VRAM $\ge 40\text{ GB}$, `transformers >= 4.50.0`, bitsandbytes NF4 quantization kernel, Hugging Face token authentication, and dataset partition non-contamination.
3. **Updated Colab Notebook**: [`notebooks/medgemma_finetune_colab.ipynb`](file:///Users/adityakachhot/interactmed/notebooks/medgemma_finetune_colab.ipynb)
   - Complete, self-contained interactive notebook featuring preflight hardware checks, automated dependency installation, single-batch 20-step gradient descent verification, resumable SFTTrainer checkpoints, adapter serialization, and SHA256 checksum generation.

---

## Phase 4 — Predeclared Evaluation Protocol for Cloud Fine-Tuning

Before cloud fine-tuning is launched, the candidate evaluation criteria are permanently frozen:
1. **Checkpoint Selection Protocol**:
   - Model selection is governed **strictly by validation loss (`eval_loss`) on `val_dev`** ($N=50$).
   - Neither Task 8 nor `expanded_test` will be accessed during training.
2. **Predeclared Comparative Metrics**:
   - Per-Finding Precision, Recall, and F1 across all 7 pathology classes.
   - Overall Micro-F1 and Macro-F1 with 95% bootstrap confidence intervals.
   - Finding Omission Rate and False-Normal Rate.
   - Device Precision, Recall, and F1.
   - Schema Validity Rate (100% target).
   - Consistency Contradiction Rate.
   - Median and P95 latency.
3. **One-Shot Unsealing of `expanded_test` ($N=117$)**:
   - The quarantined test set will be evaluated **once and only once** after the candidate checkpoint weights are frozen and SHA256 hashes are verified.
   - No subsequent hyperparameter tuning or prompt adjustments will be permitted following test set unsealing.

---

## Phase 5 — Production Protection & Rollout Decision

- **Active Production Configuration**: Remains locked to Baseline Prompt v3 and TransVG Task 3 Epoch 5 (`checkpoints/multipathology/run_02_continuation_to_6ep/best_model_val_iou.pth`).
- **Research Rollback Path**: Preserved via versioned configuration files (`configs/model_config.yaml`).
- **Clinical Governance Reminder**: All AI-generated reports and visual grounding boxes remain experimental research outputs and must not be presented as definitive medical diagnoses.

---

## Deliverables Summary

1. **Integrity Audit Report**: [`artifacts/medgemma_finetuning/task10_evaluation_integrity_audit.md`](file:///Users/adityakachhot/interactmed/artifacts/medgemma_finetuning/task10_evaluation_integrity_audit.md)
2. **Cohort Overlap Matrix CSV**: [`artifacts/medgemma_finetuning/task10_cohort_overlap_matrix.csv`](file:///Users/adityakachhot/interactmed/artifacts/medgemma_finetuning/task10_cohort_overlap_matrix.csv)
3. **Hybrid Case Audit CSV**: [`artifacts/medgemma_finetuning/task10_hybrid_case_audit.csv`](file:///Users/adityakachhot/interactmed/artifacts/medgemma_finetuning/task10_hybrid_case_audit.csv)
4. **Latency Profile CSV**: [`artifacts/medgemma_finetuning/task10_latency_profile.csv`](file:///Users/adityakachhot/interactmed/artifacts/medgemma_finetuning/task10_latency_profile.csv)
5. **Frozen Cloud Config**: [`configs/medgemma_cloud_finetune.yaml`](file:///Users/adityakachhot/interactmed/configs/medgemma_cloud_finetune.yaml)
6. **Cloud Preflight Script**: [`scripts/check_cloud_environment.py`](file:///Users/adityakachhot/interactmed/scripts/check_cloud_environment.py)
7. **Cloud Colab Notebook**: [`notebooks/medgemma_finetune_colab.ipynb`](file:///Users/adityakachhot/interactmed/notebooks/medgemma_finetune_colab.ipynb)
8. **Cloud Execution Guide**: [`artifacts/medgemma_finetuning/task10_execution_guide.md`](file:///Users/adityakachhot/interactmed/artifacts/medgemma_finetuning/task10_execution_guide.md)
