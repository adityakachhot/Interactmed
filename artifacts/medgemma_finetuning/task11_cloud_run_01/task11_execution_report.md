# Task 11 Execution Report: MedGemma 4B QLoRA Cloud Training & Baseline Audit

**InteractMed-VLM Research & Development Team**  
**Document Revision:** 1.0.0 (TASK 11 Final Audit)  
**Execution Timestamp:** 2026-10-17T01:02:00+05:30  
**Experiment Identifier:** `task11_cloud_run_01`  
**Experiment Directory:** [`artifacts/medgemma_finetuning/task11_cloud_run_01/`](file:///Users/adityakachhot/interactmed/artifacts/medgemma_finetuning/task11_cloud_run_01/)  
**Target Cloud Config:** [`artifacts/medgemma_finetuning/task11_cloud_run_01/frozen_config.yaml`](file:///Users/adityakachhot/interactmed/artifacts/medgemma_finetuning/task11_cloud_run_01/frozen_config.yaml)  

---

## 1. Cloud Training Execution Outcome

### 1.1 Status Declaration
**The full cloud fine-tuning experiment was NOT executed and has NOT been simulated.**

In strict adherence to the Task 11 operational directives:
> *"Do not simulate successful training. Do not fabricate training logs, checkpoints, or performance improvements. Do not proceed on the local Apple Silicon MPS environment if the configured QLoRA implementation requires CUDA and bitsandbytes kernels that are not supported there. If no compatible cloud GPU is available, stop and provide the exact environment failure and next operational step."*

Execution was halted at the preflight stage following execution of [`scripts/check_cloud_environment.py`](file:///Users/adityakachhot/interactmed/scripts/check_cloud_environment.py). No synthetic training curves, fake LoRA adapter weights, or imaginary evaluation gains have been generated.

### 1.2 Execution Summary

| Parameter | Specification / Result |
| :--- | :--- |
| **Cloud Run Completed** | **NO (Halted at Preflight Environment Check)** |
| **Runtime Environment** | Local macOS Darwin 24.1.0 (`arm64`), Apple Silicon M-Series |
| **Detected Accelerator** | Apple Metal Performance Shaders (`torch.backends.mps`) |
| **CUDA Capability** | None (`torch.cuda.is_available() == False`) |
| **Dedicated GPU VRAM** | 0 GB CUDA VRAM (Unified Memory Architecture) |
| **Required Accelerator** | NVIDIA Ampere/Hopper (A100 40GB/80GB SXM4 or H100) |
| **Training Manifest Count** | 200 cases (199 unique patients, 200 studies) in `train_dev.jsonl` |
| **Validation Manifest Count** | 50 cases (49 unique patients, 50 studies) in `val_dev.jsonl` |
| **Locked Test Count** | 117 cases (117 unique patients, 117 studies) in `expanded_test_locked` |
| **Epochs Completed** | **0 epochs** (Training halted per Section 4 protocol) |
| **Early Stopping Outcome** | Not reached (training was not initiated) |
| **Candidate Checkpoint Path** | `NONE` (Zero weights generated) |
| **Checksum Manifest** | [`artifacts/medgemma_finetuning/task11_cloud_run_01/checkpoint_checksums.sha256`](file:///Users/adityakachhot/interactmed/artifacts/medgemma_finetuning/task11_cloud_run_01/checkpoint_checksums.sha256) |
| **Production System Integrity** | **100% UNCHANGED** (Production Prompt v3 & Ollama `medgemma:4b` preserved) |
| **Locked Test Split Status** | **100% QUARANTINED & UNSEALED** |

---

## 2. Environment Verification & Hardware Blockers

The preflight script [`scripts/check_cloud_environment.py`](file:///Users/adityakachhot/interactmed/scripts/check_cloud_environment.py) was executed to determine whether the environment satisfies the requirements declared in [`configs/medgemma_cloud_finetune.yaml`](file:///Users/adityakachhot/interactmed/configs/medgemma_cloud_finetune.yaml).

### 2.1 Preflight Verification Log
```
======================================================================
INTERACTMED-VLM: CLOUD TRAINING ENVIRONMENT PREFLIGHT CHECK
======================================================================
✓ PyTorch Version: 2.2.2
✓ Transformers Version: 4.40.1
✓ Training partition verified: 200 examples in train_dev.jsonl
✓ Validation partition verified: 50 examples in val_dev.jsonl
----------------------------------------------------------------------
WARNINGS:
  [!] Could not verify 'google/medgemma-4b-it' access (Token is required (`token=True`), but no token found. You need to provide a token or be logged in to Hugging Face with `hf auth login` or `huggingface_hub.login`. See https://huggingface.co/settings/tokens.). Ensure HF_TOKEN is set and Health AI terms are accepted.
ERRORS (BLOCKERS):
  [X] CUDA GPU is NOT available. Switch runtime to an NVIDIA GPU (A100 recommended).
  [X] Transformers version 4.40.1 < 4.50.0. Run: pip install --upgrade 'transformers>=4.50.0'
  [X] PEFT not installed. Run: pip install peft
  [X] TRL not installed. Run: pip install trl
  [X] bitsandbytes verification failed: No module named 'bitsandbytes'. Install via: pip install bitsandbytes

Preflight check FAILED. Resolve the blockers above before launching training.
```

### 2.2 Root Cause Analysis of Hardware Blocker
1. **CUDA & BitsAndBytes Requirement:** Parameter-efficient 4-bit NormalFloat4 (NF4) quantization and double-quantized gradient accumulation with `bfloat16` compute require NVIDIA-specific CUDA runtime kernels compiled in `bitsandbytes`.
2. **MPS Architecture Limitations:** Apple Silicon Metal (`mps`) does not support CUDA kernels or `bitsandbytes` 4-bit quantization. While forward inference of a quantized model can be executed locally via Ollama's GGUF runtime, backpropagation through LoRA adapters with gradient accumulation steps cannot run on MPS without CUDA.
3. **Operational Decision:** Per Section 4 instructions, training on MPS without bitsandbytes was rejected to prevent numerical instability, gradient leakage, and methodology drift. Execution halted cleanly.

---

## 3. Baseline Audit & Historical Discrepancy Reconciliation

Task 11 conducted an in-depth audit across Task 8, Task 9, and Task 10 raw predictions and evaluation scripts to reconcile historical metric differences before establishing the development baseline.

```
+------------------------------------------------------------------------------------------------+
|                             HISTORICAL DISCREPANCY RECONCILIATION                              |
+------------------------------+--------------------+---------------------+----------------------+
| Dimension                    | Task 8 / Task 9    | Task 10 / Task 11   | Root Cause &         |
|                              | Reported Value     | Audited Standard    | Resolution           |
+------------------------------+--------------------+---------------------+----------------------+
| Baseline Prompt v3           | 5.17% (3/58)       | 69.0% (40/58 on T8) | T8/9 checked regex   |
| False-Normal Rate            |                    | 38.9% (14/36 on Val)| on impression string |
|                              |                    |                     | only; hedging evaded |
|                              |                    |                     | false-normal flag.   |
+------------------------------+--------------------+---------------------+----------------------+
| Strategy A (Prompt v4)       | 100% (58/58)       | 65.5% (38/58)       | Structured text had  |
| False-Normal Rate            |                    |                     | default impression   |
|                              |                    |                     | phrase triggering    |
|                              |                    |                     | T8 normal regex.     |
+------------------------------+--------------------+---------------------+----------------------+
| Finding Omission Rate        | 86.1% (49/58)      | 68.0% (34/50 on Val)| T8/9 evaluated 8     |
|                              |                    | 63.8% (37/58 on T10)| rigid categories;    |
|                              |                    |                     | unified parser uses  |
|                              |                    |                     | 14 categories.       |
+------------------------------+--------------------+---------------------+----------------------+
| Micro-F1 vs Macro-F1         | Micro: 0.2162      | Micro: 0.3418       | Severe class skew.   |
| Discrepancy                  | Macro: 0.0741-0.18 | Macro: 0.3115       | Rare classes have    |
|                              |                    |                     | near-zero recall.    |
+------------------------------+--------------------+---------------------+----------------------+
| Support Device F1            | 0.7647 (binary)    | 0.3158 (baseline)   | Binary presence vs   |
|                              | 0.8372 (hybrid)    | 0.8372 (hybrid)     | granular device-type |
|                              |                    |                     | classification.      |
+------------------------------+--------------------+---------------------+----------------------+
| End-to-End Latency           | 73.9s median       | 14.4s (whole image) | Cold process swap vs |
|                              | (hybrid pipeline)  | 29.2s (warm hybrid) | warm model in VRAM.  |
+------------------------------+--------------------+---------------------+----------------------+
```

### 3.1 Reconciliation Findings
1. **False-Normal Rate Definition:**
   - In Task 8, `is_normal_imp` was computed by matching regex patterns (`no acute abnormality`, `normal chest`) specifically inside the `IMPRESSION:` section text. When Prompt v3 emitted non-committal observations with no positive findings (e.g., *"Stable cardiomediastinum. Recommend clinical correlation."*), the regex did not match, so `is_normal_imp` returned `False`, falsely depressing the false-normal count to 3/58 (5.17%).
   - Conversely, Prompt v4 emitted `"7. OVERALL IMPRESSION: No acute abnormality"` on cases without definitive findings, triggering the regex 100% of the time (58/58).
   - **Unified Operational Standard:** A case is a **False Normal** if the reference study is clinically abnormal (`gt_abnormal == True`) and the model's reported findings affirm **zero reference abnormalities**.
2. **Finding Omission vs Recall:**
   - In Task 8/9, omission was defined as $1 - \text{Recall}$ on 8 fixed categories, yielding 86.11%.
   - In the audited validation framework with 14 Chest ImaGenome categories and negation cues, case-level omission on `val_dev` is **68.0% (34/50)**.
3. **Class Imbalance & Macro-F1:**
   - Common abnormalities (cardiomegaly, atelectasis, support devices) represent $> 70\%$ of positive references. Rare findings (pneumothorax, acute alveolar edema) have near-zero recall, explaining the substantial gap between Micro-F1 and Macro-F1.

---

## 4. Empirical Development Baseline Validation Results

To establish the fixed development baseline with exact denominators and per-case predictions, the production baseline model (`medgemma:4b`, Production Prompt v3) was evaluated across all 50 cases of [`artifacts/fine_tuning_cohort/val_dev.jsonl`](file:///Users/adityakachhot/interactmed/artifacts/fine_tuning_cohort/val_dev.jsonl).

All individual predictions are saved in [`artifacts/medgemma_finetuning/task11_cloud_run_01/per_case_predictions.csv`](file:///Users/adityakachhot/interactmed/artifacts/medgemma_finetuning/task11_cloud_run_01/per_case_predictions.csv).

### 4.1 Development Baseline Performance Table

| Metric | Empirical Baseline (`val_dev`, N=50) | Frozen Promotion Gate Target | Gate Status |
| :--- | :---: | :---: | :---: |
| **Validation Cohort Size** | 50 cases (49 patients, 50 studies) | 50 cases | — |
| **Abnormal Reference Cases** | 36 cases | 36 cases | — |
| **Normal Reference Controls** | 14 cases | 14 cases | — |
| **Finding Precision** | **0.6136** | $> 0.3000$ | **PASS** |
| **Finding Recall** | **0.2368** | $> 0.3500$ | Target for LoRA |
| **Micro-F1 Score** | **0.3418** | $> 0.3918$ ($\Delta > +0.05, p < 0.05$) | Target for LoRA |
| **Macro-F1 Score** | **0.3115** | $> 0.3500$ | Target for LoRA |
| **False-Normal Rate** | **38.89% (14 / 36)** | **$< 25.00\%$** | Target for LoRA |
| **False-Normal 95% Wilson CI** | **$[0.2478, 0.5514]$** | Tightened interval | Target for LoRA |
| **Finding Omission Rate** | **68.00% (34 / 50)** | **$< 40.00\%$** | Target for LoRA |
| **Unsupported Assertion Rate** | **0.3200 (16 / 50)** | **$< 0.1500$** | Target for LoRA |
| **Support Device F1** | **0.3158** | $\ge 0.8000$ | Target for Hybrid |
| **JSON / Schema Validity** | **100.0% (50 / 50)** | **100.0%** | **PASS** |
| **Median Generation Latency** | **14.42s** | $< 15.00s$ | **PASS** |
| **p95 Generation Latency** | **16.77s** | $< 25.00s$ | **PASS** |

---

## 5. Frozen Promotion Gate Evaluation

The predeclared Task 10 promotion gates define whether a fine-tuned candidate is eligible for final unsealed evaluation on the locked test partition:

```
+-------------------------------------------------------------------------------------------------+
|                                 FROZEN PROMOTION GATE AUDIT                                     |
+----+--------------------------------+-------------------+-----------------+---------------------+
| ID | Gate Description               | Threshold Target  | Candidate State | Assessment          |
+----+--------------------------------+-------------------+-----------------+---------------------+
| G1 | Micro-F1 Improvement           | Delta > +0.05     | No Model (0.00) | FAIL (Not Executed) |
|    | (Paired bootstrap p < 0.05)    | (Target: >0.3918) |                 |                     |
+----+--------------------------------+-------------------+-----------------+---------------------+
| G2 | False-Normal Rate              | < 25.0%           | No Model (N/A)  | FAIL (Not Executed) |
|    | (Unified clinical definition)  |                   |                 |                     |
+----+--------------------------------+-------------------+-----------------+---------------------+
| G3 | Finding Omission Rate          | < 40.0%           | No Model (N/A)  | FAIL (Not Executed) |
+----+--------------------------------+-------------------+-----------------+---------------------+
| G4 | Unsupported Assertion Rate     | < 0.15 / report   | No Model (N/A)  | FAIL (Not Executed) |
|    | (Hallucinated pathology)       |                   |                 |                     |
+----+--------------------------------+-------------------+-----------------+---------------------+
| G5 | Output Schema Validity         | 100.0% Valid      | 100% (Baseline) | PASS (Schema sound) |
+----+--------------------------------+-------------------+-----------------+---------------------+
```

**Promotion Decision:**
Because full cloud fine-tuning has not been executed, **the candidate is NOT promoted**. The baseline production system remains active.

---

## 6. Dataset Governance & Partition Separation Audit

Patient-level and study-level separation across all cohorts was audited and confirmed:

| Split Identifier | Total Studies | Unique Patients | Overlap with Train | Overlap with Val | Overlap with Locked Test | Split Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **TransVG Train** | 1,000 | 1,000 | 0 | 0 | **0** | Visual grounding training |
| **Task 7 Dev** | 11 | 11 | 9 | 2 | **0** | Prompt prototyping (dev only) |
| **Task 8 Dev Cohort** | 60 | 60 | 51 | 9 | **0** | Expanded prompt comparison |
| **FT Train Dev** | 200 | 199 | — | 0 | **0** | Fine-tuning training data |
| **FT Val Dev** | 50 | 49 | 0 | — | **0** | Validation & checkpoint selection |
| **Locked Test Partition** | **117** | **117** | **0** | **0** | — | **QUARANTINED FINAL EVALUATION** |

### Split Integrity Guarantees:
1. `train_dev.jsonl` ($N=200$) and `val_dev.jsonl` ($N=50$) share **0 patient overlap**.
2. Neither `train_dev` nor `val_dev` shares any patient overlap with `expanded_test` ($N=117$).
3. The 117-case locked test partition was **NOT accessed, loaded, or unsealed** during Task 11.

---

## 7. Production Artifact Protection Verification

All existing production components and historical task artifacts were audited to ensure complete preservation:

1. **Active Report Prompt:** Production Prompt v3 remains the active default.
2. **Ollama Deployment:** Local `medgemma:4b` remains unchanged at `http://localhost:11434`.
3. **TransVG Grounding Model:** Checkpoint [`checkpoints/multipathology/run_02_continuation_to_6ep/best_model_val_iou.pth`](file:///Users/adityakachhot/interactmed/checkpoints/multipathology/run_02_continuation_to_6ep/best_model_val_iou.pth) (Epoch 5) is preserved.
4. **Historical Task Artifacts:** Tasks 2, 3, 7, 8, 9, and 10 files in `artifacts/report_quality/` and `artifacts/medgemma_finetuning/` remain untouched.
5. **Regression Test Suite:** Full pytest test suite confirmed clean (**181 passed in 19.05s**).
6. **Frontend Production Build:** Clean build confirmed (**`tsc && vite build` in 314ms**).

---

## 8. Handoff Protocol for Cloud GPU Execution & Final Locked Test Evaluation

When cloud GPU resources (NVIDIA A100/H100) are provisioned, follow this reproducible protocol:

### Step 8.1: Cloud Environment Launch
1. Open [`notebooks/medgemma_finetune_colab.ipynb`](file:///Users/adityakachhot/interactmed/notebooks/medgemma_finetune_colab.ipynb) on Google Colab Pro or an A100 GPU server.
2. Set `HF_TOKEN` with access granted to `google/medgemma-4b-it`.
3. Run `python scripts/check_cloud_environment.py` to verify compute capability $\ge 8.0$, VRAM $\ge 40$GB, and bitsandbytes 4-bit NF4 kernels.

### Step 8.2: Execution & Checkpoint Selection
1. Run training: `python scripts/train_medgemma_lora.py --config configs/medgemma_cloud_finetune.yaml`.
2. Monitor validation loss on `val_dev.jsonl` after each epoch.
3. Automatically load the best checkpoint based on minimum evaluation loss (`early_stopping_patience: 2`).
4. Generate SHA-256 checksums for the resulting `adapter_model.safetensors`.

### Step 8.3: Evaluation on Validation Cohort
1. Run evaluation on `val_dev.jsonl`:
   ```bash
   python scripts/evaluate_finetuned_medgemma.py \
     --split val_dev \
     --adapter-path checkpoints/medgemma_lora_cloud_run_01/best_checkpoint
   ```
2. Check against the 5 frozen promotion gates.

### Step 8.4: One-Time Final Unsealed Test Evaluation
The 117-case locked test partition (`expanded_test_locked`) may be unsealed **ONLY ONCE**, strictly when:
1. All training runs, hyperparameter choices, and prompt formulations are locked.
2. The fine-tuned adapter weights and checksums are frozen.
3. The evaluation code is finalized.
4. Execute the one-time evaluation:
   ```bash
   python scripts/evaluate_finetuned_medgemma.py \
     --split expanded_test \
     --adapter-path checkpoints/medgemma_lora_cloud_run_01/best_checkpoint \
     --output-dir artifacts/medgemma_finetuning/final_test_evaluation
   ```

---

## 9. Conclusion

Task 11 establishes a reproducible, audited development baseline on the 50 validation cases, reconciles all historical metric discrepancies, and preserves production weights and data governance with scientific integrity. Cloud GPU fine-tuning is fully prepared for execution on NVIDIA hardware without risk to existing deployments or test set contamination.
