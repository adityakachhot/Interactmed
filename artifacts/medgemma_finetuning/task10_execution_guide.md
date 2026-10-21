# Task 10: MedGemma 4B LoRA Cloud Fine-Tuning Execution Guide

**InteractMed-VLM Research & Development Team**  
**Document Revision:** 1.0.0 (TASK 10 Frozen Release)  
**Target Environment:** NVIDIA Ampere/Hopper GPU (A100 40GB/80GB, H100 80GB, or L4 24GB with batch reduction)  
**Base Model:** `google/medgemma-4b-it` (Hugging Face Hub gated model)  
**Configuration File:** [`configs/medgemma_cloud_finetune.yaml`](file:///Users/adityakachhot/interactmed/configs/medgemma_cloud_finetune.yaml)  
**Preflight Verification Script:** [`scripts/check_cloud_environment.py`](file:///Users/adityakachhot/interactmed/scripts/check_cloud_environment.py)  
**Interactive Notebook:** [`notebooks/medgemma_finetune_colab.ipynb`](file:///Users/adityakachhot/interactmed/notebooks/medgemma_finetune_colab.ipynb)  

---

## 1. Executive Summary & Purpose

This execution guide defines the end-to-end operational protocol for training, validating, serializing, and evaluating a parameter-efficient Low-Rank Adaptation (LoRA / QLoRA) adapter for **MedGemma 4B** (`google/medgemma-4b-it`). 

Following the Task 9 and Task 10 audits:
1. **Local Apple Silicon Environment Constraints:** Local execution is limited to inference via Ollama and MPS PyTorch. Bitsandbytes 4-bit CUDA quantization and bfloat16 gradient backpropagation require an NVIDIA GPU with compute capability $\ge 8.0$.
2. **Strict Production Isolation:** This fine-tuning workflow produces an independent experimental adapter (`checkpoints/medgemma_lora_cloud_run_01`). It does **not** alter, overwrite, or replace the active production Ollama model (`medgemma:4b`), production prompts (v3), or TransVG grounding weights.
3. **Quarantine of Test Partition:** The 117-patient locked test partition (`expanded_test`) is strictly quarantined and must **never** be used for training, hyperparameter tuning, prompt development, or checkpoint selection.
4. **Validation Integrity:** Training is performed exclusively on `train_dev.jsonl` ($N=200$, 200 patients) and evaluated for checkpoint selection exclusively on `val_dev.jsonl` ($N=50$, 50 patients).

---

## 2. Hardware and Software Prerequisites

### 2.1 Compute Infrastructure

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **GPU Accelerator** | NVIDIA A100 (40 GB VRAM) | NVIDIA A100 (80 GB SXM4) or H100 |
| **Compute Capability** | CUDA $\ge 8.0$ (Ampere / Hopper) | CUDA $8.0$ or $9.0$ |
| **Host System RAM** | 32 GB RAM | 64+ GB RAM |
| **Disk Storage** | 50 GB free NVMe/SSD space | 100 GB free space |
| **CUDA Driver** | CUDA 12.1 or 12.4 | CUDA 12.4+ |
| **Operating System** | Ubuntu 22.04 LTS / Debian 11 | Ubuntu 22.04 LTS / Colab Enterprise |

> [!CAUTION]
> GPUs with compute capability $< 8.0$ (such as NVIDIA T4, V100, or K80) do not support native hardware `bfloat16` math. Running QLoRA on these GPUs will either trigger kernel crashes or cause numerical divergence under mixed-precision gradient accumulation.

### 2.2 Hugging Face Access & Health AI Terms
MedGemma 4B is a gated medical vision-language model. Before launching:
1. Log in to [Hugging Face](https://huggingface.co/google/medgemma-4b-it).
2. Request and obtain access by acknowledging the Google Health AI developer terms and clinical research disclaimers.
3. Generate a Hugging Face User Access Token with **read** permissions.
4. Export the token into your shell or Colab environment:
   ```bash
   export HF_TOKEN="hf_your_personal_access_token_here"
   ```

---

## 3. Directory & Partition Structure

Ensure the following directory structure is present in the working environment:

```
InteractMed-VLM/
├── configs/
│   ├── medgemma_finetune.yaml         # Local development reference config
│   └── medgemma_cloud_finetune.yaml   # FROZEN A100 Cloud Training Config
├── scripts/
│   ├── check_cloud_environment.py     # Environment preflight validator
│   ├── train_medgemma_lora.py         # Training engine (SFTTrainer / PEFT)
│   ├── evaluate_finetuned_medgemma.py # Pre-declared evaluation harness
│   └── prepare_medgemma_finetuning_data.py
├── notebooks/
│   └── medgemma_finetune_colab.ipynb  # Interactive Google Colab notebook
├── artifacts/
│   ├── fine_tuning_cohort/
│   │   ├── train_dev.jsonl            # 200 training cases (200 patients)
│   │   └── val_dev.jsonl              # 50 validation cases (50 patients)
│   └── medgemma_finetuning/
│       ├── task9_run_01/
│       │   └── frozen_partitions_manifest.json
│       ├── task10_cohort_overlap_matrix.csv
│       ├── task10_hybrid_case_audit.csv
│       ├── task10_latency_profile.csv
│       ├── task10_evaluation_integrity_audit.md
│       └── task10_execution_guide.md
└── checkpoints/
    └── medgemma_lora_cloud_run_01/    # Checkpoint output directory
```

---

## 4. Preflight Verification Workflow

Before initiating full training, run the automated verification suite to eliminate configuration, kernel, or permission failures.

### Step 4.1: Run Environment Sanity Check
Execute [`scripts/check_cloud_environment.py`](file:///Users/adityakachhot/interactmed/scripts/check_cloud_environment.py):

```bash
python scripts/check_cloud_environment.py
```

**Expected Successful Output:**
```
======================================================================
INTERACTMED-VLM: CLOUD TRAINING ENVIRONMENT PREFLIGHT CHECK
======================================================================
✓ PyTorch Version: 2.4.0+cu121
✓ Detected GPU: NVIDIA A100-SXM4-80GB
✓ Compute Capability: 8.0
✓ Total VRAM: 79.15 GB
✓ Transformers Version: 4.50.0
✓ PEFT Version: 0.14.0
✓ TRL Version: 0.12.0
✓ bitsandbytes Version: 0.45.0
✓ bitsandbytes 4-bit NF4 kernel verified on CUDA accelerator.
✓ Authenticated Hugging Face User: verified_researcher
✓ Access to gated model 'google/medgemma-4b-it' verified.
✓ Training partition verified: 200 examples in train_dev.jsonl
✓ Validation partition verified: 50 examples in val_dev.jsonl
----------------------------------------------------------------------
ALL CLOUD PREFLIGHT CHECKS PASSED SUCCESSFULLY!
The environment is fully validated for MedGemma 4B Tier 2 LoRA fine-tuning.
```

If the script returns any `[X]` blockers, resolve them before proceeding.

### Step 4.2: Execute 20-Step Numerical Preflight Gradient Check
Run the gradient sanity check to verify loss computation, backpropagation through the multimodal projector, and optimizer step execution:

```bash
python scripts/train_medgemma_lora.py \
  --config configs/medgemma_cloud_finetune.yaml \
  --preflight_only
```

**Verification Criteria:**
- Initial cross-entropy loss must be between $2.0$ and $4.5$.
- Gradients must flow to `target_modules` (`q_proj`, `v_proj`, `multi_modal_projector.linear_1`, `multi_modal_projector.linear_2`).
- Gradient norm across 20 steps must remain bounded ($\le 0.3$).
- Zero `NaN` or `Inf` loss values.

---

## 5. Training Execution Protocol

You may execute training via **Google Colab** or via **Cloud CLI / Headless Server**.

### Option A: Google Colab Pro / Pro+
1. Upload the repository or mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/InteractMed-VLM
   ```
2. Open [`notebooks/medgemma_finetune_colab.ipynb`](file:///Users/adityakachhot/interactmed/notebooks/medgemma_finetune_colab.ipynb).
3. Under **Runtime > Change runtime type**, select **A100 GPU** and **High-RAM**.
4. Set secret `HF_TOKEN` in Colab's Key Management tab.
5. Run Section 1 (Environment & Dependencies).
6. Run Section 2 (Preflight Verification).
7. Run Section 3 (Execute Cloud Fine-Tuning).
8. Run Section 4 (Checkpoint Packaging & Checksum Generation).

### Option B: Cloud CLI / Headless Instance (Lambda Labs, RunPod, GCP, AWS)
Run the training script in a persistent terminal session (`tmux` or `screen`):

```bash
# 1. Start a persistent session
tmux new -s medgemma_train

# 2. Activate environment
conda activate medgemma_env

# 3. Export HF Token
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# 4. Launch Training
python scripts/train_medgemma_lora.py \
  --config configs/medgemma_cloud_finetune.yaml \
  2>&1 | tee checkpoints/medgemma_lora_cloud_run_01/training.log
```

### 5.1 Monitoring Training with TensorBoard
Launch TensorBoard to inspect loss curves, learning rate schedule, and GPU metrics in real time:

```bash
tensorboard --logdir checkpoints/medgemma_lora_cloud_run_01/runs --port 6006
```

### 5.2 Resuming an Interrupted Run
If the cloud instance is preempted or disconnected, resume from the latest saved checkpoint by passing `--resume_from_checkpoint`:

```bash
python scripts/train_medgemma_lora.py \
  --config configs/medgemma_cloud_finetune.yaml \
  --resume_from_checkpoint checkpoints/medgemma_lora_cloud_run_01/checkpoint-25
```

---

## 6. Checkpoint Selection & Artifact Packaging

### 6.1 Validation Selection Criteria
- **Validation Loss:** Select the checkpoint that minimizes validation loss on `val_dev.jsonl` (`load_best_model_at_end: true`).
- **Early Stopping:** If validation loss fails to decrease for 2 consecutive evaluation epochs (`early_stopping_patience: 2`), training will automatically terminate to prevent overfitting.
- **Do NOT Select Solely on Training Loss:** A model whose training loss drops while validation loss diverges is overfitted and must be rejected.

### 6.2 Serialization Checklist
Upon completion, verify that the best checkpoint directory contains:
- `adapter_model.safetensors` (LoRA adapter weights)
- `adapter_config.json` (PEFT adapter configuration)
- `training_args.bin` (Hugging Face training arguments)
- `tokenizer_config.json`, `special_tokens_map.json`, `tokenizer.json`
- `preprocessor_config.json` (SigLIP processor settings)
- `training_manifest.json` (Execution metadata, commit hash, timestamp)

Generate SHA-256 checksums for reproducibility:
```bash
cd checkpoints/medgemma_lora_cloud_run_01/best_checkpoint
sha256sum adapter_model.safetensors adapter_config.json > checksums.sha256
```

---

## 7. Pre-Declared Post-Training Evaluation Protocol

Evaluate the newly trained LoRA candidate against the baseline using the frozen evaluation harness:

```bash
python scripts/evaluate_finetuned_medgemma.py \
  --adapter_path checkpoints/medgemma_lora_cloud_run_01/best_checkpoint \
  --val_file artifacts/fine_tuning_cohort/val_dev.jsonl \
  --output_dir artifacts/medgemma_finetuning/task10_cloud_run_eval
```

### 7.1 Pre-Declared Decision Gate Metrics

| Metric | Target / Acceptance Threshold | Rationale |
| :--- | :--- | :--- |
| **Micro-F1** | Statistically significant improvement ($\Delta \text{Micro-F1} > 0.05, p < 0.05$) | Validates global abnormality capture across all findings |
| **False-Normal Rate** | $< 25\%$ (compared to baseline $> 60\%$) | Critical clinical safety threshold |
| **Finding Omission Rate** | $< 40\%$ (compared to baseline $> 65\%$) | Minimizes silent misses on reference abnormalities |
| **Unsupported Assertion Rate** | $< 0.15$ per report | Prevents model hallucination of ungrounded findings |
| **Contradiction Rate** | $< 0.05$ per report | Ensures factual consistency across sections |
| **Device Recognition F1** | $\ge 0.80$ | Maintains hardware detection parity |
| **JSON Schema Validity** | $100\%$ valid parse rate | Mandatory for downstream automated triage routing |
| **Inference Latency** | Median $< 15$s on GPU | Preserves clinical interactive responsiveness |

### 7.2 Locked Test Partition Protocol
The 117-case locked test partition (`expanded_test`) must be unsealed **ONLY ONCE**, when:
1. All training epochs are finished.
2. The best checkpoint has been finalized and its weights frozen with SHA-256 hashes.
3. The prompt template, decoding parameters (temperature = 0.0), and evaluation script are fully frozen.
4. No further model adjustments, hyperparameter tuning, or prompt tweaks will be made.

---

## 8. Rollback and Production Protection Safeguards

1. **Active Production System Preserved:**
   - Production Prompt: Production Prompt v3
   - Primary Inference Engine: Ollama `medgemma:4b` (`http://localhost:11434`)
   - Grounding Engine: TransVG-AGPT Task 3 Epoch 5 (`checkpoints/multipathology/run_02_continuation_to_6ep/best_model_val_iou.pth`)
2. **Rollback Procedure:**
   If the experimental cloud LoRA model demonstrates any degradation in clinical safety, hallucination rate, or schema compliance, discard the candidate. The production backend will continue operating on the baseline configuration without interruption.
3. **Research Disclaimer:**
   All outputs generated by InteractMed-VLM, including those from fine-tuned candidates, are strictly research prototypes intended for developer evaluation and workflow profiling. They are **not** certified diagnostic medical devices.
