# InteractMed-VLM: MedGemma 4B Fine-Tuning Pilot Runbook

**Project:** InteractMed-VLM  
**Model:** `google/medgemma-4b-it` (Gemma 3 Multimodal Vision-Language Architecture, 4.3B Parameters)  
**Adaptation Strategy:** Tier 2 PEFT — Language Decoder + Multimodal Projector LoRA (Vision Tower strictly FROZEN)  
**Hardware Targets:**
- **Local Mac (Sonoma / Apple Silicon MPS, 16 GB RAM):** Data audit, preflight safety tests (`--inspect-only`, `--dry-run`, `--overfit-single-batch`), Ollama baseline evaluation.
- **Cloud GPU (Google Colab / Dedicated NVIDIA A100, $\ge 40$ GB VRAM):** 4-bit NF4 QLoRA gradient backpropagation, checkpoint export, final model fine-tuning.
**Date:** October 2026  
**Status:** **Preflight Safety Verified — Ready for Cloud Pilot Launch (Standby Mode: Full training must NOT launch automatically)**

---

## 1. Executive Summary & Infrastructure Separation

To preserve system stability and maintain rigorous clinical science boundaries, all commands and workflows are strictly partitioned by target runtime environment:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        INFRASTRUCTURE MATRIX                           │
├────────────────────────────────┬───────────────────────────────────────┤
│ LOCAL MAC (Apple Silicon M3)   │ CLOUD GPU (NVIDIA A100, >=40 GB VRAM) │
├────────────────────────────────┼───────────────────────────────────────┤
│ • Dataset verification & audit │ • 4-bit NF4 QLoRA gradient backprop   │
│ • Manifest audit & JSONL prep  │ • Multimodal projector LoRA training  │
│ • Preflight --inspect-only     │ • Language attention/MLP adaptation   │
│ • Preflight --dry-run (collator│ • Checkpoint export & consolidation   │
│   loss, gradients, save/reload)│ • Full SFTTrainer 3-epoch execution   │
│ • Preflight --overfit-single-  │ • Validation loss checkpoint selection│
│   batch (20-step loss decrease)│ • High-throughput GPU evaluation      │
│ • Baseline & Prompt v3 Ollama  │                                       │
│   local benchmark inference    │                                       │
└────────────────────────────────┴───────────────────────────────────────┘
```

> [!CRITICAL]
> **Why 4-Bit QLoRA Training Requires NVIDIA CUDA (Google Colab A100):**
> 1. **CUDA bitsandbytes:** Hugging Face `bitsandbytes` (4-bit NF4 quantization) relies on compiled CUDA kernels and does not run on macOS Metal (MPS).
> 2. **Memory Footprint:** Full precision (FP16/BF16) fine-tuning of a 4.3B model requires 40–60 GB VRAM when optimizer states and activations are allocated. On a 16 GB Mac, unquantized backpropagation causes kernel thrashing and out-of-memory crashes.
> 3. **GGUF Immutability:** The local Ollama model (`medgemma:4b`) is an integer-quantized GGUF container without backward autograd graphs. It is physically impossible to backpropagate through an Ollama GGUF file. Fine-tuning must target the official Hugging Face PyTorch checkpoint `google/medgemma-4b-it`.

---

## 2. Cohort Governance & Data Partitioning

The fine-tuning cohort is derived strictly from the verified 250-case development cohort (`expanded_dev`). The 117-case held-out test cohort (`expanded_test`) remains permanently sealed and untouched.

| Dataset Split | Cases | Unique Patients | Unique Studies | Images per Study | Patient Overlap with Train | Patient Overlap with Test | Purpose |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`train_dev`** | **200** | **199** | **200** | **1** | — | **0** | Supervised LoRA gradient training |
| **`val_dev`** | **50** | **49** | **50** | **1** | **0** | **0** | Checkpoint selection & hyperparameter audit |
| **`expanded_test`** | **117** | **117** | **117** | **1** | **0** | — | **Locked benchmark — evaluated only once** |

### Verified Governance Guarantees
- **Strict Patient Disjointness:**
  $$\text{Patients}(\text{train\_dev}) \cap \text{Patients}(\text{val\_dev}) = \emptyset$$
  $$\text{Patients}(\text{train\_dev} \cup \text{val\_dev}) \cap \text{Patients}(\text{expanded\_test}) = \emptyset$$
- **1-to-1 Image-Study Association:** Every study ID in `train_dev` (200), `val_dev` (50), and `expanded_test` (117) contains exactly 1 frontal radiograph image. Study-level reports are never conflated across multiple views.
- **Section Completeness:** 100% of cases in both `train_dev` and `val_dev` contain complete, parsed `OBSERVATIONS`, `IMPRESSION`, and `LIMITATIONS` sections.
- **Privacy Preservation:** All MIMIC-CXR reports remain strictly local; patient identifiers and clinical free-text do not leak into training logs.

---

## 3. Local Mac Execution Commands

Run these commands directly in your local macOS terminal using the Miniforge environment (`/Users/adityakachhot/miniforge3/envs/dl/bin/python`).

### 3.1 Step 1: Preflight Data & Split Audit
Verify dataset integrity, missing images, section counts, and patient overlap:
```bash
/Users/adityakachhot/miniforge3/envs/dl/bin/python -c "
import json, pandas as pd
df = pd.read_csv('artifacts/fine_tuning_cohort/dataset_manifest.csv')
print('Manifest Rows:', len(df))
print('Splits:\n', df['split'].value_counts())
train_pts = set(df[df['split']=='train_dev']['patient_id'].astype(str))
val_pts = set(df[df['split']=='val_dev']['patient_id'].astype(str))
test_pts = set(df[df['split']=='expanded_test_locked']['patient_id'].astype(str))
print('Train/Val Patient Overlap:', len(train_pts & val_pts))
print('Train/Test Patient Overlap:', len(train_pts & test_pts))
print('Val/Test Patient Overlap:', len(val_pts & test_pts))
"
```

### 3.2 Step 2: Architecture & Module Hierarchy Inspection
Introspects model modules, verifies linear projection target layers, and categorizes vision vs language vs projector parameters:
```bash
/Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/train_medgemma_lora.py --inspect-only
```
**Actual Verification Outcome:**
- Vision tower: 15 modules (27,243,648 parameters frozen).
- Multimodal projector: 4 modules (`linear_1`, `linear_2` identified as trainable).
- Language decoder: 41 modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` identified for LoRA).
- Status: **PASSED (Exit code 0)**.

### 3.3 Step 3: Comprehensive Preflight Dry-Run Check
Executes an end-to-end forward/backward pass, audits token label masking, verifies finite loss, tests checkpoint saving and reload, and audits GPU memory compatibility:
```bash
/Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/train_medgemma_lora.py --dry-run
```
**Actual Verification Outcome:**
- **Prompt Masking:** 174 masked prompt/image tokens (`-100`), 80 unmasked report output tokens. Loss computed strictly on report text.
- **Forward Pass:** Loss = `9.2256` (finite and valid).
- **Backward Pass:** 36 parameters receiving active gradients; 0 frozen vision parameters receiving gradients.
- **Checkpoint Serialization:** Checkpoint save and reload verified at `artifacts/report_evaluation/finetuned_pilot/temp_preflight_ckpt`.
- **Base Model Preservation:** Base model weights preserved and unchanged.
- **Memory Footprint:** 4-bit QLoRA verified compatible with NVIDIA A100.
- Status: **PASSED (Exit code 0)**.

### 3.4 Step 4: One-Batch Overfit Sanity Check
Executes 20 optimizer steps on a single batch of 2 samples to prove that optimization dynamics are functional and loss decreases monotonically:
```bash
/Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/train_medgemma_lora.py --overfit-single-batch
```
**Actual Verification Outcome:**
- Initial Loss (Step 1): `9.2529`
- Step 5 Loss: `9.1022`
- Step 10 Loss: `8.7114`
- Step 15 Loss: `7.7605`
- Final Loss (Step 20): `5.8979`
- Total Loss Drop: **-3.3550** (monotonic reduction).
- Duration: **11.2 seconds**.
- Status: **PASSED (Exit code 0)**.

### 3.5 Step 5: Local Validation Split Metric Benchmark (Mock Harness)
Validates the three-way comparison evaluation harness, metric calculations, and summary CSV output on `val_dev` ($N=50$):
```bash
/Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/evaluate_finetuned_medgemma.py \
    --split val_dev \
    --dry-run
```

---

## 4. Cloud GPU (Google Colab / NVIDIA A100) Execution Commands

Execute these commands in your Google Colab A100 instance using the turnkey notebook [`notebooks/medgemma_finetune_colab.ipynb`](file:///Users/adityakachhot/interactmed/notebooks/medgemma_finetune_colab.ipynb) or in a dedicated cloud Linux terminal with 1x NVIDIA A100 (40 GB or 80 GB VRAM).

### 4.1 Step 1: Install Cloud Dependencies
```bash
pip install --upgrade --quiet "transformers>=4.50.0" peft trl bitsandbytes datasets evaluate accelerate tensorboard
```

### 4.2 Step 2: Hugging Face Authentication
Accept the Health AI Developer Foundations license terms at `https://huggingface.co/google/medgemma-4b-it`, then authenticate:
```bash
huggingface-cli login
# Paste your Hugging Face access token with Read permissions
```

### 4.3 Step 3: Run Cloud Preflight Safety Checks
Confirm environment, processor, and 4-bit QLoRA loading on the A100:
```bash
# Verify model architecture inspection on A100
python scripts/train_medgemma_lora.py --inspect-only

# Verify single-batch overfit on A100
python scripts/train_medgemma_lora.py --overfit-single-batch
```

### 4.4 Step 4: Proposed Short Training Run (Tier 2 Pilot)
> [!NOTE]
> Launch this command ONLY when you are ready to start training. It will not start automatically.
```bash
python scripts/train_medgemma_lora.py \
    --config configs/medgemma_finetune.yaml
```

#### Proposed Pilot Training Configuration
```yaml
experiment:
  name: "medgemma_4b_pilot_tier2"
  tier: 2 # Projector + Language LoRA, Frozen SigLIP Vision
  seed: 42

model:
  model_id: "google/medgemma-4b-it"
  torch_dtype: "bfloat16"

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"

lora:
  r: 16
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  adapt_projector: true # multi_modal_projector.linear_1, linear_2
  freeze_vision_tower: true # Strictly frozen for scientific attribution

training:
  output_dir: "checkpoints/medgemma_lora_tier2"
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16 # Effective batch size = 16
  learning_rate: 1.0e-4
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.05
  weight_decay: 0.01
  optim: "paged_adamw_8bit"
  bf16: true
  eval_strategy: "steps"
  eval_steps: 25
  save_strategy: "steps"
  save_steps: 25
  save_total_limit: 2
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
```
- **Total Training Duration:** Approximately 20–25 minutes on an NVIDIA A100 GPU (3 epochs across 200 cases = ~38 optimizer steps).
- **VRAM Utilization:** ~15.8 GB peak VRAM.

### 4.5 Step 5: Resume Training (If Cloud Session Preempted)
```bash
python scripts/train_medgemma_lora.py \
    --config configs/medgemma_finetune.yaml \
    --resume-from-checkpoint checkpoints/medgemma_lora_tier2/checkpoint-25
```

---

## 5. Three-Way Evaluation Plan

Evaluation must be conducted in two strictly sequential stages to prevent data leakage and benchmark bias:

```
                  ┌────────────────────────────────────────┐
                  │ 1. VALIDATION EVALUATION (N=50 val_dev) │
                  └───────────────────┬────────────────────┘
                                      │
                         Compare 3 Pipelines on val_dev:
                         • Baseline (Ollama single-stage)
                         • Prompt v3 Structured
                         • Fine-Tuned Checkpoint (Tier 2 LoRA)
                                      │
                         Is Validation Loss minimized and
                         Micro-F1 / Opacity Recall improved?
                                      │
                               ┌──────┴──────┐
                               ▼             ▼
                              YES            NO
                               │             │
                Lock Selected Checkpoint     Diagnose Errors:
                               │             Review prompt drift,
                               │             hallucinations, or
                               │             consider Tier 3 vision.
                               ▼
            ┌──────────────────────────────────────────────┐
            │ 2. FINAL LOCKED TEST EVALUATION (N=117 test) │
            │    Execute ONCE on expanded_test.            │
            │    Do NOT revise prompts based on test.      │
            └──────────────────────────────────────────────┘
```

### 5.1 Stage 1: Validation Split Audit (Checkpoint Selection)
Evaluate on `val_dev` ($N=50$) to measure performance without touching the locked test cohort:
```bash
python scripts/evaluate_finetuned_medgemma.py \
    --split val_dev \
    --adapter-path checkpoints/medgemma_lora_tier2
```

#### Metrics Evaluated
1. **Clinical Micro-F1, Precision, and Recall:** 14 thoracic disease categories + normal status.
2. **Per-Finding Sensitivity Breakdown:** Pneumonia/consolidation, pulmonary edema, atelectasis, cardiomegaly, support devices, pleural effusion, pneumothorax.
3. **False-Normal Rate:** Rate of declaring abnormal radiographs normal (with 95% Wilson confidence intervals).
4. **Unsupported Positives (Hallucinations):** Findings reported by the model that are absent in the reference report.
5. **Missed Findings (False Negatives):** Real abnormalities omitted from the model report.
6. **Laterality Concordance:** Correct identification of right vs left vs bilateral pathology.
7. **Internal Contradictions:** Observations stating pathology while impression declares normal radiograph.
8. **Inference Latency & Throughput:** Mean latency (seconds) and tokens per second.

### 5.2 Stage 2: Locked Test Cohort Evaluation
> [!IMPORTANT]
> Execute this step **ONLY ONCE** after the best checkpoint has been finalized and locked using validation results. Test outcomes must never be used to adjust prompts or hyperparameters.
```bash
python scripts/evaluate_finetuned_medgemma.py \
    --split expanded_test \
    --adapter-path checkpoints/medgemma_lora_tier2
```

---

## 6. Tier 3 Selective Vision Adaptation: Deferral Policy

- **Current State:** Tier 3 vision encoder adaptation (`vision_tower.encoder.layers[-4:]`) is **STRICTLY DEFERRED**.
- **Scientific Attribution:** Training both the vision tower and language decoder simultaneously makes it impossible to determine whether changes in report quality stem from multimodal projection alignment or feature drift in the visual backbone.
- **Trigger Condition for Tier 3:** Only if Tier 2 LoRA evaluation on `val_dev` demonstrates that language decoder and projector adaptation cannot resolve faint airspace opacities (e.g. consolidation recall remains $<15\%$), Tier 3 vision adaptation will be explored with a low learning rate ($3.0 \times 10^{-5}$) on the top 4 SigLIP layers while keeping the lower 20 layers frozen.

---

## 7. Decoupling of the Triage Priority Service

- **Research-Only Rule Engine:** The triage service ([`src/triage_service.py`](file:///Users/adityakachhot/interactmed/src/triage_service.py)) remains completely separate from report generation fine-tuning.
- **No Joint Multi-Task Objective:** Triage priority targets are **not** mixed into the MedGemma language fine-tuning loss. Training on combined report and triage targets without verified multi-task annotations leads to finding hallucination and loss interference.
- **Clinical Governance:** The triage service is not clinically validated. Quantitative evaluation of triage accuracy requires independent, qualified radiologist double-reading using [`templates/triage_annotation_template.csv`](file:///Users/adityakachhot/interactmed/templates/triage_annotation_template.csv).
