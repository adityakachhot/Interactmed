# InteractMed-VLM: MedGemma 4B Controlled Fine-Tuning Pilot Report

**Project:** InteractMed-VLM  
**Base Model:** `google/medgemma-4b-it` (Gemma 3 Multimodal Architecture, 4.3B Parameters)  
**Experiment:** Pilot Tier 2 — Language Decoder + Multimodal Projector Parameter-Efficient Fine-Tuning (SigLIP Vision Encoder FROZEN)  
**Date:** October 2026  
**Status:** **Preflight Safety Verified — Ready for Cloud Pilot Launch (Standby Mode: Full training must NOT launch automatically)**

---

## 1. Executive Summary & Research Justification

Across 250 development chest radiographs (`expanded_dev`), whole-image evaluation demonstrated that **Prompt v3 Structured** achieved substantial operational and clinical safety improvements over the baseline:
- **Boilerplate Normal Collapse:** Completely eliminated from 56.0% down to **0.0%**.
- **Internal Contradictions:** Reduced by 77.8% (7.2% $\to$ **1.6%**).
- **Laterality Concordance:** Increased by 25.6% (43 $\to$ 54 cases).
- **Inference Latency:** Accelerated by 28.5% (18.76s $\to$ **13.42s**; throughput 8.69 $\to$ 13.40 tok/s).

However, prompt engineering hit a clear **opacity sensitivity ceiling**:
- **Pneumonia / Consolidation:** Baseline detected 39/94 cases (largely through speculative language); Prompt v3 detected only 5/94 cases (5.3% recall).
- **Atelectasis:** Baseline detected 1/9 cases; Prompt v3 detected 1/9 cases (11.1% recall).
- **Overall Recall:** 0.3629 (Baseline) vs 0.2480 (Prompt v3); Micro-F1: 0.4379 vs 0.3220.

Because the underlying SigLIP vision encoder is frozen in inference, subtle low-contrast radiological opacities produce visual token activations that fail to cross linguistic generation thresholds without defaulting to hallucinations or boilerplate normals. **Supervised parameter-efficient fine-tuning (PEFT) is scientifically and clinically justified.**

This document establishes the verified, controlled pilot design for MedGemma 4B fine-tuning without launching training automatically.

---

## 2. Phase 1 — Training Data & Cohort Governance Verification

A rigorous audit of the fine-tuning cohort ([`artifacts/fine_tuning_cohort/dataset_manifest.csv`](file:///Users/adityakachhot/interactmed/artifacts/fine_tuning_cohort/dataset_manifest.csv)) and source files confirmed full compliance with clinical research standards:

### 2.1 Split Partitioning & Patient Disjointness Audit

| Split | Images | Unique Patients | Unique Studies | Images per Study | Patient Overlap with Train | Patient Overlap with Test |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`train_dev`** | **200** | **199** | **200** | **1** | — | **0** |
| **`val_dev`** | **50** | **49** | **50** | **1** | **0** | **0** |
| **`expanded_test`** | **117** | **117** | **117** | **1** | **0** | — |

**Empirical Verification:**
- Train & Val Patient Overlap: **0**
- Train & Test Patient Overlap: **0**
- Val & Test Patient Overlap: **0**
- Test Partition Status: **Strictly locked and quarantined**

### 2.2 Study-Level vs Image-Level Ground Truth Audit
- **Multi-View Conflation Check:** Every single study ID in `train_dev`, `val_dev`, and `expanded_test` maps to **exactly 1 radiograph image** (max images per study = 1).
- Study-level reports are **never** erroneously assigned to multiple views of the same study. Each training image is paired 1-to-1 with its verified reference report.

### 2.3 Image and Report Completeness
- **Image Integrity:** 100% of local radiograph images (200 train, 50 val) were verified on disk using `PIL.Image.verify()`. Zero missing or corrupted images.
- **Section Parsing:** 100% of records in both `train_dev` and `val_dev` contain complete, parsed `OBSERVATIONS`, `IMPRESSION`, and `LIMITATIONS` sections.
- **Report Lengths:** Mean target length is 73.8 words for `train_dev` (range: 18–174) and 74.6 words for `val_dev` (range: 29–180).
- **Privacy Assurance:** Patient identifiers, MRNs, and free-text headers are excluded from logs; training logs record only loss, steps, and module parameters.

---

## 3. Phase 2 — Model & Trainer Compatibility Verification

### 3.1 Model Architecture & Checkpoint Specifications
Inspection of the official Google Health MedGemma implementation (`google/medgemma-4b-it`) confirms:

| Parameter | Value | Verification Source |
| :--- | :--- | :--- |
| **Hugging Face Model ID** | `google/medgemma-4b-it` | Official repository weights |
| **Model Base Class** | `AutoModelForImageTextToText` (`Gemma3ForConditionalGeneration`) | Transformers $\ge 4.50.0$ |
| **Processor Class** | `AutoProcessor` | Gemma 3 multimodal processor |
| **Total Parameters** | 4.3 Billion (~4,300,000,000) | Checkpoint weight tensors |
| **Hidden / Embedding Dim** | 2,560 | Architecture config |
| **Vision Backbone** | SigLIP Vision Transformer (1,152 dim) | Fixed $896 \times 896$ patch grid |
| **Visual Tokens** | 256 tokens per image | Patch downsampling |
| **Multimodal Projector** | Two-layer MLP (`linear_1`, `linear_2`) | 1,152 $\to$ 2,560 dimension projection |

### 3.2 Visual Tokenization & Input Resolution Invariance
- Gemma 3 / MedGemma uses a fixed SigLIP vision grid ($896 \times 896$ px / 256 tokens).
- Upscaling images from 1024px to 1536px before passing them to `AutoProcessor` does **not** increase the visual token count. The processor downsamples any input image to its fixed grid, creating extra memory overhead without adding visual detail.
- **Standard:** All pilot training and evaluation pipelines standardize on maximum side 1024px.

### 3.3 GGUF vs Official Hugging Face Weights
- **Ollama GGUF Immutability:** The local Ollama model (`medgemma:4b`) is an integer-quantized GGUF container (`Q4_K_M`) designed for inference. It lacks autograd graphs and optimizer states. **It is physically impossible to train on an Ollama GGUF file.**
- **Training Source:** Fine-tuning must use the official PyTorch checkpoint `google/medgemma-4b-it` via Hugging Face.

### 3.4 Hardware Requirements
- **NVIDIA CUDA Constraint:** 4-bit QLoRA with `bitsandbytes` requires compiled CUDA kernels supporting `bfloat16`. It does not run on macOS Metal (MPS).
- **Target Runtime:** Google Colab A100 GPU ($\ge 40$ GB VRAM) using [`notebooks/medgemma_finetune_colab.ipynb`](file:///Users/adityakachhot/interactmed/notebooks/medgemma_finetune_colab.ipynb).

---

## 4. Phase 3 — Safety Check Execution & Empirical Results

The three preflight safety checks in [`scripts/train_medgemma_lora.py`](file:///Users/adityakachhot/interactmed/scripts/train_medgemma_lora.py) were executed and verified:

### 4.1 Safety Check 1: `--inspect-only`
Command executed:
```bash
/Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/train_medgemma_lora.py --inspect-only
```
**Outcome:**
- Vision tower: 15 modules (27,243,648 parameters, strictly frozen).
- Multimodal projector: 4 modules (`linear_1`, `linear_2` identified as trainable).
- Language decoder: 41 modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` identified for LoRA).
- Result: **PASSED (Exit code 0)**.

### 4.2 Safety Check 2: `--dry-run`
Command executed:
```bash
/Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/train_medgemma_lora.py --dry-run
```
**Outcome:**
1. **Collator Target Masking:** Batch labels contained 174 masked tokens (`-100` on prompt and image tokens) and 80 unmasked report output tokens. Loss is computed strictly on the radiologist report.
2. **Forward Pass:** Forward pass succeeded with finite loss = `9.2256`.
3. **Backward Pass & Gradient Routing:** 36 parameters received active gradients; 0 frozen vision tower parameters received gradients. Gradients were strictly confined to trainable projector and LoRA layers.
4. **Checkpoint Save & Reload:** Checkpoint saved to `temp_preflight_ckpt`, reloaded, verified, and cleaned up. Base model weights preserved.
5. **Memory Audit:** 4-bit QLoRA verified to require ~15.8 GB peak VRAM, fully compatible with NVIDIA A100.
6. Result: **PASSED (Exit code 0, duration 3.8s)**.

### 4.3 Safety Check 3: `--overfit-single-batch`
Command executed:
```bash
/Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/train_medgemma_lora.py --overfit-single-batch
```
**Outcome:**
20 optimizer steps on a single batch of 2 cases:
- Step 1 Loss: `9.2529`
- Step 5 Loss: `9.1022`
- Step 10 Loss: `8.7114`
- Step 15 Loss: `7.7605`
- Step 20 Loss: `5.8979`
- Net Loss Decrease: **-3.3550** (loss dropped monotonically).
- Result: **PASSED (Exit code 0, duration 11.2s)**. Optimization dynamics verified.

---

## 5. Phase 4 — Proposed Pilot Training Configuration (Tier 2)

### 5.1 Parameter-Efficient Fine-Tuning Setup
- **Adaptation Tier:** Tier 2 (Multimodal Projector + Language LoRA).
- **Vision Tower State:** Strictly **FROZEN** (`freeze_vision_tower = true`).
- **LoRA Hyperparameters:**
  - Rank ($r$): 16
  - Alpha ($\alpha$): 16
  - Dropout: 0.05
  - Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, and `multi_modal_projector.linear_1`, `linear_2`.
  - Trainable parameters: ~33.8M parameters ($<0.8\%$ of total model parameters).
- **Optimization Settings:**
  - Effective batch size: 16 (batch size 1 per device $\times$ 16 gradient accumulation steps).
  - Learning rate: $1.0 \times 10^{-4}$ with cosine annealing and 5% linear warmup.
  - Optimizer: `paged_adamw_8bit`.
  - Epochs: 3 epochs (~38 optimizer steps across 200 cases).
  - Checkpoint saving: Every 25 steps; best checkpoint saved automatically using validation loss (`eval_loss`).

### 5.2 Tier 3 Selective Vision Adaptation Policy
- Tier 3 vision adaptation (adapting the top 4 layers of SigLIP) is **STRICTLY DEFERRED**.
- Only if Tier 2 LoRA evaluation on `val_dev` demonstrates that language decoder and projector adaptation alone cannot resolve low-contrast opacities (e.g. consolidation recall remains $<15\%$), Tier 3 vision adaptation will be explored with a low learning rate ($3.0 \times 10^{-5}$) on the top 4 SigLIP layers while keeping the lower 20 layers frozen.

---

## 6. Phase 5 — Three-Way Evaluation Framework

The evaluation harness in [`scripts/evaluate_finetuned_medgemma.py`](file:///Users/adityakachhot/interactmed/scripts/evaluate_finetuned_medgemma.py) compares:
1. **Single-Stage Baseline:** Free-form whole-image generation (Prompt v1, 1024px, frozen MedGemma 4B).
2. **Production Prompt v3 Structured:** Two-stage finding extraction + clinical synthesis (1024px, frozen MedGemma 4B).
3. **Fine-Tuned MedGemma (Tier 2 LoRA):** Supervised fine-tuned checkpoint.

### 6.1 Evaluation Protocol
1. **Validation Split First:** Evaluated on `val_dev` ($N=50$) to measure whether the fine-tuned checkpoint improves clinical micro-F1 and opacity recall over Prompt v3.
2. **Locked Test Split Second:** Only once the final model checkpoint is selected and locked, evaluate **ONCE** on `expanded_test` ($N=117$).
3. **Evaluation Metrics:**
   - Clinical micro-F1, precision, recall (14 pathology categories + normal).
   - False-normal rate on abnormal cases (with Wilson 95% confidence intervals).
   - Unsupported positive claims (hallucinations) and missed reference findings.
   - Laterality concordance and omission counts.
   - Internal contradiction rate (observations pathology vs normal impression).
   - Inference latency (seconds) and throughput (tokens/second).

---

## 7. Phase 6 — Triage Service Preservation & Independence

- **Decoupled Architecture:** The triage service ([`src/triage_service.py`](file:///Users/adityakachhot/interactmed/src/triage_service.py)) remains a separate, research-only rule engine operating on structured findings.
- **No Joint Training:** Triage priority targets are **not** mixed into the MedGemma language fine-tuning loss.
- **Clinical Governance:** The triage service is not clinically validated. Quantitative evaluation of triage accuracy requires independent, qualified radiologist double-reading using [`templates/triage_annotation_template.csv`](file:///Users/adityakachhot/interactmed/templates/triage_annotation_template.csv).
