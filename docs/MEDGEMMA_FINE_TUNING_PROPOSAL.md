# InteractMed-VLM: MedGemma 4B Fine-Tuning Feasibility Proposal

## 1. Executive Summary & Research Justification

The expanded whole-image evaluation on development data demonstrated that **Prompt v3 Structured** achieves significant operational and structural gains over the single-stage baseline:
- **Boilerplate Normal Collapse:** Eliminated from 50.0% to **0.0%**.
- **Internal Contradictions:** Eliminated from 3.3% to **0.0%**.
- **Hallucinated Positive Claims:** Reduced from 43.3% to **33.3%**.
- **Inference Latency:** Accelerated by 28.2% (18.85s $\to$ 13.54s).

However, both prompt pipelines encountered a rigid **opacity sensitivity bottleneck**:
- **Pneumonia / Consolidation:** 0 / 7 detected (Prompt v3), 1 / 7 detected (Baseline, via speculative language).
- **Atelectasis:** 0 / 2 detected (both pipelines).

Because MedGemma 4B relies on a frozen SigLIP vision transformer, subtle low-contrast parenchymal opacities produce visual token activations that fail to cross text generation thresholds. **Prompt engineering has reached its empirical ceiling.** To detect subtle consolidations and atelectasis without compromising precision, **supervised parameter-efficient fine-tuning (PEFT)** is technically justified.

---

## 2. Adaptation Hierarchy: Three Distinct Approaches

To maintain rigorous scientific attribution, any future fine-tuning effort must evaluate and distinguish between three separate tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ADAPTATION HIERARCHY                            │
├────────────────────────────────────────────────────────────────────────┤
│ Tier 1: Prompt-Only Baseline (Prompt v3 Structured, 1024px)            │
│   • Frozen SigLIP vision encoder + frozen Gemma 4B LM                  │
│   • Status: Complete. Establishes the frozen-weights performance ceiling.│
├────────────────────────────────────────────────────────────────────────┤
│ Tier 2: Multimodal Projector & Language LoRA Adaptation                │
│   • Frozen SigLIP vision encoder                                       │
│   • Trainable multi_modal_projector (linear / MLP projection)          │
│   • Trainable LoRA on Gemma attention (q_proj, v_proj)                 │
│   • Goal: Align existing visual embeddings with clinical terminology   │
├────────────────────────────────────────────────────────────────────────┤
│ Tier 3: Selective Vision Encoder + Projector Adaptation                │
│   • Trainable LoRA on top 2–4 SigLIP transformer blocks                │
│   • Trainable multi_modal_projector                                    │
│   • Trainable LoRA on Gemma attention                                  │
│   • Goal: DIRECTLY target low-contrast radiological opacity features   │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Tier 1 — Prompt-Only Baseline (Current Production)
- **Architecture:** Zero model weight updates. Frozen weights running in local Ollama.
- **Role in Research:** The definitive baseline against which any fine-tuned checkpoint must demonstrate statistically significant improvement in recall on opacities without degrading specificity or increasing boilerplate.

### 2.2 Tier 2 — Multimodal Projector & Language Adapter
- **Components Trained:**
  - `multi_modal_projector` (the bridge mapping 1152-dim SigLIP visual patch tokens to 2560-dim Gemma embedding space).
  - Low-Rank Adaptation (LoRA: $r=16, \alpha=32$) on language attention matrices (`q_proj`, `v_proj`).
- **Trainable Parameters:** ~25M to 45M parameters ($<1\%$ of total 4.3B weights).
- **Hypothesis:** Improves radiologist report synthesis, structural alignment, and severity grading from already perceived features.
- **Limitation:** Cannot resolve opacities that the frozen SigLIP vision encoder fails to differentiate from normal lung parenchyma.

### 2.3 Tier 3 — Selective Vision Encoder Adaptation
- **Components Trained:**
  - LoRA adapters ($r=8, \alpha=16$) on the final 2–4 vision transformer layers of SigLIP (`vision_tower.encoder.layers[-4:]`).
  - Multimodal projector (`multi_modal_projector`).
  - Language attention LoRA (`q_proj`, `v_proj`).
- **Hypothesis:** Modifies vision transformer spatial self-attention to become sensitive to low-contrast grayscale radiological gradients (e.g., air bronchograms, perihilar haziness, plate-like discoid atelectasis).
- **Risk & Safeguard:** Risk of catastrophic forgetting of natural image features. Safeguarded by low learning rates ($5 \times 10^{-5}$) and frozen lower SigLIP layers.

---

## 3. Training Checkpoint & Implementation Reality Check

### 3.1 Why the Local Ollama Model CANNOT Be Fine-Tuned Directly
- **Model in Ollama:** `medgemma:4b` is a 4-bit block-quantized GGUF container (`Q4_K_M`, 5.3 GB) running inside `llama.cpp`.
- **Technical Barrier:** GGUF is an integer-quantized, inference-only binary format. It contains pre-quantized weight matrices without autograd computation graphs, gradient buffers, or backward-pass support. **It is physically impossible to fine-tune a GGUF model in Ollama.**
- **Requirement:** Training requires the unquantized PyTorch source checkpoint from HuggingFace (e.g. `google/medgemma-4b` or `google/gemma-3-4b-it` vision variant).

### 3.2 Local Hardware & macOS MPS Constraints
An inspection of the local environment (`MacBook Air M3, 16 GB unified memory`, PyTorch 2.2.2, macOS Metal/MPS) reveals critical training constraints:

| Component | Status on macOS M3 | Impact on Training |
| :--- | :--- | :--- |
| **`bitsandbytes` (4-bit/8-bit QLoRA)** | **NOT INSTALLED / CUDA-ONLY** | `bitsandbytes` does not run natively on macOS Metal (MPS). Standard CUDA QLoRA scripts fail immediately on macOS. |
| **`peft` / `accelerate`** | **NOT INSTALLED** | Must be installed before any PEFT pipeline can execute. |
| **`transformers` Version** | `4.40.1` installed | PaliGemma/Gemma-3 vision classes require `transformers >= 4.41.0`. |
| **MPS FP16 Memory Footprint** | 4.3B parameters in FP16 = 8.6 GB weights | Fits in RAM only if Ollama (5.3 GB) is unloaded. |
| **Local Memory Budget (16 GB)** | macOS OS/UI: ~5.5 GB<br>Ollama resident: ~5.3 GB<br>Available RAM: **~5.2 GB** | Loading an 8.6 GB PyTorch model while Ollama is running causes instant Out-Of-Memory kernel thrashing. |

### 3.3 Compute Recommendation: Two Viable Pathways

#### Pathway A: Local Apple Silicon Training via Apple MLX (Recommended for Local Dev)
- **Framework:** Apple MLX framework (`mlx-lm` / MLX Vision LoRA).
- **Mechanism:** MLX provides native 4-bit quantization and LoRA gradient computation optimized directly for Apple Silicon unified memory without CUDA dependencies.
- **Prerequisite:** Unload the resident Ollama model (`ollama stop medgemma:4b`) to free 5.3 GB of unified memory before launching MLX training.
- **Estimated Training Memory:** ~6.8 GB unified RAM at batch size 1 with gradient accumulation.

#### Pathway B: Dedicated Cloud GPU (Recommended for Production / Research Rigor)
- **Compute:** 1x NVIDIA A10G (24 GB VRAM) or 1x NVIDIA A100 (40 GB / 80 GB VRAM).
- **Framework:** Standard PyTorch + HuggingFace `transformers` + `peft` + `bitsandbytes` 4-bit QLoRA.
- **Advantage:** Native CUDA acceleration, native bitsandbytes QLoRA, full bfloat16 support, and multi-image batch training with FlashAttention-2.

---

## 4. Dataset Architecture & Patient-Disjoint Partitioning

### 4.1 Strict Governance & Test Isolation
- **The 117-Case Test Cohort (`expanded_test`):** Must remain **100% UNTOUCHED, QUARANTINED, AND UNSEEN**. It will only be evaluated once after training is finalized.
- **Eligible Training Pool:** Strictly drawn from `expanded_dev` (250 images across 248 patients).

### 4.2 Proposed Training / Validation Split (Patient-Grouped)

To prevent intra-patient data leakage between training and validation:

| Split | Images | Unique Patients | Category Distribution | Role |
| :--- | :---: | :---: | :--- | :--- |
| **`train_dev`** | **200** | **198** | ~75 Pneumonia, ~57 Normal, ~46 Edema, ~8 Devices, ~7 Cardiomegaly, ~7 Atelectasis | LoRA adapter gradient optimization |
| **`val_dev`** | **50** | **50** | ~19 Pneumonia, ~14 Normal, ~11 Edema, ~2 Devices, ~2 Cardiomegaly, ~2 Atelectasis | Checkpoint selection & early stopping |
| **`expanded_test`** | **117** | **117** | Locked & held out | Final unpolluted evaluation |

- **Zero Patient Overlap:** $\text{Patients}(\text{train\_dev}) \cap \text{Patients}(\text{val\_dev}) = \emptyset$.
- **Zero Test Overlap:** $\text{Patients}(\text{train\_dev} \cup \text{val\_dev}) \cap \text{Patients}(\text{expanded\_test}) = \emptyset$.

### 4.3 Training Record Format
Each training sample is paired from verified local images and gold annotations:
```json
{
  "image_path": "/Users/adityakachhot/Downloads/data/mimic-cxr/files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e38b223.jpg",
  "prompt": "You are an expert radiologist conducting structured visual feature extraction on this chest radiograph.\nFor each category below, state: Observed (<details and laterality>) | Not identified | Indeterminate | Not assessable.\n\n1. Support apparatus: Inspect for endotracheal tube, enteric tube, vascular catheter/line, chest tube, pacemaker, or surgical clips.\n2. Cardiac silhouette: Normal, Enlarged (CTR > 0.5), or Indeterminate.\n3. Lungs & opacities: Consolidation, focal opacity, atelectasis, congestion, or edema. Specify laterality (right, left, bilateral, or unspecified laterality).\n4. Pleural spaces: Costophrenic angles sharp, blunted/effusion, or obscured.\n5. Mediastinum: Normal contour, widened, or indeterminate.\nBegin directly with 1. Support apparatus:",
  "target": "1. Support apparatus: Not identified.\n2. Cardiac silhouette: Normal.\n3. Lungs & opacities: Observed (streaky retrocardiac opacity, left base atelectasis).\n4. Pleural spaces: Costophrenic angles sharp.\n5. Mediastinum: Normal contour."
}
```

---

## 5. Hyperparameter Specification & Guardrails

| Parameter | Recommended Setting | Rationale |
| :--- | :--- | :--- |
| **Base Model** | `google/medgemma-4b` | Upstream PyTorch weights |
| **LoRA Rank ($r$)** | 16 | Sufficient capacity for medical domain alignment |
| **LoRA Alpha ($\alpha$)** | 32 | Standard scaling ($\alpha = 2r$) |
| **LoRA Target Modules** | `q_proj`, `v_proj`, `multi_modal_projector` | Projector + language attention heads |
| **Batch Size** | 1 (per device) | Conserves VRAM / unified memory |
| **Gradient Accumulation** | 16 steps | Effective batch size of 16 |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$) | Stable convergence |
| **Learning Rate** | $1 \times 10^{-4}$ (projector), $5 \times 10^{-5}$ (vision) | Prevents catastrophic forgetting |
| **LR Schedule** | Cosine with 10% linear warmup | Smooth adaptation |
| **Precision** | bfloat16 (or float16 with loss scaling) | Stable gradient dynamic range |
| **Max Steps / Epochs** | 3–5 epochs (~60–100 optimizer steps) | Prevents overfitting on 200 cases |

---

## 6. Reproducible Step-by-Step Execution Plan

### Step 1: Prepare Environment & Install Dependencies
```bash
# Upgrade transformers and install PEFT
/Users/adityakachhot/miniforge3/envs/dl/bin/pip install --upgrade "transformers>=4.41.0" peft accelerate
```

### Step 2: Unload Resident Ollama Model (If Training Locally)
```bash
# Release 5.3 GB Metal GPU memory before PyTorch / MLX allocation
ollama stop medgemma:4b
```

### Step 3: Construct Patient-Disjoint Training Datasets
```bash
# Generate train_dev.jsonl and val_dev.jsonl strictly from expanded_dev
/Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/prepare_fine_tuning_cohort.py \
  --manifest artifacts/report_evaluation/expanded_cohort/expanded_manifest.csv \
  --ref-store artifacts/report_evaluation/expanded_cohort/expanded_reference_store.json \
  --output-dir artifacts/fine_tuning_cohort
```

### Step 4: Execute Fine-Tuning Script
```bash
# Launch parameter-efficient fine-tuning (Tier 2: Projector + LoRA)
/Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/train_medgemma_adapter.py \
  --train-file artifacts/fine_tuning_cohort/train_dev.jsonl \
  --val-file artifacts/fine_tuning_cohort/val_dev.jsonl \
  --output-dir checkpoints/medgemma_lora_tier2 \
  --epochs 3 \
  --lr 1e-4
```

### Step 5: Evaluate on Locked Test Cohort (`expanded_test`)
```bash
# Compare Fine-Tuned Checkpoint vs Prompt v3 Baseline on untouched test set
/Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/evaluate_fine_tuned_model.py \
  --adapter-path checkpoints/medgemma_lora_tier2 \
  --split expanded_test
```

---

## 7. Research Governance & Clinical Safeguards

1. **No Automatic Training:** Training must never be launched without explicit user review of baseline results and hardware confirmation.
2. **Untouched Test Benchmark:** `expanded_test` ($N=117$) must remain strictly locked throughout training and validation.
3. **Data Localization:** All MIMIC-CXR and Chest-ImaGenome files remain on local disk. Zero patient data transmitted externally.
4. **Clinical Disclaimer:** Fine-tuned models remain research artifacts for evaluating visual grounding and report synthesis; they do not constitute diagnostic medical devices.
