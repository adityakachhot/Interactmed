# InteractMed-VLM: Fine-Tuning and Model Training Architecture

> **Authoritative Training, Adaptation & Provenance Reference**  
> **Project:** InteractMed-VLM  
> **Document:** `docs/architecture/FINE_TUNING_AND_TRAINING.md`  
> **Status:** Audited against Training Scripts, Configs, Logs, and Checkpoint Weights

---

## 1. Overview & Training Track Separation

Model training in InteractMed-VLM is divided into two completely separate engineering tracks:
1. **TransVG-AGPT Visual Grounding Track:** Fully executed and evaluated domain-adaptation fine-tuning on the RSNA Pneumonia Detection Challenge dataset using local Apple Metal (MPS) acceleration.
2. **MedGemma 4B Parameter-Efficient Fine-Tuning (PEFT/QLoRA) Track:** Fully prepared, audited, and preflight-verified training pipeline for Google MedGemma 4B, currently held in **Standby Mode** for execution on cloud NVIDIA CUDA GPUs (A100).

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 TRAINING TRACKS                                        │
├───────────────────────────────────────────┬────────────────────────────────────────────┤
│   Track 1: TransVG-AGPT Visual Grounding  │   Track 2: MedGemma 4B Multimodal LoRA     │
├───────────────────────────────────────────┼────────────────────────────────────────────┤
│ • Task: Phrase Bounding Box Localization  │ • Task: Whole-Image Report Drafting        │
│ • Dataset: RSNA Pneumonia Challenge 2018  │ • Dataset: Local Gold MIMIC-CXR Pairs      │
│ • Hardware: Local Apple Silicon M3 (MPS)  │ • Hardware: Cloud NVIDIA A100 GPU (CUDA)   │
│ • Checkpoint: best_model_val_iou.pth      │ • Script: scripts/train_medgemma_lora.py   │
│ • Status: COMPLETED, TESTED & EVALUATED   │ • Status: PREPARED (Preflight Passed)      │
└───────────────────────────────────────────┴────────────────────────────────────────────┘
```

---

## 2. Track 1: TransVG-AGPT Visual Grounding Fine-Tuning

### 2.1 Checkpoint Provenance & Lineage

| Checkpoint Name | Local File Path | File Size | Epochs | Training Provenance | Role in Active Application |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pretrained Baseline** | [`checkpoints/transvg.pth`](file:///Users/adityakachhot/interactmed/checkpoints/transvg.pth) | 2,099,188,914 B (~2.1 GB) | — | AGPT official weights pre-trained on RefCOCO and MS-CXR | Baseline comparative model; available via `checkpoint=baseline` |
| **RSNA Fine-Tuned (Run 01)** | [`checkpoints/finetuned_rsna/run_01_3ep/best_model_val_iou.pth`](file:///Users/adityakachhot/interactmed/checkpoints/finetuned_rsna/run_01_3ep/best_model_val_iou.pth) | 767,932,253 B (~768 MB) | 3 | Fine-tuned from baseline on RSNA Pneumonia training split | **Production default**; loaded automatically at API startup |

### 2.2 Dataset Split & Patient Disjointness Governance
The fine-tuning dataset is derived from the RSNA Pneumonia Detection Challenge 2018 (`stage_2_train_images`):
- **Cohort Selection:** Restricted to positive pneumonia cases (`Target == 1`, `class == "Lung Opacity"`), containing verified radiologist bounding boxes.
- **Partitioning Scheme:** Stratified, patient-disjoint split using random seed `42`:
  - **`train_positive`:** 4,210 unique patients (11,897 ground truth boxes).
  - **`val_positive`:** 901 unique patients (2,485 ground truth boxes; 15% split).
  - **`test_positive`:** 901 unique patients (2,575 ground truth boxes; 15% split; strictly held out).
- **Patient Leakage Audit:** Programmatic intersection verification confirmed:
  $$\operatorname{Patients}(\text{Train}) \cap \operatorname{Patients}(\text{Val}) = \emptyset, \quad \operatorname{Patients}(\text{Train}) \cap \operatorname{Patients}(\text{Test}) = \emptyset, \quad \operatorname{Patients}(\text{Val}) \cap \operatorname{Patients}(\text{Test}) = \emptyset$$

### 2.3 Training Pipeline & Optimization Hyperparameters
Fine-tuning was executed via [`scripts/train_rsna.py`](file:///Users/adityakachhot/interactmed/scripts/train_rsna.py) using configuration [`configs/rsna_train.yaml`](file:///Users/adityakachhot/interactmed/configs/rsna_train.yaml):

```yaml
# configs/rsna_train.yaml extract
experiment:
  name: "rsna_finetuning_stage1"
  phase: "4B"
  task: "pneumonia_localization_finetuning"
model:
  base_checkpoint: "checkpoints/transvg.pth"
  freeze_encoders: true          # ResNet-50, DETR encoder, ClinicalBERT frozen
  device: "auto"                 # Apple Metal (MPS)
training:
  epochs: 8                      # Maximum epochs
  batch_size: 4                  # Memory-safe for 16 GB unified RAM
  lr: 5.0e-5                     # Peak learning rate
  weight_decay: 1.0e-4
  clip_max_norm: 0.1
  l1_loss_coef: 5.0
  giou_loss_coef: 2.0
  lr_scheduler: "cosine"         # Cosine annealing with 1 epoch linear warmup
  warmup_epochs: 1
  min_lr: 1.0e-6
  save_best_metric: "mean_best_match_iou"
```

#### Loss Function Formulation
The optimization objective combines Smooth L1 loss and Generalized IoU (GIoU) loss over the normalized box coordinates $\hat{b} = [\hat{c}_x, \hat{c}_y, \hat{w}, \hat{h}]$ and ground truth $b$:
$$\mathcal{L}_{\text{box}}(\hat{b}, b) = \lambda_{\text{L1}} \mathcal{L}_{\text{L1}}(\hat{b}, b) + \lambda_{\text{giou}} \mathcal{L}_{\text{giou}}(\hat{b}, b)$$
with coefficients $\lambda_{\text{L1}} = 5.0$ and $\lambda_{\text{giou}} = 2.0$.

#### Frozen vs. Trainable Parameters
- **Frozen Encoders:** The visual backbone (ResNet-50 and DETR transformer encoder, 23.5M params) and linguistic backbone (Bio_ClinicalBERT, 108M params) were frozen.
- **Trainable Layers:** Only the 4-layer cross-modal fusion transformer and MLP regression head (~14.2M params) were updated.

---

### 2.4 Verified Epoch-by-Epoch Training Logs
The training history recorded in [`artifacts/rsna/finetuning/run_01_3ep/training_history.json`](file:///Users/adityakachhot/interactmed/artifacts/rsna/finetuning/run_01_3ep/training_history.json) establishes the exact progression over the 3 completed epochs:

| Epoch | Duration (s) | Train Total Loss | Train L1 Loss | Train GIoU Loss | Val Mean Best-Match IoU | Val Recall@0.50 | Val Recall@0.75 | Learning Rate | Best Model Checkpoint Action |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Epoch 1** | 708.15 s | 2.9026 | 0.2993 | 0.7030 | 0.3789 | 0.3751 | 0.0400 | $3.775 \times 10^{-5}$ | Saved (`best_model_val_iou.pth`) |
| **Epoch 2** | 895.51 s | 2.6819 | 0.2744 | 0.6550 | 0.4168 | 0.4473 | 0.0499 | $1.325 \times 10^{-5}$ | Saved (`best_model_val_iou.pth`) |
| **Epoch 3** | 823.64 s | **2.5420** | **0.2593** | **0.6227** | **0.4493** | **0.4950** | **0.0688** | $1.000 \times 10^{-6}$ | **Saved as Best Checkpoint (Val IoU: 0.4493)** |

**Total Training Duration:** 2,427.3 seconds (~40.5 minutes) on Apple Silicon M3.

---

### 2.5 Validation vs. Final Held-Out Test Evaluation
The best checkpoint selected from validation (`best_model_val_iou.pth`) was subsequently evaluated on the independent 901-patient held-out test split using [`scripts/evaluate_rsna.py`](file:///Users/adityakachhot/interactmed/scripts/evaluate_rsna.py):

| Evaluation Metric | Validation: Baseline | Validation: Fine-Tuned (3 ep) | **Test: Pretrained Baseline** | **Test: Fine-Tuned (3 ep)** | Absolute Gain (Test) | Relative Gain (Test) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Evaluated Images** | 901 | 901 | **901** | **901** | — | — |
| **Inference Success** | 100% | 100% | **100%** | **100%** | — | — |
| **Mean Best-Match IoU** | 0.3124 | 0.4494 | **0.3130** | **0.4530** | **+0.1400** | **+44.7%** |
| **Recall @ IoU ≥ 0.50** | 0.2331 | 0.4928 | **0.2397** | **0.4972** | **+0.2575** | **+107.5%** |
| **Recall @ IoU ≥ 0.75** | 0.0433 | 0.0688 | **0.0555** | **0.0810** | **+0.0255** | **+45.9%** |
| **Mean Latency (MPS)** | 139.1 ms | 134.3 ms | **145.9 ms** | **140.4 ms** | −5.5 ms | — |

#### Critical Scope Boundary & Generalization Caveat
- **Pneumonia Domain Adaptation:** These results definitively confirm that fine-tuning adapted the cross-modal fusion transformer to pneumonia opacities under the fixed query `"pneumonia"`.
- **No General Phrase Grounding Claim:** This training does **not** prove general visual grounding for other clinical pathologies (e.g., pneumothorax, cardiomegaly, rib fractures). Broader phrase grounding across other conditions remains uncharacterized and would require multi-pathology training on MS-CXR.

---

## 3. Track 2: MedGemma 4B Parameter-Efficient Fine-Tuning (QLoRA)

### 3.1 Motivation & The "Opacity Ceiling"
Across the 250-case development evaluation (`expanded_dev`), prompt engineering achieved high clinical safety (0.0% boilerplate collapse, 1.6% internal contradictions), but hit an **opacity sensitivity ceiling**:
- The pre-trained SigLIP vision encoder weights are frozen during inference.
- Low-contrast parenchymal opacities produce weak visual token activations that fail to trigger finding assertions in the language decoder without fine-tuned projection weights.
- As a result, Prompt v3 detected only 5 of 94 pneumonia consolidations (5.3% recall).
- Parameter-efficient fine-tuning (PEFT) targeting the multimodal projector and language decoder is scientifically necessary to bridge this sensitivity gap.

---

### 3.2 Fine-Tuning Architecture & Tier Definition

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   MEDGEMMA 4B ADAPTATION TIERS (configs/medgemma_finetune.yaml)        │
├─────────┬───────────────────────────────┬────────────────────────────┬─────────────────┤
│ Tier    │ Scope                         │ Target Modules             │ Implementation  │
├─────────┼───────────────────────────────┼────────────────────────────┼─────────────────┤
│ Tier 1  │ Prompt Engineering (Baseline) │ Zero parameter updates     │ Completed (v3)  │
│ Tier 2  │ Multimodal Projector + LoRA   │ MLP Projector + Q,K,V,O,MLP│ PREPARED        │
│ Tier 3  │ Selective Vision LoRA         │ SigLIP Attention + LoRA    │ Proposed Future │
└─────────┴───────────────────────────────┴────────────────────────────┴─────────────────┘
```

The repository has prepared **Tier 2 Adaptation**:
- **Vision Tower State:** Strictly **FROZEN** (`freeze_vision_tower: true`).
- **Multimodal Projector State:** **Trainable** (`adapt_projector: true`; 2-layer MLP linear layers updated).
- **Language Decoder State:** **LoRA Adapters** applied to attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and feed-forward MLP projections (`gate_proj`, `up_proj`, `down_proj`).
- **Quantization:** 4-bit NormalFloat (NF4) base weights with double quantization; 16-bit brain float (`bfloat16`) compute kernels.

---

### 3.3 Training Dataset Preparation & Patient-Grouped Partitioning
The dataset preparation script ([`scripts/prepare_medgemma_finetuning_data.py`](file:///Users/adityakachhot/interactmed/scripts/prepare_medgemma_finetuning_data.py)) generated standardized JSONL records from verified local Chest-ImaGenome Gold pairings:

| Dataset Partition | Case Count | Unique Patients | Unique Studies | Images per Study | Frontal Projections | Source File |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`train_dev`** | **200** | **199** | **200** | Exactly 1 | 124 AP / 76 PA | [`artifacts/fine_tuning_cohort/train_dev.jsonl`](file:///Users/adityakachhot/interactmed/artifacts/fine_tuning_cohort/train_dev.jsonl) |
| **`val_dev`** | **50** | **49** | **50** | Exactly 1 | 30 AP / 20 PA | [`artifacts/fine_tuning_cohort/val_dev.jsonl`](file:///Users/adityakachhot/interactmed/artifacts/fine_tuning_cohort/val_dev.jsonl) |
| **`expanded_test`** | **117** | **117** | **117** | Exactly 1 | 75 AP / 42 PA | **Strictly locked final test manifest** |

#### Governance & Anti-Leakage Verification
- **Zero Patient Overlap:** $\text{Patients}(\text{train\_dev}) \cap \text{Patients}(\text{val\_dev}) = \emptyset$, and both are disjoint from `expanded_test`.
- **1-to-1 Image-Report Mapping:** Every study ID maps to exactly 1 image. There is zero conflation of multi-view radiographs with study-level reports.
- **Section Completeness:** 100% of cases contain valid `OBSERVATIONS`, `IMPRESSION`, and `LIMITATIONS` sections.

---

### 3.4 Preflight Safety Checks & Verified Outputs
Before any full model training can be permitted, three automated safety gates in [`scripts/train_medgemma_lora.py`](file:///Users/adityakachhot/interactmed/scripts/train_medgemma_lora.py) were executed and verified:

#### Check 1: Model & Module Inspection (`--inspect-only`)
- **Command:** `python scripts/train_medgemma_lora.py --inspect-only`
- **Verification:**
  - Vision tower: 15 modules (27,243,648 parameters), verified **100% frozen**.
  - Multimodal projector: 4 modules (`linear_1`, `linear_2`), verified **trainable**.
  - Language decoder: 41 modules identified for LoRA injection.
- **Result:** **PASSED (Exit code 0)**.

#### Check 2: Dry-Run Forward/Backward Pass (`--dry-run`)
- **Command:** `python scripts/train_medgemma_lora.py --dry-run`
- **Verification:**
  - Label masking verified: 174 tokens masked (`-100` on image and prompt tokens); 80 unmasked tokens on report target. Loss is computed strictly on radiologist text output.
  - Forward pass loss: finite value = **`9.2256`**.
  - Gradient flow: Exactly 36 parameters received active gradients; **zero gradients** flowed to the frozen vision tower.
  - Checkpoint serialization: Verified save and reload of adapter weights to `temp_preflight_ckpt`.
- **Result:** **PASSED (Exit code 0, 3.8s runtime)**.

#### Check 3: Single-Batch Overfit Test (`--overfit-single-batch`)
- **Command:** `python scripts/train_medgemma_lora.py --overfit-single-batch`
- **Verification:** Optimization dynamics tested over 20 steps on a 2-sample batch:
  - Step 1 Loss: **`9.2529`**
  - Step 5 Loss: **`9.1022`**
  - Step 10 Loss: **`8.7114`**
  - Step 15 Loss: **`7.7605`**
  - Step 20 Loss: **`5.8979`**
  - Net Loss Reduction: **−3.3550** (monotonic decrease confirming valid gradient updates).
- **Result:** **PASSED (Exit code 0, 11.2s runtime)**.

---

### 3.5 Critical Training Status Rule

> [!IMPORTANT]
> **CRITICAL ARCHITECTURAL STATUS:**  
> The MedGemma 4B fine-tuning pipeline is **PREPARED; full training has NOT yet been completed.**  
> Successful preflight checks, gradient-flow verification, and single-batch overfit demonstrations verify code and architecture correctness, but do **not** constitute completed model fine-tuning.  
> The 3-epoch full LoRA training run awaits execution on cloud NVIDIA A100 infrastructure ([`notebooks/medgemma_finetune_colab.ipynb`](file:///Users/adityakachhot/interactmed/notebooks/medgemma_finetune_colab.ipynb)). It must **not** be described as completed in documentation or research publications.

---

## 4. Master Experiment Provenance Table

| Experiment ID | Date | Target Model | Dataset & Split | Configuration / Prompt | Hyperparameters | Execution Status | Output Artefact Path | Key Metric / Result | Next Action / Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `EXP-RSNA-01` | 2026-10-10 | TransVG-AGPT | RSNA Pneumonia (`rsna_train_positive`, 4,210 pts) | `configs/rsna_train.yaml`, query="pneumonia" | 3 ep, batch=4, lr=5e-5, L1=5.0, GIoU=2.0 | **COMPLETED** | `checkpoints/finetuned_rsna/run_01_3ep/` | Val IoU: 0.3124 $\to$ **0.4494**; Recall@0.5: 0.2331 $\to$ **0.4928** | Adopted as default production grounding checkpoint. |
| `EXP-RSNA-TEST` | 2026-10-11 | TransVG-AGPT | RSNA Held-Out Test (`rsna_test_positive`, 901 pts) | `configs/rsna_eval.yaml`, query="pneumonia" | Evaluation only, canvas=640 | **COMPLETED** | `artifacts/rsna/runs/test_finetuned_run01/` | Test IoU: 0.3130 $\to$ **0.4530**; Recall@0.5: 0.2397 $\to$ **0.4972** | Confirms domain adaptation on pneumonia; held-out test sealed. |
| `EXP-REP-BASE` | 2026-10-12 | MedGemma 4B | Expanded Dev (`expanded_dev`, 250 cases) | Single-Stage Baseline Prompt | temp=0.1, max_side=1024, num_predict=220 | **COMPLETED** | `artifacts/report_evaluation/expanded_cohort_full/` | Micro-F1: 0.4379, Contradictions: 7.2%, Boilerplate: 56.0% | Replaced by Prompt v3 due to boilerplate template collapse. |
| `EXP-REP-V3` | 2026-10-12 | MedGemma 4B | Expanded Dev (`expanded_dev`, 250 cases) | Prompt v3 Structured (Two-Stage Grounded) | temp=0.1, max_side=1024, Stage 1+2 | **COMPLETED** | `artifacts/report_evaluation/expanded_cohort_full/` | Micro-F1: 0.3220, Contradictions: **1.6%**, Boilerplate: **0.0%**, Latency: **13.42s** | Deployed in production backend (`report_service.py`). |
| `EXP-RES-ABL` | 2026-10-13 | MedGemma 4B | Historical Dev (25 cases) | Resolution Ablation (1024px vs 1536px) | Prompt v3, temp=0.1, keep_alive=30m | **COMPLETED** | `docs/RESOLUTION_ABLATION.md` | F1: 0.3478 vs 0.3478 (Identical); 96% label agreement | Standardized on 1024 px; eliminated memory payload inflation. |
| `EXP-MEDG-PRE` | 2026-10-14 | MedGemma 4B | Fine-Tuning Cohort (`train_dev`, 200 cases) | `configs/medgemma_finetune.yaml` | 4-bit QLoRA, bfloat16, inspect/dry-run/overfit | **COMPLETED** | `docs/MEDGEMMA_FINE_TUNING_RESULTS.md` | Preflight passed; single-batch overfit loss: 9.25 $\to$ 5.89 | Runbook and Colab notebook staged for cloud A100 execution. |
| `EXP-MEDG-FULL`| Staged | MedGemma 4B | Full Fine-Tuning Cohort (200 train / 50 val) | `configs/medgemma_finetune.yaml` | 3 ep, batch=1, grad_accum=16, lr=1e-4 | **PREPARED (STANDBY)** | `checkpoints/medgemma_lora_tier2/` (Pending) | Pending Cloud A100 Execution | Awaiting human engineering authorization. |
