# InteractMed-VLM — Phase 4A: Fine-Tuning Feasibility Audit

**Project:** InteractMed-VLM  
**Model:** TransVG-AGPT (`TransVG_ca`)  
**Target Benchmark:** RSNA Pneumonia Detection Challenge 2018  
**Checkpoint:** `checkpoints/transvg.pth` (2.10 GB, 175,005,380 parameters)  
**Date:** October 10, 2026  
**Status:** Audit Complete — Awaiting User Confirmation Before Implementation  

---

## Executive Summary

This feasibility audit investigates the internal architecture, loss formulations, parameter distribution, and training requirements for fine-tuning the pretrained **TransVG-AGPT** phrase-grounding model on the **RSNA Pneumonia Detection Challenge** dataset.

The core objective is **task adaptation of medical phrase grounding** to pneumonia localization, **not** causal language model fine-tuning (QLoRA). The model maps a chest radiograph and the text query `"pneumonia"` to bounding box coordinates $[x_c, y_c, w, h] \in [0, 1]^4$.

---

## 1. Verified Model Architecture & Checkpoint Structure

### 1.1 Forward Method & Tensor Flow (`TransVG_ca`)

Defined in [`third_party/AGPT/models/transvg/trans_vg_ca.py`](file:///Users/adityakachhot/interactmed/third_party/AGPT/models/transvg/trans_vg_ca.py):

```
Input Image (3×640×640) ────► Visual Backbone (ResNet-50 + DETR Encoder) ────► visu_src (400, B, 256)
                                                                                  ▲
Input Text ("pneumonia") ───► Text Backbone (ClinicalBERT) ─────────────────► text_src (20, B, 256)
                                                                                  ▲
Learnable Token ────────────► reg_token Embedding (1, 256) ──────────────────► tgt_src  (1, B, 256)
                                                                                  │
                                     ┌────────────────────────────────────────────┘
                                     ▼
                      Concat [tgt_src, text_src, visu_src] (421, B, 256)
                                     │ + vl_pos_embed (421, B, 256)
                                     ▼
                      Multimodal Transformer (6 layers, 8 heads)
                                     │
                                     ▼
                      Output Tokens vg_hs (421, B, 256)
                                     │
                         vg_hs[0] = vg_reg (B, 256)
                                     │
                                     ▼
                         MLP Head (256 ─► 256 ─► 256 ─► 4) + Sigmoid
                                     │
                                     ▼
                     pred_boxes: [cx, cy, w, h] ∈ [0, 1]  (B, 4)
```

### 1.2 Parameter Distribution & Component Breakdown

Total parameters: **175,005,380** (all 523 weight tensors verified):

| Component | Sub-modules | Parameters | % of Total | Trainable in Stage 1 |
|---|---|---|---|---|
| **Text Encoder (`textmodel`)** | ClinicalBERT (`medicalai/ClinicalBERT`) | 134,734,080 | 76.99% | ❌ Frozen |
| **Visual Encoder (`visumodel`)** | ResNet-50 CNN + DETR Encoder | 31,869,888 | 18.21% | ❌ Frozen |
| **Multimodal Transformer (`vl_transformer`)** | 6 Encoder layers (d=256, nhead=8, ffn=2048) | 7,890,432 | 4.51% | ✅ **Trainable** |
| **Visual Projection (`visu_proj`)** | `nn.Linear(256, 256)` | 65,792 | 0.04% | ✅ **Trainable** |
| **Text Projection (`text_proj`)** | `nn.Linear(768, 256)` | 196,864 | 0.11% | ✅ **Trainable** |
| **Position Embedding (`vl_pos_embed`)** | `nn.Embedding(451, 256)` | 115,456 | 0.07% | ✅ **Trainable** |
| **Regression Token (`reg_token`)** | `nn.Embedding(1, 256)` | 256 | 0.00% | ✅ **Trainable** |
| **Box Head (`bbox_embed`)** | 3-layer MLP (`Linear` + `ReLU`) | 132,612 | 0.08% | ✅ **Trainable** |
| **Total Multimodal Fusion & Head** | — | **8,401,412** | **4.80%** | **Stage 1 Target** |

### 1.3 Checkpoint Architecture State (`checkpoints/transvg.pth`)

Inspection of the pretrained file revealed its complete training state:
- **Trained Epoch:** 54 (of 90 maximum epochs)
- **Validation Metrics at Checkpoint:** Accu = $69.41\%$, mIoU = $0.6110$
- **Original Optimizer:** AdamW (`lr=5e-5`, `weight_decay=1e-4`) with 4 parameter groups:
  - Group 0 (Fusion & Head): `lr = 5e-5` (84 parameters)
  - Group 1 (Visual CNN): `lr = 1e-5` (42 parameters)
  - Group 2 (Visual Transformer): `lr = 1e-5` (74 parameters)
  - Group 3 (ClinicalBERT): `lr = 1e-5` (100 parameters)
- **LR Scheduler:** StepLR (`step_size=60`, `gamma=0.1`)

---

## 2. What Code Can Be Reused Safely

1. **Model Definition:**
   [`models/transvg/trans_vg_ca.py`](file:///Users/adityakachhot/interactmed/third_party/AGPT/models/transvg/trans_vg_ca.py), [`models/transvg/vl_transformer.py`](file:///Users/adityakachhot/interactmed/third_party/AGPT/models/transvg/vl_transformer.py), and sub-modules require **zero architectural modifications**.
2. **Loss Math:**
   [`utils/losses.py`](file:///Users/adityakachhot/interactmed/third_party/AGPT/utils/losses.py) and [`utils/box_ops.py`](file:///Users/adityakachhot/interactmed/third_party/AGPT/utils/box_ops.py) implement coordinate conversions (`xywh2xyxy`), `generalized_box_iou`, and L1 loss.
3. **Albumentations Augmentation Pipelines:**
   The training transforms from [`dataloader/mscxr_dataloader.py:get_transforms(args, 'train')`](file:///Users/adityakachhot/interactmed/third_party/AGPT/dataloader/mscxr_dataloader.py#L122-L153) can be reused:
   - `LongestMaxSize(640)`
   - `Affine(translate_percent=(-0.02, 0.02), rotate=(-2, 2))`
   - `RandomCrop(height=635, width=635, p=0.5)`
   - `PadIfNeeded(640, 640)`
   - `Normalize(mean=0.485, std=0.229)`
4. **Coordinate De-letterboxing:**
   The inverse letterbox transformation verified in Phase 3B ([`scripts/evaluate_rsna.py`](file:///Users/adityakachhot/interactmed/scripts/evaluate_rsna.py)) correctly maps model normalized canvas coordinates back to native DICOM image pixels.

---

## 3. What Training Components Are Missing

The upstream `third_party/AGPT` repository is an **inference/demo release** only; it does not contain an active training pipeline:

1. **No Training Loop (`train.py`):** Missing entirely from `third_party/AGPT/`.
2. **No RSNA Dataset Loader:** Existing loaders (`mscxr_dataloader.py`, `mimic_dataloader.py`) assume pre-saved `.pth` pickle dumps and JPEG paths, whereas RSNA uses native DICOM `.dcm` files and CSV label tables.
3. **No Validation / Checkpointing Callback:** No code to evaluate mIoU / Recall@0.50 during training or checkpoint only the best validation model.
4. **No Mixed-Precision (AMP) Context:** Upstream code did not wrap forward passes in `torch.cuda.amp.autocast()` or `torch.autocast(device_type="mps")`.

---

## 4. Single-Box vs. Multi-Box RSNA Ground Truth

### 4.1 Structural Constraint
TransVG uses **a single regression token (`reg_token`, shape `(1, B, 256)`)** passing into an MLP head. It outputs **exactly one bounding box per forward pass**. It does not possess multiple object queries or candidate heads.

### 4.2 Multi-Box Data in RSNA
In the RSNA training split (`rsna_train_positive.csv`), 3,398 positive patients have $> 1$ bounding box (bilateral or multi-focal pneumonia, up to 4 boxes per image).

### 4.3 Proposed Supervised Objective: Dynamic Minimum-Loss Assignment (Best-Match GT)

We evaluated three potential strategies:

| Strategy | Description | Pros | Cons / Scientific Risk |
|---|---|---|---|
| **A. Bounding Box Union** | Enclose all lesions into one large bounding box. | Single target per image. | ❌ Flawed for bilateral pneumonia: the box encompasses the heart and mediastinum, teaching the model to predict the entire chest. |
| **B. Instance Replication** | Duplicate the image $K$ times for $K$ boxes. | Trivial to implement. | ❌ Contradictory gradients: identical visual/text input is pulled in opposite directions in consecutive steps. |
| **C. Dynamic Best-Match (Recommended)** | For an image with GT boxes $\{g_1, \dots, g_K\}$ and prediction $\hat{b}$, select $g^* = \arg\min_k \mathcal{L}_{\text{box}}(\hat{b}, g_k)$. | Aligns training objective with the best-overlap inference evaluation rule. Stable convergence. | Requires computing loss against all $K$ boxes before backprop. |

**Formulation for Strategy C:**
$$\mathcal{L}_{\text{sample}}(\hat{b}, \{g_k\}_{k=1}^K) = \min_{k \in \{1,\dots,K\}} \left[ 5 \cdot \|\hat{b} - g_k\|_1 + 2 \cdot (1 - \text{GIoU}(\hat{b}, g_k)) \right]$$

This objective penalizes the model based on its proximity to the *nearest* true pneumonia lesion, encouraging confident focal localization rather than spatial blurring.

---

## 5. Handling Negative Images Without Inventing Boxes

1. **Model Fact:** TransVG contains **no classification branch**. It outputs an unconditional coordinate tuple $[cx, cy, w, h] \in [0, 1]^4$.
2. **Negative Image Fact:** RSNA has 20,672 negative images without annotations.
3. **Resolution:**
   - **Do not invent artificial ground-truth boxes** (e.g., $[0,0,0,0]$ or $[0,0,1,1]$) for negative images. Doing so disrupts spatial regression manifolds.
   - Following standard visual phrase grounding literature (RefCOCO, MS-CXR), phrase grounding training is conducted strictly on **positive (phrase-present) images**.
   - The 4,210 training positive images in `data/processed/rsna/rsna_train_positive.csv` provide sufficient supervision for bounding box refinement.
   - Negative images remain reserved for the post-grounding urgency triage/classification layer in Phase 5.

---

## 6. Staged Training Strategy & Encoder Freezing

To prevent catastrophic forgetting and maximize computational efficiency, fine-tuning will follow a staged schedule:

### Stage 1: Fusion & Box Head Tuning (Lightweight, Recommended First Step)
- **Freeze:**
  - `visumodel` (ResNet-50 + DETR Encoder): `requires_grad = False` (31.9M params)
  - `textmodel` (ClinicalBERT): `requires_grad = False` (134.7M params)
- **Train Only:**
  - `vl_transformer` (multimodal transformer)
  - `bbox_embed` (MLP head)
  - `visu_proj`, `text_proj`, `reg_token`, `vl_pos_embed`
- **Trainable Parameters:** **8,401,412 (4.80%)**
- **Benefits:**
  - Memory footprint $< 3.5\text{ GB}$.
  - Zero degradation to general visual feature extraction and clinical phrase semantics.
  - Can run quickly on Google Colab T4 GPU or locally on Apple Silicon MPS for smoke testing.

### Stage 2: End-to-End Fine-Tuning (Optional Cloud Ablation)
- Unfreeze the visual transformer (`lr_visu_tra = 5e-6`) while keeping ResNet-50 backbone and ClinicalBERT lower layers frozen.
- Train with cosine decay and linear warmup on Google Colab GPU.

---

## 7. Checkpoint Loading & Safety Protocol

1. **Read-Only Master Checkpoint:**
   `checkpoints/transvg.pth` is treated as strictly read-only.
2. **Loading Verification:**
   Weights are loaded via:
   ```python
   ckp = torch.load("checkpoints/transvg.pth", map_location="cpu")
   model.load_state_dict(ckp["model"], strict=True)
   ```
   `strict=True` must pass with 0 missing and 0 unexpected keys.
3. **New Output Hierarchy:**
   All fine-tuning artifacts will be saved to dedicated, git-ignored directories:
   `checkpoints/finetuned_rsna/`
   - `best_model_val_iou.pth` (saved whenever validation mIoU improves)
   - `latest_checkpoint.pth`
   - `training_log.json`

---

## 8. Hardware Feasibility & Compute Budgets

| Execution Target | Mode | Batch Size | Memory Required | Estimated Epoch Time |
|---|---|---|---|---|
| **Apple M3 (MPS)** | Stage 1 (Frozen Encoders) Smoke Test (100 steps) | 4 | ~3.2 GB | ~45 seconds |
| **Apple M3 (MPS)** | Full Training (4,210 images × 10 epochs) | 4 | ~4.5 GB | ~25 min/epoch (Thermal risk) |
| **Google Colab (NVIDIA T4)** | Stage 1 & 2 Full Training | 8 or 16 | ~6.5 GB (AMP FP16) | ~4 min/epoch |

**Recommendation:** Execute smoke-testing on Apple M3 (MPS), and run full 15-epoch training on Google Colab with NVIDIA GPU.

---

## 9. Next Steps: Commands to Be Implemented

Upon confirmation, Phase 4B will implement:

1. **Training Script:** [`scripts/train_rsna.py`](file:///Users/adityakachhot/interactmed/scripts/train_rsna.py)
   - Supports `--freeze-encoders` (Stage 1 default).
   - Supports `--device mps` and `--device cuda`.
   - Implements dynamic best-match GIoU+L1 loss.
   - Evaluates on `rsna_val_positive.csv` every epoch.
2. **Smoke Test Command:**
   ```zsh
   /Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/train_rsna.py \
       --config configs/rsna_eval.yaml \
       --smoke-test \
       --epochs 1 \
       --batch-size 4 \
       --freeze-encoders \
       --device mps
   ```
3. **Colab Fine-Tuning Notebook:** [`notebooks/rsna_finetuning_colab.ipynb`](file:///Users/adityakachhot/interactmed/notebooks/rsna_finetuning_colab.ipynb)
   - Complete standalone notebook for high-throughput GPU training on Google Colab.
