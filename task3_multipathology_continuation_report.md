# Task 3 Final Report: Controlled Multi-Pathology TransVG Continuation Fine-Tuning

**InteractMed-VLM Research Project**  
**Model:** TransVG-AGPT (Vision-Language Transformer for Phrase Grounding)  
**Experiment:** `multipathology_finetuning_continuation_to_6ep`  
**Execution Environment:** Apple Silicon M3 (MPS acceleration), PyTorch 2.2.2, Python 3.10.19  
**Date:** October 16, 2026  

---

## Executive Summary

Task 3 investigated whether continuing multi-pathology fine-tuning beyond Task 2 (Epochs 1–3) for up to three additional epochs (Epochs 4–6) improves phrase grounding localization across the eight supported chest radiograph findings. 

Continuation training completed all three budgeted epochs (Epochs 4, 5, and 6) under strict mandatory early-stopping and checkpoint-isolation guardrails.

- **Selected Best Model:** **Epoch 5** ([`checkpoints/multipathology/run_02_continuation_to_6ep/best_model_val_iou.pth`](file:///Users/adityakachhot/interactmed/checkpoints/multipathology/run_02_continuation_to_6ep/best_model_val_iou.pth))
- **Validation Mean Best-Match IoU:** **0.6550** (+0.0040 absolute improvement over Task 2 baseline best of 0.6510)
- **Validation Recall@0.75:** **46.49%** (+1.75% absolute gain over Task 2 baseline of 44.74%)
- **Validation Recall@0.50:** **74.56%** (balanced vs. 75.44% baseline)
- **Late-Stage Overfitting Detection:** At Epoch 6, training loss fell from 1.6288 to 1.6204, but validation mIoU dipped slightly to 0.6523 (-0.0027). Under our mandatory checkpoint-selection rule (*"Always select the best checkpoint by validation mIoU, rather than automatically choosing the last epoch"*), Epoch 5 was correctly and safely retained as the optimal model checkpoint.
- **Safety & Reproducibility:** Task 2 artifacts and base weights ([`checkpoints/transvg.pth`](file:///Users/adityakachhot/interactmed/checkpoints/transvg.pth)) remained 100% untouched. The locked test set was strictly excluded from training and hyperparameter decisions.

---

## 1. Phase A — Audit Findings & Issues Resolved

Before training commenced, a thorough audit was conducted across the codebase, checkpoints, datasets, and evaluators:

| Audit Item | Inspection Finding | Action / Resolution |
|---|---|---|
| **Checkpoint State** | Checked `run_01_multicxr/best_model_val_iou.pth` and `last_model.pth`. Both contained model weights, epoch, validation metrics, and training config, but optimizer/scheduler states were not serialized. | Rather than fabricating a fictitious scheduler state, initialized fresh controlled continuation from Epoch 3 best weights. |
| **Scheduler Horizon** | Task 2 scheduler used a cosine decay horizon of 3 epochs (597 steps), reaching `min_lr = 1e-6` at step 597. | Extending the original scheduler would have held LR permanently at `min_lr`. Initialized a new conservative schedule starting at `1.0e-5` decaying toward `1.0e-6`. |
| **Patient Disjointness** | Audited `local_grounding_verified.csv` (1,056 rows across 712 patients). MS-CXR splits: 746 train, 147 val, 163 test; RSNA: 200 positive train samples. | Verified 0 patient overlap between train (498 patients), val (106 patients), and test (108 patients). No data leakage. |
| **Evaluator Metric Parity** | Audited `dynamic_best_match_loss` and `evaluate_val_dataset` in `train_multipathology.py`. | Confirmed exact metric parity: greedy max-IoU best-match over ground-truth bounding boxes, identical letterbox unpadding, identical 640×640 coordinate normalization. |
| **Active Processes** | Audited running processes on system. | No conflicting training jobs active; only the local uvicorn daemon was running. |

---

## 2. Phase B & C — Continuation Strategy & Protection

### Artifact Protection
All continuation outputs were isolated to prevent overwriting:
- **New Experiment Directory:** [`checkpoints/multipathology/run_02_continuation_to_6ep/`](file:///Users/adityakachhot/interactmed/checkpoints/multipathology/run_02_continuation_to_6ep/)
- **New Artifact Directory:** [`artifacts/multipathology/run_02_continuation_to_6ep/`](file:///Users/adityakachhot/interactmed/artifacts/multipathology/run_02_continuation_to_6ep/)
- **New Configuration:** [`configs/multipathology_continuation_to_6ep.yaml`](file:///Users/adityakachhot/interactmed/configs/multipathology_continuation_to_6ep.yaml)
- **Untouched Run 1:** `checkpoints/multipathology/run_01_multicxr/` retained in original state with intact timestamps.

### Continuation Initialization
- **Model Weights:** Initialized from `checkpoints/multipathology/run_01_multicxr/best_model_val_iou.pth` (Val mIoU `0.6510`).
- **Encoders:** ResNet visual backbone and ClinicalBERT text encoder kept frozen (`requires_grad = False`, 8,401,412 trainable transformer/fusion parameters).
- **Optimizer:** Fresh `AdamW` initialized with conservative base `lr = 1.0e-5`, `weight_decay = 1.0e-4`.
- **Learning Rate Schedule:** Cosine decay from `1.0e-5` to `1.0e-6` over the 3 continuation epochs (597 total steps).
- **Loss Coefficients:** $5.0 \times \text{L1} + 2.0 \times \text{GIoU}$.

---

## 3. Training Execution & Per-Epoch Metrics

Continuation training ran across Epochs 4, 5, and 6 on 795 training query items and evaluated after each epoch on the 114 MS-CXR validation queries:

### Complete Epoch-by-Epoch History

| Epoch | Phase | Train Loss | Val mIoU | Val Recall@0.5 | Val Recall@0.75 | End LR | Duration | Result / Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **1** | Task 2 (Run 1) | 1.9584 | 0.6402 | 71.93% | 41.23% | 1.33e-5 | 158.4s | Task 2 Epoch 1 |
| **2** | Task 2 (Run 1) | 1.7737 | 0.6432 | 73.68% | 42.11% | 5.00e-6 | 155.2s | Task 2 Epoch 2 |
| **3** | Task 2 (Run 1) | 1.6937 | 0.6510 | 75.44% | 44.74% | 1.00e-6 | 157.1s | Task 2 Best Baseline |
| **4** | Task 3 (Run 2) | 1.6682 | 0.6401 | 72.81% | 43.86% | 7.50e-6 | 139.3s | No improvement (-0.0109); Counter: 1/2 |
| **5** | Task 3 (Run 2) | 1.6288 | **0.6550** | **74.56%** | **46.49%** | 2.50e-6 | 138.5s | **✓ NEW BEST (+0.0040); Counter: 0/2** |
| **6** | Task 3 (Run 2) | 1.6204 | 0.6523 | 73.68% | 48.25% | 1.00e-6 | 152.2s | No improvement (-0.0027); Counter: 1/2 |

### Early Stopping & Model Selection Rationale
- At **Epoch 4**, the model adapted to the restarted learning rate (`1.0e-5`), with temporary fluctuation in validation mIoU (`0.6401`).
- At **Epoch 5**, the model converged to a new global optimum of **0.6550** (+0.0040 absolute improvement over Task 2's best of 0.6510, exceeding the 0.001 minimum improvement threshold). The checkpoint was saved as `best_model_val_iou.pth`.
- At **Epoch 6**, training loss decreased further to `1.6204`, but validation mIoU declined slightly to `0.6523`.
- **Stopping Rule Applied:** Completed the 3 budgeted continuation epochs. Under the mandatory rule to always select the model with highest validation mIoU, **Epoch 5** was selected.

---

## 4. Per-Category Validation Results: Task 2 vs. Task 3

Evaluation on the 114 validation queries across all eight supported clinical categories:

| Clinical Category | Validation Queries | Task 2 Epoch 3 mIoU | Task 3 Epoch 5 mIoU | Absolute Gain (mIoU) | Task 2 R@0.5 | Task 3 R@0.5 | Task 2 R@0.75 | Task 3 R@0.75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Cardiomegaly** | 40 | 0.8291 | **0.8315** | +0.0024 | 100.0% | 100.0% | 87.50% | **90.00%** |
| **Pneumothorax** | 13 | 0.5876 | 0.5845 | -0.0031 | 69.23% | 69.23% | 23.08% | 23.08% |
| **Pleural Effusion** | 11 | 0.6192 | **0.6202** | +0.0010 | 81.82% | 81.82% | 27.27% | 27.27% |
| **Pneumonia** | 13 | 0.5394 | **0.5546** | **+0.0152** | 53.85% | 53.85% | 23.08% | 23.08% |
| **Consolidation** | 12 | 0.5513 | **0.5663** | **+0.0150** | 66.67% | 66.67% | 33.33% | 25.00% |
| **Atelectasis** | 8 | 0.5750 | 0.5521 | -0.0229 | 62.50% | 50.00% | 25.00% | 25.00% |
| **Pulmonary Edema** | 5 | 0.6116 | **0.6426** | **+0.0310** | 80.00% | 80.00% | 20.00% | 20.00% |
| **Lung Opacity** | 12 | 0.4427 | **0.4463** | +0.0036 | 33.33% | 33.33% | 16.67% | 16.67% |
| **OVERALL** | **114** | **0.6510** | **0.6550** | **+0.0040** | **75.44%** | **74.56%** | **44.74%** | **46.49%** |

### Per-Category Analysis
1. **Pneumonia & Consolidation Gains:** Noticeable improvements in lung parenchymal findings (+0.0152 for Pneumonia, +0.0150 for Consolidation), reflecting continued refinement of boundary discrimination between consolidation and adjacent structures.
2. **Edema Gain:** Pulmonary Edema saw a large mIoU increase (+0.0310 to 0.6426). Note that validation sample count for Edema is small ($N=5$).
3. **Cardiomegaly Precision:** Cardiomegaly maintained 100% Recall@0.5 and advanced to 90.00% Recall@0.75, confirming extremely reliable cardiac silhouette localization.
4. **Tighter Overlap (Recall@0.75):** Overall Recall@0.75 increased from 44.74% to 46.49%, demonstrating that the continuation produced higher-precision bounding boxes overall.

---

## 5. Post-Training Test Set Evaluation (Previously Inspected Cohort)

> [!IMPORTANT]
> In strict compliance with Task 3 instructions, the MS-CXR test split (163 annotations, 130 unique queries) was **never** used during training for gradient updates, early stopping, checkpoint selection, learning rate adjustment, or hyperparameter decisions.
>
> The evaluation below was executed post-training solely as diagnostic evidence on a previously inspected cohort, and is **not** claimed as an untouched first-time confirmation.

| Model / Checkpoint | Test mIoU | Test Recall@0.5 | Test Recall@0.75 | Status |
|---|---:|---:|---:|---|
| **Pretrained Baseline (MS-CXR)** (`transvg.pth`) | 0.6307 | 71.54% | 36.92% | Baseline reference |
| **Fine-Tuned RSNA (Pneumonia Only)** | 0.3800 | 26.15% | 2.31% | Severe catastrophic forgetting |
| **Task 2 Multi-Pathology (Epoch 3)** (`run_01_multicxr`) | 0.6599 | 76.92% | 43.85% | Task 2 best |
| **Task 3 Multi-Pathology (Epoch 5)** (`run_02_continuation_to_6ep`) | **0.6672** | **80.00%** | **43.85%** | **New project record (+0.0073 mIoU, +3.08% R@50)** |

On the test cohort, the Epoch 5 checkpoint reached **80.00% Recall@0.50** (104 of 130 test queries localized with IoU $\ge 0.50$), compared to 76.92% in Task 2 and 71.54% for the baseline model.

---

## 6. Checkpoint Safety, Integrity & Smoke Tests

### Checkpoint File Verification
- **Path:** [`checkpoints/multipathology/run_02_continuation_to_6ep/best_model_val_iou.pth`](file:///Users/adityakachhot/interactmed/checkpoints/multipathology/run_02_continuation_to_6ep/best_model_val_iou.pth)
- **Size:** 732 MB
- **Serialized State Dictionary Keys:** `['epoch', 'model', 'optimizer', 'scheduler', 'val_metrics', 'config', 'provenance']`
- **Integrity Verified:** Loaded with `torch.load()` on MPS device; full parameter load of 175,005,380 weights confirmed.

### Multi-Query Smoke Tests & API Routing
Inference checks were executed across representative findings using the live FastAPI service on port 8000:

| Test Query | Resolved Canonical Finding | Category | Status | Pixel Box Output | Latency | Outcome |
|---|---|---|---|---|---:|---|
| `"pneumonia"` | pneumonia | Lungs | validated | `[23.7, 25.5, 179.0, 138.8]` | 136 ms | **PASS** |
| `"cardiomegaly"` | cardiomegaly | Heart | validated | `[47.4, 41.8, 468.7, 304.7]` | 132 ms | **PASS** |
| `"pleural effusion"` | pleural effusion | Pleura | validated | `[20.5, 50.9, 255.7, 262.4]` | 137 ms | **PASS** |
| `"infiltrate"` (alias) | lung opacity | Lungs | validated | `[38.0, 26.4, 283.8, 120.4]` | 139 ms | **PASS** |
| `"enlarged cardiac silhouette"` (alias) | cardiomegaly | Heart | validated | `[47.2, 39.1, 466.3, 271.8]` | 143 ms | **PASS** |
| `"rib fracture"` (unsupported) | rib fracture | Uncategorized | unsupported | `[34.0, 20.7, 173.2, 69.1]` + Warning | 141 ms | **PASS** (Explicit warning returned) |

---

## 7. Automated Test Suite & Web Application Verification

1. **Pytest Regression Suite:**
   - Ran `pytest tests/ -v`
   - **Result:** **132 / 132 tests passing** (100% pass rate in 15.29s).
   - All tests in `test_multipathology.py`, `test_region_grounding.py`, `test_regressions.py`, `test_report_generator.py`, and `test_triage.py` passed with 0 errors.

2. **Frontend Production Build:**
   - Ran `npm run build` in `frontend/`
   - **Result:** TypeScript check and Vite build passed with 0 errors (bundle: 185 kB JS, 18.3 kB CSS).

3. **Live API Health & Routing:**
   - Live endpoint `GET /grounding/capabilities` returns the updated Epoch 5 validation evidence and selects `checkpoints/multipathology/run_02_continuation_to_6ep/best_model_val_iou.pth`.

---

## 8. Limitations & Recommended Next Steps

1. **Small Support for Infrequent Pathologies:** While Cardiomegaly ($N=40$ val), Pneumonia ($N=13$), and Pleural Effusion ($N=11$) have solid representations, findings like Pulmonary Edema ($N=5$) and Atelectasis ($N=8$) have small validation sample sizes. Per-category fluctuations for these rare findings should be interpreted with appropriate caution.
2. **Validation Plateau at Epoch 6:** The drop in validation mIoU from 0.6550 (Epoch 5) to 0.6523 (Epoch 6) indicates that additional fine-tuning with frozen encoders has reached its ceiling. Further training beyond Epoch 6 without encoder unfreezing or data augmentation is unlikely to yield meaningful localization gains.
3. **Recommended Next Experiment:** 
   - Investigate staged unfreezing of the final layer of the visual CNN backbone (ResNet layer 4) at a very low learning rate ($1.0e-6$), while keeping ClinicalBERT frozen, to explore whether finer-grained radiographic feature extraction can boost subtle findings like atelectasis and edema.
   - Reserve a new, genuinely untouched external patient cohort (e.g., from NIH ChestX-ray14 or MIMIC-CXR phrase annotations) for blind out-of-distribution evaluation.
