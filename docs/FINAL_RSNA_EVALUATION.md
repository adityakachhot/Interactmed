# Final RSNA Pneumonia Localization Evaluation

**Model:** TransVG-AGPT (Multimodal Medical Phrase Grounding)  
**Dataset:** RSNA Pneumonia Detection Challenge 2018  
**Task:** Phrase-grounded pneumonia bounding-box localization  
**Query:** `"pneumonia"` (frozen before any test evaluation)  
**Date:** 2026-10-11  

---

## 1. Integrity Checks (All Passed Before Test Evaluation)

| # | Check | Evidence | Status |
|---|---|---|---|
| 1 | Test images have ground-truth annotations | 901 unique patients, 2,575 boxes, all `Target==1 class=="Lung Opacity"`, 0 null coordinates | ✅ PASS |
| 2 | Test IDs disjoint from train and val | Overlap(Train∩Test)=0, Overlap(Val∩Test)=0 (verified programmatically) | ✅ PASS |
| 3 | Test data not used for training or checkpoint selection | `best_model_val_iou.pth` selected on val IoU only; test CSV never read during training | ✅ PASS |
| 4 | Query frozen | `"pneumonia"` fixed in `rsna_eval.yaml` before any val or test runs | ✅ PASS |
| 5 | Preprocessing identical across both checkpoints | Same 640×640 letterbox, ClinicalBERT tokenizer, coordinate conversion | ✅ PASS |
| 6 | Original checkpoint unchanged | `checkpoints/transvg.pth` untouched (2.1 GB, sha unchanged) | ✅ PASS |
| 7 | No overwriting of validation outputs | Test saved to separate `test_baseline/` and `test_finetuned_run01/` dirs | ✅ PASS |

---

## 2. Split Statistics

| Split | Unique Patients / Images | Ground-Truth Boxes | Source |
|---|---|---|---|
| **Train** | 4,210 | 11,897 | `rsna_train_positive.csv` |
| **Validation** | 901 | 2,485 | `rsna_val_positive.csv` |
| **Test (held-out)** | 901 | 2,575 | `rsna_test_positive.csv` |
| Seed | 42 | Val fraction 15% | Test fraction 15% |

---

## 3. Evaluation Protocol

- **Matching rule (frozen):** For images with multiple ground-truth boxes (multi-focal pneumonia), the single predicted box is matched to the GT box with the **highest IoU** (`max(IoU(pred, gt_i))` over all GT boxes for the same image). This is the standard PASCAL VOC / MS-COCO single-prediction matching convention.
- **Metrics:** Mean Best-Match IoU, Recall@IoU≥0.50, Recall@IoU≥0.75, Inference Success Rate, Mean Latency.
- **Negative images:** Not evaluated for recall/IoU (model always outputs a box; negatives excluded from metrics).

---

## 4. Final Results

### 4.1 Primary Results Table

| Metric | Val: Pretrained Baseline | Val: Fine-Tuned (3 ep) | **Test: Pretrained Baseline** | **Test: Fine-Tuned (3 ep)** | Test Abs Δ | Test Rel Δ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Evaluated Images | 901 | 901 | **901** | **901** | — | — |
| Excluded | 0 | 0 | **0** | **0** | — | — |
| Inference Success Rate | 100% | 100% | **100%** | **100%** | — | — |
| **Mean Best-Match IoU** | 0.3124 | 0.4494 | **0.3130** | **0.4530** | +0.1400 | +**44.7%** |
| **Recall @ IoU ≥ 0.50** | 0.2331 | 0.4928 | **0.2397** | **0.4972** | +0.2575 | +**107.5%** |
| **Recall @ IoU ≥ 0.75** | 0.0433 | 0.0688 | **0.0555** | **0.0810** | +0.0255 | +**45.9%** |
| Mean Latency | 139.1 ms | 134.3 ms | **145.9 ms** | **140.4 ms** | −5.5 ms | — |

> **Reading the table:** Val metrics were not used to choose test settings. Test results are the independent, held-out, final numbers. Val and test metrics are consistent — no overfitting inflation.

### 4.2 Matching Rule Note for Multi-GT Images

901 test images contain 2,575 ground-truth boxes (≈2.86 boxes/image on average for multi-focal cases). The best-match rule awards the prediction the maximum available IoU across all GT boxes for that image. This rule was frozen before validation evaluation and applied identically to both checkpoints on both splits.

---

## 5. Output Files

| File | Description |
|---|---|
| [`artifacts/rsna/runs/test_baseline/baseline_test.csv`](file:///Users/adityakachhot/interactmed/artifacts/rsna/runs/test_baseline/baseline_test.csv) | Per-image predictions — pretrained baseline |
| [`artifacts/rsna/runs/test_baseline/baseline_test.json`](file:///Users/adityakachhot/interactmed/artifacts/rsna/runs/test_baseline/baseline_test.json) | Summary JSON — pretrained baseline |
| [`artifacts/rsna/runs/test_finetuned_run01/finetuned_test.csv`](file:///Users/adityakachhot/interactmed/artifacts/rsna/runs/test_finetuned_run01/finetuned_test.csv) | Per-image predictions — fine-tuned |
| [`artifacts/rsna/runs/test_finetuned_run01/finetuned_test.json`](file:///Users/adityakachhot/interactmed/artifacts/rsna/runs/test_finetuned_run01/finetuned_test.json) | Summary JSON — fine-tuned |
| [`artifacts/rsna/runs/test_metrics_comparison.png`](file:///Users/adityakachhot/interactmed/artifacts/rsna/runs/test_metrics_comparison.png) | Val vs Test bar chart comparison |
| [`artifacts/rsna/runs/test_finetuned_run01/visualizations/`](file:///Users/adityakachhot/interactmed/artifacts/rsna/runs/test_finetuned_run01/visualizations/) | Per-image prediction overlays |
| [`artifacts/rsna/runs/test_qualitative_comparisons/`](file:///Users/adityakachhot/interactmed/artifacts/rsna/runs/test_qualitative_comparisons/) | Side-by-side baseline vs fine-tuned overlays |

---

## 6. Key Findings

1. **Fine-tuning improves all metrics on the held-out test set**, with Recall@0.50 more than doubling (+107.5%) and mean IoU improving by +44.7%.
2. **Results are consistent between val and test splits**, indicating that fine-tuning did not overfit to the validation selection criterion. The test metrics closely track their validation counterparts (test IoU 0.453 vs val 0.449).
3. **Zero exclusions** — all 901 test images were successfully processed by both checkpoints.
4. **Latency** is essentially unchanged (~140 ms on Apple MPS) since only the fusion transformer and box head were fine-tuned, not the backbone.
5. **Recall@0.75 remains low** (0.081 fine-tuned vs 0.056 baseline). Precise localization to IoU≥0.75 is a known limitation for a single-box grounding model on multi-focal pneumonia.

---

## 7. Limitations

- **Single-box output:** TransVG produces one box per query. Multi-focal pneumonia (bilateral opacities) is inherently under-addressed by a single prediction.
- **No objectness/presence score:** The model always outputs a box regardless of whether the pathology is present. Negative-image handling requires a separate classifier.
- **Frozen backbone:** Only fusion layers and the box head were fine-tuned. Further gains may be achievable by partially unfreezing the DETR encoder.
- **Three-epoch training:** Model was trained for 3 epochs only due to hardware constraints. Additional epochs may improve recall@0.75.
- **RSNA-specific:** These results are specific to the RSNA 2018 pneumonia task. Performance on other pathologies is not characterized here.

---

## 8. Evaluation Commands Used

```bash
# Baseline test evaluation
/Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/evaluate_rsna.py \
    --config configs/rsna_eval.yaml \
    --split test \
    --checkpoint checkpoints/transvg.pth \
    --output-dir artifacts/rsna/runs/test_baseline \
    --run-name baseline_test \
    --model-label "Pretrained Baseline TransVG-AGPT"

# Fine-tuned test evaluation
/Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/evaluate_rsna.py \
    --config configs/rsna_eval.yaml \
    --split test \
    --checkpoint checkpoints/finetuned_rsna/run_01_3ep/best_model_val_iou.pth \
    --output-dir artifacts/rsna/runs/test_finetuned_run01 \
    --run-name finetuned_test \
    --model-label "Fine-Tuned TransVG-AGPT (Run 01, 3 Epochs)" \
    --save-viz
```
