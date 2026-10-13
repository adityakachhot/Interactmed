# Final Held-Out RSNA Test Evaluation Report
**Model:** TransVG-AGPT (Multimodal Medical Phrase Grounding)  
**Dataset:** RSNA Pneumonia Detection Challenge 2018 (Held-Out Test Split, N = 901 positive images)  
**Query:** `"pneumonia"`  
**Task:** Pneumonia Bounding Box Localization  

---

## 1. Executive Summary

This report documents the final evaluation of the **TransVG-AGPT** model on the **held-out RSNA test split**. The test split contains 901 unique positive chest radiographs (2,575 ground-truth annotations) completely disjoint from training (4,210 images) and validation (901 images).

Both the original pretrained checkpoint (`checkpoints/transvg.pth`) and the fine-tuned checkpoint (`checkpoints/finetuned_rsna/run_01_3ep/best_model_val_iou.pth`) were evaluated using identical preprocessing, queries, and matching protocols.

---

## 2. Quantitative Comparison Table

| Metric | Val: Pretrained Baseline | Val: Fine-Tuned (Epoch 3) | Test: Pretrained Baseline | Test: Fine-Tuned (Epoch 3) | Test Relative Δ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Evaluated Images** | 901 | 901 | 901 | 901 | — |
| **Mean Best-Match IoU** | 0.3124 | 0.4494 | 0.3130 | 0.4530 | **+44.73%** |
| **Recall @ IoU ≥ 0.50** | 0.2331 | 0.4928 | 0.2397 | 0.4972 | **+107.43%** |
| **Recall @ IoU ≥ 0.75** | 0.0433 | 0.0688 | 0.0555 | 0.0810 | **+45.95%** |
| **Success Rate** | 100.0% | 100.0% | 100.0% | 100.0% | — |
| **Mean Latency (ms)** | 139.1 ms | 134.3 ms | 145.9 ms | 140.3 ms | — |

---

## 3. Metrics Visualization

![Test Metrics Comparison](artifacts/rsna/runs/test_metrics_comparison.png)

---

## 4. Key Findings & Analysis

1. **Generalization to Unseen Patient Cohort:**  
   The fine-tuned model's performance on the held-out test split tracks its validation performance closely, confirming that the parameter updates in the multimodal fusion transformer and box prediction head generalize without severe overfitting.

2. **Recall@0.50 Impact:**  
   Fine-tuning delivers substantial gains in clinical diagnostic alignment (IoU ≥ 0.50), significantly improving true-positive regional localization.

3. **Inference Latency Consistency:**  
   Because fine-tuning updated only the fusion layers and box prediction head without altering architecture dimensions or FLOPS, per-image latency remained virtually identical (~135-140 ms on Apple Silicon MPS).

---

## 5. Limitations & Future Work

- **Single Box Output:** TransVG produces a single bounding box per query. For patients with multi-focal pneumonia (bilateral opacity), only one focus can be bound.
- **Negative Rejection:** As a grounding model, TransVG does not incorporate an objectness or presence classification head; it assumes the queried pathology is present.
