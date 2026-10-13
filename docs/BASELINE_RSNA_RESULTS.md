# InteractMed-VLM: Zero-Shot RSNA Pneumonia Localization Baseline

**Phase:** Phase 3B — Zero-Shot External Evaluation  
**Model:** TransVG-AGPT (pretrained, weights frozen — no fine-tuning)  
**Dataset:** RSNA Pneumonia Detection Challenge 2018  
**Task:** Pneumonia localization (external grounding evaluation)  
**Config:** [`configs/rsna_eval.yaml`](file:///Users/adityakachhot/interactmed/configs/rsna_eval.yaml)  
**Eval script:** [`scripts/evaluate_rsna.py`](file:///Users/adityakachhot/interactmed/scripts/evaluate_rsna.py)  

> [!IMPORTANT]
> This is an **external pneumonia-localization evaluation only**. It assesses how well
> the pretrained TransVG-AGPT checkpoint can localize the query "pneumonia" on an unseen
> dataset. It does NOT evaluate general chest X-ray finding localization, anatomy grounding,
> or multi-finding performance.

---

## 1. Evaluation Design & Threat Model

### 1.1 What This Evaluation Measures

TransVG-AGPT was trained on MS-CXR (clinical phrase grounding of 8 pathology categories)
and Chest ImaGenome (anatomical region grounding). The RSNA Pneumonia Detection Challenge
dataset provides an **independent, external benchmark** with:

- Bounding boxes annotated by board-certified radiologists
- ~26,684 images (6,012 positive / 20,672 negative)
- A single pathology class: pneumonia (airspace opacity / lung opacity)
- No patient overlap with MS-CXR or Chest ImaGenome (verified by patientId)

This evaluation quantifies **zero-shot transfer** — how well a model trained on a related
but distinct dataset generalizes to an unseen annotation source with the same clinical domain.

### 1.2 Scope Limitations (Do Not Overstate)

| Claim | Status |
|---|---|
| Model localizes pneumonia in chest X-rays | ✅ Measured here |
| Model was trained on RSNA data | ❌ False — zero-shot only |
| Results generalize to all chest X-ray findings | ❌ Out of scope |
| Model detects presence/absence of pneumonia | ❌ Out of scope — grounding only |
| Negative images confirm pneumonia absence | ❌ Explicitly not inferred |

### 1.3 Negative Image Protocol

The model always outputs a predicted bounding box regardless of whether pneumonia is present.
On negative images (class ≠ "Lung Opacity"), **no recall or IoU metric is computed**.
Negative-image predictions are:
- Processed and recorded in a separate section of the results JSON
- **Not** used to infer pneumonia absence or presence
- Reported as "inference on negative image — box meaningless" in the results

---

## 2. Fixed Evaluation Protocol (Frozen Before Test Evaluation)

> [!WARNING]
> The query, preprocessing, matching rule, and IoU thresholds below were
> fixed **before the test split was evaluated**. Any changes after test evaluation
> would invalidate the protocol and constitute result cherry-picking.

### 2.1 Primary Query

```
"pneumonia"
```

This query is passed verbatim to the ClinicalBERT tokenizer inside TransVG-AGPT.
It is the single word that best matches the RSNA annotation class label ("Lung Opacity"
annotated as pneumonia). It was chosen before any evaluation was run.

### 2.2 Secondary Query (Val Split Only)

```
"airspace opacity"
```

This secondary query was evaluated on the **validation split only** to characterize
query sensitivity. Its results do **not** influence the primary metrics or the test split.

### 2.3 Preprocessing Pipeline

Images are preprocessed using the **unmodified AGPT val transforms** from
`third_party/AGPT/dataloader/mscxr_dataloader.py`:

```python
transforms = [
    A.LongestMaxSize(max_size=640),          # scale so longest side = 640px
    A.PadIfNeeded(min_height=640, min_width=640,
                  border_mode=cv2.BORDER_CONSTANT),  # zero-pad to 640×640
    A.Normalize(mean=0.485, std=0.229),      # channel-wise normalization
    ToTensorV2(),
]
```

DICOM images are converted to 8-bit grayscale, normalized to [0,255], then stacked
to 3-channel RGB (channel repeat) before applying AGPT transforms.

### 2.4 Prediction Coordinate Mapping

The model outputs **normalized (cx, cy, w, h) in [0,1]** relative to the 640×640 canvas.
To recover original pixel coordinates, the letterbox is reversed:

```
scale     = 640 / max(orig_H, orig_W)
new_H     = round(orig_H × scale)
new_W     = round(orig_W × scale)
pad_y     = (640 - new_H) // 2       # top padding
pad_x     = (640 - new_W) // 2       # left padding

x1_orig = (x1_canvas - pad_x) / scale
y1_orig = (y1_canvas - pad_y) / scale
x2_orig = (x2_canvas - pad_x) / scale
y2_orig = (y2_canvas - pad_y) / scale
```

All final predicted boxes are clamped to `[0, orig_W] × [0, orig_H]`.

### 2.5 Multi-GT Matching Rule (Best-Overlap)

RSNA positive images may contain **multiple bounding boxes** (multi-focal pneumonia).
The model produces a **single prediction** per image. The matching rule is:

> **Best-overlap:** Match the prediction to the GT box with the highest IoU.
> `IoU*(pred) = max_i IoU(pred, GT_i)` over all GT boxes in the same image.

This is the standard PASCAL VOC / MS-COCO single-prediction matching convention.
It is **more lenient** than worst-case matching and **more strict** than union-GT matching.
All multi-GT cases are logged with `n_gt_boxes > 1` in the per-image CSV.

If no GT box achieves IoU > 0 with the prediction (e.g., prediction is entirely outside
the annotated region), the result is recorded as `best_match_iou = 0.0` and is
**not excluded** — it contributes to the denominator of recall.

### 2.6 Metrics (Frozen)

| Metric | Definition |
|---|---|
| **Mean best-match IoU** | Mean of `IoU*(pred)` over all evaluable positive images |
| **Recall @ IoU ≥ 0.50** | Fraction of positive images with `IoU*(pred) ≥ 0.50` |
| **Recall @ IoU ≥ 0.75** | Fraction of positive images with `IoU*(pred) ≥ 0.75` |
| **Inference success rate** | Fraction of images successfully processed (DICOM readable, no crash) |
| **Mean inference latency** | Wall-clock time per image (ms) — includes DICOM loading |

**Evaluable images:** positive images (class == "Lung Opacity", Target == 1) where:
- The DICOM file is present on disk
- The inference pipeline completes without exception

Images where DICOM is missing or inference fails are recorded with
`exclusion_reason` populated and are **excluded from the denominator** of all metrics.

---

## 3. Split Construction

Splits are created deterministically using seed=42 via `--prepare-splits`.
Splits are stratified by image, not patient (RSNA does not provide patient-level groupings
beyond the patientId field, which maps 1:1 to images).

| Split | Positive Images | Fraction |
|---|---|---|
| Train | ~4,208 | 70% |
| Val | ~902 | 15% |
| Test | ~902 | 15% |

Split metadata is stored in `data/processed/rsna/split_metadata.json`.

> [!NOTE]
> The train split is unused in Phase 3B (zero-shot). It is pre-reserved for
> Phase 4A (optional RSNA fine-tuning).

---

## 4. Results (Populated After Evaluation)

> These tables will be filled in by `evaluate_rsna.py`. The structure and metric
> definitions are fixed; actual numbers are not yet available.

### 4.1 Primary Query: "pneumonia" — Validation Split

| Metric | Value |
|---|---|
| Total positive images (val) | 901 |
| Evaluable images | 901 |
| Excluded images (DICOM missing / failed) | 0 |
| Inference success rate | 100.0% |
| **Mean best-match IoU** | **0.3124** |
| **Recall @ IoU ≥ 0.50** | **0.2331 (23.31%)** |
| **Recall @ IoU ≥ 0.75** | **0.0433 (4.33%)** |
| Mean inference latency (ms) | 164.89 ms |

### 4.2 Secondary Query: "airspace opacity" — Validation Split Only

| Metric | Value |
|---|---|
| **Mean best-match IoU** | **0.3431** |
| **Recall @ IoU ≥ 0.50** | **0.2952 (29.52%)** |
| **Recall @ IoU ≥ 0.75** | **0.0233 (2.33%)** |
| Total positive images (val) | 901 |
| Inference success rate | 100.0% |
| Mean inference latency (ms) | 250.77 ms |

### 4.3 Primary Query: "pneumonia" — Held-Out Test Split

> [!CAUTION]
> The test split is evaluated **exactly once**. The following results are final.
> Query, preprocessing, and matching rules were frozen before execution.

| Metric | Value |
|---|---|
| Total positive images (test) | 901 |
| Evaluable images | 901 |
| Excluded images | 0 |
| Inference success rate | 100.0% |
| **Mean best-match IoU** | **0.3130** |
| **Recall @ IoU ≥ 0.50** | **0.2397 (23.97%)** |
| **Recall @ IoU ≥ 0.75** | **0.0555 (5.55%)** |
| Mean inference latency (ms) | 141.03 ms |

### 4.4 Negative Image Behavior

> Reported separately. Model outputs boxes on negative images; these are meaningless
> for localization assessment and are excluded from all recall metrics.

| Metric | Value |
|---|---|
| Negative images processed | *TBD* |
| Mean predicted box area (normalized) | *TBD* (characterizes model behavior) |

---

## 5. Artifacts

| File | Description |
|---|---|
| [`artifacts/rsna/baseline_validation.csv`](file:///Users/adityakachhot/interactmed/artifacts/rsna/baseline_validation.csv) | Per-image val results |
| [`artifacts/rsna/baseline_validation.json`](file:///Users/adityakachhot/interactmed/artifacts/rsna/baseline_validation.json) | Val aggregate metrics |
| `artifacts/rsna/baseline_test.csv` | Per-image test results (after test run) |
| `artifacts/rsna/baseline_test.json` | Test aggregate metrics (after test run) |
| `artifacts/rsna/visualizations/val/` | Overlay PNGs for val split |
| `artifacts/rsna/visualizations/test/` | Overlay PNGs for test split |
| `data/processed/rsna/split_metadata.json` | Split construction record |
| `data/processed/rsna/rsna_val_positive.csv` | Val split patient IDs |
| `data/processed/rsna/rsna_test_positive.csv` | Test split patient IDs (held out) |

---

## 6. Terminal Commands

```zsh
cd /Users/adityakachhot/interactmed

# Step 1: Prepare deterministic splits (run once — idempotent)
/Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/evaluate_rsna.py \
    --config configs/rsna_eval.yaml \
    --prepare-splits

# Step 2: Smoke test — verify pipeline on 5 images
/Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/evaluate_rsna.py \
    --config configs/rsna_eval.yaml \
    --smoke-test

# Step 3: Validation split — primary query + secondary query + visualizations
/Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/evaluate_rsna.py \
    --config configs/rsna_eval.yaml \
    --split val \
    --save-viz

# Step 4: Secondary query on val only (optional, for reporting)
/Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/evaluate_rsna.py \
    --config configs/rsna_eval.yaml \
    --split val \
    --query "airspace opacity"

# Step 5: Generate standalone visualizations from results CSV
/Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/visualize_rsna_predictions.py \
    --config configs/rsna_eval.yaml \
    --results-csv artifacts/rsna/baseline_validation.csv \
    --split val \
    --max-images 100

# Step 6: Final held-out test evaluation (run exactly once — final results)
/Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/evaluate_rsna.py \
    --config configs/rsna_eval.yaml \
    --split test
```

---

## 7. Interpretation Notes

1. **TransVG-AGPT was not trained on RSNA data.** Any positive result demonstrates
   genuine zero-shot generalization from MS-CXR and Chest ImaGenome annotations.

2. **Domain shift factors:**
   - MS-CXR uses clinical free-text phrases (e.g., "patchy consolidation in right lower lobe")
   - RSNA evaluation uses a single word: "pneumonia"
   - RSNA images are sourced from a different health system with potentially different
     imaging protocols

3. **Multi-focal pneumonia:** 612 images have ≥ 2 GT boxes. The best-overlap rule
   is appropriate but favors cases where the model fires near any lesion. Results for
   multi-box images are always logged with `n_gt_boxes` for downstream analysis.

4. **Negative images:** A grounding model should not be used as a binary classifier.
   The fact that the model always outputs a box is expected and does not imply the
   model predicts pneumonia is present.

5. **Phase 4A connection:** If Phase 4A (RSNA fine-tuning) is executed, the train
   split defined here will be used for training. The test split defined here will
   be used for the final controlled comparison between zero-shot and fine-tuned models.

---

## 8. Attribution

**RSNA Dataset:**
> Shih, G., Wu, C.C., et al. (2019). Augmenting the NIH Chest Radiograph Dataset with
> Expert Annotations of Possible Pneumonia. *Radiology: Artificial Intelligence*, 1(1).
> https://doi.org/10.1148/ryai.2019180041

**TransVG-AGPT:**
> Anatomical Grounding Pre-training (AGPT). https://github.com/Claire1217/AGPT
