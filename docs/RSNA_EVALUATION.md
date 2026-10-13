# RSNA Model Evaluation Guide & Command Reference

**Project:** InteractMed-VLM  
**Component:** RSNA Pneumonia Localization Evaluation  
**Script:** [`scripts/evaluate_rsna.py`](file:///Users/adityakachhot/interactmed/scripts/evaluate_rsna.py)  
**Configuration:** [`configs/rsna_eval.yaml`](file:///Users/adityakachhot/interactmed/configs/rsna_eval.yaml)  

---

## 1. Overview & Purpose

This document specifies the command syntax, directory conventions, and output-path safety rules for evaluating phrase-grounding models (TransVG-AGPT) on the **RSNA Pneumonia Detection Challenge 2018** external benchmark.

The evaluation measures **pneumonia localization accuracy** on positive chest X-ray images (`Target == 1`, `class == "Lung Opacity"`):
- Each positive image is evaluated against its reference radiologist bounding boxes.
- For multi-focal pneumonia (multiple ground-truth boxes per image), the **best-overlap matching policy** is used:  
  $$\text{IoU}(\text{pred}, \text{gt}^*) = \max_{i} \text{IoU}(\text{pred}, \text{gt}_i)$$
- Primary query (frozen): `"pneumonia"`
- Secondary query (val only): `"airspace opacity"`
- Metrics computed: Mean best-match IoU, Recall @ IoU $\ge 0.50$, Recall @ IoU $\ge 0.75$, inference latency, and success rate.

---

## 2. Command Line Interface Reference

`scripts/evaluate_rsna.py` provides a unified interface for evaluating both pretrained zero-shot baseline and fine-tuned checkpoints with strict output isolation and anti-overwrite protection.

```bash
python scripts/evaluate_rsna.py [OPTIONS]
```

### Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--config` | Path | `configs/rsna_eval.yaml` | Path to evaluation config YAML. |
| `--split` | `{val, test}` | `None` | Evaluation split to run. |
| `--checkpoint` | Path | Config default | Path to model checkpoint file (`.pth`). |
| `--output-dir` | Path | `artifacts/rsna/runs/<run_name>` | Dedicated directory where all run outputs are stored. |
| `--run-name` | String | Auto-inferred | Run name; forms prefixes for `.csv`, `.json`, and `.log`. |
| `--model-label` | String | Auto-inferred | Descriptive model label used in banners and JSON summary. |
| `--save-viz` | Flag | `False` | Save visualization overlay PNGs into `<output-dir>/visualizations/`. |
| `--overwrite` | Flag | `False` | Explicitly permit overwriting existing evaluation files in `--output-dir`. |
| `--smoke-test` | Flag | `False` | Run a 5-image verification without evaluating full split. |
| `--query` | String | Config default (`"pneumonia"`) | Override query (allowed on val split only). |
| `--max-images` | Integer | `None` | Restrict number of positive images (for debugging). |

---

## 3. Strict Output Isolation & Anti-Overwrite Policy

> [!IMPORTANT]
> **Zero Silent Overwrites**: `scripts/evaluate_rsna.py` verifies whether destination files (`<run_name>.csv` or `<run_name>.json`) already exist **before loading the model or running inference**.
> If existing results are detected and `--overwrite` is not passed, execution halts immediately with exit code 1.

Each run encapsulates all of its artifacts inside its own `--output-dir`:
```
<output-dir>/
├── <run_name>.csv           # Per-image prediction table (patient_id, IoU, coordinates, latency)
├── <run_name>.json          # Summary JSON with exact checkpoint path, query, and metrics
├── <run_name>.log           # Complete execution stdout/stderr log
└── visualizations/          # (Optional) Ground-truth vs predicted bounding-box overlays
```

### Exact Checkpoint & Query Tracking in Summary JSON

Every summary JSON records:
- `checkpoint_path`: Absolute filesystem path to the exact checkpoint evaluated.
- `checkpoint_name`: Checkpoint filename (e.g. `best_model_val_iou.pth` vs `transvg.pth`).
- `model_label`: Exact label supplied or inferred.
- `primary_query` and `query_used`: Verbatim text prompt fed into the ClinicalBERT tokenizer.
- `output_directory` and `run_name`: Filesystem provenance.
- `primary_metrics`: IoU and Recall metrics.

---

## 4. Evaluation Directories & Preserved Artifacts

### 4.1 Directory Layout

```
artifacts/rsna/
├── runs/
│   ├── original_baseline/                  # Pretrained TransVG-AGPT baseline evaluation
│   │   ├── baseline_validation.csv
│   │   ├── baseline_validation.json
│   │   └── baseline_validation.log
│   └── run_01_3ep/                         # Fine-tuned TransVG (3 epochs) evaluation
│       ├── finetuned_validation.csv        # Preserved previous evaluation results
│       ├── finetuned_validation.json       # Preserved previous evaluation results
│       └── finetuned_validation.log
├── baseline_test.csv                       # Frozen test evaluation (from Phase 3B)
├── baseline_test.json                      # Frozen test evaluation (from Phase 3B)
└── smoke_test/                             # Smoke test outputs
```

### 4.2 Explanation of Previous Overwritten Files

During a previous run, evaluating the fine-tuned checkpoint `checkpoints/finetuned_rsna/run_01_3ep/best_model_val_iou.pth` without custom output arguments defaulted to:
- `artifacts/rsna/baseline_validation.csv`
- `artifacts/rsna/baseline_validation.json`

This inadvertently overwrote the original zero-shot baseline validation results in the root `artifacts/rsna/` folder and tagged the summary as `"rsna_zero_shot_baseline"`.

**Recovery and Preservation:**
1. The fine-tuned outputs were preserved under:
   - `artifacts/rsna/runs/run_01_3ep/finetuned_validation.csv`
   - `artifacts/rsna/runs/run_01_3ep/finetuned_validation.json`
2. The original zero-shot baseline metrics remain safely recorded in [`docs/BASELINE_RSNA_RESULTS.md`](file:///Users/adityakachhot/interactmed/docs/BASELINE_RSNA_RESULTS.md) (Mean IoU: 0.3124, Recall@0.50: 0.2331).
3. The new evaluation syntax ensures that each checkpoint writes strictly to its dedicated run directory.

---

## 5. Verification of Model Checkpoints

Both checkpoints have been verified for integrity, architecture compatibility, and weight divergence:

### Original Pretrained Checkpoint
- **Path:** [`checkpoints/transvg.pth`](file:///Users/adityakachhot/interactmed/checkpoints/transvg.pth)
- **Epoch:** 54 (trained upstream on MS-CXR)
- **Parameters:** 175,005,380 across 523 state dict keys
- **Backbones:** ResNet-50 + DETR visual backbone; ClinicalBERT text encoder

### Fine-Tuned Checkpoint
- **Path:** [`checkpoints/finetuned_rsna/run_01_3ep/best_model_val_iou.pth`](file:///Users/adityakachhot/interactmed/checkpoints/finetuned_rsna/run_01_3ep/best_model_val_iou.pth)
- **Epoch:** 3 (fine-tuned on RSNA positive train split with frozen encoders)
- **Recorded Val Metrics:** Mean IoU = 0.4493, Recall@0.50 = 0.4950
- **Weight Check:** Backbones remained frozen (0.0000 diff); multimodal fusion layers (`visu_proj`, `vl_transformer`) adapted to RSNA.

---

## 6. Commands to Run Separately on Validation Split

> [!CAUTION]
> **Do NOT evaluate the final test split yet.** Only evaluate the validation split (`--split val`).

### 6.1 Evaluate Original Pretrained Baseline Checkpoint

To evaluate the original pretrained checkpoint on the validation split and store results inside `artifacts/rsna/runs/original_baseline`:

```bash
python scripts/evaluate_rsna.py \
    --config configs/rsna_eval.yaml \
    --split val \
    --checkpoint checkpoints/transvg.pth \
    --output-dir artifacts/rsna/runs/original_baseline \
    --run-name baseline_validation \
    --model-label "Pretrained Baseline TransVG-AGPT"
```

Outputs produced:
- `artifacts/rsna/runs/original_baseline/baseline_validation.csv`
- `artifacts/rsna/runs/original_baseline/baseline_validation.json`
- `artifacts/rsna/runs/original_baseline/baseline_validation.log`

---

### 6.2 Evaluate Fine-Tuned Checkpoint (Run 01, 3 Epochs)

To evaluate the fine-tuned checkpoint on the validation split into its dedicated run directory:

> [!NOTE]
> `artifacts/rsna/runs/run_01_3ep/finetuned_validation.csv` and `.json` already exist (preserved from the earlier run).  
> To re-run and refresh the run log, add `--overwrite`. Alternatively, specify a distinct `--output-dir` or `--run-name`.

```bash
python scripts/evaluate_rsna.py \
    --config configs/rsna_eval.yaml \
    --split val \
    --checkpoint checkpoints/finetuned_rsna/run_01_3ep/best_model_val_iou.pth \
    --output-dir artifacts/rsna/runs/run_01_3ep \
    --run-name finetuned_validation \
    --model-label "Fine-Tuned TransVG (Run 01, 3 epochs)" \
    --overwrite
```

Outputs produced:
- `artifacts/rsna/runs/run_01_3ep/finetuned_validation.csv`
- `artifacts/rsna/runs/run_01_3ep/finetuned_validation.json`
- `artifacts/rsna/runs/run_01_3ep/finetuned_validation.log`
