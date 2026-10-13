# InteractMed-VLM

**Region-aware phrase grounding and AI-assisted report drafting for chest radiographs.**

Built on TransVG-AGPT fine-tuned on RSNA Pneumonia Detection Challenge 2018.

---

## Quick Start

### 1. Environment

```bash
conda activate dl  # Python 3.10, PyTorch 2.2, Apple MPS
```

### 2. Start the API Backend

```bash
cd /Users/adityakachhot/interactmed
/Users/adityakachhot/miniforge3/envs/dl/bin/python -m uvicorn backend.main:app \
    --host 0.0.0.0 --port 8000
```

The fine-tuned TransVG-AGPT model loads automatically at startup (~30 s on MPS).

### 3. Start the Frontend

```bash
cd /Users/adityakachhot/interactmed/frontend
npm run dev
# → http://localhost:5173
```

### 4. (Optional) Ensure Ollama is running for report generation

```bash
ollama serve   # in a separate terminal
# Models confirmed present: llama3:latest, mistral:latest
```

---

## Architecture

```
frontend/          React + TypeScript (Vite)  →  http://localhost:5173
backend/main.py    FastAPI                     →  http://localhost:8000
src/
  grounding_service.py   TransVG-AGPT singleton (loads once at startup)
  report_service.py      Ollama llama3/mistral for report generation
  overlay_service.py     Bounding-box overlay rendering
scripts/
  evaluate_rsna.py       Full evaluation pipeline (coordinate math reused by backend)
  train_rsna.py          Fine-tuning training loop
  summarize_test_results.py  Chart + report generation
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Model status, Ollama availability |
| POST | `/ground` | Phrase grounding → bounding box + annotated image |
| POST | `/report/region` | Region-specific report draft from grounded box |
| POST | `/report/study` | Study-level report draft from clinical context |

### Example: Ground a phrase

```bash
curl -X POST http://localhost:8000/ground \
  -F "file=@path/to/cxr.png" \
  -F "query=pneumonia" \
  -F "checkpoint=finetuned" \
  -F "return_overlay=true"
```

---

## RSNA Final Evaluation Results

**Query:** `"pneumonia"` · **N = 901 held-out test images** · patient-disjoint splits

| Metric | Pretrained Baseline | Fine-Tuned (3 ep) | Rel Δ |
|---|:---:|:---:|:---:|
| Mean Best-Match IoU | 0.3130 | **0.4530** | +44.7% |
| Recall @ IoU ≥ 0.50 | 0.2397 | **0.4972** | +107.5% |
| Recall @ IoU ≥ 0.75 | 0.0555 | **0.0810** | +45.9% |
| Inference Success | 100% | 100% | — |
| Mean Latency | 145.9 ms | 140.4 ms | — |

See [`docs/FINAL_RSNA_EVALUATION.md`](docs/FINAL_RSNA_EVALUATION.md) for full methodology.

---

## Run Tests

```bash
/Users/adityakachhot/miniforge3/envs/dl/bin/python -m pytest tests/ -v
# 31 tests passing
```

---

## Project Structure

```
checkpoints/
  transvg.pth                          Pretrained baseline (2.1 GB, unchanged)
  finetuned_rsna/run_01_3ep/
    best_model_val_iou.pth             Fine-tuned checkpoint (768 MB, epoch 3)

data/processed/rsna/
  rsna_train_positive.csv              4,210 patients, 11,897 boxes
  rsna_val_positive.csv                901 patients, 2,485 boxes
  rsna_test_positive.csv               901 patients, 2,575 boxes (held-out)

artifacts/rsna/runs/
  original_baseline/                   Val evaluation — baseline
  run_01_3ep/                          Val evaluation — fine-tuned
  test_baseline/                       Test evaluation — baseline
  test_finetuned_run01/                Test evaluation — fine-tuned (with viz)
  test_metrics_comparison.png          4-panel comparison chart
  test_qualitative_comparisons/        Side-by-side overlay images

docs/
  FINAL_RSNA_EVALUATION.md             Complete evaluation report with real numbers
  FINETUNING_AUDIT.md                  Architecture + training decisions
  DATA_INVENTORY.md                    Dataset inventory
```

---

## Memory Requirements

| Component | Memory |
|---|---|
| TransVG-AGPT (MPS) | ~5–7 GB unified memory |
| Ollama llama3 (CPU) | ~5 GB |
| React frontend | negligible |
| **Total** | ~12 GB (fits 16 GB M3) |

---

## Known Limitations

- **Single-box output** — one predicted box per query; multi-focal pneumonia partially handled via best-match IoU.
- **No objectness classifier** — the model always produces a box; negative images must be filtered separately.
- **Report generation is text-only** — llama3/mistral are text models; the report prompt uses spatial metadata, not pixel values. A multimodal local model (e.g., LLaVA) can replace the adapter in `src/report_service.py`.
- **3-epoch fine-tune** — Recall@0.75 remains low; additional epochs or partial backbone unfreezing may help.
- **RSNA-specific** — generalization to other pathology tasks not characterized.

---

## Data Privacy

MIMIC-CXR images, reports, and patient identifiers are never uploaded to any external API.
All report generation runs locally via Ollama. No patient data appears in logs.
