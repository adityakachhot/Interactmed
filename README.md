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

## Complete Technical Architecture Documentation

The complete, authoritative technical architecture documentation package is available in the [`docs/`](docs/) directory:

- 🏛️ **[System Architecture Specification](docs/ARCHITECTURE.md)** (`docs/ARCHITECTURE.md`) — Primary system overview, Mermaid architecture diagrams, component responsibility table, runtime deployment, and security.
- 🔬 **[Models and Methodologies Reference](docs/architecture/MODELS_AND_METHODS.md)** (`docs/architecture/MODELS_AND_METHODS.md`) — Model inventory (TransVG-AGPT, MedGemma 4B), mathematical formulations, two-stage grounded reasoning, and evaluation equations.
- ⚙️ **[Fine-Tuning and Training Runbook](docs/architecture/FINE_TUNING_AND_TRAINING.md)** (`docs/architecture/FINE_TUNING_AND_TRAINING.md`) — RSNA fine-tuning epoch logs, MedGemma 4-bit QLoRA preflight checks, gradient routing, and master experiment tables.
- 📊 **[Evaluation and Improvements Report](docs/architecture/EVALUATION_AND_IMPROVEMENTS.md)** (`docs/architecture/EVALUATION_AND_IMPROVEMENTS.md`) — RSNA held-out test IoU results, 250-case development report benchmarks, Wilson intervals, and improvements matrix.
- 📁 **[Datasets and Research Governance](docs/architecture/DATASETS_AND_RESEARCH_GOVERNANCE.md)** (`docs/architecture/DATASETS_AND_RESEARCH_GOVERNANCE.md`) — Audit of RSNA, Chest-ImaGenome Gold, MS-CXR, and MIMIC-CXR; patient-disjoint split governance; multi-view vs single-view integrity.
- 🔌 **[API, UI, and Runtime Workflows](docs/architecture/API_UI_AND_WORKFLOWS.md)** (`docs/architecture/API_UI_AND_WORKFLOWS.md`) — Execution traces for grounding, reporting, batch processing, and research triage; double-blind annotation protocol; master REST API table.
- 🛠️ **[Operations and Reproducibility Guide](docs/architecture/OPERATIONS_AND_REPRODUCIBILITY.md)** (`docs/architecture/OPERATIONS_AND_REPRODUCIBILITY.md`) — Installation, environment requirements, Ollama setup, test suites, smoke tests, and troubleshooting runbook.
- 🗺️ **[Limitations and Technical Roadmap](docs/architecture/ROADMAP_AND_LIMITATIONS.md)** (`docs/architecture/ROADMAP_AND_LIMITATIONS.md`) — Honest assessment of limitations (opacity ceiling, strict IoU) and 8 prioritized research milestones.
- 📖 **[Master Documentation Index](docs/INDEX.md)** (`docs/INDEX.md`) — Full sitemap and diagram links.

---

## Architecture Overview

```
frontend/          React 18 + TypeScript (Vite)  →  http://localhost:5173
backend/main.py    FastAPI (Uvicorn)              →  http://localhost:8000
src/
  grounding_service.py   TransVG-AGPT singleton (loads once at startup via lifespan)
  report_service.py      MedGemma 4B / Ollama 2-stage grounded report generator
  batch_service.py       Sequential batch orchestration (up to 10 images, max_workers=1)
  triage_service.py      Deterministic rule-based clinical research triage engine
  overlay_service.py     Bounding-box overlay rendering and high-resolution crop extraction
configs/
  triage_rules.yaml      Configurable triage precedence rules and regex negation patterns
  rsna_eval.yaml         RSNA grounding evaluation parameters and frozen query settings
  medgemma_finetune.yaml 4-bit QLoRA fine-tuning hyperparameters (Tier 2 adaptation)
scripts/
  evaluate_rsna.py       Full grounding evaluation pipeline on RSNA test split
  evaluate_expanded_reports.py Full report evaluation across 250 development cases
  train_rsna.py          TransVG-AGPT fine-tuning training loop
  train_medgemma_lora.py MedGemma 4B QLoRA trainer with 3 preflight safety gates
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health, model status, Ollama availability |
| GET | `/vision-model-status` | Vision-capable model detection in Ollama |
| POST | `/ground` | Phrase grounding → bounding box + annotated overlay + region crop |
| POST | `/report/region` | Region finding draft from crop + full radiograph (requires vision model) |
| POST | `/report/patient-friendly` | Plain-language 4-section patient explanation + research triage card |
| POST | `/report/study` | Study-level report synthesis (alias for patient-friendly) |
| POST | `/batch/create` | Submit sequential batch job for up to 10 radiographs |
| GET | `/batch/{batch_id}` | Poll batch progress, item statuses, and live streaming reports |
| POST | `/batch/{batch_id}/cancel` | Immediately cancel remaining queued batch jobs |
| POST | `/batch/{batch_id}/retry` | Retry failed or cancelled jobs without re-running finished items |
| GET | `/batch/{batch_id}/download` | Export batch summary as CSV or JSON file attachment |

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

See [`docs/FINAL_RSNA_EVALUATION.md`](docs/FINAL_RSNA_EVALUATION.md) and [`docs/architecture/EVALUATION_AND_IMPROVEMENTS.md`](docs/architecture/EVALUATION_AND_IMPROVEMENTS.md) for full methodology.

---

## Run Tests

```bash
/Users/adityakachhot/miniforge3/envs/dl/bin/python -m pytest tests/ -v
# 132 tests passing in ~15s
```

---

## Project Structure

```
checkpoints/
  transvg.pth                          Pretrained baseline (2.1 GB, MS-CXR AGPT master weights)
  multipathology/run_01_multicxr/
    best_model_val_iou.pth             Multi-Pathology Unified checkpoint (768 MB, Val IoU 0.651, Test IoU 0.660)
  finetuned_rsna/run_01_3ep/
    best_model_val_iou.pth             RSNA pneumonia fine-tuned checkpoint (768 MB, epoch 3)

configs/
  grounding_capabilities.yaml          Transparent capability catalog, alias mappings, and benchmark evidence
  multipathology_train.yaml            Multi-Pathology balanced training configuration

data/processed/
  local_grounding_verified.csv         1,056 MS-CXR annotations (train 746, val 147, test 163)
  rsna/                                26,684 RSNA DICOM manifests (train 4,210, val 901, test 901)

artifacts/
  multi_pathology_eval/                Validation and locked test evaluation JSON summaries
  multipathology/                      Multi-Pathology training logs and history
  rsna/runs/                           Val & Test evaluation outputs & comparison plots

docs/
  ARCHITECTURE.md                      Comprehensive System Architecture
  INDEX.md                             Documentation Index & Sitemap
  architecture/                        Modular Architecture Reference Package (8 documents)
  architecture/diagrams/               Standalone Mermaid Architecture Diagrams (8 diagrams)
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
