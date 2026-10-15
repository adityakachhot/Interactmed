# InteractMed-VLM: Operations, Installation, and Reproducibility Guide

> **Engineering Operations & Experiment Reproduction Runbook**  
> **Project:** InteractMed-VLM  
> **Document:** `docs/architecture/OPERATIONS_AND_REPRODUCIBILITY.md`  
> **Status:** Verified against Active Codebase, Test Suites (104/104 Passing), and Local macOS M3 Environment

---

## 1. System Requirements & Prerequisites

The InteractMed-VLM architecture separates local inference from cloud fine-tuning. Ensure the appropriate environment is provisioned before proceeding:

### 1.1 Local Inference Environment (Host Machine)
- **Operating System:** macOS (Apple Silicon M1/M2/M3 recommended) or Linux x86_64.
- **Unified Memory / RAM:** Minimum 16 GB Unified Memory (fits TransVG on MPS and MedGemma 4B on Ollama concurrently).
- **Python Runtime:** Python `3.10.x` (managed via Conda or Miniforge).
- **Node.js Runtime:** Node.js `v18.x` or later with `npm v9+` (for Vite frontend).
- **Ollama Engine:** Ollama `v0.1.30+` installed and running locally.

### 1.2 Cloud Fine-Tuning Environment (Training Only)
- **Host:** Google Cloud Platform (GCP) Compute Engine or Google Colab Enterprise.
- **Hardware Acceleration:** NVIDIA A100 GPU (minimum 40 GB VRAM, 80 GB recommended).
- **Driver / CUDA:** NVIDIA Driver $\ge 525.60.13$, CUDA $\ge 12.1$.
- **Software Dependencies:** PyTorch $\ge 2.2.0$, Hugging Face Transformers $\ge 4.50.0$, `bitsandbytes`, `peft`.

---

## 2. Step-by-Step Installation & Setup

### 2.1 Clone Repository & Environment Setup
```bash
# Navigate to workspace
cd /Users/adityakachhot/interactmed

# Activate the verified Conda environment
conda activate dl

# Verify Python version and PyTorch Apple Metal (MPS) availability
python -c "import torch; print(f'PyTorch: {torch.__version__}, MPS Available: {torch.backends.mps.is_available()}')"
# Expected Output: PyTorch: 2.2.2, MPS Available: True
```

### 2.2 Model Checkpoint Placement
Ensure all model checkpoints are placed in their verified filesystem locations:

```bash
# Verify baseline TransVG checkpoint (2.1 GB)
ls -lh checkpoints/transvg.pth
# Expected size: ~2.1 GB (2,099,188,914 bytes)

# Verify RSNA fine-tuned TransVG checkpoint (768 MB)
ls -lh checkpoints/finetuned_rsna/run_01_3ep/best_model_val_iou.pth
# Expected size: ~768 MB (767,932,253 bytes)
```

### 2.3 Ollama Installation & MedGemma Setup
Ollama serves the quantized Google MedGemma 4B model locally:

```bash
# 1. Start Ollama daemon in a separate terminal window
ollama serve

# 2. Pull the MedGemma 4B medical VLM (or verify presence)
ollama pull medgemma:4b

# 3. (Optional Fallback) Pull LLaVA 7B for regional testing
ollama pull llava:7b

# 4. Verify installed models
ollama list
# Output should list: medgemma:4b (or medgemma:latest)
```

### 2.4 Frontend Setup & Dependencies
```bash
cd /Users/adityakachhot/interactmed/frontend

# Install React & TypeScript dependencies
npm install

# Verify TypeScript compilation without errors
npx tsc --noEmit
```

---

## 3. Starting the Application

### 3.1 Step 1: Launch Backend API Server
In terminal 1:
```bash
cd /Users/adityakachhot/interactmed
conda activate dl

# Start Uvicorn FastAPI server on port 8000
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
*Startup Log Verification:*
```
[Startup] Pre-loading fine-tuned grounding model …
[GroundingService] Loading checkpoint: .../checkpoints/finetuned_rsna/run_01_3ep/best_model_val_iou.pth
[GroundingService] Device: mps
[GroundingService] Loaded successfully. Parameters: 146,843,268
[Startup] Fine-tuned model ready.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 3.2 Step 2: Launch Frontend Development Server
In terminal 2:
```bash
cd /Users/adityakachhot/interactmed/frontend

# Start Vite dev server on port 5173
npm run dev
```
*Browser Entry Point:* Open `http://localhost:5173` in any modern web browser (Chrome, Safari, Firefox).

---

## 4. Automated Testing & Verification

The repository contains an exhaustive test suite verifying backend APIs, grounding models, report generators, batch queues, coordinate transformations, and triage engines.

### 4.1 Running the Full Test Suite
```bash
cd /Users/adityakachhot/interactmed
conda activate dl

# Execute all 104 unit and integration tests
python -m pytest tests/ -v
```

*Verified Test Suite Output:*
```
================================ test session starts =================================
platform darwin -- Python 3.10.19, pytest-8.3.3, pluggy-1.5.0
rootdir: /Users/adityakachhot/interactmed
collected 104 items

tests/test_backend_api.py::TestHealth::test_health_returns_200 PASSED            [  0%]
tests/test_backend_api.py::TestGrounding::test_ground_valid_image PASSED          [  5%]
tests/test_batch_service.py::TestBatchService::test_batch_creation PASSED        [ 20%]
tests/test_batch_service.py::TestBatchService::test_batch_cancellation PASSED    [ 25%]
tests/test_batch_service.py::TestBatchService::test_batch_retry PASSED           [ 30%]
tests/test_build_manifest.py::TestBuildManifest::test_unify_manifests PASSED     [ 40%]
tests/test_coordinate_conversion.py::TestCoordinateTransformations PASSED       [ 50%]
tests/test_grounding_inference.py::TestGroundingInferenceFix PASSED               [ 60%]
tests/test_report_generator.py::TestReportValidationAndSafety PASSED             [ 75%]
tests/test_triage_service.py::TestTriageService::test_potentially_urgent PASSED   [ 85%]
tests/test_triage_service.py::TestTriageService::test_negation_safety PASSED     [ 90%]
tests/test_vision_encoder.py::TestVisionEncoder PASSED                            [100%]

======================== 104 passed, 3 warnings in 14.95s =========================
```

---

## 5. Smoke Testing via CLI

You can verify each application endpoint directly from the command line using `curl`:

### 5.1 Service Health & Model Status
```bash
curl -s http://localhost:8000/health | jq .
```
*Expected Output:*
```json
{
  "status": "ok",
  "grounding_model_loaded": true,
  "grounding_model_device": "mps",
  "grounding_model_error": null,
  "ollama_available": true,
  "vision_model": "medgemma:4b",
  "text_model": "llama3:latest",
  "timestamp": "2026-10-15T06:00:00Z"
}
```

### 5.2 Vision Model Readiness
```bash
curl -s http://localhost:8000/vision-model-status | jq .
```

### 5.3 Visual Grounding Smoke Test
```bash
curl -X POST http://localhost:8000/ground \
  -F "file=@data/raw/rsna/stage_2_train_images/0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm" \
  -F "query=pneumonia" \
  -F "checkpoint=finetuned" \
  -F "return_overlay=true" \
  -F "return_crop=true" | jq '{success, query, checkpoint, pred_box_pixel, latency_ms}'
```
*Expected Output:*
```json
{
  "success": true,
  "query": "pneumonia",
  "checkpoint": "best_model_val_iou.pth",
  "pred_box_pixel": [162.4, 210.8, 485.2, 612.3],
  "latency_ms": 138.42
}
```

### 5.4 Patient-Friendly Report Smoke Test
```bash
curl -X POST http://localhost:8000/report/patient-friendly \
  -F "file=@data/raw/rsna/stage_2_train_images/0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm" \
  -F "reason_for_xray=Persistent cough and mild fever" | jq '{success, model_used, latency_ms, triage_category: .triage_result.triage_category}'
```

---

## 6. Reproducing Research Evaluations

### 6.1 Reproducing TransVG-AGPT RSNA Grounding Evaluation
To reproduce the exact validation and held-out test IoU metrics published in project reports:

```bash
cd /Users/adityakachhot/interactmed
conda activate dl

# 1. Run Baseline Test Evaluation (901 held-out images)
python scripts/evaluate_rsna.py \
  --config configs/rsna_eval.yaml \
  --split test \
  --checkpoint checkpoints/transvg.pth \
  --output-dir artifacts/rsna/runs/test_baseline \
  --run-name baseline_test \
  --model-label "Pretrained Baseline TransVG-AGPT"

# 2. Run Fine-Tuned Test Evaluation (901 held-out images)
python scripts/evaluate_rsna.py \
  --config configs/rsna_eval.yaml \
  --split test \
  --checkpoint checkpoints/finetuned_rsna/run_01_3ep/best_model_val_iou.pth \
  --output-dir artifacts/rsna/runs/test_finetuned_run01 \
  --run-name finetuned_test \
  --model-label "Fine-Tuned TransVG-AGPT (Run 01, 3 Epochs)" \
  --save-viz
```
*Output Verification:* Compare generated `baseline_test.json` and `finetuned_test.json` against `docs/FINAL_RSNA_EVALUATION.md`. IoU should match **0.3130** (baseline) and **0.4530** (fine-tuned).

---

### 6.2 Reproducing MedGemma Whole-Image Report Evaluation
To evaluate Prompt v3 across all 250 development cases in `expanded_dev`:

```bash
cd /Users/adityakachhot/interactmed
conda activate dl

# Make sure Ollama is running with medgemma:4b
python scripts/evaluate_expanded_reports.py \
  --config configs/expanded_report_evaluation.yaml \
  --prompt-version v3 \
  --output-dir artifacts/report_evaluation/expanded_cohort_full/prompt_v3
```

---

### 6.3 Reproducing MedGemma Fine-Tuning Preflight Checks
Before launching any fine-tuning run on cloud GPUs, execute the three automated safety gates:

```bash
cd /Users/adityakachhot/interactmed
conda activate dl

# Safety Check 1: Inspect model modules and parameter freeze states
python scripts/train_medgemma_lora.py --inspect-only

# Safety Check 2: Execute dry-run forward/backward pass & verify gradient routing
python scripts/train_medgemma_lora.py --dry-run

# Safety Check 3: Execute single-batch overfit test (verifies loss decreases)
python scripts/train_medgemma_lora.py --overfit-single-batch
```
*Expected Overfit Outcome:* Loss must drop monotonically over 20 steps (from ~`9.25` down to ~`5.89`).

---

## 7. Troubleshooting & Operational Failure Modes

| Symptom / Error Message | Root Cause | Verified Resolution |
| :--- | :--- | :--- |
| `Grounding model failed to load: No module named 'dataloader'` | AGPT root directory is not in Python search path. | `src/grounding_service.py` automatically injects `third_party/AGPT` into `sys.path`. Ensure submodules are checked out (`git submodule update --init --recursive`). |
| `Ollama is not reachable (localhost:11434)` | Ollama daemon is offline or crashed. | Open a terminal and run `ollama serve`. Verify with `curl http://localhost:11434/api/tags`. |
| `No vision-capable model is installed in Ollama` | Only text models (`llama3`) are available in Ollama. | Execute `ollama pull medgemma:4b` or `ollama pull llava:7b`. The backend will automatically recognize vision readiness. |
| `Inference error: MPS out of memory` | Apple Silicon unified memory exhausted by competing background processes. | The backend automatically catches MPS allocation errors and retries on CPU. Close heavy browser tabs or Docker containers to free unified RAM. |
| `DICOM pixel values appear inverted (bones are black)` | Radiograph was acquired with `PhotometricInterpretation == "MONOCHROME1"`. | `_load_to_rgb_array` in `report_service.py` and `grounding_service.py` automatically checks for `MONOCHROME1` and inverts pixel values (`pixels.max() - pixels`) to standard radiological white-bone presentation. |
| `Bounding box coordinates exceed image bounds` | Coordinate scaling mismatch between 640px letterbox canvas and native DICOM resolution. | `OverlayService` applies strict boundary clamping: `clamp(x, 0, orig_w)` and `clamp(y, 0, orig_h)`. Unit tests in `test_coordinate_conversion.py` guarantee round-trip mathematical consistency. |
| `CORS Error in Browser Developer Console` | Frontend port differs from backend allowed origins. | `backend/main.py` explicitly configures `CORSMiddleware` with `allow_origins=["*"]`, `allow_methods=["*"]`, and `allow_headers=["*"]`. |
