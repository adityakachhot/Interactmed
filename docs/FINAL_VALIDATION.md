# Final Validation Report

**Date:** 2026-10-11  
**Machine:** MacBook Air M3 (Apple Silicon, 16 GB unified memory)  
**Environment:** miniforge3 `dl` — Python 3.10.19, PyTorch 2.2.2, Apple MPS  

---

## 1. RSNA Test Evaluation — Confirmed Final Numbers

| Metric | Pretrained Baseline | Fine-Tuned (3 ep) | Abs Δ | Rel Δ |
|---|:---:|:---:|:---:|:---:|
| **N evaluated** | 901 | 901 | — | — |
| **Excluded** | 0 | 0 | — | — |
| **Inference success** | 100% | 100% | — | — |
| **Mean Best-Match IoU** | 0.3130 | **0.4530** | +0.1400 | **+44.7%** |
| **Recall @ IoU ≥ 0.50** | 0.2397 | **0.4972** | +0.2575 | **+107.5%** |
| **Recall @ IoU ≥ 0.75** | 0.0555 | **0.0810** | +0.0255 | **+45.9%** |
| Mean latency | 145.9 ms | 140.4 ms | — | — |

Query: `"pneumonia"` · Matching: best-overlap (max IoU over all GT boxes per image) · Frozen before any test evaluation.

---

## 2. Automated Test Suite

```
pytest tests/test_backend_api.py tests/test_coordinate_conversion.py -v
```

| Test | Result |
|---|---|
| `TestHealth::test_health_returns_200` | ✅ PASS |
| `TestHealth::test_health_schema` | ✅ PASS |
| `TestHealth::test_health_status_ok` | ✅ PASS |
| `TestGround::test_ground_returns_200_with_valid_png` | ✅ PASS |
| `TestGround::test_ground_invalid_checkpoint` | ✅ PASS |
| `TestGround::test_ground_empty_query` | ✅ PASS |
| `TestGround::test_ground_unsupported_filetype` | ✅ PASS |
| `TestGround::test_ground_baseline_checkpoint` | ✅ PASS |
| `TestGround::test_ground_response_contains_latency` | ✅ PASS |
| `TestReportRegion::test_region_report_returns_200` | ✅ PASS |
| `TestReportRegion::test_region_report_invalid_box` | ✅ PASS |
| `TestReportRegion::test_region_report_zero_image_dims` | ✅ PASS |
| `TestReportStudy::test_study_report_returns_200` | ✅ PASS |
| `TestReportStudy::test_study_report_missing_indication` | ✅ PASS |
| `TestReportStudy::test_study_report_with_prior_context` | ✅ PASS |
| `TestReportStudy::test_study_report_no_invention_no_prior` | ✅ PASS |
| `TestCoordinateConversion::test_letterbox_params_square_image` | ✅ PASS |
| `TestCoordinateConversion::test_letterbox_params_tall_image` | ✅ PASS |
| `TestCoordinateConversion::test_pred_norm_to_pixel_identity_square` | ✅ PASS |
| `TestCoordinateConversion::test_iou_identical_boxes` | ✅ PASS |
| `TestCoordinateConversion::test_iou_no_overlap` | ✅ PASS |
| `TestCoordinateTransformations::*` (5 tests) | ✅ PASS |
| `TestProcessedMetadataIntegrity::*` (5 tests) | ✅ PASS |
| **Total** | **31 / 31 PASSED** |

---

## 3. Backend Smoke Test

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
# → GET /health → 200 OK
```

```json
{
  "status": "ok",
  "grounding_model_loaded": true,
  "grounding_model_device": "mps",
  "grounding_model_error": null,
  "ollama_available": true,
  "timestamp": "2026-10-10T21:05:19Z"
}
```

- Fine-tuned checkpoint (`best_model_val_iou.pth`) loads on MPS at startup.
- 175,005,380 parameters loaded successfully.
- Ollama reachable at `localhost:11434` with `llama3` and `mistral`.

---

## 4. Frontend Build Check

```bash
cd frontend && npm run build
# ✓ built in 284ms
# dist/assets/index.css   7.75 kB │ gzip: 2.32 kB
# dist/assets/index.js  156.26 kB │ gzip: 50.07 kB
```

TypeScript check: `tsc --noEmit` → **0 errors**.

---

## 5. Data Integrity Summary

| Check | Result |
|---|---|
| Train/Val/Test patient-disjoint | ✅ Zero overlap |
| Test has GT annotations | ✅ 2,575 boxes, all Target=1 |
| Test not used in training or ckpt selection | ✅ Confirmed |
| Original checkpoint unchanged | ✅ `transvg.pth` unmodified |
| Validation outputs not overwritten | ✅ Separate directories used |

---

## 6. Known Remaining Limitations

| Issue | Severity | Notes |
|---|---|---|
| LLM report is text-only (no image pixels) | Medium | Replace with `llava` via Ollama for true visual reports |
| Single-box grounding for multi-focal pneumonia | Medium | By design; best-match IoU mitigates for evaluation |
| No objectness score | Medium | Model always outputs a box; negative filtering is external |
| Recall@0.75 low (0.081) | Low | 3-epoch training limitation; more epochs or backbone unfreezing may help |
| RSNA-only evaluation | Low | Generalization to other datasets not characterized |
