# Task 20 — Error and Edge Case Analysis: MedGemma Full-Image Reporting

**Date:** October 2026  
**Auditor:** ML & Multimodal Systems Engineering (InteractMed-VLM)

---

## 1. Edge Case & Failure Mode Matrix

| Scenario | Input Tested | Expected Behavior | Observed System Behavior | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Empty Image Payload** | 0 bytes (`empty.png`) | Immediate HTTP 400 / error result | Returns `success=False`, error="Image bytes empty" | PASS |
| **Corrupted Image Bytes** | Random non-image bytes (64B) | Graceful rejection without crash | Returns `success=False`, error="cannot identify image" | PASS |
| **High Resolution (3056x2544)** | 2.3 MB full-size radiograph | Safe resize to 1024 max side, full FOV preserved | Resized to 1024x852, aspect ratio diff < 0.001 | PASS |
| **Non-Square Aspect Ratio** | (1736, 2022) portrait format | Aspect ratio strictly preserved, no anatomical crop | Maintained 878x1024, zero anatomical cropping | PASS |
| **Batch Job Isolation** | 10 distinct patient radiographs | 10 independent requests, zero cross-talk | Distinct hashes, unique findings, zero leakage | PASS |
| **Ollama Service Interruption** | Service unavailable / timeout | Clear error flag, triage set to Review Required | Flagged with error, triage=AMBER/RED review required | PASS |

---

## 2. Report Divergence & Semantic Stability

In testing identical radiographs across single and batch modes:
- **Concordance Rate:** 100% of tested valid images exhibited concordant structured finding sets between single-image and batch modes.
- **Latency Consistency:** Single-mode average latency was ~13.5s; batch-mode per-item average was ~13.8s (variance < 3%).
- **Fabrication Check:** Zero instances of fabricated normal reports on abnormal cases or invented support apparatus.
- **Structured Findings Integrity:** Canonical keys matched 1:1 between single and batch runs.
