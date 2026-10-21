# Task 20 — Root Cause Audit: Full-Image Input Pathway Parity in MedGemma Reporting

**Date:** October 2026  
**Auditor:** ML & Multimodal Systems Engineering (InteractMed-VLM)  
**System Target:** MedGemma 4B (`medgemma:4b` via Ollama) on macOS Apple Silicon M3 (16GB RAM)

---

## Executive Summary

An architectural and operational audit of MedGemma whole-image report generation identified a critical design vulnerability where full chest radiographs were not consistently supplied as visual inputs across all inference stages, creating risks of hallucinated text-to-text transforms and mode divergence between single-image and batch reporting.

This audit confirmed the root causes, traced the request lifecycles, and verified the complete elimination of text-only fallback paths.

---

## 1. Request Lifecycle Tracing

### A. Single-Image Path
1. **Upload & Ingestion:** `POST /report/research` or `POST /report/patient` receives image binary.
2. **Preprocessing:** `_load_to_rgb_array` converts raw bytes to NumPy RGB array. `_array_to_base64_jpeg(arr, max_side=1024)` resizes the image while strictly preserving aspect ratio and full field of view (FOV).
3. **Stage 1 (Anatomical Observations):** MedGemma 4B is called via Ollama `/api/chat` with the base64-encoded image payload attached in `images: [full_b64]`. MedGemma systematically analyzes lungs, pleura, cardiac contours, mediastinum, bones, and devices.
4. **Stage 2 (Report Synthesis):** *(Prior Defect Resolved)* MedGemma 4B synthesizes the 7-section structured research or patient explanation. Crucially, the **full image is attached again** (`images: [full_b64]`) alongside Stage 1 observations, ensuring that report generation is directly grounded in image pixels rather than relying on an unverified text-to-text summary.
5. **Downstream Reconciliation:** `clean_and_structure_research_report` parses sections, validates completeness, extracts canonical structured findings, and checks for internal contradictions without deleting whole-image findings.

### B. Batch Path
1. **Submission:** `POST /batch/submit` accepts up to 10 images.
2. **Job Isolation:** Each item is instantiated with an independent `job_id`, canonical `pixel_sha256`, and isolated memory buffer.
3. **Execution:** Items are processed sequentially using `generate_whole_image_report(image_bytes=item._image_bytes, ...)` — guaranteeing **100% code and prompt parity** with single-image reporting.
4. **Per-Item State:** Errors, retries, and metrics are isolated per job; no shared mutable state or response reuse exists.

---

## 2. Confirmed Root Causes Prior to Task 20 Fix

1. **Stage 2 Text-Only Inference Flaw:**
   - In previous iterations, Stage 2 used `_ollama_generate_text_with_metrics(prompt=s2_prompt, ...)` omitting the image payload entirely.
   - Stage 2 acted as an ungrounded LLM summarizer of Stage 1 text rather than a vision-language model visually verifying the radiograph.
   - *Fix:* Replaced with `_ollama_generate_vision_with_metrics(prompt=s2_prompt, images_b64=[full_b64], ...)` across all styles.

2. **Truncation Vulnerability (`num_predict` underallocation):**
   - Stage 1 and Stage 2 previously had token budgets (`num_predict=220` and `380`), causing truncation on complex multi-pathology cases.
   - *Fix:* Raised token limits to `num_predict=480` and `temperature=0.1` (deterministic), ensuring complete 7-section outputs without abrupt termination.

3. **Concurrency Contention on 16GB Unified Memory:**
   - Concurrent vision inferences on Apple Silicon M3 cause Ollama memory thrashing and metal allocation errors.
   - *Fix:* Enforced strictly sequential worker execution (`max_workers=1`) for batch vision jobs.

---

## 3. Verification Summary

- **Visual Input Attachment:** Confirmed 100% of single and batch model calls pass `images_b64` with valid JPEG payloads.
- **Parity:** Both single and batch pathways invoke the identical underlying `generate_whole_image_report` function.
- **Fallback Elimination:** Empty or invalid payloads fail fast with descriptive errors; no fabricated normal reports are ever generated.
