# InteractMed-VLM: API, UI, and Workflow Execution Paths

> **Authoritative Runtime Workflows & API Reference**  
> **Project:** InteractMed-VLM  
> **Document:** `docs/architecture/API_UI_AND_WORKFLOWS.md`  
> **Status:** Audited against FastAPI Endpoints, React Components, and Service Classes

---

## 1. Visual Grounding Workflow (`/ground` & `/report/region`)

The visual grounding workflow enables interactive phrase-conditioned localization and region-specific finding analysis.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               VISUAL GROUNDING EXECUTION TRACE                         │
└────────────────────────────────────────────────────────────────────────────────────────┘

 [Browser Client]
       │
       │ 1. Multipart Form Upload: file (CXR), query="pneumonia", checkpoint="finetuned"
       ▼
 [FastAPI Gateway: backend/main.py]
       │
       │ 2. Validation: Suffix check (.dcm, .png, .jpg, .tiff), file size (< 50 MB)
       │    Query sanitization, Checkpoint validation ("finetuned" or "baseline")
       ▼
 [GroundingService: src/grounding_service.py]
       │
       │ 3. Decode image bytes → uint8 RGB array (H0, W0, 3)
       │    (Auto-handles MONOCHROME1 DICOM inversion and grayscale channel expansion)
       │
       │ 4. Albumentations Validation Transforms:
       │    LongestMaxSize(640) + PadIfNeeded(640, 640) + ImageNet Normalize + ToTensor
       │
       │ 5. Construct NestedTensor(image_tensor (1, 3, 640, 640), zero_mask (1, 640, 640))
       │
       │ 6. Model Forward Pass: outputs = model(image_nested, [query])
       │    (BERT tokenization executed internally; returns normalized [cx, cy, w, h])
       │
       │ 7. Coordinate Inversion:
       │    pred_norm_to_canvas_xyxy() → Canvas Box [x1, y1, x2, y2] on 640x640
       │    pred_norm_to_pixel()       → Original Pixel Box [x1, y1, x2, y2]
       ▼
 [OverlayService: src/overlay_service.py]
       │
       │ 8. draw_overlay(): Renders SVG/Pillow bounding box and query banner on radiograph
       │    extract_crop(): Slices original pixel box with margin_px=40 (margin-clamped)
       ▼
 [FastAPI Gateway Response]
       │
       │ 9. Returns GroundingResponse (JSON + Base64 JPEG overlay and crop images)
       ▼
 [Browser Client UI]
       Displays interactive side-by-side overlay, coordinates, and region crop thumbnail.
```

### 1.1 Region-Specific Report Drafting (`/report/region`)
Once a bounding box is established, the user can trigger a region-specific report draft:
- **Input Context:** Two distinct image payloads are sent to Ollama:
  - **Image 1 (Primary Context):** The complete thoracic radiograph, providing global anatomical orientation, thoracic cage dimensions, and bilateral lung symmetry.
  - **Image 2 (Supplementary Context):** The magnified region crop centered on the predicted box.
- **Hypothesis Evaluation:** The queried phrase is explicitly presented to the model as an **unproven hypothesis to evaluate**, not an established diagnosis.
- **Cardiac Evaluation Guard:** For cardiac queries (e.g. *"cardiomegaly"*), the prompt specifically forbids diagnosing heart enlargement from the crop alone, requiring the model to assess the cardiothoracic ratio against the inner thoracic margins of Image 1.
- **Vision Model Policy:** If no vision model is installed in Ollama, the endpoint returns `success: false` with a clear installation prompt (`ollama pull medgemma:4b`). **It never fabricates a report from text metadata alone.**

---

## 2. Whole-Image Report Generation Workflow (`/report/patient-friendly`)

The full-image report generation workflow produces comprehensive, structured drafts from complete thoracic radiographs without relying on bounding boxes or region crops.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                         WHOLE-IMAGE REPORT GENERATION TRACE                            │
└────────────────────────────────────────────────────────────────────────────────────────┘

 [Browser Client]
       │
       │ 1. Multipart Form Upload: file (CXR), reason_for_xray, relevant_history
       ▼
 [FastAPI Gateway: backend/main.py]
       │
       │ 2. Validation & Read Upload (max 50 MB, allowed image types)
       ▼
 [ReportService: src/report_service.py]
       │
       │ 3. Image Preprocessing:
       │    Decode to uint8 RGB array; downsample to max_side=1024 with LANCZOS filter;
       │    Encode single-pass JPEG Base64 payload (Quality=85).
       │
       │ 4. Two-Stage Grounded Execution:
       │    ├──► Stage 1 (Objective Visual Inspection via Ollama /api/generate):
       │    │    Prompt: Inspect lungs, cardiac silhouette, pleural spaces, lines/tubes.
       │    │    num_predict=130, temperature=0.1.
       │    │    Yields: Raw physical visual observations.
       │    │
       │    └──► Stage 2 (Structured Translation & Sectioning):
       │         Prompt: Translate Stage 1 observations into four structured sections.
       │         Yields: Draft report text.
       │
       │ 5. Deterministic Factual Sanitization:
       │    clean_and_structure_patient_report():
       │    - Intercepts and corrects the "frontal view only shows one side" fallacy.
       │    - Strips prompt template leaks and fabricated millimeter measurements.
       │    - Enforces cautious indeterminate fallbacks if visual reading is ambiguous.
       ▼
 [TriageService: src/triage_service.py]
       │
       │ 6. Evaluates structured findings through configurable precedence rules.
       │    Extracts affirmed patterns, verifies negation cues, and flags contradictions.
       │    Assigns provisional category: potentially_urgent, abnormality_flagged,
       │    no_priority_finding_flagged, or indeterminate_review_required.
       ▼
 [FastAPI Gateway Response]
       │
       │ 7. Returns PatientFriendlyReportResponse (JSON with structured sections + triage)
       ▼
 [Browser Client UI]
       Renders 4 styled accordion cards, latency telemetry, and research triage card.
```

---

## 3. Patient-Friendly Explanation Workflow

Designed specifically for patient communication, this workflow transforms complex radiographic observations into accessible, reassuring language:

### 3.1 Four Standardized Sections
Every patient-friendly explanation implements four mandatory headings:
1. **Your X-ray in simple words:** A plain-language summary of what is visible across the lungs, heart, and chest wall.
2. **What the findings may mean:** Translation of visible features into everyday terms, immediately defining any necessary medical words (e.g. explaining that cardiomegaly refers to an enlarged heart).
3. **What is uncertain:** Clear communication of technical constraints (e.g. explaining that a single 2D frontal projection cannot confirm depth or subtle findings without a lateral view).
4. **What happens next:** Gentle, reassuring guidance encouraging the patient to discuss their imaging results with their ordering physician. Prescribing medications or recommending specific treatments is strictly forbidden.

### 3.2 Medical Terminology & Reading Level
- **Target Reading Level:** 6th-grade Flesch-Kincaid reading level.
- **Tone:** Calm, respectful, and objective; avoids alarmist language.
- **Disclaimers:** Mandatory educational disclaimers are placed outside the narrative body in dedicated UI banners.

---

## 4. Sequential Batch Processing Workflow (`/batch/*`)

The batch processing system enables sequential whole-image report drafting across up to 10 radiographs with fault tolerance and progress streaming.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              BATCH PROCESSING ENGINE TRACE                             │
└────────────────────────────────────────────────────────────────────────────────────────┘

 [Browser Client]
       │
       │ 1. POST /batch/create with up to 10 CXR files + report style ('patient'|'research')
       ▼
 [BatchService: src/batch_service.py]
       │
       │ 2. Validates batch size (len <= 10; > 10 rejected with HTTP 400).
       │    Generates batch_id ("batch_<uuid>") and job_id ("job_<uuid>") per file.
       │    Generates lightweight thumbnail preview (max_side=256) per item.
       │    Submits each job to ThreadPoolExecutor(max_workers=1).
       ▼
 [Worker Thread: _process_single_item]
       │
       │ 3. Sequential Execution Loop (Single Worker):
       │    ├── Check if batch cancelled -> mark job "cancelled"
       │    ├── Update job status to "processing", record started_at
       │    ├── Check Ollama availability & persistent keep_alive="30m"
       │    ├── Encode image (1024px, JPEG Q85)
       │    ├── Invoke _ollama_generate_vision_with_metrics():
       │    │   Extracts load_duration, prompt_eval_duration, eval_count, eval_duration.
       │    ├── Clean & structure report according to selected style
       │    ├── Evaluate research triage priority via TriageService
       │    └── Update job status to "completed" (or "failed" on error)
       ▼
 [Polling Loop / UI Streaming]
       │
       │ 4. Client polls GET /batch/{batch_id} every 1.5 seconds.
       │    BatchStatusResponse returns progress counters and completed job cards.
       ▼
 [Post-Batch Controls]
       ├── POST /batch/{batch_id}/retry   -> Re-queues only failed/cancelled jobs
       ├── POST /batch/{batch_id}/cancel  -> Halts remaining queued jobs
       └── GET  /batch/{batch_id}/download -> Exports CSV or JSON summary
```

### 4.1 Telemetry & Performance Tracking
Every batch item records granular performance telemetry:
- `prep_ms`: Image decoding and thumbnail resizing.
- `load_duration_ms`: Model memory load time (143 ms warm load vs 12s cold load).
- `prompt_eval_ms`: SigLIP visual tokenization and prompt evaluation.
- `eval_duration_ms`: Autoregressive token generation time.
- `tokens_per_second`: Actual text generation throughput (~28 tokens/sec on Apple M3).

---

## 5. Research-Only Triage Workflow

The research triage service operates as an independent post-processing layer that classifies radiographs into preliminary urgency categories based on structured observations:

```mermaid
flowchart TD
    Start["Structured Findings Input\n(From Report Service or User)"] --> CheckErrors{"Inference Error or\nCorrupted Image?"}

    CheckErrors -->|Yes| CatIndet1["indeterminate_review_required\nPriority: urgent\nRule: RULE_INDETERMINATE_FAILED_INFERENCE"]
    CheckErrors -->|No| CheckContradiction{"Internal Contradiction\nDetected Between Sections?"}

    CheckContradiction -->|Yes| CatIndet2["indeterminate_review_required\nPriority: urgent\nRule: RULE_INDETERMINATE_CONTRADICTORY_FINDINGS"]
    CheckContradiction -->|No| CheckUrgent{"Potentially Serious Findings Present?\n(Pneumothorax, Severe Edema,\nLine Malposition, Massive Effusion)"}

    CheckUrgent -->|Yes (Affirmed & Unnegated)| CatUrgent["potentially_urgent\nPriority: urgent\nRule: Precedence 100-140"]
    CheckUrgent -->|No| CheckAbnormal{"Non-Critical Abnormality Present?\n(Consolidation, Cardiomegaly,\nAtelectasis, Routine Devices)"}

    CheckAbnormal -->|Yes (Affirmed & Unnegated)| CatAbnormal["abnormality_flagged\nPriority: routine\nRule: Precedence 200-250"]
    CheckAbnormal -->|No| CatNoPriority["no_priority_finding_flagged\nPriority: standard_review\nRule: RULE_NO_PRIORITY_FINDINGS_FLAGGED\n(MANDATORY: Not Certified Normal)"]

    CatIndet1 --> Disclaimer["Attach Mandatory Research Disclaimer\nrequires_human_review = true"]
    CatIndet2 --> Disclaimer
    CatUrgent --> Disclaimer
    CatAbnormal --> Disclaimer
    CatNoPriority --> Disclaimer
```

### 5.1 Clinical Double-Blind Annotation Protocol
To validate the triage rule engine against clinical reality, a formal human annotation protocol and template have been established in [`templates/triage_annotation_template.csv`](file:///Users/adityakachhot/interactmed/templates/triage_annotation_template.csv):
- **Reviewer Credentials:** Board-certified radiologists or senior radiology residents under faculty supervision.
- **Double-Reading:** Two blinded radiologists independently categorize each case (`reviewer_1_category`, `reviewer_2_category`).
- **Disagreement Adjudication:** Cases with discordant reviews (`disagreement_flag == TRUE`) are routed to a senior radiologist for binding adjudication (`adjudicated_category`, `adjudication_rationale`).
- **Image Quality Audit:** Each case records technical adequacy (`adequate`, `limited_penetration`, `motion_blur`, `unassessable`).

---

## 6. Comprehensive API Endpoint Reference

The backend exposes 10 REST endpoints running on `http://localhost:8000`:

| HTTP Method | Route Path | Purpose | Request Parameters / Body | Response Model / Body | Error Codes | Implementing Service | Associated Tests |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` | Service health, model memory status, Ollama availability | None | `HealthResponse` (status, model loaded, device, Ollama flag) | 200 OK | `GroundingService`, `report_service` | [`tests/test_backend_api.py`](file:///Users/adityakachhot/interactmed/tests/test_backend_api.py) |
| `GET` | `/vision-model-status` | Detects installed vision models in Ollama | None | `VisionModelStatusResponse` (vision model name, install cmd) | 200 OK | `report_service._vision_model_available` | `test_backend_api.py` |
| `POST` | `/ground` | Phrase visual grounding to bounding box + overlay + crop | Multipart Form: `file`, `query`, `checkpoint`, `return_overlay`, `return_crop` | `GroundingResponse` (box coordinates, overlay b64, crop b64) | 400 (empty query/bad checkpoint), 413 (>50MB), 415 (bad ext) | `GroundingService`, `OverlayService` | [`tests/test_grounding_inference.py`](file:///Users/adityakachhot/interactmed/tests/test_grounding_inference.py) |
| `POST` | `/report/region` | Region finding draft from crop + full image | Multipart Form: `file`, `query`, `box_pixel_xyxy` (JSON), `orig_h`, `orig_w` | `RegionReportResponse` (4 sections, crop_info, timing, warnings) | 400 (invalid box coords/dims), 413, 415 | `report_service.region_report` | [`tests/test_report_generator.py`](file:///Users/adityakachhot/interactmed/tests/test_report_generator.py) |
| `POST` | `/report/patient-friendly` | Accessible 4-section patient explanation | Multipart Form: `file`, `reason_for_xray`, `relevant_history` | `PatientFriendlyReportResponse` (4 sections, triage_result, timing) | 413 (>50MB), 415 (bad ext), 503 (Ollama offline) | `report_service.patient_friendly_report`, `TriageService` | `test_report_generator.py`, `test_triage_service.py` |
| `POST` | `/report/study` | Study-level report synthesis (Alias for `/report/patient-friendly`) | Same as above | `StudyReportResponse` (identical to PatientFriendlyReportResponse) | Same as above | Same as above | `test_backend_api.py` |
| `POST` | `/batch/create` | Initializes sequential batch report job (up to 10 images) | Multipart Form: `files` (List of uploads), `style` ('patient'\|'research') | `BatchStatusResponse` (batch_id, job items with previews) | 400 (>10 images or empty files), 413, 415 | `BatchService.create_batch` | [`tests/test_batch_service.py`](file:///Users/adityakachhot/interactmed/tests/test_batch_service.py) |
| `GET` | `/batch/{batch_id}` | Polls status and streaming reports for all jobs in batch | Path Parameter: `batch_id` | `BatchStatusResponse` (counters, per-job statuses, reports) | 404 (batch not found) | `BatchService.get_batch` | `test_batch_service.py` |
| `POST` | `/batch/{batch_id}/cancel` | Immediately cancels remaining queued jobs in batch | Path Parameter: `batch_id` | `BatchStatusResponse` (cancelled: true, queued marked cancelled) | 404 (batch not found) | `BatchService.cancel_batch` | `test_batch_service.py` |
| `POST` | `/batch/{batch_id}/retry` | Retries failed or cancelled jobs without re-running finished | Path Parameter: `batch_id`, Optional Body: `job_ids` (List) | `BatchStatusResponse` (re-queued jobs) | 404 (batch not found) | `BatchService.retry_jobs` | `test_batch_service.py` |
| `GET` | `/batch/{batch_id}/download` | Exports batch summary as CSV or JSON file attachment | Path Parameter: `batch_id`, Query: `format` ('csv'\|'json') | Attachment file (`text/csv` or `application/json`) | 404 (batch not found) | `BatchService.export_summary` | `test_batch_service.py` |
