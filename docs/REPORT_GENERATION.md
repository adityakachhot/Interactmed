# Report Generation — Architecture, Medical Reliability & Verification

## 1. Overview

**InteractMed-VLM** implements a two-stage local phrase-grounded radiology assistant:

1. **Phrase Grounding (TransVG-AGPT):** Localizes a clinical phrase (e.g. *"cardiomegaly"*, *"pneumonia"*) onto chest radiograph coordinates on Apple MPS / CPU.
2. **Region-Specific Draft Report:** Directly on the Grounding page, generates a structured 4-part medical finding draft using an image-capable vision-language model (VLM).

```
[Chest Radiograph (DICOM/PNG)] + [Clinical Phrase]
                        │
                        ▼
            TransVG-AGPT Grounding (MPS)
                        │
                        ├─────────────────────────────────────────────────┐
                        ▼                                                 ▼
             Canonical Pixel Bounding Box                     Annotated Visual Overlay
                 [x1, y1, x2, y2]                              (Cyan 4px + Label)
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
Derived Normalized Box         Extracted Region Crop
[cx, cy, w, h] in [0, 1]      (+40 px margin clamped)
         │                             │
         └──────────────┬──────────────┘
                        ▼
          [Generate Region Report]
         (Directly on Grounding Page)
                        │
                        ▼
         Local Medical Vision-Language Model
        (e.g., MedGemma:4b / LLaVA:7b via Ollama)
        Inputs: Region Crop + Full Context CXR
                        │
                        ▼
            Lightweight Validation Layer
            • Empty response check
            • Unsupported measurements check
            • Anatomical contradiction check
            • Phrase echo check
                        │
                        ▼
       Structured Report Card on Grounding Page
```

---

## 2. Vision Model Policy & Evaluation

### Why Text-Only Models (e.g. Llama 3) Were Rejected for Image Description
Previous versions displayed `llama3` for report drafting. However, standard LLaMA-3 models are **text-only**:
- They receive only text prompts and bounding-box numbers.
- They cannot see image pixels, opacity, lung markings, or silhouettes.
- Any clinical descriptions produced by text-only models are **fabrications** hallucinated from the user's prompt rather than observed in the radiograph.

### Image-Capable VLM Integration
InteractMed-VLM strictly enforces the following policy:
- **No text-only fallback for region reports:** If no image-capable vision model is installed locally, the backend returns an explicit error (`success: False`) explaining that a vision model is required. It **never** fabricates a report from text metadata alone.
- **Evaluated Models:**
  - **MedGemma:4b** (`medgemma:4b`): A specialized Gemma-3 medical vision-language model trained on medical text and image comprehension. Recommended for clinical radiology tasks.
  - **LLaVA 7B** (`llava:7b`): A 7-billion parameter multimodal model that accepts base64-encoded image payloads alongside text prompts.
- **Model Discovery:** The backend automatically queries Ollama (`GET /api/tags`) and detects installed vision models prioritizing medical models (`medgemma:4b`, `medgemma:latest`, `medgemma`), followed by general multimodal models (`llava:7b`, `llava:13b`, `llava`, `llama3.2-vision`, `minicpm-v`, `moondream`, `bakllava`).

### Data Privacy & Local Execution
- **Strictly Local:** All grounding and VLM inference runs on `localhost` (Ollama at `http://localhost:11434` and TransVG on MPS/CPU).
- **No External Services:** Radiograph pixels and report text are never transmitted over the internet or to external cloud APIs.
- **No MIMIC-CXR Leakage:** MIMIC-CXR images and annotations remain strictly on local disks and are never transmitted externally.

---

## 3. Region Report Pipeline & Image Inputs

When the user clicks **"Generate Region Report"** on the Grounding page, the pipeline passes:

1. **Extracted Region Crop (`crop_b64`):** Extracted using the canonical pixel bounding box `[x1, y1, x2, y2]` with an adaptive `+40 px` margin, clamped to image borders `[0, W]` and `[0, H]`. Encoded as base64 JPEG (max 512 px).
2. **Original Full Radiograph (`full_b64`):** Rescaled preserving aspect ratio (max 512 px) to provide anatomical context (contralateral comparison, overall lung expansion, cardiothoracic ratio).
3. **Hypothesis Framing:** The user's target phrase is framed as a **question or localization request to evaluate**, never as a confirmed diagnosis.
4. **No Coordinate Leakage in Clinical Narrative:** Pixel coordinates, bounding-box area percentages, and raw image dimensions are excluded from the clinical narrative prompt to prevent the model from generating confusing geometric jargon.

---

## 4. Structured Medical Report Prompt & Rules

The generation pipeline enforces four standardized sections targeting approximately 80–150 words:

```
TARGET FINDING:
State the clinical phrase being evaluated (e.g. "cardiomegaly").
Treated strictly as a hypothesis to assess, not a confirmed diagnosis.

IMAGE OBSERVATIONS:
Describe physical radiographic density, opacity, borders, and anatomical structures.
Primary context (Image 1) is the complete thoracic radiograph; supplementary context (Image 2) is the magnified crop.
For cardiac queries, evaluates heart size relative to maximum inner thoracic width, never crop alone.
Incidental findings are strictly controlled; if relevant and visible, labeled as "Tentative incidental observation (requires review)".

ASSESSMENT AND UNCERTAINTY:
Categorized explicitly as: "Supported:", "Indeterminate:", or "Not supported:".
Summarizes diagnostic stance in a non-redundant sentence with clinical uncertainty.
Does NOT repeat observations verbatim.

LIMITATIONS:
Reports genuine technical limitations (e.g. single-view projection, inspiratory depth, positioning).
Does not falsely claim "limited field of view" because the complete radiograph is supplied.
```

### Prompt Guardrails & Pipeline Controls
- **Image Roles:** Image 1 is explicitly designated as **PRIMARY CONTEXT** (full thoracic CXR), and Image 2 as **SUPPLEMENTARY CONTEXT** (magnified crop).
- **Deduplication Engine:** Automatic sentence-overlap analysis removes redundant observation sentences from the assessment, ensuring distinct diagnostic synthesis.
- **Incidental Finding Control:** Prevents hallucinated lower lung opacities from being injected into unrelated regional or cardiac queries. If an incidental finding is detected, it is labeled as tentative requiring review.
- **Latency Profiling:** Preprocessing (`prep_ms`), model generation (`inference_ms`), and post-processing (`post_ms`) are measured and reported separately.
- **Single Outside Disclaimer:** The research-use disclaimer is displayed as a dedicated UI banner outside the report narrative body, avoiding cluttering the clinical text.
- **Localization Data Separation:** Pixel coordinates, dimensions, box area percentage, and latency numbers are stored in structured response fields and kept out of the medical text.

---

## 5. Lightweight Output Validation Layer

Before returning or rendering any generated report, `validate_report()` and `clean_and_structure_regional_report()` in `src/report_service.py` execute:

| Check | Condition Checked | Action Taken |
|---|---|---|
| **Empty Response** | Length < 30 characters or blank | Flags response (`flagged=True`, `valid=False`) |
| **Echo Detection** | Report repeats query phrase with < 30 words and no narrative | Flags warning: "merely repeats target phrase without image analysis" |
| **Sentence Redundancy** | Assessment repeats image observations (>50% word overlap) | Strips duplicate sentence; synthesizes clean diagnostic stance |
| **Incidental Abnormality** | Incidental lower lung opacity claimed during cardiac query | Labels as `Tentative incidental observation (requires review)` and flags warning |
| **Unsupported Measurements** | Regex detects numbers with units (`cm`, `mm`, `HU`, `mmHg`, `ml`) | Flags warning: "contains specific measurements ... cannot be derived from image metadata" |
| **Anatomical Contradictions** | Cardiac terms (`cardiomegaly`, `heart`) paired with `upper lung zone`, `apical zone`, `lung apex`, or `pneumothorax` paired with `cardiac silhouette` | Flags warning: "Potential anatomical contradiction detected" |
| **Crop Limitation Claim** | Erroneously states "limited by cropped view" when full CXR was passed | Replaces with genuine single-view projection limitation |

Validation flags and warnings are transmitted in `validation_warnings` and `validation_flagged` and displayed as warning badges on the frontend report card.

---

## 6. Canonical Coordinate Frame & Display Alignment

To prevent inconsistencies where displayed normalized coordinates do not match image pixel dimensions:

1. **Single Canonical Representation:** The predicted bounding box in original pixel space `[x1, y1, x2, y2]` (clamped to `[0, orig_w]` and `[0, orig_h]`) is the **single source of truth**.
2. **Derived Normalized Coordinates:** Normalized values are mathematically derived directly from the canonical pixel box:
   $$\text{norm\_cx} = \frac{(x_1 + x_2) / 2}{\text{orig\_w}}, \quad \text{norm\_cy} = \frac{(y_1 + y_2) / 2}{\text{orig\_h}}$$
   $$\text{norm\_w} = \frac{x_2 - x_1}{\text{orig\_w}}, \quad \text{norm\_h} = \frac{y_2 - y_1}{\text{orig\_h}}$$
   This guarantees that $\text{norm\_cx} \times \text{orig\_w} \equiv c_x$ and $\text{norm\_w} \times \text{orig\_w} \equiv \text{width}$, eliminating previous discrepancies between canvas-normalized and image-pixel values.
3. **Overlay & Crop Alignment:**
   - Visual overlay in `src/overlay_service.py` draws a cyan rectangle at `[(x1, y1), (x2, y2)]`.
   - Region crop in `src/report_service.py` crops pixels at `[x1-40, y1-40, x2+40, y2+40]` clamped to image bounds.
4. **Area Semantic Labeling:** The area metric is explicitly labeled **"Box area (% of image pixels)"**, preventing it from being misinterpreted as disease severity or extent.

---

## 7. Study-Level vs. Region-Specific Reports

The application strictly separates two distinct clinical workflows:

| Attribute | Region-Specific Report (Grounding Tab) | Study-Level Report (Reports Tab) |
|---|---|---|
| **Location** | Displayed on the **Grounding Tab** alongside the prediction & crop | Dedicated **Study Report Tab** |
| **Trigger** | "Generate Region Report" button | "Generate Study Report" button |
| **Inputs** | Image crop + full CXR + localized phrase | Full radiograph + required clinical indication + patient history + optional prior report |
| **Output Focus** | Hypothesized finding evaluation in localized region | Comprehensive multi-field radiology report (TECHNIQUE, COMPARISON, FINDINGS, IMPRESSION) |
| **Prior Studies** | None (focused on localized region) | User-supplied text; explicitly states *"No prior available for comparison"* if omitted |

---

## 8. Remaining Limitations

1. **Clinical Validation:** The vision-language models and grounding models are research prototypes. **No automated checks guarantee medical correctness.** Direct review by a qualified radiologist is always mandatory.
2. **Model Availability & Latency:** If no local VLM is installed in Ollama, region report generation will refuse to run until a vision model is pulled (`ollama pull medgemma:4b` or `ollama pull llava:7b`). On Apple Silicon (16 GB unified memory), VLM inference takes 15–45 seconds depending on model quantization and context length.
3. **Small Focal Lesions:** Sub-centimeter lung nodules or subtle rib fractures near the edge of a crop may have limited resolution when downscaled to 512 px. Radiologists must always review the native full-resolution DICOM.
