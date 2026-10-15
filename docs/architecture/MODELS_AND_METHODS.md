# InteractMed-VLM: Models and Methodologies Reference

> **Authoritative Technical Model & Methodology Reference**  
> **Project:** InteractMed-VLM  
> **Document:** `docs/architecture/MODELS_AND_METHODS.md`  
> **Status:** Audited against Checkpoints, Source Implementations, and Runtime Environments

---

## 1. Model Inventory & Specification

The InteractMed-VLM platform integrates domain-specific computer vision models, large vision-language models (VLMs), and specialized clinical rule engines. Each model serves an isolated, mathematically distinct role.

### 1.1 Master Model Inventory Table

| Model Component | Architecture & Family | Primary Research Role | Source & Checkpoint Path | Format & Precision | Training Provenance | Input / Output Contract | Inference Framework | Implementation Status | Fine-Tuning Status | Verified Evaluation Evidence | Key Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TransVG-AGPT (Fine-Tuned)** | ResNet-50 + DETR Encoder + ClinicalBERT + 4-layer Fusion Transformer + MLP Head | Phrase-conditioned anatomical / pathology bounding box localization | `checkpoints/finetuned_rsna/run_01_3ep/best_model_val_iou.pth` | PyTorch State Dict (768 MB, Float32/Float16) | Pre-trained on MS-CXR & Chest ImaGenome; fine-tuned 3 epochs on RSNA Pneumonia | **Input:** (1, 3, 640, 640) NestedTensor + raw phrase list `[phrase]`<br>**Output:** Normalized box `[cx, cy, w, h]` $\in [0, 1]$ | PyTorch 2.2.2 with Apple Metal (MPS) / CUDA | **IMPLEMENTED & TESTED** | **Fine-Tuned (Completed: 3 Epochs)** | Mean Best-Match IoU = **0.4530**, Recall@0.5 = **0.4972** on 901 held-out test cases | Outputs exactly 1 box per query; no objectness score (always predicts a box); multi-focal pneumonia partially captured. |
| **TransVG-AGPT (Baseline)** | Same as above | Baseline comparative model for phrase localization | `checkpoints/transvg.pth` | PyTorch State Dict (2.1 GB, Float32) | Checkpoint from AGPT authors (pre-trained on MS-CXR / RefCOCO) | Same as above | PyTorch 2.2.2 (MPS / CPU) | **IMPLEMENTED & TESTED** | **Pretrained Baseline (Frozen)** | Mean Best-Match IoU = **0.3130**, Recall@0.5 = **0.2397** on 901 held-out test cases | Poor localization precision on pneumonia opacities prior to domain adaptation. |
| **Google MedGemma 4B** | Gemma 3 Multimodal Architecture (4.3B params, 2560 hidden dim, SigLIP Vision Grid) | Whole-image radiograph report drafting & patient-friendly translation | Local Ollama store: `medgemma:4b` (derived from `google/medgemma-4b-it`) | GGUF Quantized Container (`Q4_K_M`, 4-bit) | Pre-trained by Google Health on multimodal biomedical corpora | **Input:** Full radiograph (1024 px base64) + structured prompt<br>**Output:** 4-section structured text report | Local Ollama Runtime (Port 11434) with Metal GPU acceleration | **IMPLEMENTED & TESTED** | **Inference: Quantized GGUF; Training: Prepared LoRA (Standby)** | 0.0% boilerplate collapse, 1.6% contradictions, 13.42s latency on 250 dev cases | Frozen SigLIP vision encoder exhibits low sensitivity to subtle, low-contrast grayscale opacities (5.3% pneumonia recall). |
| **LLaVA 7B (`llava:7b`)** | LLaVA v1.5 (CLIP ViT-L/14 + Vicuna 7B + Linear Projector) | Secondary multimodal fallback for regional report drafting | Local Ollama store: `llava:7b` | GGUF Quantized Container (`Q4_0`) | General-domain vision-language pretraining | **Input:** 2 images (Full CXR + Region Crop) + prompt<br>**Output:** 4-section draft finding | Ollama Runtime | **IMPLEMENTED & TESTED** | **Pretrained Baseline (Frozen)** | Tested in regional report fallback when MedGemma unavailable | Lacks biomedical domain specialization; prone to non-clinical descriptive language. |
| **Llama 3 8B (`llama3:latest`)** | Llama 3 Dense Causal LLM | Text-only fallback for study-level report synthesis from metadata | Local Ollama store: `llama3:latest` | GGUF Quantized Container (`Q4_0`) | General-domain pre-training by Meta | **Input:** Text metadata & clinical indication<br>**Output:** Text report narrative | Ollama Runtime | **IMPLEMENTED & TESTED** | **Pretrained Baseline (Frozen)** | Baseline text generation benchmarked in early v0.1.0 audits | Cannot inspect image pixels; strictly disabled for regional image analysis. |
| **Mistral 7B (`mistral:latest`)**| Mistral 7B Dense Causal LLM | Secondary text-only fallback | Local Ollama store: `mistral:latest` | GGUF Quantized Container (`Q4_0`) | General-domain pre-training by Mistral AI | **Input:** Text metadata<br>**Output:** Text report narrative | Ollama Runtime | **IMPLEMENTED & TESTED** | **Pretrained Baseline (Frozen)** | Functional fallback when Llama 3 is uninstalled | Cannot inspect image pixels. |

---

## 2. In-Depth Model Architectures

### 2.1 TransVG-AGPT Visual Grounding Model
TransVG-AGPT is a visual grounding transformer designed to localize the bounding box of a specific anatomical landmark or pathological finding conditioned on an arbitrary clinical phrase query.

```
                              ┌───────────────────────────────────────────────┐
                              │            Input Radiograph (RGB)             │
                              └──────────────────────┬────────────────────────┘
                                                     │
                                                     ▼
                                      Albumentations Letterbox
                                        LongestMaxSize(640)
                                        + PadIfNeeded(640, 640)
                                                     │
                                                     ▼
                              ┌───────────────────────────────────────────────┐
                              │            NestedTensor (1, 3, 640, 640)      │
                              └──────┬─────────────────────────────────┬──────┘
                                     │                                 │
                 Image Tensor        │                                 │ Zero-Mask
                                     ▼                                 ▼
                     ┌───────────────────────────────┐   ┌───────────────────────────────┐
                     │    ResNet-50 + DETR Encoder   │   │     ClinicalBERT Tokenizer    │
                     │    Spatial Visual Features    │   │      Query: "pneumonia"       │
                     │    Shape: (1, 256, 20, 20)    │   │      Language Embeddings      │
                     └───────────────┬───────────────┘   └───────────────┬───────────────┘
                                     │                                   │
                                     └─────────────────┬─────────────────┘
                                                       │
                                                       ▼
                                      ┌─────────────────────────────────┐
                                      │   Multimodal Transformer Fusion │
                                      │   (4-layer Cross-Attention)     │
                                      └────────────────┬────────────────┘
                                                       │
                                                       ▼
                                      ┌─────────────────────────────────┐
                                      │       MLP Box Prediction Head   │
                                      │       Linear -> ReLU -> Linear  │
                                      └────────────────┬────────────────┘
                                                       │
                                                       ▼
                                          Normalized Box: [cx, cy, w, h]
                                                       │
                                                       ▼
                                          Inverse Letterbox Transform
                                                       │
                                                       ▼
                                          Pixel Box: [x1, y1, x2, y2]
```

#### Mathematical Formulation & Forward Pass
1. **Visual Representation:** Given an input image $I \in \mathbb{R}^{H_0 \times W_0 \times 3}$, it is letterboxed onto a square canvas $I_{\text{canvas}} \in \mathbb{R}^{640 \times 640 \times 3}$. A ResNet-50 backbone coupled with a DETR transformer encoder extracts visual feature tokens $F_v \in \mathbb{R}^{N_v \times C}$, where $N_v = \frac{640}{32} \times \frac{640}{32} = 20 \times 20 = 400$ spatial tokens with channel dimension $C = 256$.
2. **Linguistic Representation:** The input query phrase $Q$ (e.g., `"pneumonia"`) is tokenized using `emilyalsentzer/Bio_ClinicalBERT` inside the model, yielding word embeddings $F_t \in \mathbb{R}^{N_t \times C}$, where $N_t \le 40$ word tokens.
3. **Cross-Modal Attention Fusion:** A 4-layer transformer fusion module applies multi-head cross-attention between visual tokens $F_v$ and language tokens $F_t$:
   $$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
   A learnable `[REG]` (region) token attends across all multimodal tokens to synthesize the target object's spatial context.
4. **Bounding Box Regression:** The updated `[REG]` token embedding $h_{\text{reg}} \in \mathbb{R}^C$ is passed to a multi-layer perceptron (MLP) with a sigmoid activation function to output normalized bounding box coordinates:
   $$\hat{b}_{\text{norm}} = \sigma\left(\operatorname{MLP}(h_{\text{reg}})\right) = \left[\hat{c}_x, \hat{c}_y, \hat{w}, \hat{h}\right] \in [0, 1]^4$$
   where $(\hat{c}_x, \hat{c}_y)$ is the normalized box center and $(\hat{w}, \hat{h})$ are normalized box dimensions relative to the $640 \times 640$ canvas.

#### Coordinate Inverse Transformation
To map the normalized prediction $\hat{b}_{\text{norm}}$ back to original image pixel coordinates $[x_1, y_1, x_2, y_2]$ without distortion:
1. Canvas pixel bounding box on the $640 \times 640$ canvas:
   $$x_1^{\text{canvas}} = (\hat{c}_x - 0.5 \hat{w}) \times 640, \quad y_1^{\text{canvas}} = (\hat{c}_y - 0.5 \hat{h}) \times 640$$
   $$x_2^{\text{canvas}} = (\hat{c}_x + 0.5 \hat{w}) \times 640, \quad y_2^{\text{canvas}} = (\hat{c}_y + 0.5 \hat{h}) \times 640$$
2. Letterbox scale factor and padding offsets:
   $$s = \frac{640}{\max(H_0, W_0)}, \quad \text{pad}_x = \frac{640 - s W_0}{2}, \quad \text{pad}_y = \frac{640 - s H_0}{2}$$
3. Original pixel bounding box:
   $$x_1 = \operatorname{clamp}\left(\frac{x_1^{\text{canvas}} - \text{pad}_x}{s}, 0, W_0\right), \quad y_1 = \operatorname{clamp}\left(\frac{y_1^{\text{canvas}} - \text{pad}_y}{s}, 0, H_0\right)$$
   $$x_2 = \operatorname{clamp}\left(\frac{x_2^{\text{canvas}} - \text{pad}_x}{s}, 0, W_0\right), \quad y_2 = \operatorname{clamp}\left(\frac{y_2^{\text{canvas}} - \text{pad}_y}{s}, 0, H_0\right)$$

---

### 2.2 Google MedGemma 4B Report Generation Model

Google MedGemma 4B is an open medical vision-language model built on the Gemma 3 multimodal architecture. In InteractMed-VLM, it functions as the primary visual reasoning engine for full-radiograph report generation and patient-friendly translation.

```
                              ┌───────────────────────────────────────────────┐
                              │            Input Radiograph (RGB)             │
                              │           Standardized to 1024 px             │
                              └──────────────────────┬────────────────────────┘
                                                     │
                                                     ▼
                                      AutoProcessor Image Resampling
                                        Fixed 896 x 896 Spatial Grid
                                                     │
                                                     ▼
                                      SigLIP Vision Encoder (Frozen)
                                        1,152 Hidden Dimensions
                                        Patch Grid: 256 Visual Tokens
                                                     │
                                                     ▼
                                      Multimodal Projector (2-Layer MLP)
                                        Linear(1152 -> 2560) -> GELU ->
                                        Linear(2560 -> 2560)
                                                     │
                                                     ▼
                              ┌───────────────────────────────────────────────┐
                              │    Interleaved Token Sequence:                │
                              │    [Visual Tokens (256)] + [Prompt Tokens]    │
                              └──────────────────────┬────────────────────────┘
                                                     │
                                                     ▼
                                      Gemma 3 Language Decoder (4B)
                                        26 Transformer Decoder Layers
                                        Hidden Size: 2,560 | Heads: 8
                                                     │
                                                     ▼
                                      Autoregressive Text Output
                                        (Prompt v3 Structured Sections)
```

#### Key Architecture Specifications
- **Total Parameters:** ~4.3 Billion ($4,300,000,000$).
- **Language Transformer Backbone:** 26 decoder layers, hidden dimension $D = 2,560$, multi-query attention with 8 query heads.
- **Vision Transformer Backbone:** SigLIP vision transformer pre-trained at $896 \times 896$ px resolution, projecting to 256 visual tokens per image.
- **Multimodal Projector:** Two-layer MLP mapping 1,152-dimensional SigLIP embeddings into the 2,560-dimensional Gemma text embedding space.

#### Local Ollama Runtime vs. Trainable PyTorch Source
A critical technical distinction governs how MedGemma is utilized in this repository:
1. **Local Inference (Ollama Q4_K_M GGUF):** For production inference, the model runs locally via Ollama (`medgemma:4b`). The weights are quantized to 4 bits using integer quantization (`Q4_K_M`), enabling real-time execution on 16 GB Apple Silicon unified memory with Metal acceleration. **GGUF containers are read-only, non-differentiable files that cannot be trained using PyTorch autograd.**
2. **Trainable Source (`google/medgemma-4b-it`):** The proposed parameter-efficient fine-tuning (PEFT/LoRA) pipeline uses the official Hugging Face PyTorch checkpoint (`google/medgemma-4b-it`) in 4-bit NormalFloat (NF4) with 16-bit brain floating point (`bfloat16`) compute kernels on NVIDIA GPUs.

---

## 3. Engineering Methodologies

InteractMed-VLM implements specific clinical engineering methodologies to ensure safety, reproducibility, and reliability.

### 3.1 Two-Stage Grounded Report Generation (Prompt v3 Pipeline)
When small VLMs (e.g. 4B parameters) attempt single-stage end-to-end report generation from complex prompts, they exhibit severe **linguistic bias collapse**: the language decoder attends primarily to formatting instructions and defaults to pre-trained text templates (e.g., *"The lungs are clear, no acute cardiopulmonary disease"*), completely ignoring subtle visual opacities in the image.

To resolve this, InteractMed-VLM implements a **two-stage grounded pipeline**:

```
                              ┌───────────────────────────────────────────────┐
                              │            Full Radiograph (1024 px)          │
                              └──────────────────────┬────────────────────────┘
                                                     │
                                                     ▼
                                     ┌─────────────────────────────────┐
                                     │  Stage 1: Direct Visual Feature │
                                     │  Extraction (No Medical Jargon) │
                                     └───────────────┬─────────────────┘
                                                     │
                                                     ▼
                                        Raw Visual Observations:
                                        "Bilateral lower lobe haziness,
                                         right costophrenic angle blunted,
                                         endotracheal tube tip 3cm carina"
                                                     │
                                                     ▼
                                     ┌─────────────────────────────────┐
                                     │  Stage 2: Structured Clinical   │
                                     │  Synthesis & Section Parsing    │
                                     └───────────────┬─────────────────┘
                                                     │
                                                     ▼
                                     ┌─────────────────────────────────┐
                                     │  Deterministic Factual          │
                                     │  Sanitizer & Contradiction Guard│
                                     └───────────────┬─────────────────┘
                                                     │
                                                     ▼
                                     ┌─────────────────────────────────┐
                                     │  Final 4-Section Output:        │
                                     │  - OBSERVATIONS                 │
                                     │  - POSSIBLE FINDINGS            │
                                     │  - ASSESSMENT AND UNCERTAINTY   │
                                     │  - LIMITATIONS                  │
                                     └─────────────────────────────────┘
```

1. **Stage 1 (Visual Feature Extraction):** The VLM is presented with the full radiograph and an objective prompt focused exclusively on visible physical patterns:
   - *"Describe what visible abnormalities, densities, or opacities are present in the lungs, cardiac silhouette, and any lines, tubes, or devices. Be objective and specific about whether opacities are present."*
   - Generation is constrained to a short token budget (`num_predict=130`), forcing the model to describe what it actually sees rather than synthesizing boilerplate prose.
2. **Stage 2 (Clinical Synthesis & Sectioning):** The extracted visual observations from Stage 1 are passed into a second prompt that structures the findings into four explicit sections (`OBSERVATIONS`, `POSSIBLE FINDINGS`, `ASSESSMENT AND UNCERTAINTY`, `LIMITATIONS`).
3. **Deterministic Sanitizer:** A post-processing regex engine strips prompt leaks, removes fabricated millimeter measurements, and corrects anatomical misconceptions.

---

### 3.2 Anatomical Misconception & Hallucination Sanitization
Empirical testing revealed that small VLMs frequently hallucinate a specific anatomical error when generating patient-friendly explanations for frontal radiographs:
> *"Because this is a frontal view of the chest, it only shows one side of your chest at a time."*

This statement is factually and radiologically false: a frontal projection (AP or PA) visualizes **both hemithoraces simultaneously**, but lacks depth/lateral dimensions. 

`src/report_service.py` (`clean_and_structure_patient_report`) intercepts this error using targeted regular expressions and substitutes an accurate, reassuring explanation:
> *"This frontal view displays both the left and right sides of the chest, but as a single 2D projection, it does not provide a side (lateral) view."*

Furthermore, ungrounded claims asserting absolute normal bone integrity without dedicated rib examinations are flagged with `validation_warnings`.

---

### 3.3 Rule-Based Research Triage Engine
Rather than using an unvalidated neural classifier to predict clinical urgency, InteractMed-VLM uses a **deterministic, transparent rule engine** (`src/triage_service.py`, `configs/triage_rules.yaml`).

#### Precedence Sorting
Rules are evaluated in strict priority order (lowest precedence integer fires first):
1. **Precedence 10–40 (Indeterminate Safeguards):** Failed inference, unreadable images, internal contradictions, or severe motion artifacts trigger `indeterminate_review_required`.
2. **Precedence 100–140 (Potentially Urgent):** Pneumothorax, severe pulmonary edema, critical line malposition, massive effusion, or lobar collapse trigger `potentially_urgent` with human review priority `urgent`.
3. **Precedence 200–250 (Abnormality Flagged):** Non-critical opacities, cardiomegaly, atelectasis, mild effusions, or routine support apparatus trigger `abnormality_flagged` with review priority `routine`.
4. **Precedence 999 (Default Fallback):** When no positive patterns are matched, the system assigns `no_priority_finding_flagged`.

#### Negation and Contradiction Safety Guards
- **Negation Cues:** The rule engine matches finding terms against pre-negation cues (`no `, `without `, `negative for `, `clear of `, `denies `, `absence of `) and post-negation cues (` is ruled out`, ` not identified`, ` not seen`, ` absent`). Mentions such as *"no pneumothorax"* are filtered out and will **never** trigger an urgent rule.
- **Contradiction Detection:** If Stage 1 observations identify active pathology (e.g., *"dense consolidation in right lower lobe"*) but Stage 2 outputs claim *"normal radiograph"*, the rule engine detects the discrepancy and assigns `indeterminate_review_required`.

---

### 3.4 Bounded Batch Orchestration
For processing multiple radiographs, `src/batch_service.py` provides sequential queue orchestration:
- **Max Batch Limit:** Exactly 10 images per batch (strictly enforced at API gateway).
- **Concurrency Cap (`max_workers=1`):** Processing is sequential. On unified memory architectures (16 GB Apple Silicon), concurrent multimodal VLM invocations cause memory spikes and thermal throttling. Sequential execution ensures flat memory consumption (< 190 MB process RSS) and constant generation speed (~28 tokens/sec).
- **Persistent Model In-Memory (`keep_alive: "30m"`):** Pins MedGemma in GPU memory across batch items, reducing per-image model load overhead from ~12 seconds down to **143 ms**.
- **Per-Item Failure Isolation:** If image #3 in a 10-image batch has a corrupted DICOM header, job #3 is marked `failed`, but jobs #1–2 and #4–10 complete normally.
- **Granular Retry:** The API endpoint `POST /batch/{id}/retry` re-executes only `failed` or `cancelled` jobs without re-running successful items.

---

### 3.5 Quantitative Evaluation Mathematical Taxonomy

The evaluation pipeline (`scripts/evaluate_rsna.py`, `scripts/evaluate_expanded_reports.py`) uses strict mathematical formulations:

#### 1. Intersection over Union (IoU)
For a predicted bounding box $B_p = [x_1, y_1, x_2, y_2]$ and ground truth box $B_g = [x_1', y_1', x_2', y_2']$:
$$\operatorname{IoU}(B_p, B_g) = \frac{\operatorname{Area}(B_p \cap B_g)}{\operatorname{Area}(B_p \cup B_g)} = \frac{\operatorname{Area}(B_p \cap B_g)}{\operatorname{Area}(B_p) + \operatorname{Area}(B_g) - \operatorname{Area}(B_p \cap B_g)}$$

#### 2. Mean Best-Match IoU (Multi-Box Images)
In multi-focal pathology (e.g. bilateral pneumonia) where an image has $K \ge 1$ ground truth boxes $\{B_{g, 1}, \dots, B_{g, K}\}$, the single predicted box $B_p$ is matched using the standard VOC / COCO convention:
$$\operatorname{BestMatchIoU}(B_p) = \max_{k \in \{1, \dots, K\}} \operatorname{IoU}(B_p, B_{g, k})$$
The Mean Best-Match IoU across an evaluation cohort of $N$ images is:
$$\overline{\operatorname{IoU}} = \frac{1}{N} \sum_{i=1}^N \operatorname{BestMatchIoU}(B_{p, i})$$

#### 3. Recall at IoU Threshold $\tau$ ($\operatorname{Recall}@\tau$)
$$\operatorname{Recall}@\tau = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left(\operatorname{BestMatchIoU}(B_{p, i}) \ge \tau\right), \quad \tau \in \{0.50, 0.75\}$$

#### 4. Clinical Finding Micro-Precision, Recall, and F1
For $C$ clinical finding categories across $N$ reports:
$$\text{Precision} = \frac{\sum_{c=1}^C \text{TP}_c}{\sum_{c=1}^C \text{TP}_c + \sum_{c=1}^C \text{FP}_c}, \quad \text{Recall} = \frac{\sum_{c=1}^C \text{TP}_c}{\sum_{c=1}^C \text{TP}_c + \sum_{c=1}^C \text{FN}_c}$$
$$\text{Micro-F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

#### 5. Asymmetric Wilson Score 95% Confidence Interval
To account for small cohort sizes and rare positive clinical findings without Gaussian approximation distortion:
$$w = \frac{p + \frac{z^2}{2n} \pm z \sqrt{\frac{p(1-p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$
where $p$ is the observed proportion, $n$ is the sample size, and $z = 1.95996$ for a 95% confidence level.
