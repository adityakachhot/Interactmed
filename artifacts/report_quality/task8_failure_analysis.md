# Task 8 — Failure Analysis, Metric Operational Audits, and Failure Modes

**Author:** Senior Medical Vision-Language Model Researcher & ML Evaluation Engineer  
**Date:** October 2026  
**Status:** In-Depth Empirical Failure Mode Audit & Theoretical Reconciliation  
**Evaluation Set:** 60 Patient-Disjoint Cases (`expanded_dev`) & 11 Quarantined Development Cases  
**Models:** `medgemma:4b` via Ollama | TransVG-AGPT Task 3 Epoch 5 Checkpoint  

---

## 1. Metric Operational Audit: Resolving the Prompt v6 Metric Coexistence Paradox

In Task 7, evaluation of Grounding-Assisted Prompt v6 reported the following headline figures on the 11 development cases:
- **False-Normal Rate:** **0.0%**
- **Finding Omission Rate:** **80.95%**
- **Micro-Averaged Recall:** **0.1905 (19.05%)**

At first glance, a 0% false-normal rate might seem to imply near-perfect pathology detection. However, this interpretation conflates two fundamentally different units of analysis: **study-level report classification** versus **slot-level multi-label finding recall**.

### 1.1 Mathematical & Operational Definitions

1. **Study-Level False-Normal Rate:**
   $$\text{False-Normal Rate} = \frac{\sum_{i=1}^N \mathbb{I}(\text{Reference Abnormal}_i \land \text{Report Stated Normal}_i)}{\sum_{i=1}^N \mathbb{I}(\text{Reference Abnormal}_i)}$$
   - **Unit of Analysis:** Entire examination / patient report ($N=11$ in Task 7; $N=50$ abnormal cases in Task 8).
   - **Operational Rule:** A report is graded as a "False Normal" if and only if the patient has confirmed reference abnormalities, but the generated report's overall impression concludes *"Normal chest radiograph"*, *"No acute findings"*, or *"Unremarkable"* without identifying *any* positive abnormality across all report sections.
   - **Why Prompt v6 Achieved 0%:** Prompt v6 combines whole-image evaluation with TransVG-AGPT regional review. In every single case, at least one feature was asserted (such as an indwelling catheter, sternotomy wire, or cardiomegaly). Because the report never declared the scan completely normal, the study-level false-normal count was exactly 0.

2. **Finding-Level Omission Rate:**
   $$\text{Finding Omission Rate} = \frac{\text{Total False Negatives (FN)}}{\text{Total Affirmed Reference Findings}} = 1.0 - \text{Micro Recall}$$
   - **Unit of Analysis:** Individual multi-label pathology slots (e.g., cardiomegaly, pleural effusion, pneumothorax, consolidation, atelectasis, edema, support devices).
   - In complex ICU or post-operative patients (such as Case 1 of Task 7), a single radiograph frequently contains **4 to 6 co-occurring abnormalities**:
     $$\text{Case 1 Ground Truth} = \{\text{PTX}, \text{Atelectasis}, \text{Edema}, \text{Widened Mediastinum}, \text{ET Tube}, \text{Chest Tube}, \text{Swan-Ganz}\}$$
   - When Prompt v6 successfully identified the endotracheal tube but missed the small apical pneumothorax, the interstitial edema, the atelectasis, and the chest tube:
     - The study was **not normal** (0 false normal).
     - But **4 out of 5 findings were omitted** (80% omission rate for that case).

3. **Reconciliation:**
   - A model can readily achieve a 0% study-level false-normal rate simply by consistently detecting high-contrast support hardware or cardiomegaly, while remaining completely blind to life-threatening parenchymal conditions (pneumothorax, pneumonia, pulmonary edema).
   - **Governance Policy:** Study-level false-normal rates must **never** be cited in isolation as proof of clinical efficacy. Granular per-finding sensitivity and omission rates are mandatory.

---

## 2. Technical Failure Audit of Prompt v5 (Two-Stage JSON Pipeline)

In Task 7, Strategy B (Prompt v5 Two-Stage Extraction) experienced a catastrophic 0% schema validity rate across the 11 development cases (`Strategy B JSON valid: False`).

### 2.1 Root Cause of JSON Generation Breakdown
1. **Schema Over-Specification:**
   The initial Stage 1 schema (`prompt_v5_two_stage_extraction_schema.json`) requested a deeply nested object comprising 9 top-level keys, arrays of structured device objects, four distinct string enumerations, and technical limitation fields.
2. **Autoregressive Token Budget & Context Exhaustion:**
   MedGemma 4B is a 4-billion parameter model. Without constrained context-free grammar (CFG) decoding in Ollama, generating 450+ tokens of deeply nested JSON syntax caused the model to:
   - Prematurely output an EOS token after generating 1 or 2 keys.
   - Omit closing brackets `}` or quotes `"`.
   - Wrap output in Markdown code blocks (````json ... ````) that broke naive string decoders.
3. **Flawed Fallback Calculation:**
   In Task 7, when JSON validation failed, fallback parsing attempted regex recovery on partial text. This introduced high false-positive noise and spurious assertions (FP=12), resulting in a distorted micro F1 score (0.3125) that reflected hallucinated fallback matches rather than genuine structured extraction.

### 2.2 Task 8 Streamlined Remediation
To fairly evaluate the two-stage paradigm in Task 8, `prompt_v5_streamlined_extraction.txt` and `prompt_v5_streamlined_schema.json` were engineered:
- Reduced schema depth to a flat, robust JSON dictionary evaluating the 7 core target categories as `["present", "absent", "indeterminate"]`.
- Added regex fence stripping (`re.search(r"(\{.*\})", raw_resp, re.DOTALL)`).
- Separated:
  - **Schema Validity Rate:** % of runs producing syntactically valid JSON conforming to the schema.
  - **Conditional Quality:** Finding-level F1 on valid JSON generations.
  - **End-to-End Usable Rate:** Overall proportion of inference calls yielding a medically sound, parseable report.

---

## 3. Visual Representation Ceiling of MedGemma 4B

The central empirical insight of Tasks 7 and 8 is that **prompt engineering alone cannot overcome the fundamental vision encoder limitations of un-finetuned MedGemma 4B**.

### 3.1 Anatomical Resolution Attenuation
- **Patch Resolution:** Vision transformers split images into fixed patch embeddings (typically 14×14 or 16×16 pixels). On a 1024×1024 radiograph, fine line structures—such as the visceral pleural line in a 1.5 cm apical pneumothorax or surgical clips—occupy only 2 to 4 patches.
- **Global Context Smoothing:** In whole-image self-attention, the dominant signal from normal rib shadows, clavicles, and clear lung parenchyma drowns out the faint localized edge displacement of a pneumothorax or plate-like atelectasis.
- **Conservative Prior:** Because MedGemma was pre-trained on broad web and medical corpora where "normal" dictations vastly outnumber acute pneumothoraces or consolidations, its language decoder has an overwhelming statistical prior toward predicting "clear" or "normal size".

### 3.2 TransVG-AGPT Objectness Head Deficit
- TransVG-AGPT is an attention-guided visual grounding model designed to answer: *"Where is phrase X?"*
- It is **not** a detection or classification model; it lacks a calibrated classification head (such as a sigmoid presence probability or objectness score).
- When queried for *"pneumothorax"* on an image of a perfectly healthy patient, TransVG-AGPT will highlight the region with the highest relative attention (frequently the lung apex) and return a bounding box.
- Therefore, bounding boxes from TransVG-AGPT represent **spatial hypotheses, not diagnostic proofs**. Strategy C correctly decouples spatial proposal from radiological verification by passing the clamped regional crop back to MedGemma 4B.

---

## 4. Synthesis Loss and Inter-Section Discordance

Task 7 identified widespread discordance within single model outputs:
1. **Observation vs. Impression Contradiction:**
   In 8 out of 11 development cases, MedGemma observed a chest tube or vascular catheter in Section 1, but concluded in Section 6: *"No acute cardiopulmonary abnormality"* or *"Normal chest radiograph"*.
2. **Laterality Reversal:**
   The vision encoder occasionally detected abnormal opacity in the left lower lobe, but the text decoder summarized it as right lower lobe consolidation in the impression.
3. **Synthesis Omission:**
   In two-stage pipelines, positive findings validated in Stage 1 were occasionally dropped in Stage 2 because the report synthesizer treated them as incidental.

### Remediation via Automated Consistency Validation:
The updated `src/consistency_service.py` evaluates all five consistency dimensions and flags discrepancies with `review_required: True` without mutating the clinical narrative. This ensures complete transparency for human radiologists.
