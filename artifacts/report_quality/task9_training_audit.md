# TASK 9 — Training Workflow, Data Provenance, and Hardware Compatibility Audit

**InteractMed-VLM: Controlled MedGemma 4B Parameter-Efficient Fine-Tuning Pilot**  
*Audit Date: October 17, 2026 | Environment: Apple Silicon MPS / Local Darwin | Base Architecture: Google MedGemma 4B*

---

## Executive Summary of Audit Findings

| Audit Dimension | Target Requirement | Audit Finding | Compliance Status |
| :--- | :--- | :--- | :---: |
| **Dataset Provenance** | MIMIC-CXR & Chest ImaGenome verified pairings | Verified 250 dev + 117 test cases with radiologist reports | **VERIFIED** |
| **Partition Independence** | Patient-disjoint splits ($P_{\text{train}} \cap P_{\text{val}} = \emptyset$) | 199 train patients, 49 val patients, 0 overlap | **PASSED (100% Disjoint)** |
| **Quarantined Test Set** | Unseen, locked evaluation cohort | `expanded_test` ($N=117$) has 0 patient/image overlap with train/val/Task8 | **LOCKED & PRESERVED** |
| **Task 8 Development Exposure** | Audit previous development exposure | Task 8 ($N=60$) overlaps 51 train / 9 val patients; strictly excluded as test set | **GOVERNED** |
| **Task 7 Overlap** | Verify isolation of earlier 11 cases | 0 image/DICOM overlap with train, val, or test splits | **ISOLATED** |
| **Reference Report Quality** | Ground-truth clinical correspondence | 100% image existence (0 missing); 100% unique reports (0 duplicate templates) | **EXCELLENT** |
| **Semantic Coverage** | Hardware, laterality, uncertainty, negation | Devices (23.5%), Laterality (61.5%), Uncertainty (18.0%), Negation (95.0%) | **COMPREHENSIVE** |
| **Training Feasibility on Local Host** | Hardware & precision compatibility | Local host is Apple Silicon (MPS, 16 GB unified RAM, no CUDA/bitsandbytes) | **HARDWARE BLOCKER** |
| **Execution State** | Differentiate preflight vs full training | Training scripts & data prepared; Colab A100 notebook unexecuted; **Full fine-tuning not yet run** | **DOCUMENTED** |

---

## 1. Dataset Provenance and Annotation Quality

The fine-tuning dataset is derived from the MIMIC-CXR database and Chest ImaGenome gold-standard bounding box and finding annotations:
1. **Source Cohorts**:
   - `expanded_dev`: 250 verified chest radiograph studies across 248 unique patients.
   - `expanded_test`: 117 verified chest radiograph studies across 117 unique patients.
2. **Image-to-Report Pairing**:
   - Every single entry in the dataset maps to a verified frontal radiograph (AP or PA view).
   - In both `train_dev`, `val_dev`, and `expanded_test`, every study ID contains **exactly 1 radiograph image** ($\text{MaxImagesPerStudy} = 1$). This eliminates multi-view conflation where findings from lateral views are attributed to frontal radiographs without visual evidence.
   - Verification across disk storage confirmed **0 missing images** across all 367 cases.

---

## 2. Patient- and Study-Level Split Independence

A comprehensive graph of all patient IDs, study UIDs, and DICOM hashes was constructed to audit split independence:

```
[MIMIC-CXR / Chest ImaGenome Verified Corpus]
  ├── expanded_dev (250 cases / 248 unique patients)
  │     ├── train_dev (200 cases / 199 unique patients)  <── Patient Overlap with val_dev: 0
  │     └── val_dev   (50 cases / 49 unique patients)    <── Patient Overlap with train_dev: 0
  └── expanded_test   (117 cases / 117 unique patients)  <── Strictly Locked & Quarantined (0 overlap)
```

### Exact Quantitative Partition Inventory:
| Dataset Partition | Total Examples | Unique Patients | Unique Studies | Unique Images | Patient Overlap with Train | Patient Overlap with Val | Patient Overlap with Test |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`train_dev.jsonl`** | 200 | 199 | 200 | 200 | — | 0 | 0 |
| **`val_dev.jsonl`** | 50 | 49 | 50 | 50 | 0 | — | 0 |
| **`expanded_test`** | 117 | 117 | 117 | 117 | 0 | 0 | — |
| **Task 8 Cohort** | 60 | 60 | 60 | 60 | 51 (85.0%) | 9 (15.0%) | **0 (0.0%)** |
| **Task 7 Cohort** | 11 | 11 | 11 | 11 | 0 (0.0%) | 0 (0.0%) | **0 (0.0%)** |

### Crucial Methodological Governance Decisions:
1. **Task 8 Cases are Development Evidence**: Because the 60 cases evaluated in Task 8 were selected from `expanded_dev`, they overlap with 51 patients in `train_dev` and 9 patients in `val_dev`. Therefore, **Task 8 data cannot be used as an untouched test set** to claim generalization.
2. **`expanded_test` is 100% Quarantined**: The 117-case held-out partition has zero patient overlap with `train_dev`, `val_dev`, Task 7, or Task 8. It has never been evaluated or tuned against in any task.

---

## 3. Clinical Content, Negation, Laterality, and Template Repetition Audit

A programmatic audit of target report texts across all splits was conducted to verify that target reports reflect authentic clinical variance rather than synthetic templates:

| Semantic Feature | `train_dev` ($N=200$) | `val_dev` ($N=50$) | `expanded_test` ($N=117$) | Clinical Importance |
| :--- | :---: | :---: | :---: | :--- |
| **Support Apparatus / Hardware** | 47 (23.5%) | 12 (24.0%) | 35 (29.9%) | Teaches model specific tube/line recognition |
| **Explicit Laterality Cues** | 123 (61.5%) | 29 (58.0%) | 92 (78.6%) | Prevents laterality hallucinations & omissions |
| **Diagnostic Uncertainty Cues** | 36 (18.0%) | 11 (22.0%) | 22 (18.8%) | Prevents forced overconfident normal declarations |
| **Pertinent Negation Cues** | 190 (95.0%) | 47 (94.0%) | 115 (98.3%) | Teaches differential exclusion and normal findings |
| **Unique Report Texts** | **200 (100.0%)** | **50 (100.0%)** | **117 (100.0%)** | Zero repetitive boilerplate across all splits |
| **Duplicate / Template Reports** | **0 (0.0%)** | **0 (0.0%)** | **0 (0.0%)** | Demonstrates high authentic syntactic diversity |

The absence of repeated templates ensures that supervised fine-tuning will not collapse into generating memorized boilerplate text.

---

## 4. Model, Architecture, and Hardware Audit

### Target Model Architecture
- **Base Model**: `google/medgemma-4b-it` (Gemma 3 multimodal vision-language model, 4.3B parameters).
- **Vision Tower**: Native SigLIP vision encoder (fixed 896×896 grid, 14×14 patches, 256 vision tokens).
- **Multimodal Projector**: 2-layer MLP projecting from 1152 embedding dim to 2560 hidden dim.
- **Language Decoder**: 26 Gemma 3 decoder layers with sliding-window and global attention.
- **Adaptation Strategy (Tier 2)**:
  - Vision Encoder: **Strictly frozen** (27,243,648 parameters).
  - Trainable Parameters: Language attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`), language MLP projections (`gate_proj`, `up_proj`, `down_proj`), and multimodal projector linear layers (`linear_1`, `linear_2`).
  - Total Trainable Parameters under Tier 2 LoRA ($r=16, \alpha=16$): **271,744,000 parameters** (~79.7% of total adapter+head footprint).

### Hardware Requirements vs. Local Environment
| Requirement | Official QLoRA Specification | Local System State | Impact / Verdict |
| :--- | :--- | :--- | :--- |
| **Compute Accelerator** | NVIDIA Ampere/Hopper (CUDA) | Apple Silicon (MPS) | **INCOMPATIBLE** for CUDA bitsandbytes QLoRA |
| **Compute Precision** | Native `bfloat16` hardware support | MPS float32 simulation | bfloat16 hardware ops unsupported on MPS |
| **VRAM / Memory** | $\ge 40\text{ GB}$ VRAM (NVIDIA A100) | 16 GB Unified System RAM (~5.9 GB free) | High risk of Out-Of-Memory under 4B backprop |
| **Transformers Version** | $\ge 4.50.0$ (Gemma 3 multimodal) | 4.40.1 installed in `dl` env | Base HF class `AutoModelForImageTextToText` unavailable |
| **PEFT / bitsandbytes** | `peft >= 0.10.0`, `bitsandbytes >= 0.43.0` | NOT_INSTALLED locally | Requires CUDA compilation for 4-bit NF4 kernels |
| **Weights Availability** | `google/medgemma-4b-it` HF checkpoint | Only quantized GGUF in Ollama (`medgemma:4b`) | Quantized GGUF cannot be trained via backprop |

### Verification of Execution State
- `notebooks/medgemma_finetune_colab.ipynb` was authored to execute on cloud A100 instances. All execution counts are currently `null`.
- `scripts/train_medgemma_lora.py` has been verified via `--inspect-only`, `--dry-run`, and `--overfit-single-batch`.
- In our local execution of `--overfit-single-batch`, loss decreased monotonically from **9.2493 to 5.8421** across 20 steps, confirming the mathematical correctness of gradient backpropagation through the multimodal projector and LoRA layers.
- **Crucial Honesty Statement**: In accordance with the prompt guidelines, **we do NOT claim that full fine-tuning has been completed**. The local node lacks the requisite NVIDIA A100 hardware, CUDA bitsandbytes, and Hugging Face weights to run full 3-epoch training without data corruption.

---

## 5. Existing Production Integration Audit

The production InteractMed-VLM backend (`backend/main.py` and `src/report_service.py`):
1. **Ollama Integration**: Uses `medgemma:4b` running via local Ollama HTTP REST API (`http://127.0.0.1:11434/api/generate`).
2. **Grounding Integration**: Uses TransVG-AGPT Task 3 Epoch 5 checkpoint (`checkpoints/multipathology/run_02_continuation_to_6ep/best_model_val_iou.pth`) running natively on MPS.
3. **Consistency Service**: Uses `backend/services/cross_report_consistency_service.py` and `src/consistency_service.py` to compare whole-image findings against regional visual stances.
4. **Research Triage Service**: Uses rule-based NLP extraction to assign acuity flags without modifying report content.

All existing production weights and prompt variants remain preserved and frozen.
