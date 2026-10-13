# InteractMed-VLM: Comprehensive Dataset Inventory & Audit

**Execution Date:** October 10, 2026  
**Auditor:** Lead ML Engineer  
**Host Environment:** Apple Silicon M3 MacBook Air (16 GB Unified Memory, macOS)  
**Python Environment:** `conda` environment `dl` (`Python 3.10.19`, `PyTorch 2.2.2`, `MPS` accelerated)  
**Verification Script:** [`scripts/inventory_datasets.py`](file:///Users/adityakachhot/interactmed/scripts/inventory_datasets.py)  
**Configuration File:** [`configs/data_paths.yaml`](file:///Users/adityakachhot/interactmed/configs/data_paths.yaml)

---

## 1. Executive Summary of Dataset Holdings

| Dataset | Local Status | Total Records | Unique Images (DICOM) | Unique Studies | Images Present Locally | Local Image Coverage (%) | Bounding Box Convention |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MIMIC-CXR-JPG (Local Store)** | Available (Partial) | N/A | **1,174** | 1,174 | 1,174 | 100.0% | N/A (Image Store) |
| **MS-CXR (v1.1.0)** | Available | **1,448** | **1,047** | ~1,047 | **764** | **72.97%** | `[x, y, w, h]` (Native Pixels) |
| **Chest ImaGenome Gold (BBox)** | Available | **25,989** | **1,000** | 1,000 | **417** | **41.70%** | `[x1, y1, x2, y2]` (224 & Native) |
| **Prior Unified Manifest** | Available | **17,135** | **1,541** | 1,541 | **1,174** | **76.18%** | `[x, y, w, h]` (Native Pixels) |
| **MIMIC-CXR-VQA (Val / Test / Train)** | Available | **87,360+** | Multi-view | ~40,000+ | Linked to MIMIC | Question-Answer Pairs | N/A (QA Tuples) |
| **RSNA Pneumonia Detection** | **Missing** | 0 | 0 | 0 | 0 | 0.0% | `[x, y, w, h]` (1024x1024 Pixels) |
| **TransVG-AGPT Checkpoint** | **Missing** | 0 | 0 | 0 | 0 | 0.0% | Model Weights (`transvg.pth`) |

---

## 2. Dataset-by-Dataset Audit

### 2.1 Local MIMIC-CXR Image Store
* **Discovered Directory:** `/Users/adityakachhot/Downloads/data/mimic-cxr`
* **Hierarchy:** `files/p{prefix}/p{subject_id}/s{study_id}/{dicom_id}.jpg`
* **Partitions Discovered:** 10 patient prefix subdirectories (`p10` through `p19`).
* **Available File Count:** Exactly **1,174** `.jpg` files (total non-hidden image files).
* **Format:** Compressed standard JPEG (RGB/Grayscale).
* **Integrity Audit:** 100% of these 1,174 local images are referenced in the unified grounding manifest; zero orphan image files exist on disk.
* **Missing Files:** Out of 1,541 images requested in the earlier project download list, **367** images are missing from local disk.

---

### 2.2 MS-CXR (Local Alignment v1.1.0)
* **Metadata Paths:**
  * CSV: `/Users/adityakachhot/Downloads/data/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.csv` (311 KB)
  * COCO JSON: `/Users/adityakachhot/Downloads/data/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.json` (3.33 MB)
* **Total Records:** **1,448** annotation records.
* **Unique Image Identifiers:** **1,047** unique `dicom_id`s across 851 subjects.
* **Local Image Match:**
  * **1,056** out of 1,448 records (72.93%) have their corresponding `.jpg` file present locally.
  * **764** out of 1,047 unique images (72.97%) exist on local storage.
  * **283** unique images (392 annotation records) are missing.
* **Schema & Columns:**
  * CSV Columns: `['dicom_id', 'category_name', 'label_text', 'path', 'x', 'y', 'w', 'h', 'image_width', 'image_height', 'split']`
  * JSON Keys: `['info', 'licenses', 'categories', 'images', 'annotations']`
* **Bounding Box Format & Coordinate System:**
  * Coordinates: `[x, y, w, h]` (top-left $x$, top-left $y$, box width $w$, box height $h$).
  * Space: Unnormalized absolute pixel coordinates based on the native DICOM dimensions (`image_width`, `image_height`).
* **Categories (8 Pathology Classes):**
  * `cardiomegaly`, `pneumothorax`, `pleural effusion`, `pneumonia`, `consolidation`, `lung opacity`, `atelectasis`, `edema`.
* **Clinical Text & Reports:**
  * Fine-grained medical phrases are present in the `label_text` field (phrase-level annotations).
  * Full raw MIMIC free-text reports (`.txt`) are not bundled with MS-CXR.

---

### 2.3 Chest ImaGenome Dataset (v1.0.0 Gold & Silver)
* **Metadata Paths:**
  * Gold BBox CSV: `/Users/adityakachhot/Downloads/data/chest-imagenome/gold_dataset/gold_bbox_coordinate_annotations_1000images.csv` (6.64 MB)
  * Gold Attributes & Relations: `/Users/adityakachhot/Downloads/data/chest-imagenome/gold_dataset/gold_object_attribute_with_coordinates.txt` (7.42 MB)
  * Gold Sentence Extraction: `/Users/adityakachhot/Downloads/data/chest-imagenome/gold_dataset/gold_all_sentences_500pts_1000studies.txt` (1.66 MB)
  * Scaling Factors: `/Users/adityakachhot/Downloads/data/chest-imagenome/gold_dataset/gold_bbox_scaling_factors_original_to_224x224.csv` (78.8 KB)
  * Master View Record List: `/Users/adityakachhot/Downloads/chest-imagenome-dataset-1.0.0/utils/cxr-record-list_view.csv` (55.0 MB)
  * Demographics & Patient Context: `/Users/adityakachhot/Downloads/chest-imagenome-dataset-1.0.0/utils/cxr-study-list_v2.csv` (13.7 MB)
* **Total Records:**
  * Gold Bounding Boxes: **25,989** records across **1,000** gold studies/images.
  * Gold Sentences: **11,885** extracted sentence records across 500 patients and 1,000 studies.
* **Local Image Match:**
  * **417** out of the 1,000 gold images (41.7%) exist locally on disk as `.jpg`.
  * **10,840** out of 25,989 bounding box annotations (41.7%) correspond to locally existing images.
  * **583** gold images are missing locally.
* **Schema & Columns:**
  * BBox CSV: `['image_id', 'x1', 'y1', 'x2', 'y2', 'width', 'height', 'bbox_name', 'annot_id', 'original_x1', 'original_x2', 'original_y1', 'original_y2', 'original_width', 'original_height', 'annotator', 'coord224', 'coord_original']`
  * Study List CSV: `['subject_id', 'study_id', 'path', 'gender', 'age_decile']`
* **Bounding Box Formats & Conventions:**
  * Two distinct coordinate formats are explicitly precomputed in the dataset:
    1. `coord224`: `[x1, y1, x2, y2]` resized into a standard $224 \times 224$ image space.
    2. `coord_original`: `[original_x1, original_y1, original_x2, original_y2]` in native image pixel resolution.
  * Component columns: `x1`, `y1` (top-left), `x2`, `y2` (bottom-right), `width`, `height`.
* **Clinical Text & Patient Context Links:**
  * Sentences are partitioned by section (`FINDINGS` vs `IMPRESSION`) and mapped to specific bounding box objects and attributes.
  * Master demographics file (`cxr-study-list_v2.csv`) provides `gender` and `age_decile`, directly unlocking **patient-context-conditioned study report generation**.

---

### 2.4 Prior Unified Grounding Manifest
* **File Paths:**
  * Manifest: `/Users/adityakachhot/interactmed/manifests/unified_manifest.csv` (2.4 MB)
  * Download List: `/Users/adityakachhot/interactmed/manifests/download_list.txt` (104 KB)
* **Total Harmonized Records:** **17,135** rows.
* **Breakdown by Source:**
  * `chest_imagenome_gold`: 15,687 rows across 500 unique DICOM images (412 images present locally, 82.4% record coverage).
  * `ms_cxr`: 1,448 rows across 1,047 unique DICOM images (764 images present locally, 72.9% record coverage).
* **Overall Manifest Image Coverage:**
  * **13,992** out of 17,135 records (81.66%) have locally present image files.
  * **1,174** out of 1,541 unique referenced images (76.18%) exist locally.
  * **367** unique images are pending download.
* **Standardized Schema:**
  * Columns: `['dicom_id', 'subject_id', 'study_id', 'image_relative_path', 'region_label', 'bbox', 'phrase', 'finding_text', 'split', 'source', 'raw_region', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']`
  * Standardized BBox: `[bbox_x, bbox_y, bbox_w, bbox_h]` in original pixel coordinates.

---

### 2.5 MIMIC-CXR-VQA Dataset
* **File Paths:**
  * `valid.json`: 73,567 QA records (65 MB)
  * `test.json`: 13,793 QA records (12 MB)
  * `train.json`: ~270,000+ QA records (257 MB in Downloads raw package)
* **Schema:** `['split', 'idx', 'subject_id', 'study_id', 'image_id', 'image_path', 'question', 'semantic_type', 'content_type', 'template', 'template_program', 'template_arguments', 'answer']`
* **Utility:** Provides secondary validation for multimodal understanding and question-answering evaluation.

---

## 3. Cross-Dataset Overlaps & Union Analysis

Using set arithmetic on verified unique DICOM identifiers:

```
MS-CXR Unique DICOMs:                    1,047
Chest ImaGenome Gold Unique DICOMs:      1,000
---------------------------------------------------
Overlap (MS-CXR ∩ Chest ImaGenome Gold):     9 DICOMs
Total Unique Benchmark Images:           2,038 DICOMs
Total Locally Present Images:            1,174 DICOMs
---------------------------------------------------
Images Present in Both Benchmarks:           7 DICOMs
MS-CXR Images Present Locally:             764 DICOMs (73.0%)
Chest ImaGenome Gold Images Present:       417 DICOMs (41.7%)
```

### Key Insight:
* The overlap between MS-CXR and Chest ImaGenome Gold is minimal (**only 9 images**). They represent complementary evaluation distributions:
  * **Chest ImaGenome Gold**: Evaluates anatomical region grounding (e.g., *cardiac silhouette*, *right middle lung zone*, *costophrenic angle*).
  * **MS-CXR**: Evaluates pathology phrase grounding (e.g., *left lower lobe consolidation*, *enlarged cardiomediastinum*).
* Combining both yields **2,038 distinct radiological benchmark images**, of which **1,174 (57.6%)** are immediately evaluable offline without downloading extra images.

---

## 4. Missing Files, Unresolved Paths, & Blockers

### Blocker 1: Pretrained TransVG-AGPT Checkpoint
* **Current State:** No `.pth` checkpoint for TransVG-AGPT exists locally.
* **Impact:** Offline phrase grounding evaluation and inference cannot run with real weights until the checkpoint is downloaded.
* **Resolution Path:**
  * Official AGPT Google Drive folder ID: `1lFaqAlG6AJExCjkW3Pbb24CX-LWSstrK`
  * Target local path: `checkpoints/transvg.pth`
  * Official repository: `https://github.com/Claire1217/AGPT`

### Blocker 2: RSNA Pneumonia Detection Challenge
* **Current State:** No RSNA labels (`stage_2_train_labels.csv`) or images (`stage_2_train_images/*.dcm`) exist on the local machine.
* **Impact:** External validation and domain adaptation fine-tuning cannot begin without this dataset.
* **Resolution Path:**
  * RSNA / Kaggle: `kaggle competitions download -c rsna-pneumonia-detection-challenge`
  * Target local directory: `data/rsna/`

### Blocker 3: Raw MIMIC-CXR Text Reports
* **Current State:** Raw unformatted full-study radiology reports (`.txt` files) are not present locally.
* **Workaround:**
  * Chest ImaGenome provides sentence-level annotations mapped to studies and sections (`gold_all_sentences_500pts_1000studies.txt`).
  * `cxr-study-list_v2.csv` provides patient demographic attributes (`gender`, `age_decile`).
  * These metadata sources fully suffice for report generation experiments without requiring full MIMIC report downloads.

---

## 5. Privacy, Ethics & Governance Compliance
* **Zero PHI Emission:** No patient names, MRNs, study texts, or clinical findings have been logged or stored in versioned repository files.
* **Strict Local Processing:** MIMIC-CXR images and text sentences remain strictly confined to the local filesystem and are not passed to external APIs or public cloud services.
