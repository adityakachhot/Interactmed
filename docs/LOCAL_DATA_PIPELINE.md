# InteractMed-VLM: Validated Local Data Layer & Audit Pipeline

**Phase:** Phase 2 Implementation  
**Auditor:** Lead ML Engineer  
**Date:** October 10, 2026  
**Environment:** MacBook Air M3 (16 GB Unified Memory, macOS)  
**Configuration:** [`configs/data_paths.yaml`](file:///Users/adityakachhot/interactmed/configs/data_paths.yaml)  
**Master Catalog:** [`data/processed/unified_grounding_master.csv`](file:///Users/adityakachhot/interactmed/data/processed/unified_grounding_master.csv) (27,437 records)  
**Verified Local Subset:** [`data/processed/local_grounding_verified.csv`](file:///Users/adityakachhot/interactmed/data/processed/local_grounding_verified.csv) (11,896 records)  

---

## 1. Executive Summary & Design Principles

The local data layer for **InteractMed-VLM** establishes a unified, reproducible, and clinically sound foundation for vision-language phrase grounding and region-specific report generation without redownloading data or modifying original source files.

### Core Principles Enforced:
1. **Zero Redownload & Non-Destructive Ingestion:** Raw metadata files (`MS_CXR_Local_Alignment_v1.1.0.csv`, `gold_bbox_coordinate_annotations_1000images.csv`, `cxr-record-list_view.csv`, `cxr-study-list_v2.csv`) are strictly treated as read-only.
2. **Explicit Verification vs. Silent Discarding:** Every annotation record is cataloged in the master table (`unified_grounding_master.csv`). Missing images are flagged with `image_exists: False`, guaranteeing that downstream pipelines never silently drop unverified data.
3. **Dual Coordinate Preservation:** Both raw native coordinates (`orig_x1`, `orig_y1`, `orig_w`, `orig_h`) and normalized bounding boxes (`norm_x1`, `norm_y1`, `norm_x2`, `norm_y2` in $[0.0, 1.0]$) are simultaneously stored and verified.
4. **Separation of Pathology vs. Anatomy:** Clinical finding phrases (e.g., *"patchy consolidation in the right lower"*) are rigorously distinguished from anatomical structures (e.g., *"cardiac silhouette"*).
5. **Split Contamination & Leakage Prevention:** Discovered cross-dataset intersections and evaluated potential data leakage against the official TransVG-AGPT benchmark.
6. **Strict PHI Privacy Governance:** Patient identifiers and free-text clinical report bodies are isolated from Antigravity logs and public APIs.

---

## 2. Dataset Roles & Schema Standardization

| Dataset | Role | Total Records | Local Records | Unique DICOMs | Unique Patients | Label Type | Coordinate Space |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MS-CXR (v1.1.0)** | Clinical phrase grounding benchmark | **1,448** | **1,056** (72.9%) | 1,047 | 851 | `pathology` | `[x, y, w, h]` (Native DICOM) |
| **Chest ImaGenome Gold** | Anatomical prior & sanity-check grounding | **25,989** | **10,840** (41.7%) | 1,000 | 500 | `anatomy` | `[x1, y1, x2, y2]` (224 & Native) |
| **Combined Master** | Unified Grounding Layer | **27,437** | **11,896** (43.4%) | 2,038 | 1,335 | Mixed | Standardized $[0, 1]$ & Native |

### Standardized Master Table Schema:
* **Identifiers:** `row_id`, `dataset_source`, `label_type`, `dicom_id`, `subject_id`, `study_id`
* **Text / Queries:** `phrase`, `category_name`, `organ_region` (mapped to *heart*, *lungs*, *pleura*)
* **Split Allocation:** `split` (`train`, `val`, `test`, `gold_sanity_check`)
* **Native Box Coordinates:** `orig_coord_format`, `orig_x1`, `orig_y1`, `orig_x2`, `orig_y2`, `orig_w`, `orig_h`, `image_width`, `image_height`
* **Normalized Box Coordinates:** `norm_x1`, `norm_y1`, `norm_x2`, `norm_y2` (strictly clamped in $[0.0, 1.0]$)
* **Integrity Flags:** `is_valid_bbox` (bool), `validation_error` (str), `image_exists` (bool), `image_path` (str)
* **Patient Demographics:** `gender` (`M`/`F`), `age_decile` (e.g., `50-60`), `view_position` (`PA`/`AP`/`LATERAL`)

---

## 3. Coordinate Transformation & Geometric Validation Audit

The automated coordinate audit script ([`scripts/audit_coordinates.py`](file:///Users/adityakachhot/interactmed/scripts/audit_coordinates.py)) validated all 27,437 bounding boxes against boundary invariants:

```
+-----------------------------------------------------------------------------------+
|                        COORDINATE AUDIT SUMMARY METRICS                           |
+-----------------------------------------------------------------------------------+
| Total Records Audited:                     27,437                                 |
| Strictly Bounded in [0, 1] Unit Square:    27,437 (100.0%)                        |
| Valid Positive Bounding Box Area:          27,433 (99.985%)                       |
| Degenerate / Zero-Area Boxes:              4 (Annotator anomalies flagged)        |
| Extreme Aspect Ratios (>10:1 or <1:10):    19                                     |
| Reconstruction Discrepancy (Native - Norm):0.0000 px                              |
+-----------------------------------------------------------------------------------+
```

### Anomaly Resolutions:
1. **MS-CXR Conversion:** All 1,448 records use unnormalized native $x, y, w, h$. $x_2 = x + w$ and $y_2 = y + h$ yielded **100% valid, in-bounds bounding boxes** with zero anomalies.
2. **Chest ImaGenome Anomalies Handled:**
   - **Flipped Coordinates (21 records):** Annotators drew bounding boxes from right-to-left or bottom-to-top ($x_2 < x_1$ or $y_2 < y_1$). Automatically re-ordered such that $x_1 = \min(x_1, x_2)$ and $x_2 = \max(x_1, x_2)$.
   - **Negative Padding Offsets (212 records):** When resizing native images to $224 \times 224$ with letterbox padding, reverse projections produced negative margins (e.g. $x_1 = -54.0$). Clamped non-negative boundaries to $0.0$.
   - **Degenerate Boxes (4 records):** Bounding boxes with identical start and end coordinates ($w \le 0$ or $h \le 0$) flagged with `is_valid_bbox = False` and `validation_error = 'zero_or_negative_area'`.

---

## 4. Cross-Dataset Overlap & Data Leakage Analysis

The overlap audit ([`scripts/audit_overlap.py`](file:///Users/adityakachhot/interactmed/scripts/audit_overlap.py)) evaluated patient, study, and image intersections between MS-CXR and Chest ImaGenome Gold:

```
MS-CXR Unique Patients:                     851
Chest ImaGenome Gold Unique Patients:       500
-----------------------------------------------
Shared Patients (Overlap):                   16 patients
Shared Studies (Overlap):                     9 studies
Shared Images (Overlap):                      9 images (7 locally present on disk)
Total Unique Patients (Union):            1,335 patients
Total Unique Images (Union):              2,038 images
```

### Cross-Split Contamination Audit:
We cross-referenced the 16 shared patients against the official splits of MS-CXR:

| MS-CXR Split | Split Images | Split Patients | Shared Patients with CIG Gold | Contamination Status |
| :--- | :--- | :--- | :--- | :--- |
| **`test`** | **155** | **128** | **0** | **CLEAN (Zero Leakage, 100% Disjoint)** |
| **`train`** | 737 | 595 | 14 | Contaminated (Overlap with Pretraining Pool) |
| **`val`** | 155 | 128 | 2 | Minor Contamination |

---

## 5. Protocol Determination on Evaluation Splits (Task 11)

Per project guidelines, we explicitly investigated whether local gold-set images can be treated as an independent benchmark for TransVG-AGPT:

### Clinical ML Finding:
* **Pretraining Distribution:** TransVG-AGPT was pre-trained on Chest ImaGenome anatomical scene graphs before fine-tuning on MS-CXR. The published AGPT repository does not release the exact patient IDs excluded during anatomical pre-training.
* **Risk of Data Snooping:** If Chest ImaGenome Gold studies were included in the pretraining set, evaluating TransVG-AGPT on them would measure in-domain recall rather than out-of-distribution phrase grounding generalization.
* **Formal Classification:**
  1. **MS-CXR `test` Split:** Classified as the **Primary Independent Grounding Benchmark**. It has **zero patient leakage** with Chest ImaGenome Gold, and matches the official evaluation protocol of TransVG-AGPT ($mIoU = 61.2$).
  2. **Chest ImaGenome Gold:** Formally classified as a **Visual Sanity-Check & Anatomical Consistency Set**, NOT an independent evaluation benchmark. It serves to verify anatomical alignment and qualitative model interpretability.

---

## 6. Execution Commands

### 1. Execute Local Metadata Preparation Pipeline
```zsh
/Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/prepare_local_metadata.py \
    --config configs/data_paths.yaml \
    --output-dir data/processed
```

### 2. Run Geometric Coordinate & Bounding Box Audit
```zsh
/Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/audit_coordinates.py \
    --table data/processed/unified_grounding_master.csv \
    --output docs/coordinate_audit_report.json
```

### 3. Run Dataset Overlap & Data Leakage Audit
```zsh
/Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/audit_overlap.py \
    --table data/processed/unified_grounding_master.csv \
    --output docs/dataset_overlap_report.json
```

### 4. Run Coordinate Conversion & Pipeline Unit Tests
```zsh
/Users/adityakachhot/miniforge3/envs/dl/bin/pytest tests/test_coordinate_conversion.py
/Users/adityakachhot/miniforge3/envs/dl/bin/pytest tests
```
