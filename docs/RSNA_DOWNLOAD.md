# RSNA Pneumonia Detection Challenge — Download & Setup Guide

**Phase:** Phase 3A — External Dataset Acquisition  
**Lead ML Engineer:** InteractMed-VLM  
**Dataset:** RSNA Pneumonia Detection Challenge 2018  
**Destination:** `data/raw/rsna/`  
**Configuration:** `configs/data_paths.yaml` → `rsna_pneumonia`  
**Validation Script:** `scripts/validate_rsna.py`  

---

## 1. Official Source & Licensing

| Item | Details |
|---|---|
| **Challenge Name** | RSNA Pneumonia Detection Challenge 2018 |
| **Organizer** | Radiological Society of North America (RSNA) |
| **Official Page** | https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018 |
| **Kaggle Page** | https://www.kaggle.com/c/rsna-pneumonia-detection-challenge |
| **Data Page (Kaggle)** | https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data |
| **Access Requirement** | Must accept competition **Rules & Data Use Agreement** on Kaggle |
| **License** | RSNA Data Use Agreement (non-commercial research use; no redistribution) |
| **Citation Required** | Yes — see Section 7 |

> [!IMPORTANT]
> You **must** be logged into Kaggle and have accepted the competition rules before downloading.
> The dataset is for **non-commercial research purposes only** and must not be redistributed.

---

## 2. Dataset Files & Archive Layout

After full extraction, `data/raw/rsna/` will have this layout:

```
data/raw/rsna/
├── stage_2_train_labels.csv          # 30,227 rows — training annotations
├── stage_2_detailed_class_info.csv   # 30,227 rows — per-image class labels
├── stage_2_sample_submission.csv     # Submission format template
├── stage_2_train_images/             # ~26,684 DICOM files (.dcm) — ~8.7 GB
│   ├── 00436515-870c-4b36-a041-de91049b9ab4.dcm
│   └── ... (~26,684 files total)
└── stage_2_test_images/              # ~3,000 DICOM files (.dcm) — ~1.0 GB
    ├── 0000a175-0e68-4ca4-b1af-167204a7e0bc.dcm
    └── ... (~3,000 files total)
```

### Annotation File: `stage_2_train_labels.csv`

| Column | Type | Description |
|---|---|---|
| `patientId` | string | UUID matching the DICOM filename (without `.dcm`) |
| `x` | float | Left edge of bounding box (pixels, native DICOM res ~1024×1024) |
| `y` | float | Top edge of bounding box (pixels) |
| `width` | float | Width of bounding box (pixels) |
| `height` | float | Height of bounding box (pixels) |
| `Target` | int (0/1) | `1` = pneumonia present; `0` = no pneumonia (x/y/w/h are NaN) |

**Key Facts:**
- 6,012 positive images (Target=1), some with **multiple boxes** (multi-focal pneumonia)
- 20,672 negative images (Target=0) — x/y/w/h will be NaN
- Total annotation rows: 30,227 (more rows than images due to multi-box positives)
- Image resolution: ~1024×1024 pixels (varies slightly per DICOM)
- Bounding box coordinate format: `[x, y, width, height]` — same as MS-CXR v1.1.0

### Annotation File: `stage_2_detailed_class_info.csv`

| Column | Type | Description |
|---|---|---|
| `patientId` | string | UUID (same as `stage_2_train_labels.csv`) |
| `class` | string | One of: `"Normal"`, `"Lung Opacity"`, `"No Lung Opacity / Not Normal"` |

**Usage note:** Only rows with `class == "Lung Opacity"` and `Target == 1` should be
used as positive grounding examples. `"No Lung Opacity / Not Normal"` images are
abnormal but have no spatial annotation.

### Total Dataset Size

| Component | Files | Approx. Size |
|---|---|---|
| `stage_2_train_images/` | ~26,684 DICOMs | ~8.7 GB |
| `stage_2_test_images/` | ~3,000 DICOMs | ~1.0 GB |
| CSV annotation files | 3 CSVs | ~3 MB |
| **Total** | **~29,684 files** | **~9.7 GB** |

> [!WARNING]
> Ensure you have at least **12 GB** of free disk space before extracting.
> Kaggle downloads as a single zip; temporary space is needed during extraction.

---

## 3. Method A: Browser-Based Download (Recommended)

This method requires only a web browser and your Kaggle account.
No API tokens or credentials are stored anywhere in the project.

### Step-by-Step Instructions

**Step 1 — Accept the Data Use Agreement (one-time)**
1. Go to: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/rules
2. Scroll to the bottom and click **"I Understand and Accept"** if you haven't already.

**Step 2 — Navigate to the Data page**
1. Go to: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data

**Step 3 — Download the dataset**
1. Click the **"Download All"** button (top right of the file listing).
2. This downloads a single file: `rsna-pneumonia-detection-challenge.zip` (~9.7 GB).
3. Your browser will save it to your `~/Downloads/` folder by default.

**Step 4 — Move to the project destination and extract**

Open **Mac Terminal** (⌘+Space → "Terminal"):

```zsh
# Navigate to the RSNA destination
cd /Users/adityakachhot/interactmed/data/raw/rsna

# Move the downloaded archive here
mv ~/Downloads/rsna-pneumonia-detection-challenge.zip .

# Extract the archive (expect 10–20 minutes for ~9.7 GB)
unzip -q rsna-pneumonia-detection-challenge.zip

# Optional: Remove the zip after extraction to reclaim ~9.7 GB
# rm rsna-pneumonia-detection-challenge.zip
```

**Step 5 — Quick sanity check before running the validator**

```zsh
# Confirm directory structure
ls -la /Users/adityakachhot/interactmed/data/raw/rsna/

# Count training DICOM files (expect ~26,684)
ls /Users/adityakachhot/interactmed/data/raw/rsna/stage_2_train_images/*.dcm \
    2>/dev/null | wc -l

# Count test DICOM files (expect ~3,000)
ls /Users/adityakachhot/interactmed/data/raw/rsna/stage_2_test_images/*.dcm \
    2>/dev/null | wc -l

# Inspect annotation header (expect: patientId,x,y,width,height,Target)
head -5 /Users/adityakachhot/interactmed/data/raw/rsna/stage_2_train_labels.csv
```

---

## 4. Method B: Kaggle CLI (Optional — Requires API Token)

The Kaggle CLI provides a programmatic, resumable download.
This requires setting up a Kaggle API token first.

> [!NOTE]
> Kaggle credentials are **never stored in this repository**.
> The token file must be placed at `~/.kaggle/kaggle.json` on your local machine only.

### Step 1: Create a Kaggle API Token
1. Log in at https://www.kaggle.com
2. Go to Account Settings: https://www.kaggle.com/settings/account
3. Scroll to the **API** section → click **"Create New Token"**
4. This downloads `kaggle.json` to your machine.
5. Move it and restrict permissions:

```zsh
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### Step 2: Accept Competition Rules (via Browser — Required)

You **must** accept the rules in the browser before the CLI download will work:
1. Visit: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/rules
2. Click **"I Understand and Accept"**.

### Step 3: Install Kaggle CLI & Download

```zsh
# Install Kaggle CLI in the project's dl environment
/Users/adityakachhot/miniforge3/envs/dl/bin/pip install kaggle

# Verify installation
/Users/adityakachhot/miniforge3/envs/dl/bin/kaggle --version

# Download competition data to the RSNA destination
/Users/adityakachhot/miniforge3/envs/dl/bin/kaggle competitions download \
    -c rsna-pneumonia-detection-challenge \
    -p /Users/adityakachhot/interactmed/data/raw/rsna

# Extract
cd /Users/adityakachhot/interactmed/data/raw/rsna
unzip -q rsna-pneumonia-detection-challenge.zip
```

> [!TIP]
> The CLI download is resumable: if interrupted, re-running the same command
> will continue from where it left off.

---

## 5. Extraction Validation

After extraction, run the automated validation script to verify file counts,
annotation schema, DICOM integrity, and bounding box geometry:

```zsh
cd /Users/adityakachhot/interactmed

/Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/validate_rsna.py \
    --rsna-root data/raw/rsna \
    --output docs/rsna_validation_report.json \
    --verbose
```

### Expected Validation Output

```
RSNA Dataset Validation Report
================================
rsna_root:                 data/raw/rsna
labels_csv_exists:         True
class_info_csv_exists:     True
sample_submission_exists:  True
train_images_dir_exists:   True
test_images_dir_exists:    True

label_rows_total:          30,227
unique_train_patients:     26,684
positive_cases:            6,012
negative_cases:            20,672
multi_box_patients:        612
max_boxes_per_patient:     4

dicom_sample_valid:        True (10/10 sampled DICOMs readable)
bbox_range_valid:          True (all boxes within 0–1024 px)
columns_valid:             True

OVERALL STATUS: PASS ✅
```

---

## 6. Integration with InteractMed-VLM Pipeline

After validation passes, update `configs/data_paths.yaml`:

```yaml
rsna_pneumonia:
  root_dir: "data/raw/rsna"
  labels_csv: "data/raw/rsna/stage_2_train_labels.csv"
  class_info_csv: "data/raw/rsna/stage_2_detailed_class_info.csv"
  images_dir: "data/raw/rsna/stage_2_train_images"
  test_images_dir: "data/raw/rsna/stage_2_test_images"
  status: "present_and_validated"
  coordinate_format: "[x, y, width, height]"   # native pixel coordinates
  image_resolution: "1024x1024"
  positive_phrase: "airspace opacity"           # grounding query phrase
```

The Phase 3B pipeline will then:
1. Load `stage_2_train_labels.csv`, filtering to `Target == 1` (positive only)
2. Parse `patientId` → locate `stage_2_train_images/{patientId}.dcm`
3. Convert DICOM pixel array → RGB PIL image (grayscale → 3-channel via repeat)
4. Convert RSNA `[x, y, w, h]` → normalized `[x1, y1, x2, y2]` (divide by image width/height)
5. Feed each `(image, phrase="airspace opacity", bbox)` tuple through TransVG-AGPT
6. Compute IoU and Precision@0.5 for zero-shot evaluation

---

## 7. Attribution & Use Requirements

When using this dataset in any publication, presentation, or report, include:

```
Shih, G., Wu, C.C., Halabi, S.S., Kohli, M.D., Prevedello, L.M., Cook, T.S.,
Sharma, A., Amorosa, J.K., Arteaga, V., Galperin-Aizenberg, M., et al. (2019).
Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert
Annotations of Possible Pneumonia. Radiology: Artificial Intelligence, 1(1).
https://doi.org/10.1148/ryai.2019180041
```

**Additional Requirements:**
- **No redistribution** of raw DICOM images or annotations outside your lab
- **Non-commercial research use only**
- Reference the Kaggle competition page and accept the RSNA Data Use Agreement
- Do **not** publish patient identifiers or attempt to re-identify any patient

---

## 8. Status Tracking

| Step | Action | Status |
|---|---|---|
| 3A.1 | Create `data/raw/rsna/` destination directory | ✅ Complete |
| 3A.2 | Create `docs/RSNA_DOWNLOAD.md` | ✅ Complete (this file) |
| 3A.3 | Create `scripts/validate_rsna.py` | ✅ Complete |
| 3A.4 | Register RSNA paths in `configs/data_paths.yaml` | ✅ Pre-registered (status=missing) |
| 3A.5 | **Accept Kaggle competition rules (browser)** | ⏳ **Requires user action** |
| 3A.6 | **Download RSNA archive (~9.7 GB)** | ⏳ **Requires user action** |
| 3A.7 | **Extract archive to `data/raw/rsna/`** | ⏳ **Requires user action** |
| 3A.8 | Run `scripts/validate_rsna.py` | ⏳ After download |
| 3A.9 | Update `configs/data_paths.yaml` status → `present_and_validated` | ⏳ After validation |
| 3B | Zero-shot AGPT evaluation on RSNA test split | 🔲 Next phase |
