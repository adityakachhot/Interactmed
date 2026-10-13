# InteractMed-VLM: Pretrained Checkpoint Setup Guide

**Target Model:** TransVG-AGPT (Medical Phrase Grounding)  
**Host Environment:** MacBook Air M3 (macOS, 16 GB Unified Memory)  
**Canonical Local Path:** [`checkpoints/transvg.pth`](file:///Users/adityakachhot/interactmed/checkpoints/transvg.pth)  
**AGPT Repository Symlink:** [`third_party/AGPT/model_weight/transvg.pth`](file:///Users/adityakachhot/interactmed/third_party/AGPT/model_weight/transvg.pth)

---

## 1. Upstream Source & Access Reconcilement

The official model weights are released by the authors of **Anatomical Grounding Pre-training (AGPT)**:
* **Official Repository:** [`https://github.com/Claire1217/AGPT`](https://github.com/Claire1217/AGPT)
* **Official Google Drive Folder:** [`https://drive.google.com/drive/folders/1lFaqAlG6AJExCjkW3Pbb24CX-LWSstrK?usp=sharing`](https://drive.google.com/drive/folders/1lFaqAlG6AJExCjkW3Pbb24CX-LWSstrK?usp=sharing)
* **Target Checkpoint File:** `transvg.pth` (~2.10 GB)
* **Google Drive Unique File ID:** `1INXf6rhAizBKf4IfC_Zi-pmyIbcdHKW9`

> [!NOTE]
> The upstream Google Drive folder also contains `mdetr.pth` (~2.22 GB) and pretraining weights. Per project requirements, **only `transvg.pth`** is downloaded for the phrase grounding pipeline.

---

## 2. Checkpoint Directory Reconcilement

There is a slight difference between the repository README and `demo.py`:
* **README description:** References `agpt/model_weight/transvg.pth`
* **`demo.py` implementation:** Hardcodes `model_weight/transvg.pth` relative to working directory.

To ensure consistency across our modular pipeline, repository demo, and future fine-tuning scripts, we establish:
1. **Canonical Project Storage:** Store the persistent, protected weights at `checkpoints/transvg.pth`.
2. **Backward-Compatibility Symlink:** Create a symlink from `checkpoints/transvg.pth` into `third_party/AGPT/model_weight/transvg.pth`.

---

## 3. Automated Download via CLI (`gdown`)

In the project's `dl` Python 3.10 environment:

```zsh
# 1. Create destination directories
cd /Users/adityakachhot/interactmed
mkdir -p checkpoints third_party/AGPT/model_weight

# 2. Download transvg.pth via direct file ID
/Users/adityakachhot/miniforge3/envs/dl/bin/python -m gdown "https://drive.google.com/uc?id=1INXf6rhAizBKf4IfC_Zi-pmyIbcdHKW9" -O checkpoints/transvg.pth

# 3. Create symlink for AGPT native code
ln -sf /Users/adityakachhot/interactmed/checkpoints/transvg.pth /Users/adityakachhot/interactmed/third_party/AGPT/model_weight/transvg.pth
```

---

## 4. Manual Web Browser Download (If Google Drive Requires Manual Sign-In)

If Google Drive rate-limits CLI requests or requires an interactive browser session:

1. Open your web browser and navigate to:
   [`https://drive.google.com/drive/folders/1lFaqAlG6AJExCjkW3Pbb24CX-LWSstrK?usp=sharing`](https://drive.google.com/drive/folders/1lFaqAlG6AJExCjkW3Pbb24CX-LWSstrK?usp=sharing)
2. In the folder listing, find the file named **`transvg.pth`** (file size approximately 2.10 GB).
3. Right-click on `transvg.pth` and select **Download** (select "Download anyway" when Google warns that it cannot scan large files for viruses).
4. Once the download completes in your `~/Downloads` folder, move and link it using the Mac Terminal:

```zsh
# In Mac Terminal:
cd /Users/adityakachhot/interactmed
mkdir -p checkpoints third_party/AGPT/model_weight

# Move downloaded weight into canonical project storage
mv ~/Downloads/transvg.pth /Users/adityakachhot/interactmed/checkpoints/transvg.pth

# Create symlink
ln -sf /Users/adityakachhot/interactmed/checkpoints/transvg.pth /Users/adityakachhot/interactmed/third_party/AGPT/model_weight/transvg.pth
```

---

## 5. Verification & Integrity Check

Verify file presence and load state dictionary into the model architecture:

```zsh
# Check file size (~2.1 GB)
ls -lh /Users/adityakachhot/interactmed/checkpoints/transvg.pth

# Run loading smoke test
/Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/test_transvg_loading.py \
    --checkpoint checkpoints/transvg.pth \
    --device auto \
    --save-dir artifacts/model_smoke_test
```
