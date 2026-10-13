# InteractMed-VLM: Local Development & Environment Setup Guide

**Target Machine:** MacBook Air M3 (Apple Silicon, 16 GB Unified Memory)  
**Operating System:** macOS (Darwin arm64)  
**Python Version:** Python 3.10.19 (Conda environment: `dl`)  
**Workspace Directory:** `/Users/adityakachhot/interactmed`  
**Date:** October 10, 2026  

---

## 1. System Architecture & Environment Isolation

All development for **InteractMed-VLM** is isolated within a dedicated Python 3.10 virtual environment to avoid conflicts with system utilities or alternative Python distributions on macOS:

* **Active Python Interpreter:** `/Users/adityakachhot/miniforge3/envs/dl/bin/python`
* **Python Version:** `3.10.19` (matches upstream AGPT repository recommendation)
* **System Python Safety:** System Python (`/opt/homebrew/bin/python3` and `/usr/bin/python3`) is strictly preserved and unmodified.

---

## 2. Dependency Matrix & Compatibility Resolutions

During dependency inspection, several cross-library conflicts common to macOS Apple Silicon were diagnosed and reconciled:

| Package | Resolved Version | Rationale & Compatibility Constraints |
| :--- | :--- | :--- |
| **PyTorch** | `2.2.2` | Native Apple Silicon build with Metal Performance Shaders (`mps`) support. |
| **Torchvision** | `0.17.2` | Precompiled binary matched to PyTorch 2.2.2. |
| **NumPy** | `1.26.4` | **Pinned to `<2.0.0`**. PyTorch 2.2.2 C-API is built against NumPy 1.x; NumPy 2.x throws `_ARRAY_API not found` warnings. |
| **OpenCV** | `4.8.1.78` | Precompiled macOS arm64 wheel compatible with NumPy 1.26.4. |
| **Albumentations** | `1.4.4` | Avoids recent `albucore` 2.x namespace issues on Apple Silicon. |
| **Transformers** | `4.40.1` | Pinned to AGPT specification for HuggingFace `medicalai/ClinicalBERT` loading. |
| **Timm** | `1.0.30` | Backbone utilities for vision models. |
| **Pillow** | `12.0.0` | CXR image decompression and patch rendering. |
| **gdown** | `6.4.2` | Single-checkpoint CLI download from Google Drive. |

---

## 3. Apple Silicon Metal (MPS) Acceleration

Apple Metal Performance Shaders (`mps`) acceleration is detected and functional:

```python
import torch
print("MPS available:", torch.backends.mps.is_available())  # True
print("MPS built:", torch.backends.mps.is_built())          # True
```

### Hardware Policy:
1. **Local Mac Terminal:**
   - Single-image inference, baseline evaluation on local subsets (MS-CXR, Chest ImaGenome), and interactive Gradio frontend.
   - Initial smoke tests evaluate both `cpu` and `mps` to benchmark numerical consistency.
2. **Google Colab (GPU):**
   - TransVG full-parameter fine-tuning on large RSNA training splits.

---

## 4. Third-Party Repository Integration (`third_party/AGPT`)

The official AGPT repository is cloned locally at `third_party/AGPT`:

```zsh
cd /Users/adityakachhot/interactmed
mkdir -p third_party
git clone https://github.com/Claire1217/AGPT.git third_party/AGPT
```

### Key Modules:
* `models/transvg/trans_vg_ca.py`: Core TransVG vision-language transformer.
* `models/transvg/language_model/bert.py`: ClinicalBERT language encoder.
* `models/transvg/visual_model/detr.py`: ResNet-50 visual backbone and spatial transformer.
* `dataloader/mscxr_dataloader.py`: Normalization and bounding box transforms.

---

## 5. Verification Commands

To verify your environment and dependencies at any time:

```zsh
# Run environment audit
/Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/check_environment.py --json-output docs/env_check.json

# Run unit test suite
/Users/adityakachhot/miniforge3/envs/dl/bin/pytest tests
```
