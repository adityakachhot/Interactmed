import os
import glob
from pathlib import Path
from src.batch_service import (
    _ollama_generate_vision_with_metrics,
    _load_to_rgb_array,
    _array_to_base64_jpeg,
    clean_and_structure_research_report,
    clean_and_structure_patient_report,
)
from src.report_service import _vision_model_available

vision_model = _vision_model_available()
print("Vision model:", vision_model)

img_paths = [
    "artifacts/rsna/smoke_test/000fe35a-2649-43d4-b027-e67796d412e0_iou049.png",
    "artifacts/rsna/smoke_test/018951e6-9fb3-4e92-8ce7-e3a018daf93f_iou048.png",
    "artifacts/rsna/smoke_test/010ccb9f-6d46-4380-af11-84f87397a1b8_iou058.png",
]

for p in img_paths:
    data = Path(p).read_bytes()
    arr = _load_to_rgb_array(data, Path(p).name)
    b64_img = _array_to_base64_jpeg(arr, max_side=1024, quality=85)
    
    # Test v_prompt
    v_prompt = (
        "You are an expert radiologist conducting structured visual feature extraction on this chest radiograph.\n"
        "For each category below, state: Observed (<details and laterality>) | Not identified | Indeterminate | Not assessable.\n\n"
        "1. Support apparatus: Inspect for endotracheal tube, enteric tube, vascular catheter/line, chest tube, pacemaker, or surgical clips.\n"
        "2. Cardiac silhouette: Normal, Enlarged (CTR > 0.5), or Indeterminate.\n"
        "3. Lungs & opacities: Consolidation, focal opacity, atelectasis, congestion, or edema. Specify laterality (right, left, bilateral, or unspecified laterality).\n"
        "4. Pleural spaces: Costophrenic angles sharp, blunted/effusion (specify laterality: right, left, bilateral, or unspecified laterality), or obscured.\n"
        "5. Mediastinum: Normal contour, widened, or indeterminate.\n\n"
        "RULES:\n"
        "- State laterality (right/left/bilateral) only when visually supported; if uncertain, state 'unspecified laterality'.\n"
        "- If equivocal or confidence is low, classify as Indeterminate; DO NOT default to normal.\n"
        "- Concise bullet list, no conversational filler.\n"
        "Begin directly with 1. Support apparatus:"
    )
    raw_obs, metrics = _ollama_generate_vision_with_metrics(
        prompt=v_prompt,
        images_b64=[b64_img],
        model=vision_model,
        num_predict=180,
        temperature=0.1,
        keep_alive="30m",
    )
    print(f"\n=================== Image: {Path(p).name} ===================")
    print("RAW OBS:")
    print(raw_obs)
