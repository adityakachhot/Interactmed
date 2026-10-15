import textwrap
from pathlib import Path
from src.report_service import (
    _load_to_rgb_array,
    _array_to_base64_jpeg,
    _vision_model_available,
    _ollama_generate_vision,
)

vision_model = _vision_model_available()

img_paths = [
    ("pneumonia_ett_2", "artifacts/rsna/smoke_test/010ccb9f-6d46-4380-af11-84f87397a1b8_iou058.png"),
    ("pneumonia_1", "artifacts/rsna/smoke_test/000fe35a-2649-43d4-b027-e67796d412e0_iou049.png"),
]

# We prompt the vision model with specific questions for each of the 5 visual sections,
# then synthesize into the 7-section structured research report.

v_prompt_7sec = textwrap.dedent("""
    You are an expert radiologist conducting structured visual analysis of this frontal chest radiograph.
    Systematically inspect each anatomical region and report findings:

    1. Support apparatus: Inspect for endotracheal tube, enteric tube, vascular catheter/line, chest tube, pacemaker, or surgical clips. Note presence and position, or state "None identified".
    2. Cardiac silhouette: Evaluate cardiothoracic ratio and borders (Normal, Enlarged, or Indeterminate).
    3. Lungs and opacities: Inspect all zones bilaterally for consolidation, focal opacity, atelectasis, or edema. Detail any opacity and specify laterality (right, left, or bilateral). If clear, state "No focal consolidation or acute opacity identified".
    4. Pleural spaces: Inspect costophrenic angles and pleural margins (Sharp, Blunted/Effusion with laterality, or Pneumothorax).
    5. Mediastinum: Evaluate mediastinal width and contours (Normal contour, Widened, or Indeterminate).

    Begin directly with 1. Support apparatus:
""").strip()

for label, p in img_paths:
    data = Path(p).read_bytes()
    arr = _load_to_rgb_array(data, Path(p).name)
    b64_img = _array_to_base64_jpeg(arr, max_side=1024, quality=85)
    
    print(f"\n=================== Visual Extraction for {label} ===================")
    raw_obs = _ollama_generate_vision(v_prompt_7sec, [b64_img], model=vision_model)
    print("RAW OBS:")
    print(raw_obs)
    
    s_prompt = textwrap.dedent(f"""
        Visual observations from this chest radiograph:
        \"\"\"{raw_obs}\"\"\"

        Format these findings into a structured research report with EXACTLY these SEVEN headings:

        SUPPORT APPARATUS
        Describe any observed lines, tubes, or devices and their visible position, or state if none are identified on the available image.

        CARDIAC SILHOUETTE
        State observed cardiac silhouette size (normal, enlarged, or indeterminate).

        LUNGS AND OPACITIES
        Detail observed parenchymal findings across lung fields with laterality.

        PLEURAL SPACES
        Describe costophrenic angles and pleural margins.

        MEDIASTINUM
        Describe mediastinal and hilar contour.

        OVERALL IMPRESSION
        Provide a concise, grounded clinical summary of observed findings. Do not declare normal if abnormalities or devices are present.

        UNCERTAINTY AND LIMITATIONS
        State technical factors (single frontal projection, absence of lateral view).

        STRICT RULES:
        - Do not invent measurements or prior comparisons.
        - Every device or abnormality noted in observations must be reflected in the relevant section and impression.
        - Do not output markdown asterisks or bullet characters in section titles.

        Begin directly with SUPPORT APPARATUS:
    """).strip()
    
    import urllib.request, json
    payload = json.dumps({
        "model": vision_model,
        "prompt": s_prompt,
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0.1, "num_predict": 280},
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode())
        raw_report = body.get("response", "").strip()
    print("\nSTRUCTURED 7-SECTION REPORT:")
    print(raw_report)
