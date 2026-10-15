import textwrap
from pathlib import Path
from src.report_service import (
    _load_to_rgb_array,
    _array_to_base64_jpeg,
    _vision_model_available,
    _ollama_generate_vision,
)

vision_model = _vision_model_available()
print("Vision model:", vision_model)

img_paths = [
    ("pneumonia_1", "artifacts/rsna/smoke_test/000fe35a-2649-43d4-b027-e67796d412e0_iou049.png"),
    ("pneumonia_ett_2", "artifacts/rsna/smoke_test/010ccb9f-6d46-4380-af11-84f87397a1b8_iou058.png"),
]

prompt_7section = textwrap.dedent("""
    You are an expert AI radiologist performing structured research interpretation of a chest radiograph.
    Inspect the entire thoracic image systematically. Ground every statement in visible visual evidence.

    For each section, classify findings as one of:
    - Observed (<describe details, density, and laterality: right, left, or bilateral>)
    - Not identified on available image
    - Indeterminate (<explain visual ambiguity or technical obscuration>)
    - Not assessable (<state limiting factor>)

    YOUR REPORT MUST CONTAIN EXACTLY THESE SEVEN SECTION HEADINGS:

    1. SUPPORT APPARATUS
    Assess for presence, type, and position of lines, tubes, clips, or pacemakers. If none visible, state "Not identified on available image".

    2. CARDIAC SILHOUETTE
    Assess cardiac size and borders relative to thoracic width. If enlarged (CTR > 0.5), state "Observed: Cardiomegaly / enlargement". If borders obscured, state "Indeterminate".

    3. LUNGS AND OPACITIES
    Inspect all lung zones (apices, mid-zones, bases, retrocardiac areas). Describe any focal opacity, consolidation, atelectasis, or edema, specifying laterality. If lungs appear clear, state "Not identified on available image".

    4. PLEURAL SPACES
    Assess bilateral costophrenic angles and pleural margins for effusion, blunting, or pneumothorax, specifying laterality.

    5. MEDIASTINUM
    Assess mediastinal and hilar contours (normal contour, widened, or indeterminate).

    6. OVERALL IMPRESSION
    Concisely summarize key findings and uncertainty. Do not declare normal if abnormalities or devices were observed above.

    7. UNCERTAINTY AND LIMITATIONS
    State technical limitations (single frontal view, absence of lateral projection, patient positioning).

    STRICT RULES:
    - Do not invent measurements, patient history, or prior studies.
    - Do not collapse indeterminate or unassessable areas into normal.
    - Begin directly with "1. SUPPORT APPARATUS".
""").strip()

for label, p in img_paths:
    data = Path(p).read_bytes()
    arr = _load_to_rgb_array(data, Path(p).name)
    b64_img = _array_to_base64_jpeg(arr, max_side=1024, quality=85)
    
    print(f"\n=================== Generating for {label} ({Path(p).name}) ===================")
    resp = _ollama_generate_vision(prompt_7section, [b64_img], model=vision_model)
    print(resp)
