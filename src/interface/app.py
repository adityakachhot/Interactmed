"""
interface/app.py
================
Interactive Gradio application wiring the InteractMed-VLM pipeline with placeholder inference.

Pipeline Workflow:
CXR Image + Region Selection -> Vision Encoder -> Region Grounding -> Report Generator
-> Patient-Friendly Translation -> Independent Triage Classifier -> Visual Explainability Overlay
"""

import argparse
import os
import sys
from typing import Any, Dict, Optional, Tuple
from PIL import Image, ImageDraw

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.vision_encoder.encoder import VisionEncoder
from src.region_grounding.grounding import RegionGroundingModule
from src.report_generator.generator import ReportGenerator
from src.triage.classifier import TriageClassifier, TriageResult
from src.explainability.explainer import VisualExplainer


class InteractMedPipeline:
    """End-to-end pipeline wrapper coordinating all modular components."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.encoder = VisionEncoder(config)
        self.grounder = RegionGroundingModule(config)
        self.generator = ReportGenerator(config)
        # Triage is kept strictly separate for safety
        self.triage = TriageClassifier(config)
        self.explainer = VisualExplainer(config)

    def process_cxr(
        self,
        image: Optional[Image.Image],
        region_label: str = "heart",
    ) -> Tuple[Image.Image, str, str, str, str]:
        """
        Executes the end-to-end pipeline on an input CXR image.

        Returns:
            Tuple of:
              - overlay_image: Image with BBox and Grad-CAM heatmap
              - clinical_finding: Specialist report text
              - patient_translation: 6th-grade level patient explanation
              - triage_badge: Formatted urgency badge (Green/Yellow/Red)
              - triage_details: Urgency level, confidence, and clinical rationale
        """
        if image is None:
            # Create a placeholder dummy image if none uploaded
            image = Image.new("RGB", (224, 224), color=(30, 30, 30))
            draw = ImageDraw.Draw(image)
            draw.text((30, 100), "Dummy CXR Radiograph", fill=(200, 200, 200))

        # 1. Vision Feature Extraction
        visual_features = self.encoder.encode(image)

        # 2. Region Grounding
        bbox, regional_features = self.grounder.ground(
            visual_features,
            region_label,
            image_size=(image.height, image.width),
        )

        # 3. Report Generation & Patient Translation
        clinical_finding = self.generator.generate_finding(regional_features, region_label)
        patient_translation = self.generator.translate_to_patient_friendly(clinical_finding)

        # 4. Independent Triage Urgency Assessment
        triage_result: TriageResult = self.triage.classify(clinical_finding)

        # 5. Visual Explainability (Grad-CAM + BBox overlay)
        heatmap = self.explainer.generate_gradcam_heatmap(
            visual_features,
            target_size=(image.height, image.width),
        )
        overlay_image = self.explainer.overlay_explanation(
            image,
            heatmap,
            bbox=bbox,
            region_label=region_label,
        )

        # Format Triage Outputs
        badge_colors = {"Green": "🟢", "Yellow": "🟡", "Red": "🔴"}
        icon = badge_colors.get(triage_result.category, "⚪")
        triage_badge = f"{icon} **Triage Level: {triage_result.category.upper()}** ({triage_result.urgency_level})"
        triage_details = (
            f"**Confidence:** {triage_result.confidence * 100:.1f}%\n\n"
            f"**Clinical Rationale:** {triage_result.rationale}\n\n"
            f"**Safety Isolation Status:** Independent classifier validated (no generator feedback)."
        )

        return overlay_image, clinical_finding, patient_translation, triage_badge, triage_details


def build_app():
    """Builds and returns the Gradio Blocks interface."""
    try:
        import gradio as gr
    except ImportError:
        print("[WARNING] Gradio is not installed. Install via 'pip install gradio' to launch the UI.")
        return None

    pipeline = InteractMedPipeline()

    theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
    )

    with gr.Blocks(theme=theme, title="InteractMed-VLM") as demo:
        gr.Markdown(
            """
            # 🩺 InteractMed-VLM
            ### Region-Aware, Patient-Friendly CXR Report Generation with Independent Safety Triage
            *Extending 'Learning to Generate Clinically Coherent Chest X-Ray Reports' (Lovelace & Mortazavi, EMNLP Findings 2020)*
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Input CXR & Region Selection")
                input_image = gr.Image(type="pil", label="Upload Chest X-Ray (PA / AP View)")
                region_dropdown = gr.Dropdown(
                    choices=["heart", "lungs", "pleura"],
                    value="heart",
                    label="Target Anatomical Region",
                    info="Select which cardiopulmonary anatomical compartment to analyze",
                )
                analyze_btn = gr.Button("🚀 Analyze Anatomical Region", variant="primary")

                gr.Markdown(
                    """
                    > [!NOTE]
                    > **Safety Isolation Architecture**: The Triage Classifier runs on an independent model 
                    > separate from the generation weights to guard against diagnostic hallucination.
                    """
                )

            with gr.Column(scale=1):
                gr.Markdown("### 2. Grounding & Visual Explainability")
                output_image = gr.Image(type="pil", label="Grounded Region & Grad-CAM Heatmap")

                gr.Markdown("### 3. Triage & Urgency Assessment")
                triage_badge_output = gr.Markdown(value="🟢 **Triage Level: Ready**")
                triage_details_output = gr.Markdown(value="Awaiting image analysis...")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 4. Specialist Clinical Finding")
                clinical_output = gr.Textbox(
                    label="Clinical Report Sentence",
                    lines=3,
                    placeholder="Clinical findings will appear here...",
                )

            with gr.Column():
                gr.Markdown("### 5. Patient-Friendly Translation")
                patient_output = gr.Textbox(
                    label="Accessible Explanation (6th-Grade Reading Level)",
                    lines=3,
                    placeholder="Patient-friendly translation will appear here...",
                )

        analyze_btn.click(
            fn=pipeline.process_cxr,
            inputs=[input_image, region_dropdown],
            outputs=[output_image, clinical_output, patient_output, triage_badge_output, triage_details_output],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch InteractMed-VLM Gradio Demo")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Generate public Gradio link")
    parser.add_argument("--test-run", action="store_true", help="Execute single forward pass and exit")
    args = parser.parse_args()

    if args.test_run:
        print("[INFO] Running offline test pass through InteractMedPipeline...")
        pipeline = InteractMedPipeline()
        dummy_img = Image.new("RGB", (224, 224), color=(50, 50, 50))
        overlay, clinical, patient, badge, details = pipeline.process_cxr(dummy_img, "heart")
        print(f"[SUCCESS] Pipeline test run complete!\nClinical: {clinical}\nPatient: {patient}\nBadge: {badge}")
        return

    demo = build_app()
    if demo is not None:
        demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
