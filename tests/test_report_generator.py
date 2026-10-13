"""
test_report_generator.py
========================
Unit tests verifying the ReportGenerator contract:
Input:  regional_features tensor [B, embed_dim] + region_label
Output: clinical finding string + patient-friendly translation
"""

import unittest
import torch

from src.report_generator.generator import ReportGenerator


class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = ReportGenerator()

    def test_generate_finding(self):
        dummy_feat = torch.randn(1, 512)
        finding = self.generator.generate_finding(dummy_feat, region_label="heart")

        self.assertIsInstance(finding, str)
        self.assertTrue(len(finding) > 10)
        self.assertIn("cardiomediastinal", finding.lower())

    def test_patient_friendly_translation(self):
        clinical_text = "The cardiomediastinal silhouette is normal in size and contour."
        patient_text = self.generator.translate_to_patient_friendly(clinical_text)

        self.assertIsInstance(patient_text, str)
        self.assertTrue(len(patient_text) > 10)
        # Should be accessible, non-technical English
        self.assertIn("heart", patient_text.lower())

    def test_forward_dict_output(self):
        dummy_feat = torch.randn(1, 512)
        result = self.generator(dummy_feat, region_label="pleura")

        self.assertIn("clinical_finding", result)
        self.assertIn("patient_translation", result)
        self.assertIsInstance(result["clinical_finding"], str)
        self.assertIsInstance(result["patient_translation"], str)


class TestReportValidationAndSafety(unittest.TestCase):
    """Tests for report validation rules, safety constraints, and no-fabrication policy."""

    def test_validate_report_detects_empty(self):
        from src.report_service import validate_report
        res = validate_report("", "pneumonia")
        self.assertFalse(res["valid"])
        self.assertTrue(res["flagged"])

    def test_validate_report_detects_unsupported_measurements(self):
        from src.report_service import validate_report
        text = "There is an opacity measuring exactly 3.5 cm in the right lower lobe."
        res = validate_report(text, "pneumonia")
        self.assertTrue(any("3.5 cm" in w or "measurements" in w for w in res["warnings"]))
        self.assertTrue(res["flagged"])

    def test_validate_report_detects_anatomical_contradiction(self):
        from src.report_service import validate_report
        text = "Marked cardiomegaly is identified within the upper lung zone and apical zone."
        res = validate_report(text, "cardiomegaly")
        self.assertTrue(any("contradiction" in w for w in res["warnings"]))
        self.assertTrue(res["flagged"])

    def test_validate_report_detects_echo_phrase(self):
        from src.report_service import validate_report
        text = "Pneumonia. Pneumonia."
        res = validate_report(text, "pneumonia")
        self.assertTrue(res["flagged"])

    def test_region_report_no_vision_model_does_not_fabricate(self):
        """When vision model is unavailable, region_report must fail gracefully and not fabricate."""
        from src.report_service import region_report
        from unittest.mock import patch
        import io
        from PIL import Image

        img = Image.new("RGB", (100, 100), color=(100, 100, 100))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        with patch("src.report_service._vision_model_available", return_value=None), \
             patch("src.report_service._ollama_available", return_value=True):
            res = region_report(
                image_bytes=image_bytes,
                image_filename="test.png",
                phrase="pneumonia",
                box_pixel_xyxy=[10.0, 10.0, 50.0, 50.0],
                orig_h=100,
                orig_w=100,
            )
            self.assertFalse(res["success"])
            self.assertFalse(res["image_input_used"])
            self.assertIsNone(res["report_text"])
            self.assertIn("No vision-capable model is installed", res["error"])


if __name__ == "__main__":
    unittest.main()

