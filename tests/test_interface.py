"""
test_interface.py
=================
Unit tests verifying the end-to-end InteractMedPipeline wrapper and dummy inference flow.
"""

import unittest
from PIL import Image

from src.interface.app import InteractMedPipeline


class TestInterfacePipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = InteractMedPipeline()

    def test_pipeline_forward_heart(self):
        dummy_img = Image.new("RGB", (224, 224), color=(60, 60, 60))
        overlay, clinical, patient, badge, details = self.pipeline.process_cxr(
            image=dummy_img,
            region_label="heart",
        )

        self.assertIsInstance(overlay, Image.Image)
        self.assertEqual(overlay.size, (224, 224))
        self.assertIsInstance(clinical, str)
        self.assertIsInstance(patient, str)
        self.assertIsInstance(badge, str)
        self.assertIsInstance(details, str)

        self.assertIn("cardiomediastinal", clinical.lower())
        self.assertIn("heart", patient.lower())
        self.assertIn("Triage Level", badge)

    def test_pipeline_none_image_fallback(self):
        # Even with None image input, pipeline should generate placeholder
        overlay, clinical, patient, badge, details = self.pipeline.process_cxr(
            image=None,
            region_label="lungs",
        )

        self.assertIsInstance(overlay, Image.Image)
        self.assertTrue(len(clinical) > 0)


if __name__ == "__main__":
    unittest.main()
