"""
test_explainability.py
======================
Unit tests verifying the VisualExplainer contract:
Input:  activations tensor [B, C, H, W] + bbox [x, y, w, h] + PIL Image
Output: heatmap np.ndarray [H, W] + annotated overlay PIL.Image
"""

import unittest
import numpy as np
import torch
from PIL import Image

from src.explainability.explainer import VisualExplainer


class TestVisualExplainer(unittest.TestCase):
    def setUp(self):
        self.explainer = VisualExplainer()

    def test_heatmap_generation(self):
        dummy_activations = torch.randn(1, 512, 7, 7)
        heatmap = self.explainer.generate_gradcam_heatmap(
            dummy_activations,
            target_size=(224, 224),
        )

        self.assertIsInstance(heatmap, np.ndarray)
        self.assertEqual(heatmap.shape, (224, 224))
        self.assertTrue(heatmap.min() >= 0.0)
        self.assertTrue(heatmap.max() <= 1.0)

    def test_overlay_visualization(self):
        base_img = Image.new("RGB", (224, 224), color=(100, 100, 100))
        dummy_heatmap = np.ones((224, 224), dtype=np.float32) * 0.5
        bbox = [50.0, 60.0, 80.0, 90.0]

        overlay = self.explainer.overlay_explanation(
            base_img,
            dummy_heatmap,
            bbox=bbox,
            region_label="heart",
        )

        self.assertIsInstance(overlay, Image.Image)
        self.assertEqual(overlay.size, (224, 224))
        self.assertEqual(overlay.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
