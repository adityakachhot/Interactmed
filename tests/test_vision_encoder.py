"""
test_vision_encoder.py
======================
Unit tests verifying the VisionEncoder contract:
Input: CXR image tensor [B, 3, H, W] or PIL Image
Output: Visual feature tensor [B, feature_dim, H_feat, W_feat]
"""

import unittest
import torch
from PIL import Image

from src.vision_encoder.encoder import VisionEncoder


class TestVisionEncoder(unittest.TestCase):
    def setUp(self):
        self.config = {
            "image_size": [224, 224],
            "feature_dim": 512,
            "output_spatial_dim": [7, 7],
        }
        self.encoder = VisionEncoder(self.config)

    def test_tensor_input_shape(self):
        # Batch of 2 images, 3 channels, 224x224
        dummy_tensor = torch.randn(2, 3, 224, 224)
        features = self.encoder.encode(dummy_tensor)

        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.shape, (2, 512, 7, 7))

    def test_grayscale_tensor_expansion(self):
        # 1 channel grayscale image [1, 1, 224, 224]
        dummy_gray = torch.randn(1, 1, 224, 224)
        features = self.encoder.encode(dummy_gray)

        self.assertEqual(features.shape, (1, 512, 7, 7))

    def test_pil_image_input(self):
        # PIL RGB Image
        pil_img = Image.new("RGB", (300, 400), color=(128, 128, 128))
        features = self.encoder.encode(pil_img)

        self.assertEqual(features.shape, (1, 512, 7, 7))


if __name__ == "__main__":
    unittest.main()
