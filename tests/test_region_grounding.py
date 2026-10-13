"""
test_region_grounding.py
========================
Unit tests verifying the RegionGroundingModule contract:
Input:  visual_features [B, C, H, W] + region_label ('heart', 'lungs', 'pleura')
Output: bbox [x, y, w, h] + regional_features [B, embed_dim]
"""

import unittest
import torch

from src.region_grounding.grounding import RegionGroundingModule


class TestRegionGrounding(unittest.TestCase):
    def setUp(self):
        self.config = {
            "feature_dim": 512,
        }
        self.grounder = RegionGroundingModule(self.config)

    def test_grounding_heart(self):
        visual_features = torch.randn(1, 512, 7, 7)
        bbox, regional_feats = self.grounder.ground(
            visual_features,
            region_label="heart",
            image_size=(224, 224),
        )

        # Check bbox structure
        self.assertIsInstance(bbox, list)
        self.assertEqual(len(bbox), 4)
        x, y, w, h = bbox
        self.assertTrue(x >= 0 and y >= 0 and w > 0 and h > 0)
        self.assertTrue(x + w <= 224 and y + h <= 224)

        # Check regional feature shape
        self.assertIsInstance(regional_feats, torch.Tensor)
        self.assertEqual(regional_feats.shape, (1, 512))

    def test_grounding_all_supported_regions(self):
        visual_features = torch.randn(2, 512, 7, 7)
        for region in ["heart", "lungs", "pleura"]:
            bbox, reg_feat = self.grounder.ground(
                visual_features,
                region_label=region,
                image_size=(512, 512),
            )
            self.assertEqual(len(bbox), 4)
            self.assertEqual(reg_feat.shape, (2, 512))

    def test_unrecognized_region_fallback(self):
        visual_features = torch.randn(1, 512, 7, 7)
        # Should fallback gracefully to default region ('lungs') without error
        bbox, reg_feat = self.grounder.ground(
            visual_features,
            region_label="unknown_anatomy",
            image_size=(224, 224),
        )
        self.assertEqual(len(bbox), 4)
        self.assertEqual(reg_feat.shape, (1, 512))


if __name__ == "__main__":
    unittest.main()
