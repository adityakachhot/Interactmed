"""
region_grounding/grounding.py
=============================
Anatomical region localization and regional feature extraction for InteractMed-VLM.

Input/Output Contract:
----------------------
Input:  image (or visual_features) + region_label ('heart', 'lungs', 'pleura')
Output: (bbox [x, y, w, h], regional_features [B, embed_dim])
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from PIL import Image


class RegionGroundingModule(nn.Module):
    """
    Localizes user-selected anatomical regions (heart/lungs/pleura) and extracts
    region-pooled visual features.

    TODO: Candidate grounding mechanisms to choose from:
      1. 'Grounding DINO' / Open-vocabulary detector fine-tuned on MS-CXR / Chest ImaGenome.
      2. 'RoIAlign + Cross-Attention' conditioned on text prompt / region queries.
      3. 'Chest ImaGenome Anatomical Prior BBoxes' with deformable refinement.
    """

    # Canonical anatomical prior bounding boxes (normalized [x_norm, y_norm, w_norm, h_norm] in [0, 1])
    # Derived from population mean locations across Chest ImaGenome Gold annotations
    ANATOMICAL_PRIORS = {
        "heart": [0.30, 0.45, 0.40, 0.35],   # Centered lower-middle thorax
        "lungs": [0.10, 0.15, 0.80, 0.70],   # Bilateral thoracic cage
        "pleura": [0.08, 0.40, 0.84, 0.50],  # Bilateral costophrenic recesses & margins
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self.method: str = self.config.get("method", "anatomical_prior_stub")
        self.feature_dim: int = int(self.config.get("feature_dim", 512))

        # Stub pooling / linear transformation for regional feature extraction
        self.stub_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.stub_region_proj = nn.Linear(self.feature_dim, self.feature_dim)

    def get_prior_bbox(self, region_label: str, image_size: Tuple[int, int] = (224, 224)) -> List[float]:
        """
        Returns an initial bounding box [x, y, w, h] in absolute pixel coordinates
        for the given anatomical region based on clinical priors.
        """
        key = region_label.lower().strip()
        prior_norm = self.ANATOMICAL_PRIORS.get(key, self.ANATOMICAL_PRIORS["lungs"])
        h_img, w_img = image_size
        x = round(prior_norm[0] * w_img, 2)
        y = round(prior_norm[1] * h_img, 2)
        w = round(prior_norm[2] * w_img, 2)
        h = round(prior_norm[3] * h_img, 2)
        return [x, y, w, h]

    def ground(
        self,
        visual_features: torch.Tensor,
        region_label: str,
        image_size: Tuple[int, int] = (224, 224),
    ) -> Tuple[List[float], torch.Tensor]:
        """
        Grounds an anatomical region from visual features.

        Args:
            visual_features: Tensor of shape [B, C, H_feat, W_feat].
            region_label: String in ('heart', 'lungs', 'pleura').
            image_size: Tuple (height, width) of the original CXR image.

        Returns:
            Tuple of:
              - bbox: [x, y, w, h] in pixel coordinates.
              - regional_features: Pooled feature tensor of shape [B, embed_dim].
        """
        # Validate region label
        clean_label = region_label.lower().strip()
        if clean_label not in self.ANATOMICAL_PRIORS:
            # Fallback
            clean_label = "lungs"

        # 1. Compute bounding box coordinates
        # TODO: In production, replace with actual grounding model predictions:
        # bbox = self.detector.predict(image, prompt=region_label)
        bbox = self.get_prior_bbox(clean_label, image_size)

        # 2. Extract regional features
        # TODO: In production, use torchvision.ops.roi_align(visual_features, [rois], ...)
        # For CPU stub, pool spatial features and apply projection
        b, c, _, _ = visual_features.shape
        pooled = self.stub_pool(visual_features).view(b, c)
        regional_features = self.stub_region_proj(pooled)

        return bbox, regional_features

    def forward(
        self,
        visual_features: torch.Tensor,
        region_label: str,
        image_size: Tuple[int, int] = (224, 224),
    ) -> Tuple[List[float], torch.Tensor]:
        """Alias for ground()."""
        return self.ground(visual_features, region_label, image_size)
