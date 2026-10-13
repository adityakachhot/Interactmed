"""
vision_encoder/encoder.py
=========================
Visual feature extraction module for InteractMed-VLM.

Input/Output Contract:
----------------------
Input:  CXR image (torch.Tensor [B, C, H, W], PIL.Image.Image, or file path str)
Output: Visual feature tensor of shape [B, feature_dim, H_feat, W_feat]
"""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from PIL import Image


class VisionEncoder(nn.Module):
    """
    Vision Encoder backbone for extracting dense spatial representations from CXR images.

    TODO: When selecting a pretrained backbone, consider:
      1. 'microsoft/BiomedVLP-BioViL-T' (Recommended for domain-specific chest pathology)
      2. 'stanford-crfm/CheXzero' (Zero-shot CXR image encoder)
      3. 'radimagenet-resnet50' (Trained on 1.35M radiology images)
      4. 'openai/clip-vit-base-patch32' (General VLM baseline)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self.image_size: Tuple[int, int] = tuple(self.config.get("image_size", (224, 224)))
        self.feature_dim: int = int(self.config.get("feature_dim", 512))
        self.output_spatial_dim: Tuple[int, int] = tuple(self.config.get("output_spatial_dim", (7, 7)))
        self.model_name: str = self.config.get("model_name_or_path", "placeholder_vision_encoder")

        # Stub projection layer for CPU unit testing and dimension alignment
        # In production, this will be replaced by the chosen vision backbone
        self.stub_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.output_spatial_dim),
            nn.Conv2d(3, self.feature_dim, kernel_size=1, bias=False),
        )

    def preprocess(self, image: Union[torch.Tensor, Image.Image, str]) -> torch.Tensor:
        """
        Preprocesses an input image into a standardized tensor [B, 3, H, W].
        """
        if isinstance(image, str):
            # Load from file
            pil_img = Image.open(image).convert("RGB")
            return self._pil_to_tensor(pil_img)
        elif isinstance(image, Image.Image):
            return self._pil_to_tensor(image)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                # [C, H, W] -> [1, C, H, W]
                image = image.unsqueeze(0)
            if image.size(1) == 1:
                # Grayscale to 3-channel
                image = image.repeat(1, 3, 1, 1)
            return image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def _pil_to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        """Helper to convert PIL Image to tensor [1, 3, H, W]."""
        resized = pil_img.resize(self.image_size)
        # Scale to [0, 1]
        tensor = torch.tensor(list(resized.getdata()), dtype=torch.float32).view(
            resized.size[1], resized.size[0], len(resized.getbands())
        )
        tensor = tensor.permute(2, 0, 1) / 255.0  # [3, H, W]
        if tensor.size(0) == 1:
            tensor = tensor.repeat(3, 1, 1)
        elif tensor.size(0) > 3:
            tensor = tensor[:3, :, :]
        return tensor.unsqueeze(0)  # [1, 3, H, W]

    def forward(self, image: Union[torch.Tensor, Image.Image, str]) -> torch.Tensor:
        """
        Encodes an input CXR into visual features.

        Args:
            image: CXR image as tensor [B, C, H, W], PIL Image, or file path.

        Returns:
            Visual feature tensor of shape [B, feature_dim, H_feat, W_feat].
        """
        x = self.preprocess(image)

        # TODO: Replace stub forward pass with pretrained backbone feature extractor:
        # e.g.:
        # if "biovil" in self.model_name.lower():
        #     features = self.backbone(x).last_hidden_state
        # else:
        #     features = self.backbone(x)
        features = self.stub_projection(x)
        return features

    def encode(self, image: Union[torch.Tensor, Image.Image, str]) -> torch.Tensor:
        """Alias for forward()."""
        return self.forward(image)
