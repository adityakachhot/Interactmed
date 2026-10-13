"""
explainability/explainer.py
===========================
Visual attribution and explainability module for InteractMed-VLM.

Input/Output Contract:
----------------------
Input:  activations (torch.Tensor [B, C, H, W]) + bbox ([x, y, w, h]) + image (PIL.Image)
Output: heatmap (np.ndarray [H, W]), overlay_image (PIL.Image.Image)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


class VisualExplainer:
    """
    Computes post-hoc visual attribution heatmaps (Grad-CAM) and overlays
    grounded bounding boxes on the CXR image.

    TODO: When integrating real weights:
      - Hook target convolutional layer or cross-attention map (e.g., last block of ViT/ResNet).
      - Compute gradient of selected regional finding logits w.r.t. target layer activations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.alpha: float = float(self.config.get("alpha_overlay", 0.4))
        self.bbox_color: Tuple[int, int, int] = tuple(self.config.get("bbox_color", (255, 69, 0)))  # type: ignore
        self.bbox_thickness: int = int(self.config.get("bbox_thickness", 3))

    def generate_gradcam_heatmap(
        self,
        activations: torch.Tensor,
        gradients: Optional[torch.Tensor] = None,
        target_size: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """
        Generates a 2D Grad-CAM attribution heatmap normalized to [0, 1].

        Args:
            activations: Feature tensor of shape [B, C, H_feat, W_feat] or [C, H_feat, W_feat].
            gradients: Optional gradient tensor of matching shape.
            target_size: Desired output dimensions (height, width).

        Returns:
            Normalized 2D numpy array of shape [target_size[0], target_size[1]] with values in [0, 1].
        """
        if activations.dim() == 4:
            act = activations[0]  # Take first batch item [C, H, W]
        else:
            act = activations

        # Convert to float on CPU
        act_np = act.detach().cpu().float().numpy()  # [C, H, W]

        if gradients is not None:
            grad_np = gradients.detach().cpu().float().numpy()
            weights = np.mean(grad_np, axis=(1, 2), keepdims=True)  # [C, 1, 1]
            cam = np.sum(weights * act_np, axis=0)
        else:
            # Saliency proxy: channel-wise mean energy
            cam = np.mean(np.maximum(act_np, 0), axis=0)

        # Apply ReLU
        cam = np.maximum(cam, 0)

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Resize to target_size using PIL
        cam_img = Image.fromarray((cam * 255).astype(np.uint8), mode="L")
        cam_resized = cam_img.resize((target_size[1], target_size[0]), resample=Image.BILINEAR)
        return np.array(cam_resized, dtype=np.float32) / 255.0

    def colormap_heatmap(self, heatmap: np.ndarray) -> Image.Image:
        """
        Converts a [0, 1] 2D heatmap into an RGB colorized PIL Image (pseudo-JET colormap).
        """
        h, w = heatmap.shape
        # Simple procedural pseudo-JET palette without requiring cv2
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        val = np.clip(heatmap, 0.0, 1.0)

        # Red channel: ramps up in top half
        rgb[..., 0] = (np.clip(val * 2.0 - 0.5, 0.0, 1.0) * 255).astype(np.uint8)
        # Green channel: peaks in middle
        rgb[..., 1] = (np.clip(1.0 - np.abs(val * 2.0 - 1.0), 0.0, 1.0) * 255).astype(np.uint8)
        # Blue channel: ramps down
        rgb[..., 2] = (np.clip(1.5 - val * 2.0, 0.0, 1.0) * 255).astype(np.uint8)

        return Image.fromarray(rgb, mode="RGB")

    def overlay_explanation(
        self,
        image: Union[Image.Image, np.ndarray, str],
        heatmap: np.ndarray,
        bbox: Optional[List[float]] = None,
        region_label: str = "",
    ) -> Image.Image:
        """
        Combines original image, Grad-CAM color heatmap, and bounding box overlay.

        Args:
            image: Original CXR image.
            heatmap: 2D numpy array in [0, 1].
            bbox: Optional bounding box [x, y, w, h].
            region_label: Label tag for the bounding box.

        Returns:
            Annotated PIL Image.
        """
        if isinstance(image, str):
            base_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            base_img = Image.fromarray(image).convert("RGB")
        else:
            base_img = image.convert("RGB")

        # Resize heatmap color overlay to match base image
        target_size = (base_img.height, base_img.width)
        if heatmap.shape != target_size:
            cam_img = Image.fromarray((heatmap * 255).astype(np.uint8), mode="L")
            resized_cam = cam_img.resize((target_size[1], target_size[0]), resample=Image.BILINEAR)
            heatmap = np.array(resized_cam, dtype=np.float32) / 255.0

        color_cam = self.colormap_heatmap(heatmap)

        # Alpha blend base image with heatmap
        blended = Image.blend(base_img, color_cam, alpha=self.alpha)

        # Draw bounding box overlay
        if bbox is not None and len(bbox) == 4:
            draw = ImageDraw.Draw(blended)
            x, y, w, h = bbox
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(base_img.width, x + w)
            y2 = min(base_img.height, y + h)

            for i in range(self.bbox_thickness):
                draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=self.bbox_color)

            if region_label:
                text = f"{region_label.upper()}"
                draw.rectangle([x1, max(0, y1 - 20), x1 + 80, y1], fill=self.bbox_color)
                draw.text((x1 + 4, max(0, y1 - 18)), text, fill=(255, 255, 255))

        return blended
