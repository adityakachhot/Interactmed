"""
src/overlay_service.py
======================
Draw predicted bounding box and GT boxes over the original image.
Returns a PIL.Image with annotations, plus base64 JPEG for API responses.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont


def _load_image(image_bytes: bytes, image_filename: str) -> PILImage.Image:
    suffix = Path(image_filename).suffix.lower()
    if suffix == ".dcm":
        import pydicom
        ds = pydicom.dcmread(io.BytesIO(image_bytes))
        pixels = ds.pixel_array
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixels = pixels.max() - pixels
        if pixels.dtype != np.uint8:
            pmin, pmax = pixels.min(), pixels.max()
            pixels = ((pixels - pmin) / (pmax - pmin) * 255).astype(np.uint8) if pmax > pmin \
                      else np.zeros_like(pixels, dtype=np.uint8)
        if pixels.ndim == 2:
            pixels = np.stack([pixels] * 3, axis=-1)
        return PILImage.fromarray(pixels)
    else:
        return PILImage.open(io.BytesIO(image_bytes)).convert("RGB")


def draw_overlay(
    image_bytes: bytes,
    image_filename: str,
    pred_box_pixel_xyxy: list[float],
    gt_boxes_xywh: Optional[list[list[float]]] = None,
    query: str = "",
    iou: Optional[float] = None,
    model_label: str = "",
) -> PILImage.Image:
    """
    Draw predicted box (cyan) and optional GT boxes (green) on the original image.

    Args:
        pred_box_pixel_xyxy : [x1, y1, x2, y2] in original pixel coords.
        gt_boxes_xywh       : Optional list of [x, y, w, h] GT boxes.
        query               : Clinical phrase used for grounding.
        iou                 : Optional best-match IoU to annotate.
        model_label         : Model name string for annotation.

    Returns:
        Annotated PIL image (RGB).
    """
    img = _load_image(image_bytes, image_filename).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    # ── GT boxes (green solid) ────────────────────────────────────────────────
    if gt_boxes_xywh:
        for i, (gx, gy, gw, gh) in enumerate(gt_boxes_xywh):
            draw.rectangle(
                [(gx, gy), (gx + gw, gy + gh)],
                outline=(34, 197, 94, 255),  # green-500
                width=3,
            )
            draw.text(
                (gx + 4, gy + 2),
                f"GT {i+1}",
                fill=(34, 197, 94, 255),
            )

    # ── Predicted box (cyan dashed → drawn as solid thin) ────────────────────
    x1, y1, x2, y2 = pred_box_pixel_xyxy
    # Draw with 4px cyan border
    draw.rectangle([(x1, y1), (x2, y2)], outline=(6, 182, 212, 255), width=4)

    # ── Label ─────────────────────────────────────────────────────────────────
    label_parts = [f'Query: "{query}"']
    if model_label:
        label_parts.append(f"Model: {model_label}")
    if iou is not None:
        label_parts.append(f"Best IoU: {iou:.3f}")
    label = " | ".join(label_parts)

    # Background box for text
    W, H = img.size
    draw.rectangle([(0, H - 28), (W, H)], fill=(0, 0, 0, 180))
    draw.text((6, H - 22), label, fill=(255, 255, 255, 255))

    # ── Legend ────────────────────────────────────────────────────────────────
    if gt_boxes_xywh:
        draw.rectangle([(6, 6), (22, 22)], fill=(34, 197, 94, 200))
        draw.text((26, 6), "GT Box", fill=(34, 197, 94, 255))
    draw.rectangle([(6, 28), (22, 44)], fill=(6, 182, 212, 200))
    draw.text((26, 28), "Predicted Box", fill=(6, 182, 212, 255))

    return img


def image_to_base64_jpeg(img: PILImage.Image, quality: int = 85) -> str:
    """Convert PIL image to base64-encoded JPEG string."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()
