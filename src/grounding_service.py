"""
src/grounding_service.py
========================
Singleton grounding service wrapping TransVG-AGPT.

Loads the model ONCE at process start and exposes a predict() method.

Inference protocol (identical to scripts/evaluate_rsna.py::infer_single_image):
  1. Load image bytes → RGB numpy array
  2. Apply AGPT val transforms: LongestMaxSize(640) + PadIfNeeded(640) + Normalize + ToTensor
  3. Build NestedTensor(image_tensor, zero_mask)
  4. Call: outputs = model(image_nested, [phrase])   ← exactly 2 args
  5. Convert outputs["pred_boxes"] (cx,cy,w,h normalized) → original pixel coords

TransVG_ca.forward(img_data: NestedTensor, text_data: list[str]) -> dict
  - img_data   : NestedTensor wrapping (1,3,H,W) tensor + (1,H,W) bool mask
  - text_data  : list of raw phrase strings — BERT tokenization happens INSIDE the model
  - returns    : {"pred_boxes": Tensor(1,4), "attn_output_weights": ...}

NB: The previous implementation incorrectly called model(img_tensor, word_ids, word_mask)
    — 3 separate tensors — which fails with "takes 3 positional arguments but 4 were given".
"""

from __future__ import annotations

import io
import sys
import time
import threading
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Reuse coordinate helpers directly from evaluate_rsna.py (single source of truth)
from evaluate_rsna import (
    compute_letterbox_params,
    pred_norm_to_pixel,
    pred_norm_to_canvas_xyxy,
)

_LOCK = threading.Lock()

# ── Checkpoint paths ──────────────────────────────────────────────────────────
FINETUNED_CKPT = (
    PROJECT_ROOT / "checkpoints" / "finetuned_rsna" / "run_01_3ep" / "best_model_val_iou.pth"
)
BASELINE_CKPT = PROJECT_ROOT / "checkpoints" / "transvg.pth"
AGPT_ROOT = PROJECT_ROOT / "third_party" / "AGPT"


class GroundingService:
    """
    Loads TransVG-AGPT from disk once and serves inference requests.

    Args:
        checkpoint: "finetuned" (default) | "baseline" | absolute path str
        device:     "auto" | "mps" | "cpu"
    """

    def __init__(self, checkpoint: str = "finetuned", device: str = "auto"):
        import torch

        if device == "auto":
            self._device = (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        else:
            self._device = torch.device(device)

        if checkpoint == "finetuned":
            self.ckpt_path = FINETUNED_CKPT
        elif checkpoint == "baseline":
            self.ckpt_path = BASELINE_CKPT
        else:
            self.ckpt_path = Path(checkpoint)

        if not self.ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")

        self._model = None
        self._args = None
        self._loaded = False
        self._load_error: Optional[str] = None
        self._canvas = 640  # AGPT inference canvas size

    # ── Lazy load ─────────────────────────────────────────────────────────────
    def ensure_loaded(self) -> None:
        with _LOCK:
            if self._loaded or self._load_error:
                return
            try:
                self._load()
            except Exception as exc:
                self._load_error = str(exc)
                raise

    def _load(self) -> None:
        import torch

        # Add AGPT to sys.path so its internal imports resolve
        if str(AGPT_ROOT) not in sys.path:
            sys.path.insert(0, str(AGPT_ROOT))

        from Args import build_args
        from models.transvg import build_transvg_model

        print(f"[GroundingService] Loading checkpoint: {self.ckpt_path}")
        print(f"[GroundingService] Device: {self._device}")

        args = build_args("transvg", "ms_cxr")
        args.device = self._device

        model = build_transvg_model(args)
        ckpt = torch.load(str(self.ckpt_path), map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        model = model.to(self._device)
        model.eval()

        self._model = model
        self._args = args
        self._loaded = True

        n_params = sum(p.numel() for p in model.parameters())
        print(f"[GroundingService] Loaded successfully. Parameters: {n_params:,}")

    # ── Image loading ─────────────────────────────────────────────────────────
    @staticmethod
    def _load_image_to_rgb_array(image_bytes: bytes, image_filename: str) -> np.ndarray:
        """
        Convert raw image bytes (DICOM or standard format) to uint8 RGB numpy array.
        Mirrors the preprocessing in evaluate_rsna.py::infer_single_image.
        """
        suffix = Path(image_filename).suffix.lower()

        if suffix == ".dcm":
            import pydicom
            ds = pydicom.dcmread(io.BytesIO(image_bytes))
            pixels = ds.pixel_array
        else:
            from PIL import Image as PILImage
            pil_img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
            pixels = np.array(pil_img)

        # Normalize to uint8 [0, 255]
        if pixels.dtype != np.uint8:
            pmin, pmax = pixels.min(), pixels.max()
            pixels = (
                ((pixels - pmin) / (pmax - pmin) * 255).astype(np.uint8)
                if pmax > pmin
                else np.zeros_like(pixels, dtype=np.uint8)
            )

        # Ensure 3-channel RGB
        if pixels.ndim == 2:
            pixels = np.stack([pixels] * 3, axis=-1)
        elif pixels.ndim == 3 and pixels.shape[2] == 1:
            pixels = np.concatenate([pixels] * 3, axis=-1)
        elif pixels.ndim == 3 and pixels.shape[2] == 4:
            pixels = pixels[:, :, :3]  # drop alpha

        return pixels  # shape (H, W, 3), dtype uint8

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(
        self,
        image_bytes: bytes,
        query: str,
        image_filename: str = "upload",
    ) -> dict:
        """
        Run phrase grounding on raw image bytes.

        Calls: model(image_nested, [query])
            image_nested = NestedTensor(img_tensor (1,3,640,640), mask (1,640,640))
            [query] = list of raw phrase strings — BERT tokenization inside model

        Returns dict with keys:
            pred_box_norm     : [cx, cy, w, h] in [0,1]   (None on failure)
            pred_box_pixel    : [x1, y1, x2, y2] original pixel space  (None on failure)
            pred_box_canvas   : [x1, y1, x2, y2] on 640×640 canvas     (None on failure)
            orig_h, orig_w    : original image dimensions
            latency_ms        : wall-clock time
            success           : bool
            error             : str or None
            query             : echo of the query phrase
            checkpoint        : checkpoint filename
        """
        self.ensure_loaded()

        import torch

        t0 = time.perf_counter()

        try:
            # ── 1. Load and normalize image ───────────────────────────────────
            pixels = self._load_image_to_rgb_array(image_bytes, image_filename)
            orig_h, orig_w = pixels.shape[:2]

            # ── 2. Apply AGPT val transforms (identical to evaluate_rsna.py) ──
            # Must import from AGPT path since it's already on sys.path after _load()
            from dataloader.mscxr_dataloader import get_transforms
            from utils.misc import NestedTensor

            transform = get_transforms(self._args, "val", with_bboxes=False)
            transformed = transform(image=pixels)
            img_tensor = transformed["image"].unsqueeze(0).to(self._device)  # (1,3,640,640)
            img_mask = torch.zeros_like(img_tensor[:, 0, :, :], dtype=torch.bool)  # (1,640,640)
            image_nested = NestedTensor(img_tensor, img_mask)

            # ── 3. Forward pass — EXACT signature: model(NestedTensor, [phrase]) ──
            with torch.no_grad():
                outputs = self._model(image_nested, [query])

            # outputs["pred_boxes"] shape: (1, 4) — normalized (cx, cy, w, h) in [0,1]
            pred_norm = outputs["pred_boxes"].squeeze(0).detach().cpu().numpy()  # (4,)

            latency_ms = (time.perf_counter() - t0) * 1000.0

            # ── 4. Coordinate conversion ──────────────────────────────────────
            pred_canvas_xyxy = pred_norm_to_canvas_xyxy(pred_norm, self._canvas)
            pred_pixel_xyxy = pred_norm_to_pixel(pred_norm, orig_h, orig_w, self._canvas)

            # Derive canonical normalized representation directly from the pixel box:
            # [cx/orig_w, cy/orig_h, w/orig_w, h/orig_h]
            # This ensures displayed normalized box and pixel box are 100% mathematically consistent.
            x1_p, y1_p, x2_p, y2_p = [float(v) for v in pred_pixel_xyxy]
            box_w = max(0.0, x2_p - x1_p)
            box_h = max(0.0, y2_p - y1_p)
            cx_p = x1_p + 0.5 * box_w
            cy_p = y1_p + 0.5 * box_h

            norm_cx = round(cx_p / max(1, orig_w), 4)
            norm_cy = round(cy_p / max(1, orig_h), 4)
            norm_w  = round(box_w / max(1, orig_w), 4)
            norm_h  = round(box_h / max(1, orig_h), 4)
            canonical_norm = [norm_cx, norm_cy, norm_w, norm_h]

            return {
                "success": True,
                "error": None,
                "pred_box_norm": canonical_norm,
                "raw_canvas_box_norm": [round(float(v), 6) for v in pred_norm],
                "pred_box_canvas": [round(float(v), 2) for v in pred_canvas_xyxy],
                "pred_box_pixel": [round(float(v), 2) for v in pred_pixel_xyxy],
                "orig_h": orig_h,
                "orig_w": orig_w,
                "latency_ms": round(latency_ms, 2),
                "query": query,
                "checkpoint": str(self.ckpt_path.name),
            }

        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            warnings.warn(f"[GroundingService] Inference error: {exc}")
            return {
                "success": False,
                "error": str(exc),
                "pred_box_norm": None,
                "pred_box_canvas": None,
                "pred_box_pixel": None,
                "orig_h": None,
                "orig_w": None,
                "latency_ms": round(latency_ms, 2),
                "query": query,
                "checkpoint": str(self.ckpt_path.name),
            }

    @property
    def status(self) -> dict:
        return {
            "loaded": self._loaded,
            "error": self._load_error,
            "checkpoint": str(self.ckpt_path),
            "device": str(self._device),
        }


# ── Module-level singletons ───────────────────────────────────────────────────
_finetuned_service: Optional[GroundingService] = None
_baseline_service: Optional[GroundingService] = None


def get_grounding_service(checkpoint: str = "finetuned") -> GroundingService:
    """Return (and lazily create) the module-level singleton service."""
    global _finetuned_service, _baseline_service
    if checkpoint == "finetuned":
        if _finetuned_service is None:
            _finetuned_service = GroundingService(checkpoint="finetuned")
        return _finetuned_service
    elif checkpoint == "baseline":
        if _baseline_service is None:
            _baseline_service = GroundingService(checkpoint="baseline")
        return _baseline_service
    else:
        return GroundingService(checkpoint=checkpoint)
