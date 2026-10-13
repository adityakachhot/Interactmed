"""
tests/test_grounding_inference.py
==================================
Regression test for the TransVG_ca.forward() 2-argument API fix.

Tests:
  1. Synthetic PNG — verifies no "positional arguments" exception at all
  2. Real RSNA DICOM — verifies grounding returns a valid box on an actual DCM
  3. AGPT demo JPG — verifies grounding on a standard JPEG (non-DICOM)
  4. Both checkpoints (finetuned + baseline) smoke-tested

What we DO NOT test:
  - GT annotations are NEVER used as model inputs.
  - We do NOT check IoU against a specific GT box (that is done in evaluate_rsna.py).
    Here we only verify: success=True, pred_box_pixel is a valid finite 4-tuple,
    and no positional-argument TypeError is raised.

Run with:
    /Users/adityakachhot/miniforge3/envs/dl/bin/python -m pytest tests/test_grounding_inference.py -v
"""

from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# ── Helpers ───────────────────────────────────────────────────────────────────

def _synthetic_png_bytes(width: int = 1024, height: int = 1024) -> bytes:
    """Create a grey square PNG in memory — no real patient data."""
    img = Image.new("RGB", (width, height), color=(130, 130, 130))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _assert_valid_result(result: dict, label: str) -> None:
    """Assert inference succeeded and produced a valid bounding box."""
    assert result["success"], (
        f"[{label}] Inference failed: {result['error']}\n"
        "Expected the 'TransVG_ca.forward() takes 3 positional arguments but 4 were given' "
        "error to be fixed."
    )
    assert result["error"] is None, f"[{label}] error field not None: {result['error']}"

    box = result["pred_box_pixel"]
    assert box is not None, f"[{label}] pred_box_pixel is None"
    assert len(box) == 4, f"[{label}] pred_box_pixel has {len(box)} elements, expected 4"

    x1, y1, x2, y2 = box
    assert all(np.isfinite(v) for v in box), f"[{label}] non-finite values in box: {box}"
    assert x2 > x1, f"[{label}] x2 ({x2}) must be > x1 ({x1})"
    assert y2 > y1, f"[{label}] y2 ({y2}) must be > y1 ({y1})"

    # Box must lie within the reported image dimensions
    orig_h = result["orig_h"]
    orig_w = result["orig_w"]
    assert orig_h > 0 and orig_w > 0, f"[{label}] invalid image dims: {orig_h}x{orig_w}"
    assert x1 >= 0, f"[{label}] x1 < 0: {x1}"
    assert y1 >= 0, f"[{label}] y1 < 0: {y1}"
    assert x2 <= orig_w + 1, f"[{label}] x2 ({x2}) exceeds image width ({orig_w})"
    assert y2 <= orig_h + 1, f"[{label}] y2 ({y2}) exceeds image height ({orig_h})"

    # Normalized box should be in [0, 1]
    norm = result["pred_box_norm"]
    assert norm is not None
    assert all(0.0 <= v <= 1.0 for v in norm), (
        f"[{label}] pred_box_norm out of [0,1]: {norm}"
    )

    assert result["latency_ms"] > 0, f"[{label}] latency_ms must be positive"

    print(f"  [{label}] PASS — box={[round(v,1) for v in box]}, "
          f"norm={[round(v,4) for v in norm]}, latency={result['latency_ms']:.0f}ms")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def finetuned_service():
    """Load the fine-tuned grounding service once for the whole module."""
    from src.grounding_service import GroundingService
    svc = GroundingService(checkpoint="finetuned")
    svc.ensure_loaded()
    return svc


@pytest.fixture(scope="module")
def baseline_service():
    """Load the baseline grounding service once for the whole module."""
    from src.grounding_service import GroundingService
    svc = GroundingService(checkpoint="baseline")
    svc.ensure_loaded()
    return svc


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestGroundingInferenceFix:
    """
    Regression tests for TransVG_ca.forward() 2-argument API.
    Each test verifies: success=True, valid box, no TypeError.
    """

    def test_synthetic_png_pneumonia_finetuned(self, finetuned_service):
        """
        REGRESSION: ensures 'takes 3 positional arguments but 4 were given' is gone.
        Uses a synthetic CXR-sized image — no real patient data.
        """
        result = finetuned_service.predict(
            _synthetic_png_bytes(1024, 1024),
            query="pneumonia",
            image_filename="synthetic_1024.png",
        )
        _assert_valid_result(result, "synthetic-PNG/finetuned/pneumonia")

    def test_synthetic_png_airspace_opacity_finetuned(self, finetuned_service):
        """Verify a different phrase also works without errors."""
        result = finetuned_service.predict(
            _synthetic_png_bytes(512, 512),
            query="airspace opacity",
            image_filename="synthetic_512.png",
        )
        _assert_valid_result(result, "synthetic-PNG/finetuned/airspace opacity")

    def test_synthetic_png_pneumonia_baseline(self, baseline_service):
        """Verify the baseline checkpoint also uses the correct 2-arg API."""
        result = baseline_service.predict(
            _synthetic_png_bytes(1024, 1024),
            query="pneumonia",
            image_filename="synthetic_baseline.png",
        )
        _assert_valid_result(result, "synthetic-PNG/baseline/pneumonia")

    @pytest.mark.skipif(
        not list(Path("data/raw/rsna/stage_2_train_images").glob("*.dcm"))
        if Path("data/raw/rsna/stage_2_train_images").is_dir() else True,
        reason="No RSNA DICOM files found in data/raw/rsna/stage_2_train_images",
    )
    def test_real_rsna_dicom_pneumonia_finetuned(self, finetuned_service):
        """
        Integration test: run grounding on a real RSNA DICOM.
        Image is from the TRAINING split — no GT annotation is passed to the model.
        """
        dcm_dir = Path("data/raw/rsna/stage_2_train_images")
        dcm_files = sorted(dcm_dir.glob("*.dcm"))
        assert dcm_files, "No DCM files found"

        # Use the first available DCM
        dcm_path = dcm_files[0]
        print(f"\n  [RSNA-DCM] Using: {dcm_path.name}")

        with open(dcm_path, "rb") as f:
            image_bytes = f.read()

        result = finetuned_service.predict(
            image_bytes,
            query="pneumonia",
            image_filename=dcm_path.name,
        )
        _assert_valid_result(result, f"RSNA-DCM/{dcm_path.name}/finetuned/pneumonia")

    @pytest.mark.skipif(
        not Path("third_party/AGPT/demo/0ae1a7ad-fcc627aa-01e96e50-cb14cf17-23f441ac.jpg").is_file(),
        reason="AGPT demo JPG not found",
    )
    def test_agpt_demo_jpg_finetuned(self, finetuned_service):
        """
        Integration test: run grounding on the AGPT demo JPG.
        The demo file is a real CXR JPEG. We use 'pneumonia' as the phrase.
        (The demo.py has GT annotations but we do NOT use them here.)
        """
        jpg_path = Path(
            "third_party/AGPT/demo/0ae1a7ad-fcc627aa-01e96e50-cb14cf17-23f441ac.jpg"
        )
        print(f"\n  [AGPT-JPG] Using: {jpg_path.name}")

        with open(jpg_path, "rb") as f:
            image_bytes = f.read()

        result = finetuned_service.predict(
            image_bytes,
            query="pneumonia",
            image_filename=jpg_path.name,
        )
        _assert_valid_result(result, f"AGPT-JPG/{jpg_path.name}/finetuned/pneumonia")

    @pytest.mark.skipif(
        not Path("third_party/AGPT/demo/3aa9ee81-08355b5d-713a3a03-6cd20281-6222326c.jpg").is_file(),
        reason="AGPT demo JPG #2 not found",
    )
    def test_agpt_demo_jpg_opacity_phrase(self, finetuned_service):
        """Test a second AGPT demo image with 'Left perihilar opacity' phrase."""
        jpg_path = Path(
            "third_party/AGPT/demo/3aa9ee81-08355b5d-713a3a03-6cd20281-6222326c.jpg"
        )
        print(f"\n  [AGPT-JPG2] Using: {jpg_path.name}")

        with open(jpg_path, "rb") as f:
            image_bytes = f.read()

        result = finetuned_service.predict(
            image_bytes,
            query="Left perihilar opacity",
            image_filename=jpg_path.name,
        )
        _assert_valid_result(result, "AGPT-JPG2/finetuned/Left perihilar opacity")

    def test_checkpoint_selector_finetuned_vs_baseline_differ(
        self, finetuned_service, baseline_service
    ):
        """
        Verify that the finetuned and baseline checkpoints produce different
        predictions on the same image — confirming the checkpoint selector actually
        loads different weights (not just a single cached model).
        """
        img_bytes = _synthetic_png_bytes(1024, 1024)

        r_ft = finetuned_service.predict(img_bytes, "pneumonia", "ckpt_test.png")
        r_bl = baseline_service.predict(img_bytes, "pneumonia", "ckpt_test.png")

        assert r_ft["success"] and r_bl["success"], (
            "Both checkpoints must succeed for this comparison test"
        )
        assert r_ft["checkpoint"] != r_bl["checkpoint"], (
            f"Both services returned same checkpoint: {r_ft['checkpoint']}"
        )

        ft_box = r_ft["pred_box_norm"]
        bl_box = r_bl["pred_box_norm"]

        # The two checkpoints should produce different outputs (they have different weights)
        max_diff = max(abs(a - b) for a, b in zip(ft_box, bl_box))
        assert max_diff > 0.001, (
            f"Fine-tuned and baseline returned nearly identical boxes: {ft_box} vs {bl_box}. "
            "This may indicate the checkpoint selector is not working."
        )
        print(f"\n  [checkpoint-diff] max box diff = {max_diff:.4f} — checkpoints confirmed distinct")

    def test_invalid_box_detection_empty_query(self, finetuned_service):
        """
        Verify that an empty image still returns a result (success=True with some box)
        rather than crashing. The model should always produce output.
        """
        # 1x1 pixel PNG — extreme edge case
        img = Image.new("RGB", (1, 1), color=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result = finetuned_service.predict(
            buf.getvalue(), "pneumonia", "tiny.png"
        )
        # Model should succeed or return a structured error — never raise an exception
        assert isinstance(result, dict)
        assert "success" in result
        assert "error" in result
        # If it succeeded, the box must be valid
        if result["success"]:
            assert result["pred_box_pixel"] is not None
            print(f"\n  [tiny-image] PASS — model handled 1×1 px gracefully")
        else:
            print(f"\n  [tiny-image] PASS — model returned structured error: {result['error']}")
