"""
tests/test_task20_image_grounded_reports.py
============================================
Task 20 Regression Tests: MedGemma Full-Image Report Generation

Verifies:
1. generate_whole_image_report always sends image bytes as actual visual input
   in BOTH Stage 1 (visual extraction) and Stage 2 (synthesis).
2. Stage 2 synthesis NEVER degrades to text-only (_ollama_generate_text_with_metrics).
3. Batch uses the same generate_whole_image_report pathway as single-image.
4. Image is verified by hash at the inference boundary in batch processing.
5. Batch items are independently isolated — one item's image does not bleed into another.
6. Calling with no image bytes returns a clear error; no fabricated report is produced.
7. The code path matches the specification (static analysis of source).
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import io
import re
import time
import types
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, call

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_jpeg_bytes(width: int = 64, height: int = 64, seed: int = 0) -> bytes:
    """Return minimal valid JPEG bytes using PIL."""
    import PIL.Image
    import numpy as np
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = PIL.Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=50)
    return buf.getvalue()


def _make_mock_vision_response(text: str = "") -> tuple:
    """Return (text, metrics) matching _ollama_generate_vision_with_metrics output."""
    return (text, {
        "total_duration_ns": 1_000_000,
        "load_duration_ns": 0,
        "prompt_eval_count": 10,
        "prompt_eval_duration_ns": 0,
        "eval_count": 20,
        "eval_duration_ns": 1_000_000,
        "done_reason": "stop",
    })


def _stage1_obs() -> str:
    return (
        "1. Support apparatus: Not identified\n"
        "2. Cardiac silhouette: Normal\n"
        "3. Lungs & opacities: Not identified\n"
        "4. Pleural spaces: Costophrenic angles sharp\n"
        "5. Mediastinum: Normal contour"
    )


def _research_raw() -> str:
    return (
        "SUPPORT APPARATUS: Not identified on available image\n"
        "CARDIAC SILHOUETTE: Normal size and borders\n"
        "LUNGS AND OPACITIES: Lungs clear bilaterally\n"
        "PLEURAL SPACES: Costophrenic angles sharp bilaterally\n"
        "MEDIASTINUM: Normal contour\n"
        "OVERALL IMPRESSION: No acute cardiopulmonary process identified\n"
        "UNCERTAINTY AND LIMITATIONS: Single frontal projection; no lateral view available"
    )


def _patient_raw() -> str:
    return (
        "Your X-ray in simple words\n"
        "The X-ray of your chest appears unremarkable with no obvious abnormalities.\n\n"
        "What the findings may mean\n"
        "The automated visual reading shows no significant lung or heart abnormalities.\n\n"
        "What is uncertain\n"
        "This frontal view displays both the left and right sides of the chest, but depth "
        "cannot be confirmed without a lateral view.\n\n"
        "What happens next\n"
        "Please consult your ordering physician for qualified clinical evaluation."
    )


# ── Static Analysis Tests ─────────────────────────────────────────────────────

class TestStaticAnalysis(unittest.TestCase):
    """Verify source code structure guarantees image is always passed in Stage 2."""

    def setUp(self):
        self.src_path = PROJECT_ROOT / "src" / "report_service.py"
        with open(self.src_path) as f:
            self.src = f.read()
        self.lines = self.src.splitlines()

    def test_no_syntax_errors(self):
        """report_service.py must parse without syntax errors."""
        try:
            ast.parse(self.src)
        except SyntaxError as e:
            self.fail(f"SyntaxError in report_service.py: {e}")

    def test_stage2_is_vision_not_text(self):
        """Stage 2 synthesis must use _ollama_generate_vision_with_metrics, not text-only."""
        stage2_start = next(
            i for i, l in enumerate(self.lines) if "Stage 2: Synthesis" in l
        )
        # Examine enough lines to cover both research and patient branches
        stage2_block = "\n".join(self.lines[stage2_start : stage2_start + 250])

        self.assertIn(
            "_ollama_generate_vision_with_metrics",
            stage2_block,
            "Stage 2 must call _ollama_generate_vision_with_metrics to pass the image.",
        )
        self.assertNotIn(
            "_ollama_generate_text_with_metrics",
            stage2_block,
            "Stage 2 must NOT degrade to text-only _ollama_generate_text_with_metrics.",
        )

    def test_stage2_image_passed_with_full_b64(self):
        """Stage 2 must pass images_b64=[full_b64] — the actual radiograph."""
        stage2_start = next(
            i for i, l in enumerate(self.lines) if "Stage 2: Synthesis" in l
        )
        stage2_block = "\n".join(self.lines[stage2_start : stage2_start + 250])
        self.assertIn(
            "images_b64=[full_b64]",
            stage2_block,
            "Stage 2 must pass images_b64=[full_b64] to MedGemma.",
        )

    def test_vision_call_count(self):
        """There must be at least 3 vision calls: Stage 1, Stage 2 research, Stage 2 patient."""
        count = self.src.count("_ollama_generate_vision_with_metrics")
        self.assertGreaterEqual(
            count,
            3,
            f"Expected >= 3 vision calls (s1 + s2_research + s2_patient), found {count}.",
        )

    def test_batch_imports_generate_whole_image_report(self):
        """batch_service.py must import and use generate_whole_image_report."""
        batch_src_path = PROJECT_ROOT / "src" / "batch_service.py"
        with open(batch_src_path) as f:
            batch_src = f.read()
        self.assertIn(
            "generate_whole_image_report",
            batch_src,
            "batch_service.py must import and use generate_whole_image_report.",
        )
        self.assertIn(
            "from src.report_service import",
            batch_src,
            "batch_service.py must import from report_service.",
        )

    def test_stage2_num_predict_increased(self):
        """Stage 2 num_predict must be >= 400 to reduce truncation."""
        stage2_start = next(
            i for i, l in enumerate(self.lines) if "Stage 2: Synthesis" in l
        )
        stage2_block = "\n".join(self.lines[stage2_start : stage2_start + 250])
        nums = re.findall(r"num_predict=(\d+)", stage2_block)
        for n in nums:
            self.assertGreaterEqual(
                int(n),
                400,
                f"Stage 2 num_predict={n} is too low (must be >= 400 to avoid truncation).",
            )


# ── Unit Tests with Mocking ───────────────────────────────────────────────────

class TestGenerateWholeImageReportImageGrounding(unittest.TestCase):
    """Unit tests verifying image is passed at every inference call."""

    def setUp(self):
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from src import report_service
        self.report_service = report_service
        self.img_bytes = _make_jpeg_bytes(seed=42)

    @patch("src.report_service._ollama_available", return_value=True)
    @patch("src.report_service._vision_model_available", return_value="medgemma:4b")
    @patch("src.report_service._ollama_generate_vision_with_metrics")
    def test_patient_style_calls_vision_twice(self, mock_vision, mock_model, mock_avail):
        """Patient style must call vision model twice (Stage 1 + Stage 2)."""
        mock_vision.side_effect = [
            _make_mock_vision_response(_stage1_obs()),
            _make_mock_vision_response(_patient_raw()),
        ]
        result = self.report_service.generate_whole_image_report(
            image_bytes=self.img_bytes,
            image_filename="test.jpg",
            style="patient",
        )
        self.assertEqual(
            mock_vision.call_count,
            2,
            f"Expected 2 vision calls for patient style, got {mock_vision.call_count}.",
        )

    @patch("src.report_service._ollama_available", return_value=True)
    @patch("src.report_service._vision_model_available", return_value="medgemma:4b")
    @patch("src.report_service._ollama_generate_vision_with_metrics")
    def test_research_style_calls_vision_twice(self, mock_vision, mock_model, mock_avail):
        """Research style must call vision model twice (Stage 1 + Stage 2)."""
        mock_vision.side_effect = [
            _make_mock_vision_response(_stage1_obs()),
            _make_mock_vision_response(_research_raw()),
        ]
        result = self.report_service.generate_whole_image_report(
            image_bytes=self.img_bytes,
            image_filename="test.jpg",
            style="research",
        )
        self.assertEqual(
            mock_vision.call_count,
            2,
            f"Expected 2 vision calls for research style, got {mock_vision.call_count}.",
        )

    @patch("src.report_service._ollama_available", return_value=True)
    @patch("src.report_service._vision_model_available", return_value="medgemma:4b")
    @patch("src.report_service._ollama_generate_vision_with_metrics")
    def test_every_vision_call_receives_image(self, mock_vision, mock_model, mock_avail):
        """Every vision call in both styles must include a non-empty images_b64 list."""
        for style, side_effects in [
            ("patient", [_make_mock_vision_response(_stage1_obs()), _make_mock_vision_response(_patient_raw())]),
            ("research", [_make_mock_vision_response(_stage1_obs()), _make_mock_vision_response(_research_raw())]),
        ]:
            mock_vision.reset_mock()
            mock_vision.side_effect = side_effects
            self.report_service.generate_whole_image_report(
                image_bytes=self.img_bytes,
                image_filename="test.jpg",
                style=style,
            )
            for i, c in enumerate(mock_vision.call_args_list):
                images_b64 = c.kwargs.get("images_b64") or (c.args[1] if len(c.args) > 1 else [])
                self.assertTrue(
                    len(images_b64) > 0 and images_b64[0],
                    f"[{style}] Call #{i+1} did not receive a non-empty image.",
                )

    @patch("src.report_service._ollama_available", return_value=True)
    @patch("src.report_service._vision_model_available", return_value="medgemma:4b")
    @patch("src.report_service._ollama_generate_vision_with_metrics")
    def test_no_text_only_model_calls_in_full_pipeline(self, mock_vision, mock_model, mock_avail):
        """_ollama_generate_text_with_metrics must NOT be called in full pipeline."""
        mock_vision.side_effect = [
            _make_mock_vision_response(_stage1_obs()),
            _make_mock_vision_response(_patient_raw()),
        ]
        with patch("src.report_service._ollama_generate_text_with_metrics") as mock_text:
            self.report_service.generate_whole_image_report(
                image_bytes=self.img_bytes,
                image_filename="test.jpg",
                style="patient",
            )
            self.assertEqual(
                mock_text.call_count,
                0,
                f"Text-only model was called {mock_text.call_count} times in patient pipeline — must be 0.",
            )

    def test_no_image_bytes_returns_error_not_fabricated_report(self):
        """Passing None image_bytes must return success=False — never a fabricated report."""
        result = self.report_service.generate_whole_image_report(
            image_bytes=None,
            image_filename="test.jpg",
            style="patient",
        )
        self.assertFalse(result["success"], "Must return success=False when no image is provided.")
        self.assertIsNone(result["report_text"], "report_text must be None when no image is provided.")
        self.assertFalse(
            result.get("image_input_used", True),
            "image_input_used must be False when no image is provided.",
        )

    def test_empty_image_bytes_returns_error(self):
        """Passing empty bytes must return success=False."""
        result = self.report_service.generate_whole_image_report(
            image_bytes=b"",
            image_filename="test.jpg",
            style="patient",
        )
        self.assertFalse(result["success"])
        self.assertIsNone(result["report_text"])


# ── Batch Isolation Tests ─────────────────────────────────────────────────────

class TestBatchPerImageIsolation(unittest.TestCase):
    """Batch must treat each image as a completely independent report."""

    def setUp(self):
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))

    @patch("src.report_service._ollama_available", return_value=True)
    @patch("src.report_service._vision_model_available", return_value="medgemma:4b")
    @patch("src.report_service._ollama_generate_vision_with_metrics")
    def test_each_batch_item_uses_its_own_image(self, mock_vision, mock_model, mock_avail):
        """Each batch item must receive its own image bytes at inference time."""
        from src.batch_service import BatchService

        img_a = _make_jpeg_bytes(seed=1)
        img_b = _make_jpeg_bytes(seed=2)

        hash_a = hashlib.sha256(img_a).hexdigest()
        hash_b = hashlib.sha256(img_b).hexdigest()

        # Side effects: stage1 + stage2 for each of the two images
        mock_vision.side_effect = [
            _make_mock_vision_response(_stage1_obs()),
            _make_mock_vision_response(_patient_raw()),
            _make_mock_vision_response(_stage1_obs()),
            _make_mock_vision_response(_patient_raw()),
        ]

        svc = BatchService(max_workers=1)
        batch = svc.create_batch([(("test_a.jpg", img_a)), ("test_b.jpg", img_b)], style="patient")

        # Wait for completion (max 10 seconds in unit test)
        deadline = time.time() + 10
        while time.time() < deadline:
            statuses = [it.status for it in batch.items]
            if all(s in ("completed", "failed") for s in statuses):
                break
            time.sleep(0.1)

        # Verify each item has the correct hash
        for item in batch.items:
            if item.filename == "test_a.jpg":
                self.assertEqual(
                    item.bytes_sha256,
                    hash_a,
                    "Batch item A must preserve hash of image A.",
                )
            elif item.filename == "test_b.jpg":
                self.assertEqual(
                    item.bytes_sha256,
                    hash_b,
                    "Batch item B must preserve hash of image B.",
                )

    @patch("src.report_service._ollama_available", return_value=True)
    @patch("src.report_service._vision_model_available", return_value="medgemma:4b")
    @patch("src.report_service._ollama_generate_vision_with_metrics")
    def test_batch_fails_on_empty_image(self, mock_vision, mock_model, mock_avail):
        """Batch item with empty image bytes must fail with a clear error, not fabricate."""
        from src.batch_service import BatchService

        mock_vision.return_value = _make_mock_vision_response(_patient_raw())

        svc = BatchService(max_workers=1)
        batch = svc.create_batch([("empty.jpg", b"")], style="patient")

        deadline = time.time() + 10
        while time.time() < deadline:
            if all(it.status in ("completed", "failed") for it in batch.items):
                break
            time.sleep(0.1)

        item = batch.items[0]
        self.assertEqual(item.status, "failed", "Empty-image batch item must have status=failed.")
        self.assertIsNone(item.report_text, "Empty-image batch item must not have a fabricated report.")


# ── Integration: Single vs Batch Parity ──────────────────────────────────────

class TestSingleAndBatchParity(unittest.TestCase):
    """Batch and single-image must use the same underlying function."""

    def test_batch_calls_generate_whole_image_report(self):
        """BatchService._process_single_item must call generate_whole_image_report."""
        from src.batch_service import BatchService
        import inspect
        src = inspect.getsource(BatchService._process_single_item)
        self.assertIn(
            "generate_whole_image_report",
            src,
            "BatchService._process_single_item must call generate_whole_image_report.",
        )

    def test_single_patient_calls_generate_whole_image_report(self):
        """patient_friendly_report must delegate to generate_whole_image_report."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from src import report_service
        src = inspect.getsource(report_service.patient_friendly_report)
        self.assertIn(
            "generate_whole_image_report",
            src,
            "patient_friendly_report must call generate_whole_image_report.",
        )

    def test_single_research_calls_generate_whole_image_report(self):
        """research_report must delegate to generate_whole_image_report."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from src import report_service
        src = inspect.getsource(report_service.research_report)
        self.assertIn(
            "generate_whole_image_report",
            src,
            "research_report must call generate_whole_image_report.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
