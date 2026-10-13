"""
tests/test_backend_api.py
=========================
Automated tests for the InteractMed-VLM FastAPI backend.

Tests cover:
  - /health endpoint
  - /ground endpoint with valid and invalid inputs
  - /report/region validation
  - /report/study validation
  - Error handling: missing fields, bad files, oversized uploads

Uses httpx.AsyncClient with TestClient (no live model required for validation tests).
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Fixture: mock the grounding service to avoid loading 2GB checkpoint ───────

@pytest.fixture(autouse=True)
def mock_grounding_service():
    """
    Mock GroundingService so tests run without GPU/checkpoint.
    Real integration tests should use the actual service.
    """
    mock_svc = MagicMock()
    mock_svc._loaded = True
    mock_svc._load_error = None
    mock_svc.status = {
        "loaded": True,
        "error": None,
        "checkpoint": "checkpoints/finetuned_rsna/run_01_3ep/best_model_val_iou.pth",
        "device": "mps",
    }
    mock_svc.predict.return_value = {
        "success": True,
        "error": None,
        "pred_box_norm": [0.45, 0.55, 0.30, 0.25],
        "pred_box_pixel": [200.0, 300.0, 500.0, 600.0],
        "pred_box_canvas": [192.0, 320.0, 480.0, 608.0],
        "orig_h": 1024,
        "orig_w": 1024,
        "latency_ms": 145.0,
        "query": "pneumonia",
        "checkpoint": "best_model_val_iou.pth",
    }

    with patch("src.grounding_service.get_grounding_service", return_value=mock_svc), \
         patch("backend.main.get_grounding_service", return_value=mock_svc):
        yield mock_svc


@pytest.fixture
def client(mock_grounding_service):
    """Create a FastAPI test client with startup/shutdown lifespan skipped."""
    from fastapi.testclient import TestClient
    import backend.main as bm

    # Patch lifespan so model loading is skipped
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def no_op_lifespan(app):
        yield

    original_lifespan = bm.app.router.lifespan_context
    bm.app.router.lifespan_context = no_op_lifespan
    with TestClient(bm.app, raise_server_exceptions=True) as c:
        yield c
    bm.app.router.lifespan_context = original_lifespan


@pytest.fixture
def small_png_bytes():
    """Create a minimal valid 64×64 PNG as bytes."""
    from PIL import Image
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "grounding_model_loaded" in data
        assert "ollama_available" in data
        assert "timestamp" in data

    def test_health_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"


# ── /ground ───────────────────────────────────────────────────────────────────

class TestGround:
    def test_ground_returns_200_with_valid_png(self, client, small_png_bytes):
        with patch("backend.main.draw_overlay") as mock_overlay, \
             patch("backend.main.image_to_base64_jpeg", return_value="fakeb64"):
            mock_img = MagicMock()
            mock_overlay.return_value = mock_img
            resp = client.post(
                "/ground",
                data={"query": "pneumonia", "checkpoint": "finetuned", "return_overlay": "false"},
                files={"file": ("test.png", io.BytesIO(small_png_bytes), "image/png")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["query"] == "pneumonia"
        assert data["pred_box_pixel"] is not None
        assert len(data["pred_box_pixel"]) == 4

    def test_ground_invalid_checkpoint(self, client, small_png_bytes):
        resp = client.post(
            "/ground",
            data={"query": "pneumonia", "checkpoint": "invalid_ckpt"},
            files={"file": ("test.png", io.BytesIO(small_png_bytes), "image/png")},
        )
        assert resp.status_code == 400

    def test_ground_empty_query(self, client, small_png_bytes):
        resp = client.post(
            "/ground",
            data={"query": "   ", "checkpoint": "finetuned"},
            files={"file": ("test.png", io.BytesIO(small_png_bytes), "image/png")},
        )
        assert resp.status_code == 400

    def test_ground_unsupported_filetype(self, client):
        resp = client.post(
            "/ground",
            data={"query": "pneumonia", "checkpoint": "finetuned"},
            files={"file": ("test.gif", io.BytesIO(b"GIF89a"), "image/gif")},
        )
        assert resp.status_code == 415

    def test_ground_baseline_checkpoint(self, client, small_png_bytes):
        with patch("backend.main.draw_overlay") as mock_overlay, \
             patch("backend.main.image_to_base64_jpeg", return_value="b64"):
            mock_overlay.return_value = MagicMock()
            resp = client.post(
                "/ground",
                data={"query": "pleural effusion", "checkpoint": "baseline", "return_overlay": "false"},
                files={"file": ("cxr.png", io.BytesIO(small_png_bytes), "image/png")},
            )
        assert resp.status_code == 200

    def test_ground_response_contains_latency(self, client, small_png_bytes):
        resp = client.post(
            "/ground",
            data={"query": "pneumonia", "return_overlay": "false"},
            files={"file": ("test.png", io.BytesIO(small_png_bytes), "image/png")},
        )
        data = resp.json()
        assert "latency_ms" in data
        assert isinstance(data["latency_ms"], (int, float))


# ── /report/region ────────────────────────────────────────────────────────────

class TestReportRegion:
    def test_region_report_returns_200(self, client, small_png_bytes):
        with patch("backend.main.region_report") as mock_rr:
            mock_rr.return_value = {
                "success": True,
                "error": None,
                "report_text": "Draft finding for pneumonia region. Clinical correlation required.",
                "crop_info": {"box_pixel_xyxy": [100, 100, 400, 400]},
                "latency_ms": 2500.0,
                "model_used": "llama3",
            }
            resp = client.post(
                "/report/region",
                data={
                    "query": "pneumonia",
                    "box_pixel_xyxy": "[100, 100, 400, 400]",
                    "orig_h": "1024",
                    "orig_w": "1024",
                },
                files={"file": ("test.png", io.BytesIO(small_png_bytes), "image/png")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "report_text" in data

    def test_region_report_invalid_box(self, client, small_png_bytes):
        resp = client.post(
            "/report/region",
            data={
                "query": "pneumonia",
                "box_pixel_xyxy": "[100, 100]",  # only 2 elements
                "orig_h": "1024",
                "orig_w": "1024",
            },
            files={"file": ("test.png", io.BytesIO(small_png_bytes), "image/png")},
        )
        assert resp.status_code == 400

    def test_region_report_zero_image_dims(self, client, small_png_bytes):
        resp = client.post(
            "/report/region",
            data={
                "query": "pneumonia",
                "box_pixel_xyxy": "[100, 100, 400, 400]",
                "orig_h": "0",
                "orig_w": "0",
            },
            files={"file": ("test.png", io.BytesIO(small_png_bytes), "image/png")},
        )
        assert resp.status_code == 400


# ── /report/patient-friendly & /report/study ─────────────────────────────────

class TestReportPatientFriendly:
    def test_patient_friendly_report_returns_200(self, client, small_png_bytes):
        with patch("backend.main.patient_friendly_report") as mock_pfr:
            mock_pfr.return_value = {
                "success": True,
                "error": None,
                "report_text": "**Your X-ray in simple words**\nThe lungs appear clear.\n\n**What the findings may mean**\nNo active pneumonia is seen.\n\n**What is uncertain**\nSingle 2D view.\n\n**What happens next**\nDiscuss with your clinician.",
                "sections": {
                    "Your X-ray in simple words": "The lungs appear clear.",
                    "What the findings may mean": "No active pneumonia is seen.",
                    "What is uncertain": "Single 2D view.",
                    "What happens next": "Discuss with your clinician.",
                },
                "latency_ms": 1200.0,
                "model_used": "medgemma:4b",
                "image_input_used": True,
                "validation_warnings": [],
                "validation_flagged": False,
            }
            resp = client.post(
                "/report/patient-friendly",
                data={"reason_for_xray": "Routine checkup"},
                files={"file": ("test.png", io.BytesIO(small_png_bytes), "image/png")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "Your X-ray in simple words" in data["sections"]

    def test_patient_friendly_report_optional_fields(self, client, small_png_bytes):
        """Reason and history are completely optional."""
        with patch("backend.main.patient_friendly_report") as mock_pfr:
            mock_pfr.return_value = {
                "success": True,
                "error": None,
                "report_text": "Plain explanation.",
                "sections": {},
                "latency_ms": 1000.0,
                "model_used": "medgemma:4b",
                "image_input_used": True,
                "validation_warnings": [],
                "validation_flagged": False,
            }
            resp = client.post(
                "/report/patient-friendly",
                data={},
                files={"file": ("test.png", io.BytesIO(small_png_bytes), "image/png")},
            )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_legacy_study_endpoint_alias(self, client, small_png_bytes):
        with patch("backend.main.patient_friendly_report") as mock_pfr:
            mock_pfr.return_value = {
                "success": True,
                "error": None,
                "report_text": "Plain explanation.",
                "sections": {},
                "latency_ms": 1000.0,
                "model_used": "medgemma:4b",
                "image_input_used": True,
                "validation_warnings": [],
                "validation_flagged": False,
            }
            resp = client.post(
                "/report/study",
                data={"clinical_indication": "Persistent cough"},
                files={"file": ("test.png", io.BytesIO(small_png_bytes), "image/png")},
            )
        assert resp.status_code == 200
        assert resp.json()["success"] is True


# ── Coordinate conversion tests (unit) ────────────────────────────────────────

class TestCoordinateConversion:
    """Unit tests for the letterbox coordinate math used in the grounding service."""

    def test_letterbox_params_square_image(self):
        from scripts.evaluate_rsna import compute_letterbox_params
        scale, pad_x, pad_y = compute_letterbox_params(1024, 1024, canvas=640)
        assert abs(scale - 640 / 1024) < 1e-5
        assert pad_x == 0
        assert pad_y == 0

    def test_letterbox_params_tall_image(self):
        from scripts.evaluate_rsna import compute_letterbox_params
        # 512×1024 (W×H) — height is the long side
        scale, pad_x, pad_y = compute_letterbox_params(1024, 512, canvas=640)
        assert abs(scale - 640 / 1024) < 1e-5
        # Width after scaling: int(round(512 * scale)) = 320, pad = (640-320)/2 = 160
        assert pad_x == 160
        assert pad_y == 0

    def test_pred_norm_to_pixel_identity_square(self):
        """A centered 50% box on a square image should map back correctly."""
        from scripts.evaluate_rsna import pred_norm_to_pixel
        # cx=0.5, cy=0.5, w=0.5, h=0.5 → x1=192, y1=192, x2=448, y2=448 on 640 canvas
        # On square 1024×1024: scale=640/1024=0.625, pad_x=pad_y=0
        # canvas: x1=(0.5-0.25)*640=160, y1=160, x2=480, y2=480
        # pixel: (160-0)/0.625=256, (480-0)/0.625=768
        pred = pred_norm_to_pixel(
            pred_cx_cy_wh_norm=[0.5, 0.5, 0.5, 0.5],
            orig_h=1024, orig_w=1024, canvas=640
        )
        assert abs(pred[0] - 256.0) < 1.0   # x1
        assert abs(pred[2] - 768.0) < 1.0   # x2

    def test_iou_identical_boxes(self):
        from scripts.evaluate_rsna import compute_iou
        box = [100.0, 100.0, 300.0, 300.0]
        assert abs(compute_iou(box, box) - 1.0) < 1e-6

    def test_iou_no_overlap(self):
        from scripts.evaluate_rsna import compute_iou
        a = [0.0, 0.0, 100.0, 100.0]
        b = [200.0, 200.0, 400.0, 400.0]
        assert compute_iou(a, b) == 0.0
