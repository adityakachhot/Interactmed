"""
tests/test_multipathology.py
============================
Comprehensive test suite for Multi-Pathology Grounding, Capability Configuration,
Checkpoint Selection, Crop Alignment, Unsupported Finding Detection, and
Evidence-Based Region Report Hypothesis Testing.
"""

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.grounding_service import (
    GroundingService,
    get_grounding_service,
    load_grounding_capabilities,
    resolve_query_capability,
    MULTIPATHOLOGY_CKPT,
    BASELINE_CKPT,
    FINETUNED_CKPT,
)
from src.report_service import clean_and_structure_regional_report, extract_crop
from backend.main import app

client = TestClient(app)


def _make_dummy_image_bytes(width=512, height=512, color=(128, 128, 128)) -> bytes:
    """Create in-memory PNG bytes for testing."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestCapabilityMappingsAndAliases:
    """Test transparent capability configuration and query canonicalization."""

    def test_load_capabilities_schema(self):
        caps_data = load_grounding_capabilities()
        assert "capabilities" in caps_data
        assert "unsupported_defaults" in caps_data

        caps = caps_data["capabilities"]
        expected_categories = [
            "cardiomegaly",
            "pneumothorax",
            "pleural effusion",
            "pneumonia",
            "consolidation",
            "atelectasis",
            "pulmonary edema",
            "lung opacity",
        ]
        for cat in expected_categories:
            assert cat in caps, f"Missing capability mapping for {cat}"
            info = caps[cat]
            assert info["status"] == "validated"
            assert "validation_evidence" in info
            ev = info["validation_evidence"]
            assert ev["mean_iou"] > 0
            assert ev["recall_50"] >= 0

    def test_resolve_exact_canonical_queries(self):
        for q in ["cardiomegaly", "pneumothorax", "pleural effusion", "pneumonia", "consolidation", "atelectasis"]:
            res = resolve_query_capability(q)
            assert res["status"] == "validated"
            assert res["canonical_query"] == q
            assert res["warning"] is None

    def test_resolve_aliases(self):
        # Cardiomegaly alias
        res = resolve_query_capability("enlarged cardiac silhouette")
        assert res["status"] == "validated"
        assert res["canonical_query"] == "cardiomegaly"
        assert res["category"] == "Heart"

        # Pleural effusion alias
        res = resolve_query_capability("blunted costophrenic angle")
        assert res["status"] == "validated"
        assert res["canonical_query"] == "pleural effusion"
        assert res["category"] == "Pleura"

        # Pneumothorax alias
        res = resolve_query_capability("apical pneumothorax")
        assert res["status"] == "validated"
        assert res["canonical_query"] == "pneumothorax"

        # Edema alias
        res = resolve_query_capability("peribronchial cuffing")
        assert res["status"] == "validated"
        assert res["canonical_query"] == "pulmonary edema"

    def test_resolve_unsupported_query(self):
        unsupported_queries = ["rib fracture", "pulmonary nodule", "cholecystectomy clip", "scoliosis"]
        for q in unsupported_queries:
            res = resolve_query_capability(q)
            assert res["status"] == "unsupported"
            assert res["warning"] is not None
            assert "unsupported" in res["warning"].lower()
            assert res["validation_evidence"] is None


class TestCheckpointSelectionAndRouting:
    """Verify that checkpoint selection routes according to capability and configuration."""

    def test_auto_checkpoint_selection(self):
        svc = get_grounding_service("auto")
        assert svc is not None
        assert svc.ckpt_path == MULTIPATHOLOGY_CKPT
        assert "run_02_continuation_to_6ep" in str(svc.ckpt_path)
        assert "Task 3 Epoch 5" in svc.model_version

    def test_missing_checkpoint_fails_clearly(self):
        with pytest.raises(FileNotFoundError) as exc_info:
            GroundingService(checkpoint="nonexistent_checkpoint_path_12345.pth")
        assert "Requested checkpoint not found" in str(exc_info.value)

    def test_explicit_multipathology_checkpoint(self):
        svc = GroundingService(checkpoint="multipathology")
        assert svc.ckpt_path == MULTIPATHOLOGY_CKPT

    def test_explicit_task2_checkpoint(self):
        svc = GroundingService(checkpoint="task2")
        from src.grounding_service import MULTIPATHOLOGY_TASK2_CKPT
        assert svc.ckpt_path == MULTIPATHOLOGY_TASK2_CKPT
        assert "run_01_multicxr" in str(svc.ckpt_path)

    def test_explicit_baseline_checkpoint(self):
        svc = GroundingService(checkpoint="baseline")
        assert svc.ckpt_path == BASELINE_CKPT

    def test_explicit_finetuned_checkpoint(self):
        svc = GroundingService(checkpoint="finetuned")
        assert svc.ckpt_path == FINETUNED_CKPT


class TestInferenceAndCoordinateTransforms:
    """Verify inference pipeline, coordinate consistency, and crop extraction."""

    def test_inference_returns_valid_pixel_box(self):
        svc = get_grounding_service("auto")
        img_bytes = _make_dummy_image_bytes(width=600, height=400)
        res = svc.predict(img_bytes, query="cardiomegaly", image_filename="synthetic.png")

        assert res["success"] is True
        assert res["error"] is None
        assert res["canonical_query"] == "cardiomegaly"
        assert res["capability_status"] == "validated"

        # Verify pixel box bounds
        x1, y1, x2, y2 = res["pred_box_pixel"]
        assert 0.0 <= x1 < x2 <= 600.0
        assert 0.0 <= y1 < y2 <= 400.0

        # Verify canonical normalized box bounds
        cx, cy, w, h = res["pred_box_norm"]
        assert 0.0 <= cx <= 1.0
        assert 0.0 <= cy <= 1.0
        assert 0.0 < w <= 1.0
        assert 0.0 < h <= 1.0

    def test_unsupported_query_returns_warning_flag(self):
        svc = get_grounding_service("auto")
        img_bytes = _make_dummy_image_bytes(width=500, height=500)
        res = svc.predict(img_bytes, query="rib fracture", image_filename="synthetic.png")

        assert res["success"] is True
        assert res["capability_status"] == "unsupported"
        assert res["warning"] is not None

    def test_crop_extraction_geometry(self):
        img_bytes = _make_dummy_image_bytes(width=800, height=600)
        box = [100.0, 150.0, 300.0, 350.0]
        crop = extract_crop(img_bytes, "test.png", box, margin_px=40)

        assert crop is not None
        assert isinstance(crop, np.ndarray)
        # Expected crop size: width = (300-100) + 2*40 = 280, height = (350-150) + 2*40 = 280
        assert crop.shape == (280, 280, 3)

    def test_crop_margin_clamping_at_boundaries(self):
        img_bytes = _make_dummy_image_bytes(width=500, height=500)
        box = [10.0, 10.0, 490.0, 490.0]
        crop = extract_crop(img_bytes, "test.png", box, margin_px=50)

        assert crop is not None
        # Clamped to image dimensions (500x500)
        assert crop.shape == (500, 500, 3)

    def test_model_version_in_prediction_response(self):
        svc = get_grounding_service("auto")
        img_bytes = _make_dummy_image_bytes(width=400, height=400)
        res = svc.predict(img_bytes, query="pneumonia", image_filename="synthetic.png")
        assert res["success"] is True
        assert "model_version" in res
        assert "Task 3 Epoch 5" in res["model_version"]

    def test_clamped_coordinates_stay_inside_image(self):
        svc = get_grounding_service("auto")
        img_bytes = _make_dummy_image_bytes(width=300, height=200)
        res = svc.predict(img_bytes, query="pleural effusion", image_filename="synthetic.png")
        assert res["success"] is True
        x1, y1, x2, y2 = res["pred_box_pixel"]
        assert 0.0 <= x1 <= 300.0
        assert 0.0 <= y1 <= 200.0
        assert x1 <= x2 <= 300.0
        assert y1 <= y2 <= 200.0


class TestRegionalReportHypothesisStates:
    """Verify that region reports preserve all 4 clinical assessment states without keyword bias."""

    def test_preserves_supported_state(self):
        raw_text = """
TARGET FINDING:
Pneumothorax

IMAGE OBSERVATIONS:
A thin visceral pleural line is visible in the right lung apex with absence of peripheral lung markings.

ASSESSMENT AND UNCERTAINTY:
Supported: Apical pleural separation demonstrates small right apical pneumothorax.

LIMITATIONS:
Upright inspiratory view.
"""
        cleaned, warnings_list = clean_and_structure_regional_report(raw_text, phrase="Pneumothorax")
        assert "ASSESSMENT AND UNCERTAINTY:\nSupported:" in cleaned

    def test_preserves_not_supported_state(self):
        raw_text = """
TARGET FINDING:
Cardiomegaly

IMAGE OBSERVATIONS:
Cardiac borders are sharp and transverse cardiac diameter does not exceed 45% of maximum thoracic width.

ASSESSMENT AND UNCERTAINTY:
Not supported: Cardiac silhouette size is within normal limits.

LIMITATIONS:
Portable AP projection.
"""
        cleaned, warnings_list = clean_and_structure_regional_report(raw_text, phrase="Cardiomegaly")
        assert "ASSESSMENT AND UNCERTAINTY:\nNot supported:" in cleaned

    def test_preserves_indeterminate_state(self):
        raw_text = """
TARGET FINDING:
Pleural effusion

IMAGE OBSERVATIONS:
Hazy density at the right lateral costophrenic angle; sulcus is partially obscured.

ASSESSMENT AND UNCERTAINTY:
Indeterminate: Suboptimal inspiration and mild patient rotation obscure the costophrenic sulcus.

LIMITATIONS:
Submaximal inspiration depth.
"""
        cleaned, warnings_list = clean_and_structure_regional_report(raw_text, phrase="Pleural effusion")
        assert "ASSESSMENT AND UNCERTAINTY:\nIndeterminate:" in cleaned

    def test_preserves_not_assessable_state(self):
        raw_text = """
TARGET FINDING:
Left lower lobe consolidation

IMAGE OBSERVATIONS:
Marked patient rotation and motion artifact completely obscure the left retrocardiac lung field.

ASSESSMENT AND UNCERTAINTY:
Not assessable: Region cannot be evaluated due to severe motion artifact and suboptimal positioning.

LIMITATIONS:
Severe motion degradation.
"""
        cleaned, warnings_list = clean_and_structure_regional_report(raw_text, phrase="Left lower lobe consolidation")
        assert "ASSESSMENT AND UNCERTAINTY:\nNot assessable:" in cleaned


class TestAPIEndpointContracts:
    """Verify FastAPI routes for capabilities and multi-pathology grounding."""

    def test_get_health_endpoint_reports_model_version(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["grounding_model_loaded"] is True
        assert "grounding_model_version" in data
        assert "Task 3 Epoch 5" in (data["grounding_model_version"] or "")

    def test_get_grounding_capabilities_endpoint(self):
        resp = client.get("/grounding/capabilities")
        assert resp.status_code == 200
        data = resp.json()
        assert "capabilities" in data
        assert "cardiomegaly" in data["capabilities"]
        assert "pneumonia" in data["capabilities"]
        assert "run_02_continuation_to_6ep" in data["capabilities"]["cardiomegaly"]["selected_checkpoint"]

    def test_post_ground_auto_endpoint(self):
        dummy_img = _make_dummy_image_bytes(300, 300)
        resp = client.post(
            "/ground",
            files={"file": ("test.png", dummy_img, "image/png")},
            data={"query": "cardiomegaly", "checkpoint": "auto"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["canonical_query"] == "cardiomegaly"
        assert data["capability_status"] == "validated"
        assert data["pred_box_pixel"] is not None
        assert "model_version" in data
        assert "Task 3 Epoch 5" in data["model_version"]

    def test_post_ground_explicit_task2_endpoint(self):
        dummy_img = _make_dummy_image_bytes(300, 300)
        resp = client.post(
            "/ground",
            files={"file": ("test.png", dummy_img, "image/png")},
            data={"query": "cardiomegaly", "checkpoint": "task2"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "Task 2 Epoch 3" in (data.get("model_version") or "")

    def test_post_ground_unsupported_query_endpoint(self):
        dummy_img = _make_dummy_image_bytes(300, 300)
        resp = client.post(
            "/ground",
            files={"file": ("test.png", dummy_img, "image/png")},
            data={"query": "rib fracture", "checkpoint": "auto"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["capability_status"] == "unsupported"
        assert data["warning"] is not None
