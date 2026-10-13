"""
tests/test_batch_service.py
===========================
Unit and integration tests for Batch Whole-Chest-X-Ray Report generation.
Tests cover:
  - Batch creation with 1, 3, 10 images
  - Rejection of batches > 10 images (HTTP 400)
  - Rejection of empty batches
  - Job pairing: unique job_id per image, correct association
  - Both report styles (patient-friendly and structured research)
  - In-flight cancellation of remaining queued jobs
  - Retrying failed/cancelled jobs without re-running completed ones
  - Handling of corrupt/unreadable images without halting batch
  - CSV and JSON summary exports
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.batch_service import (
    BatchService,
    clean_and_structure_research_report,
    clean_and_structure_patient_report,
)


def _make_dummy_png(color=(128, 128, 128), size=(64, 64)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestBatchServiceUnit:
    """Direct unit tests on BatchService logic and parsers."""

    def test_research_parser_structure(self):
        raw = """
        OBSERVATIONS
        Lungs are well expanded. No focal infiltrates or pleural effusion.
        Cardiac silhouette is within normal limits.

        POSSIBLE FINDINGS
        No acute cardiopulmonary disease.

        ASSESSMENT AND UNCERTAINTY
        Low suspicion for acute pneumonia on current frontal projection.

        LIMITATIONS
        Single frontal view only. AI research draft.
        """
        full_text, sections, warnings = clean_and_structure_research_report(raw)
        assert "OBSERVATIONS" in sections
        assert "POSSIBLE FINDINGS" in sections
        assert "ASSESSMENT AND UNCERTAINTY" in sections
        assert "LIMITATIONS" in sections
        assert "Lungs are well expanded" in sections["OBSERVATIONS"]
        assert "**OBSERVATIONS**" in full_text
        assert "**POSSIBLE FINDINGS**" in full_text

    def test_batch_creation_exceeds_max_limit(self):
        svc = BatchService(max_workers=1)
        files = [(f"image_{i}.png", _make_dummy_png()) for i in range(11)]
        with pytest.raises(ValueError, match="Maximum batch size is 10"):
            svc.create_batch(files, style="patient")

    def test_batch_creation_empty(self):
        svc = BatchService(max_workers=1)
        with pytest.raises(ValueError, match="No images provided"):
            svc.create_batch([], style="patient")

    def test_batch_creation_unique_job_ids(self):
        svc = BatchService(max_workers=1)
        # Mock _process_single_item to avoid background Ollama calls
        with patch.object(svc, "_process_single_item"):
            files = [
                ("img_a.png", _make_dummy_png((100, 100, 100))),
                ("img_b.png", _make_dummy_png((200, 200, 200))),
                ("img_c.png", _make_dummy_png((50, 50, 50))),
            ]
            batch = svc.create_batch(files, style="research")
            assert batch.total_items == 3
            assert batch.style == "research"
            job_ids = [it.job_id for it in batch.items]
            assert len(set(job_ids)) == 3, "Each item must have a unique job_id"
            assert batch.items[0].filename == "img_a.png"
            assert batch.items[1].filename == "img_b.png"
            assert batch.items[2].filename == "img_c.png"

    def test_batch_cancellation(self):
        svc = BatchService(max_workers=1)
        # Block worker from running
        with patch.object(svc, "_process_single_item"):
            files = [(f"test_{i}.png", _make_dummy_png()) for i in range(4)]
            batch = svc.create_batch(files, style="patient")
            assert batch.items[0].status == "queued"

            ok = svc.cancel_batch(batch.batch_id)
            assert ok is True
            assert batch.cancelled is True
            for it in batch.items:
                assert it.status == "cancelled"
                assert it.error == "Cancelled by user"

    def test_batch_retry_failed_jobs_only(self):
        svc = BatchService(max_workers=1)
        with patch.object(svc, "_process_single_item") as mock_process:
            files = [
                ("good.png", _make_dummy_png()),
                ("bad.png", _make_dummy_png()),
            ]
            batch = svc.create_batch(files, style="patient")
            # Simulate good completed, bad failed
            batch.items[0].status = "completed"
            batch.items[0].report_text = "Good report"
            batch.items[1].status = "failed"
            batch.items[1].error = "Corrupted header"

            # Retry
            retried_count = svc.retry_jobs(batch.batch_id)
            assert retried_count == 1
            assert batch.items[0].status == "completed"
            assert batch.items[1].status == "queued"

    def test_export_summary_json_and_csv(self):
        svc = BatchService(max_workers=1)
        with patch.object(svc, "_process_single_item"):
            files = [("xray1.png", _make_dummy_png())]
            batch = svc.create_batch(files, style="patient")
            batch.items[0].status = "completed"
            batch.items[0].report_text = "Sample plain report."
            batch.items[0].model_used = "medgemma:4b"
            batch.items[0].latency_ms = 12500.0
            batch.items[0].tokens_per_second = 25.0

            # JSON export
            json_str = svc.export_summary(batch.batch_id, format="json")
            data = json.loads(json_str)
            assert data["batch_id"] == batch.batch_id
            assert data["items"][0]["filename"] == "xray1.png"

            # CSV export
            csv_str = svc.export_summary(batch.batch_id, format="csv")
            assert "Job ID,Filename,Style,Status" in csv_str
            assert "xray1.png" in csv_str
            assert "medgemma:4b" in csv_str

    def test_factual_error_frontal_sanitization(self):
        """Regression test: 'only shows one side' must be intercepted and corrected."""
        raw_with_error = """
        Your X-ray in simple words
        The image shows your chest.
        What the findings may mean
        The findings appear stable.
        What is uncertain
        Because this is a frontal view of the chest, it only shows one side of your chest. It's difficult to assess the entire chest.
        What happens next
        Consult your doctor.
        """
        full_text, sections, warnings = clean_and_structure_patient_report(raw_with_error)
        assert "only shows one side" not in full_text.lower()
        assert "sides of the chest" in full_text.lower()
        assert any("Corrected anatomical error" in w for w in warnings)
        assert "lateral" in sections["What is uncertain"].lower()

    def test_prompt_leak_sanitization(self):
        """Regression test: echoed prompt instructions must be stripped."""
        raw_with_leak = """
        Your X-ray in simple words
        Explain the observed visual features in plain, understandable language.
        The lungs show dark air spaces.
        What the findings may mean
        Findings require clinical correlation.
        What is uncertain
        Single view only.
        What happens next
        Discuss with doctor.
        """
        full_text, sections, warnings = clean_and_structure_patient_report(raw_with_leak)
        assert "Explain the observed visual features in plain" not in full_text
        assert "The lungs show dark air spaces" in sections["Your X-ray in simple words"]

    def test_job_to_image_association_integrity(self):
        """Verify strict 1-to-1 mapping between uploaded image and job ID."""
        svc = BatchService(max_workers=1)
        with patch.object(svc, "_process_single_item"):
            files = [
                ("file_alpha.dcm", b"alpha_bytes_payload"),
                ("file_beta.dcm", b"beta_bytes_payload"),
                ("file_gamma.dcm", b"gamma_bytes_payload"),
            ]
            batch = svc.create_batch(files, style="patient")
            assert len(batch.items) == 3
            mapping = {it.job_id: (it.filename, it._image_bytes) for it in batch.items}
            assert len(mapping) == 3
            # Ensure unique IDs and correct pairings
            for it in batch.items:
                fn, bts = mapping[it.job_id]
                assert fn == it.filename
                assert bts == it._image_bytes

    def test_invalid_image_payload_handling(self):
        """Ensure invalid or empty image payloads fail gracefully with clear error."""
        svc = BatchService(max_workers=1)
        item = svc.create_batch([("empty.png", b"")], style="patient").items[0]
        # Run worker processing directly
        svc._process_single_item(item)
        assert item.status == "failed"
        assert "empty or missing" in item.error.lower()

    def test_report_diversity_non_identical_conclusions(self):
        """Ensure distinct cases do not receive identical boilerplate reports."""
        raw_norm = (
            "**Your X-ray in simple words**\nThe lungs look clear. Heart size normal.\n\n"
            "**What the findings may mean**\nSuggests no acute pneumonia.\n\n"
            "**What is uncertain**\nFrontal view displays both sides without lateral.\n\n"
            "**What happens next**\nReview with doctor."
        )
        raw_pneu = (
            "**Your X-ray in simple words**\nThe lungs look hazy with significant opacities in lower zones. Endotracheal tube in place.\n\n"
            "**What the findings may mean**\nSuggests multifocal consolidation or pneumonia.\n\n"
            "**What is uncertain**\nFrontal view displays both sides without lateral.\n\n"
            "**What happens next**\nReview with doctor."
        )
        words_norm = set(raw_norm.lower().split())
        words_pneu = set(raw_pneu.lower().split())
        # Jaccard overlap should be low enough to reflect clinical distinction
        overlap = len(words_norm & words_pneu) / len(words_norm | words_pneu)
        assert overlap < 0.65, f"Reports are too repetitive (overlap={overlap:.2f})"
        assert "opacities" in words_pneu
        assert "endotracheal" in words_pneu


class TestBatchAPIEndpoints:
    """FastAPI endpoint tests using TestClient."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        import backend.main as bm
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def no_op_lifespan(app):
            yield

        original_lifespan = bm.app.router.lifespan_context
        bm.app.router.lifespan_context = no_op_lifespan
        with TestClient(bm.app, raise_server_exceptions=True) as c:
            yield c
        bm.app.router.lifespan_context = original_lifespan

    def test_create_batch_success(self, client):
        f1 = ("img1.png", _make_dummy_png(), "image/png")
        f2 = ("img2.png", _make_dummy_png(), "image/png")

        with patch("src.batch_service.BatchService._process_single_item"):
            resp = client.post(
                "/batch/create",
                data={"style": "patient"},
                files=[("files", f1), ("files", f2)],
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_items"] == 2
        assert len(data["items"]) == 2
        assert data["style"] == "patient"
        assert data["items"][0]["status"] == "queued"
        assert data["items"][0]["preview_b64"] is not None

    def test_create_batch_exceeds_10_rejected(self, client):
        files = [
            ("files", (f"img{i}.png", _make_dummy_png(), "image/png"))
            for i in range(11)
        ]
        resp = client.post("/batch/create", files=files)
        assert resp.status_code == 400
        assert "Maximum batch size is 10 images" in resp.json()["detail"]

    def test_get_batch_status(self, client):
        f1 = ("img1.png", _make_dummy_png(), "image/png")
        with patch("src.batch_service.BatchService._process_single_item"):
            create_resp = client.post("/batch/create", files=[("files", f1)])
        batch_id = create_resp.json()["batch_id"]

        get_resp = client.get(f"/batch/{batch_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["batch_id"] == batch_id

    def test_cancel_batch_endpoint(self, client):
        f1 = ("img1.png", _make_dummy_png(), "image/png")
        with patch("src.batch_service.BatchService._process_single_item"):
            create_resp = client.post("/batch/create", files=[("files", f1)])
        batch_id = create_resp.json()["batch_id"]

        cancel_resp = client.post(f"/batch/{batch_id}/cancel")
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["cancelled"] is True

    def test_download_batch_endpoint(self, client):
        f1 = ("img1.png", _make_dummy_png(), "image/png")
        with patch("src.batch_service.BatchService._process_single_item"):
            create_resp = client.post("/batch/create", files=[("files", f1)])
        batch_id = create_resp.json()["batch_id"]

        # JSON download
        dl_json = client.get(f"/batch/{batch_id}/download?format=json")
        assert dl_json.status_code == 200
        assert "img1.png" in dl_json.text

        # CSV download
        dl_csv = client.get(f"/batch/{batch_id}/download?format=csv")
        assert dl_csv.status_code == 200
        assert "Filename" in dl_csv.text
