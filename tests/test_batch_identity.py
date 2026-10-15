"""
tests/test_batch_identity.py
============================
Regression test suite for Task 5B:
  - Per-image identity verification (job ID, bytes SHA-256, pixel SHA-256, dimensions).
  - Inference-boundary validation (detects tampered/mismatched image bytes).
  - Isolation & state independence (failed jobs do not inherit reports, duplicate files handled cleanly).
  - Single-image vs Batch pipeline equivalence (same generator, prompt schema, and normalization).
  - 7-section structured research report parsing and schema integrity.
  - Triage rule integration on structured findings without false-normal conversion.
  - Elimination of raw markdown UI artifacts.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.batch_service import (
    BatchService,
    BatchItem,
    clean_and_structure_research_report,
    clean_and_structure_patient_report,
)
from src.report_service import (
    generate_whole_image_report,
    research_report,
    patient_friendly_report,
    PROMPT_VERSION_RESEARCH,
    PROMPT_VERSION_PATIENT,
)
from src.triage_service import get_triage_service


def _create_unique_image(color=(128, 128, 128), size=(100, 100), text_seed: int = 0) -> bytes:
    """Create a unique RGB PNG image with guaranteed distinct pixels and bytes."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, :] = color
    # Add unique pixel variation
    arr[0, 0, :] = text_seed % 255
    arr[1, 1, :] = (text_seed * 7) % 255
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestPerImageIdentityVerification:
    """Tests ensuring each batch item retains an immutable, independent identity."""

    def test_three_distinct_images_create_unique_payloads(self):
        svc = BatchService(max_workers=1)
        with patch.object(svc, "_process_single_item"):
            img1 = _create_unique_image(color=(255, 0, 0), size=(120, 80), text_seed=1)
            img2 = _create_unique_image(color=(0, 255, 0), size=(150, 100), text_seed=2)
            img3 = _create_unique_image(color=(0, 0, 255), size=(200, 150), text_seed=3)

            files = [("scan_a.png", img1), ("scan_b.png", img2), ("scan_c.png", img3)]
            batch = svc.create_batch(files, style="research")

            assert len(batch.items) == 3
            job_ids = [it.job_id for it in batch.items]
            assert len(set(job_ids)) == 3, "All job IDs must be unique"

            # Check SHA-256 and dimensions
            assert batch.items[0].bytes_sha256 == hashlib.sha256(img1).hexdigest()
            assert batch.items[1].bytes_sha256 == hashlib.sha256(img2).hexdigest()
            assert batch.items[2].bytes_sha256 == hashlib.sha256(img3).hexdigest()

            # Unique pixel hashes
            pixel_hashes = [it.pixel_sha256 for it in batch.items]
            assert len(set(pixel_hashes)) == 3, "All pixel SHA-256 hashes must be distinct"

            # Dimensions [H, W]
            assert batch.items[0].orig_dims == [80, 120]
            assert batch.items[1].orig_dims == [100, 150]
            assert batch.items[2].orig_dims == [150, 200]

            # Prompt version
            assert batch.items[0].prompt_version == PROMPT_VERSION_RESEARCH

    def test_inference_boundary_validation_detects_byte_mismatch(self):
        """Worker must detect if _image_bytes does not match bytes_sha256 at inference boundary."""
        svc = BatchService(max_workers=1)
        with patch.object(svc, "_process_single_item"):
            img = _create_unique_image(color=(50, 50, 50), text_seed=10)
            batch = svc.create_batch([("check.png", img)], style="patient")
            item = batch.items[0]

        # Tamper with bytes prior to worker execution (len > 100 to trigger hash check)
        item._image_bytes = b"tampered_bytes_that_do_not_match_hash_" * 5
        svc._process_single_item(item)

        assert item.status == "failed"
        assert "inference boundary identity mismatch" in item.error.lower()

    def test_duplicate_files_handled_independently(self):
        """Two identical files submitted in a batch must have distinct job IDs and not interfere."""
        svc = BatchService(max_workers=1)
        with patch.object(svc, "_process_single_item"):
            img = _create_unique_image(color=(100, 100, 100), text_seed=42)
            files = [("duplicate.png", img), ("duplicate.png", img)]
            batch = svc.create_batch(files, style="patient")

            assert len(batch.items) == 2
            assert batch.items[0].job_id != batch.items[1].job_id
            assert batch.items[0].bytes_sha256 == batch.items[1].bytes_sha256

    def test_failed_job_cannot_inherit_another_jobs_report(self):
        """Failed job retains failure status and error message without taking data from prior jobs."""
        svc = BatchService(max_workers=1)
        item_fail = BatchItem(
            job_id="job_fail",
            batch_id="batch_x",
            filename="broken.png",
            style="patient",
            bytes_sha256=hashlib.sha256(b"corrupt").hexdigest(),
            _image_bytes=b"corrupt",  # Too short, will fail decode
        )
        batch = MagicMock()
        batch.cancelled = False
        with patch.object(svc, "get_batch", return_value=batch):
            svc._process_single_item(item_fail)

        assert item_fail.status == "failed"
        assert item_fail.report_text is None
        assert item_fail.sections is None
        assert "empty or missing" in item_fail.error.lower()


class TestSingleVsBatchEquivalence:
    """Tests ensuring single-image and batch reporting invoke identical underlying logic."""

    @patch("src.report_service._ollama_available", return_value=True)
    @patch("src.report_service._vision_model_available", return_value="medgemma:4b")
    @patch("src.report_service._ollama_generate_vision_with_metrics")
    def test_single_and_batch_pipeline_call_equivalence(
        self, mock_gen, mock_vavail, mock_oavail
    ):
        raw_obs = (
            "1. Support apparatus: Endotracheal tube observed in mid trachea.\n"
            "2. Cardiac silhouette: Normal.\n"
            "3. Lungs & opacities: Bilateral base consolidations observed.\n"
            "4. Pleural spaces: Costophrenic angles sharp.\n"
            "5. Mediastinum: Normal contour."
        )
        s2_text = (
            "SUPPORT APPARATUS:\nEndotracheal tube in mid trachea.\n\n"
            "CARDIAC SILHOUETTE:\nNormal cardiac size.\n\n"
            "LUNGS AND OPACITIES:\nBilateral retrocardiac consolidations.\n\n"
            "PLEURAL SPACES:\nNo pleural effusion.\n\n"
            "MEDIASTINUM:\nNormal mediastinal silhouette.\n\n"
            "OVERALL IMPRESSION:\nMultifocal consolidation consistent with pneumonia.\n\n"
            "UNCERTAINTY AND LIMITATIONS:\nSingle frontal view; preliminary AI research draft."
        )
        # Mock vision Stage 1 and Stage 2
        mock_gen.side_effect = [
            (raw_obs, {"eval_count": 50, "eval_duration_ns": 1e9}),
            (s2_text, {"eval_count": 80, "eval_duration_ns": 1.5e9}),
            (raw_obs, {"eval_count": 50, "eval_duration_ns": 1e9}),
            (s2_text, {"eval_count": 80, "eval_duration_ns": 1.5e9}),
        ]

        img_bytes = _create_unique_image(color=(80, 80, 80), text_seed=99)

        # 1. Direct single-image call
        rep_single = research_report(image_bytes=img_bytes, image_filename="sample.png")

        # 2. Batch call
        svc = BatchService(max_workers=1)
        with patch.object(svc, "_process_single_item"):
            batch = svc.create_batch([("sample.png", img_bytes)], style="research")
            item = batch.items[0]

        # Process through worker
        svc._process_single_item(item)

        assert item.status == "completed"
        assert item.prompt_version == rep_single["prompt_version"] == PROMPT_VERSION_RESEARCH
        assert item.sections.keys() == rep_single["sections"].keys()
        assert item.triage_result["matched_rule_id"] == rep_single["triage_result"]["matched_rule_id"]


class TestReportStructureAndNoMarkdownArtifacts:
    """Tests ensuring clean, robust section parsing and absence of raw markdown artifacts."""

    def test_7section_research_parsing(self):
        raw = """
        SUPPORT APPARATUS:
        Endotracheal tube terminating 4cm above carina.

        CARDIAC SILHOUETTE:
        Normal cardiothoracic ratio.

        LUNGS AND OPACITIES:
        Right lower lobe focal opacity with air bronchograms.

        PLEURAL SPACES:
        No pleural effusion or pneumothorax identified.

        MEDIASTINUM:
        Normal mediastinal and hilar contours.

        OVERALL IMPRESSION:
        Right lower lobe consolidation concerning for pneumonia.

        UNCERTAINTY AND LIMITATIONS:
        Single AP portable view without lateral projection.
        """
        full_text, sections, warnings = clean_and_structure_research_report(raw)

        expected_sections = [
            "SUPPORT APPARATUS",
            "CARDIAC SILHOUETTE",
            "LUNGS AND OPACITIES",
            "PLEURAL SPACES",
            "MEDIASTINUM",
            "OVERALL IMPRESSION",
            "UNCERTAINTY AND LIMITATIONS",
        ]
        for sec in expected_sections:
            assert sec in sections, f"Section '{sec}' must be present in parsed dictionary"
            assert len(sections[sec]) > 5

        assert "**" not in full_text, "7-section format must not leak markdown asterisks"
        assert "Endotracheal tube" in sections["SUPPORT APPARATUS"]
        assert "Right lower lobe" in sections["LUNGS AND OPACITIES"]

    def test_legacy_4heading_backward_compatibility(self):
        raw = """
        OBSERVATIONS
        Lungs are clear. Cardiac silhouette is normal.

        POSSIBLE FINDINGS
        No acute cardiopulmonary disease.

        ASSESSMENT AND UNCERTAINTY
        Low clinical suspicion on current view.

        LIMITATIONS
        Single frontal projection only.
        """
        full_text, sections, warnings = clean_and_structure_research_report(raw)
        assert "OBSERVATIONS" in sections
        assert "POSSIBLE FINDINGS" in sections
        assert "Lungs are clear" in sections["OBSERVATIONS"]

    def test_triage_detects_abnormalities_without_false_normals(self):
        """Structured sections with endotracheal tube and consolidation must trigger appropriate triage rule."""
        findings = {
            "support_apparatus": "Observed endotracheal tube terminating in distal trachea",
            "cardiac_silhouette": "Normal",
            "lungs_and_opacities": "Observed consolidation in left retrocardiac space",
            "pleural_spaces": "Not identified",
            "mediastinum": "Normal",
        }
        triage = get_triage_service().evaluate_from_structured_findings(findings)
        assert triage.triage_category != "RULE_NORMAL_CXR"
        assert "APPARATUS" in triage.matched_rule_id or "OPACITY" in triage.matched_rule_id
