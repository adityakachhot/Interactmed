"""
tests/test_regressions.py
=========================
Targeted regression tests for InteractMed-VLM fixes:
1. GroundingService error re-raising, empty query guard, and MONOCHROME1 inversion.
2. OverlayService MONOCHROME1 inversion consistency.
3. ReportService regional report stance preservation (no keyword override of Indeterminate).
4. ReportService patient-friendly cleaner deduplication across all duplicated-prefix variants.
5. BatchService per-image isolation, image_sha256 hashing, independent storage, and CSV export.
6. TriageService unnumbered patient section heading parsing.
"""

import io
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image

from src.grounding_service import GroundingService, get_grounding_service
from src.overlay_service import _load_image
from src.report_service import (
    clean_and_structure_regional_report,
    clean_and_structure_patient_report,
    _array_to_base64_jpeg,
)
from src.batch_service import BatchService, BatchItem
from src.triage_service import get_triage_service


class TestGroundingServiceRegressions(unittest.TestCase):
    def test_ensure_loaded_reraises_previous_error(self):
        service = GroundingService.__new__(GroundingService)
        service._loaded = False
        service._load_error = "CUDA out of memory"

        with self.assertRaises(RuntimeError) as ctx:
            service.ensure_loaded()
        self.assertIn("CUDA out of memory", str(ctx.exception))

    def test_predict_rejects_empty_query(self):
        service = GroundingService.__new__(GroundingService)
        service.ckpt_path = "checkpoints/transvg.pth"

        for empty_q in ["", "   ", "\t\n"]:
            res = service.predict(b"dummy_bytes", empty_q)
            self.assertFalse(res["success"])
            self.assertIn("Query phrase cannot be empty", res["error"])

    def test_monochrome1_dicom_inversion(self):
        mock_ds = MagicMock()
        mock_ds.PhotometricInterpretation = "MONOCHROME1"
        mock_pixels = np.arange(100, dtype=np.uint16).reshape((10, 10))
        mock_ds.pixel_array = mock_pixels.copy()

        with patch("pydicom.dcmread", return_value=mock_ds):
            arr = GroundingService._load_image_to_rgb_array(b"dummy_dcm_bytes", "test.dcm")
            self.assertEqual(arr.shape, (10, 10, 3))
            self.assertEqual(arr.dtype, np.uint8)
            # In MONOCHROME1, index (0,0) had original 0, after inversion it should be max (255)
            self.assertEqual(arr[0, 0, 0], 255)
            # Index (9,9) had original 99 (max), after inversion it should be min (0)
            self.assertEqual(arr[9, 9, 0], 0)


class TestOverlayServiceRegressions(unittest.TestCase):
    def test_overlay_monochrome1_inversion(self):
        mock_ds = MagicMock()
        mock_ds.PhotometricInterpretation = "MONOCHROME1"
        mock_pixels = np.arange(100, dtype=np.uint16).reshape((10, 10))
        mock_ds.pixel_array = mock_pixels.copy()

        with patch("pydicom.dcmread", return_value=mock_ds):
            img = _load_image(b"dummy_dcm_bytes", "test.dcm")
            arr = np.array(img)
            self.assertEqual(arr.shape, (10, 10, 3))
            self.assertEqual(arr[0, 0, 0], 255)
            self.assertEqual(arr[9, 9, 0], 0)


class TestReportServiceRegressions(unittest.TestCase):
    def test_regional_report_preserves_indeterminate_assessment(self):
        """Ensure explicit 'Indeterminate' is NOT flipped to 'Supported' when raw_obs mentions opacity."""
        raw_text = (
            "IMAGE OBSERVATIONS: Faint opacity demonstrated in right lower zone; technique limited.\n"
            "ASSESSMENT AND UNCERTAINTY: Indeterminate\n"
            "LIMITATIONS: Low lung volumes."
        )
        cleaned_text, warnings = clean_and_structure_regional_report(raw_text, phrase="pneumonia")
        self.assertIn("Indeterminate:", cleaned_text)
        self.assertNotIn("Supported:", cleaned_text)

    def test_regional_report_preserves_not_supported(self):
        raw_text = (
            "IMAGE OBSERVATIONS: The visualized lung parenchyma appears normal. No opacity seen.\n"
            "ASSESSMENT AND UNCERTAINTY: Not supported\n"
            "LIMITATIONS: None."
        )
        cleaned_text, warnings = clean_and_structure_regional_report(raw_text, phrase="pneumothorax")
        self.assertIn("Not supported:", cleaned_text)

    def test_patient_report_cleaner_multiple_duplicated_prefix_variants(self):
        """
        Ensure hallucinated one-side misconception and all duplicated introductory
        prefix variations are cleanly eliminated from 'What is uncertain'.
        """
        prefix_cases = [
            "A frontal chest X-ray This frontal view displays both the left and right sides of the chest",
            "This frontal view This frontal view displays both the left and right sides of the chest",
            "This X-ray This frontal view displays both the left and right sides of the chest",
            "A frontal view This frontal view displays both the left and right sides of the chest",
            "On a frontal chest X-ray, This frontal view displays both the left and right sides of the chest",
            "Because this is a frontal view, This frontal view displays both the left and right sides of the chest",
            "A single frontal chest X-ray This frontal view displays both the left and right sides of the chest",
            "A frontal chest radiograph This frontal view displays both the left and right sides of the chest",
            "A frontal chest X-ray only shows one side of the chest, so depth cannot be confirmed",
        ]

        for case in prefix_cases:
            raw_text = (
                "Your X-ray in simple words\n"
                "The heart and lungs appear normal.\n\n"
                "What the findings may mean\n"
                "No active pneumonia is identified.\n\n"
                f"What is uncertain\n"
                f"{case}.\n\n"
                "What happens next\n"
                "Consult your ordering physician."
            )
            full_text, sections, warnings = clean_and_structure_patient_report(raw_text)
            uncertain = sections.get("What is uncertain", "")

            # Verify no stuttering prefix remains
            self.assertNotIn("A frontal chest X-ray This frontal view", uncertain)
            self.assertNotIn("This frontal view This frontal view", uncertain)
            self.assertNotIn("This X-ray This frontal view", uncertain)
            self.assertNotIn("A frontal view This frontal view", uncertain)
            self.assertNotIn("only shows one side", uncertain)
            self.assertIn("This frontal view displays both the left and right sides", uncertain)


class TestBatchServiceRegressions(unittest.TestCase):
    def test_per_image_isolation_and_sha256(self):
        # Create two different dummy images
        img1 = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img2 = Image.new("RGB", (100, 100), color=(0, 255, 0))
        buf1 = io.BytesIO()
        buf2 = io.BytesIO()
        img1.save(buf1, format="PNG")
        img2.save(buf2, format="PNG")
        data1 = buf1.getvalue()
        data2 = buf2.getvalue()

        service = BatchService(max_workers=1)
        # Mock _process_single_item to avoid network calls during batch creation test
        with patch.object(service, "_process_single_item"):
            batch = service.create_batch(
                files_data=[("scan_a.png", data1), ("scan_b.png", data2)],
                style="patient",
            )
            self.assertEqual(len(batch.items), 2)
            item1, item2 = batch.items[0], batch.items[1]

            # Verify unique job IDs and distinct image hashes
            self.assertNotEqual(item1.job_id, item2.job_id)
            self.assertIsNotNone(item1.image_sha256)
            self.assertIsNotNone(item2.image_sha256)
            self.assertNotEqual(item1.image_sha256, item2.image_sha256)

            # Test independent result storage
            item1.report_text = "Report for Image A"
            item2.report_text = "Report for Image B"
            self.assertNotEqual(item1.report_text, item2.report_text)

            # Test CSV export includes Image SHA256 column and distinct values
            csv_export = service.export_summary(batch.batch_id, format="csv")
            self.assertIn("Image SHA256", csv_export)
            self.assertIn(item1.image_sha256, csv_export)
            self.assertIn(item2.image_sha256, csv_export)
            self.assertIn("Report for Image A", csv_export)
            self.assertIn("Report for Image B", csv_export)


class TestTriageServiceRegressions(unittest.TestCase):
    def test_triage_parses_clean_unnumbered_patient_sections(self):
        triage = get_triage_service()
        sections = {
            "Your X-ray in simple words": "Large right pneumothorax observed with mediastinal shift.",
            "What the findings may mean": "Urgent collapsed lung.",
            "What is uncertain": "Exact pleural volume.",
            "What happens next": "Immediate physician consultation.",
        }
        res = triage.evaluate_from_report_text(report_text="", sections=sections)
        # Should flag urgent rule for pneumothorax
        self.assertEqual(res.triage_category, "potentially_urgent")
        self.assertEqual(res.matched_rule_id, "RULE_POTENTIALLY_URGENT_PNEUMOTHORAX")
        self.assertEqual(res.human_review_priority, "urgent")


if __name__ == "__main__":
    unittest.main()
