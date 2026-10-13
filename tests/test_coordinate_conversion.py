"""
tests/test_coordinate_conversion.py
===================================
Unit tests for coordinate conversions, bounding box boundary invariants,
normalization, and processed metadata integrity for InteractMed-VLM.
"""

import os
from typing import Dict, List, Optional, Tuple
import pytest
import numpy as np
import pandas as pd
from PIL import Image


def xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """Converts COCO format [x, y, w, h] to Pascal VOC [x1, y1, x2, y2]."""
    if w <= 0 or h <= 0:
        raise ValueError(f"Width and height must be strictly positive, got w={w}, h={h}")
    return x, y, x + w, y + h


def normalize_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: float,
    img_h: float,
) -> Tuple[float, float, float, float]:
    """Normalizes pixel coordinates to [0.0, 1.0] with boundary clamping."""
    if img_w <= 0 or img_h <= 0:
        raise ValueError("Image dimensions must be positive.")
    nx1 = max(0.0, min(1.0, x1 / img_w))
    ny1 = max(0.0, min(1.0, y1 / img_h))
    nx2 = max(0.0, min(1.0, x2 / img_w))
    ny2 = max(0.0, min(1.0, y2 / img_h))
    return nx1, ny1, nx2, ny2


class TestCoordinateTransformations:
    """Tests basic geometric coordinate transformation rules."""

    def test_xywh_to_xyxy_standard(self):
        x1, y1, x2, y2 = xywh_to_xyxy(100.0, 150.0, 50.0, 80.0)
        assert x1 == 100.0
        assert y1 == 150.0
        assert x2 == 150.0
        assert y2 == 230.0
        assert (x2 - x1) == 50.0
        assert (y2 - y1) == 80.0

    def test_xywh_to_xyxy_invalid_dimensions(self):
        with pytest.raises(ValueError, match="strictly positive"):
            xywh_to_xyxy(10.0, 10.0, 0.0, 50.0)

        with pytest.raises(ValueError, match="strictly positive"):
            xywh_to_xyxy(10.0, 10.0, 50.0, -5.0)

    def test_normalize_xyxy_bounds(self):
        nx1, ny1, nx2, ny2 = normalize_xyxy(50.0, 100.0, 150.0, 300.0, 500.0, 1000.0)
        assert pytest.approx(nx1, rel=1e-5) == 0.10
        assert pytest.approx(ny1, rel=1e-5) == 0.10
        assert pytest.approx(nx2, rel=1e-5) == 0.30
        assert pytest.approx(ny2, rel=1e-5) == 0.30
        assert 0.0 <= nx1 <= nx2 <= 1.0
        assert 0.0 <= ny1 <= ny2 <= 1.0

    def test_normalize_xyxy_clamping(self):
        # Slightly out of bounds coordinates should clamp safely to [0, 1]
        nx1, ny1, nx2, ny2 = normalize_xyxy(-20.0, -10.0, 520.0, 1010.0, 500.0, 1000.0)
        assert nx1 == 0.0
        assert ny1 == 0.0
        assert nx2 == 1.0
        assert ny2 == 1.0

    def test_round_trip_reconstruction(self):
        img_w, img_h = 2048.0, 2560.0
        orig_x, orig_y, orig_w, orig_h = 450.0, 600.0, 350.0, 500.0
        x1, y1, x2, y2 = xywh_to_xyxy(orig_x, orig_y, orig_w, orig_h)
        nx1, ny1, nx2, ny2 = normalize_xyxy(x1, y1, x2, y2, img_w, img_h)

        # Reconstruct native
        rec_x1 = nx1 * img_w
        rec_y1 = ny1 * img_h
        rec_x2 = nx2 * img_w
        rec_y2 = ny2 * img_h

        assert pytest.approx(rec_x1, abs=0.5) == x1
        assert pytest.approx(rec_y1, abs=0.5) == y1
        assert pytest.approx(rec_x2, abs=0.5) == x2
        assert pytest.approx(rec_y2, abs=0.5) == y2


class TestProcessedMetadataIntegrity:
    """Tests generated master and verified tables for schema and integrity."""

    @pytest.fixture
    def master_df(self):
        path = "data/processed/unified_grounding_master.csv"
        if not os.path.exists(path):
            pytest.skip(f"Processed master table not found at {path}")
        return pd.read_csv(path)

    @pytest.fixture
    def local_df(self):
        path = "data/processed/local_grounding_verified.csv"
        if not os.path.exists(path):
            pytest.skip(f"Processed verified table not found at {path}")
        return pd.read_csv(path)

    def test_master_schema_and_columns(self, master_df):
        required_cols = [
            "row_id", "dataset_source", "label_type", "dicom_id", "subject_id",
            "study_id", "split", "phrase", "category_name", "organ_region",
            "image_width", "image_height", "norm_x1", "norm_y1", "norm_x2",
            "norm_y2", "is_valid_bbox", "image_exists"
        ]
        for col in required_cols:
            assert col in master_df.columns, f"Missing required column: {col}"

    def test_local_verified_subset_exists_flag(self, local_df):
        assert (local_df["image_exists"] == True).all(), "Verified table contains records where image_exists is False"
        assert len(local_df) > 0

    def test_coordinate_normalization_range(self, local_df):
        eps = 1e-4
        assert (local_df["norm_x1"] >= -eps).all()
        assert (local_df["norm_y1"] >= -eps).all()
        assert (local_df["norm_x2"] <= 1.0 + eps).all()
        assert (local_df["norm_y2"] <= 1.0 + eps).all()

    def test_ms_cxr_pathology_vs_cig_anatomy_labels(self, master_df):
        ms_labels = master_df[master_df["dataset_source"] == "ms_cxr"]["label_type"].unique()
        cig_labels = master_df[master_df["dataset_source"] == "chest_imagenome_gold"]["label_type"].unique()
        assert set(ms_labels) == {"pathology"}
        assert set(cig_labels) == {"anatomy"}

    def test_physical_image_file_accessibility(self, local_df):
        # Sample 10 random verified records and verify that image files open cleanly
        sample = local_df.sample(min(10, len(local_df)), random_state=42)
        mimic_root = "/Users/adityakachhot/Downloads/data/mimic-cxr"
        for _, row in sample.iterrows():
            img_path = os.path.join(mimic_root, row["image_path"])
            assert os.path.exists(img_path), f"Referenced image does not exist: {img_path}"
            with Image.open(img_path) as img:
                assert img.size[0] > 0
                assert img.size[1] > 0


class TestCanonicalCoordinatesAndCrop:
    """
    Tests for canonical pixel bounding box derivation, crop extraction,
    box clipping, and overlay alignment.
    """

    def test_canonical_pixel_to_norm_consistency(self):
        """
        Verify that deriving normalized coordinates from canonical pixel box
        guarantees exact mathematical correspondence:
          cx_norm * orig_w == (x1 + x2) / 2
          w_norm * orig_w  == (x2 - x1)
        """
        orig_w, orig_h = 2544, 3056
        x1, y1, x2, y2 = 1264.0, 1387.0, 1949.0, 1930.0

        box_w = x2 - x1
        box_h = y2 - y1
        cx_px = x1 + 0.5 * box_w
        cy_px = y1 + 0.5 * box_h

        norm_cx = cx_px / orig_w
        norm_cy = cy_px / orig_h
        norm_w = box_w / orig_w
        norm_h = box_h / orig_h

        # Reconstructed pixel center and dimensions must match exactly
        assert pytest.approx(norm_cx * orig_w, abs=1e-3) == cx_px
        assert pytest.approx(norm_cy * orig_h, abs=1e-3) == cy_px
        assert pytest.approx(norm_w * orig_w, abs=1e-3) == box_w
        assert pytest.approx(norm_h * orig_h, abs=1e-3) == box_h

        # Normalized coordinates must all lie strictly in [0, 1]
        assert 0.0 <= norm_cx <= 1.0
        assert 0.0 <= norm_cy <= 1.0
        assert 0.0 <= norm_w <= 1.0
        assert 0.0 <= norm_h <= 1.0

    def test_crop_creation_and_margin_clamping(self):
        """
        Verify that extract_crop handles margin expansion, correctly clamps
        to image boundaries, and returns a valid numpy RGB array.
        """
        from src.report_service import extract_crop
        import io

        # Create a synthetic 100x100 RGB image
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        # Box near top-left edge: [5, 5, 25, 25] with margin 20
        # Clamped cx1=max(0, 5-20)=0, cy1=max(0, 5-20)=0, cx2=25+20=45, cy2=25+20=45
        crop = extract_crop(image_bytes, "test.png", [5.0, 5.0, 25.0, 25.0], margin_px=20)
        assert crop is not None
        assert isinstance(crop, np.ndarray)
        assert crop.shape == (45, 45, 3)

        # Box near bottom-right edge: [80, 80, 95, 95] with margin 20
        # Clamped cx1=80-20=60, cy1=80-20=60, cx2=min(100, 95+20)=100, cy2=min(100, 95+20)=100
        crop_br = extract_crop(image_bytes, "test.png", [80.0, 80.0, 95.0, 95.0], margin_px=20)
        assert crop_br is not None
        assert crop_br.shape == (40, 40, 3)

    def test_crop_invalid_box_returns_none(self):
        """Verify that inverted or zero-area boxes return None gracefully."""
        from src.report_service import extract_crop
        import io

        img = Image.new("RGB", (50, 50), color=(100, 100, 100))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        # Inverted box (x2 < x1)
        assert extract_crop(image_bytes, "test.png", [30.0, 10.0, 10.0, 40.0]) is None
        # Zero area (x1 == x2)
        assert extract_crop(image_bytes, "test.png", [20.0, 10.0, 20.0, 40.0]) is None

    def test_overlay_alignment_coordinates(self):
        """
        Verify that draw_overlay executes without error and preserves original image size.
        """
        from src.overlay_service import draw_overlay
        import io

        img = Image.new("RGB", (200, 300), color=(50, 50, 50))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        annotated = draw_overlay(
            image_bytes=image_bytes,
            image_filename="test.png",
            pred_box_pixel_xyxy=[50.0, 60.0, 150.0, 220.0],
            query="pneumonia",
            model_label="TransVG-AGPT",
        )
        assert isinstance(annotated, Image.Image)
        assert annotated.size == (200, 300)

