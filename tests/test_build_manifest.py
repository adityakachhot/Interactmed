"""
test_build_manifest.py
======================
Unit tests for the manifest builder and dataset join logic using hand-written fixtures.
"""

import os
import tempfile
import unittest
import pandas as pd

from src.preprocessing.build_manifest import (
    load_ms_cxr,
    load_chest_imagenome_gold,
    unify_manifests,
    generate_download_list,
    format_gcs_relative_path,
    map_region_label,
)


class TestBuildManifest(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test fixtures
        self.test_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.test_dir.name

        # 1. Tiny hand-written MS-CXR fixture (2 rows, 2 distinct studies)
        self.ms_cxr_csv = os.path.join(self.tmp_path, "mock_ms_cxr.csv")
        ms_data = pd.DataFrame([
            {
                "dicom_id": "dicom-001",
                "category_name": "Cardiomegaly",
                "label_text": "Heart size is moderately enlarged.",
                "path": "files/p10/p10001/s50001/dicom-001.jpg",
                "x": 100.0,
                "y": 150.0,
                "w": 200.0,
                "h": 250.0,
                "image_width": 2000,
                "image_height": 2000,
                "split": "train",
            },
            {
                "dicom_id": "dicom-002",
                "category_name": "Pneumothorax",
                "label_text": "Small apical pneumothorax on the right.",
                "path": "files/p12/p12002/s52002/dicom-002.jpg",
                "x": 50.0,
                "y": 60.0,
                "w": 80.0,
                "h": 90.0,
                "image_width": 2048,
                "image_height": 2048,
                "split": "test",
            },
        ])
        ms_data.to_csv(self.ms_cxr_csv, index=False)

        # 2. Tiny hand-written Chest ImaGenome Gold fixture
        # (1 row overlapping with dicom-001, 1 new study dicom-003)
        self.gold_txt = os.path.join(self.tmp_path, "mock_gold.txt")
        gold_data = pd.DataFrame([
            {
                "patient_id": "10001",
                "study_id": "50001",
                "image_id": "dicom-001.dcm",
                "bbox": "cardiac silhouette",
                "coord_original": "[100, 150, 300, 400]",  # x1=100, y1=150, x2=300, y2=400 -> w=200, h=250
                "sentence": "Cardiomegaly is present without acute pulmonary edema.",
                "label_name": "enlarged cardiac silhouette",
                "attribute": "anatomicalfinding|yes|enlarged cardiac silhouette",
            },
            {
                "patient_id": "15003",
                "study_id": "55003",
                "image_id": "dicom-003.dcm",
                "bbox": "left lung",
                "coord_original": "[500, 600, 900, 1200]", # x1=500, y1=600, x2=900, y2=1200 -> w=400, h=600
                "sentence": "Left lower lobe consolidation suspicious for pneumonia.",
                "label_name": "consolidation",
                "attribute": "anatomicalfinding|yes|consolidation",
            },
        ])
        gold_data.to_csv(self.gold_txt, sep="\t", index=False)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_ms_cxr_loader(self):
        df_ms = load_ms_cxr(self.ms_cxr_csv)
        self.assertEqual(len(df_ms), 2)
        self.assertIn("dicom_id", df_ms.columns)
        self.assertIn("image_relative_path", df_ms.columns)
        self.assertIn("bbox", df_ms.columns)
        self.assertIn("region_label", df_ms.columns)

        # Check mapping
        row_cardio = df_ms[df_ms["dicom_id"] == "dicom-001"].iloc[0]
        self.assertEqual(row_cardio["region_label"], "heart")
        self.assertEqual(row_cardio["subject_id"], "10001")
        self.assertEqual(row_cardio["study_id"], "50001")
        self.assertEqual(row_cardio["image_relative_path"], "files/p10/p10001/s50001/dicom-001.jpg")
        self.assertEqual(row_cardio["bbox_w"], 200.0)

        row_pneumo = df_ms[df_ms["dicom_id"] == "dicom-002"].iloc[0]
        self.assertEqual(row_pneumo["region_label"], "pleura")

    def test_chest_imagenome_loader_and_bbox_conversion(self):
        df_gold = load_chest_imagenome_gold(self.gold_txt)
        self.assertEqual(len(df_gold), 2)

        # Verify coord_original [100, 150, 300, 400] -> COCO [100, 150, 200, 250]
        row_001 = df_gold[df_gold["dicom_id"] == "dicom-001"].iloc[0]
        self.assertEqual(row_001["bbox_x"], 100.0)
        self.assertEqual(row_001["bbox_y"], 150.0)
        self.assertEqual(row_001["bbox_w"], 200.0)
        self.assertEqual(row_001["bbox_h"], 250.0)
        self.assertEqual(row_001["region_label"], "heart")
        self.assertEqual(row_001["image_relative_path"], "files/p10/p10001/s50001/dicom-001.jpg")

        row_003 = df_gold[df_gold["dicom_id"] == "dicom-003"].iloc[0]
        self.assertEqual(row_003["region_label"], "lungs")
        self.assertEqual(row_003["bbox_w"], 400.0)
        self.assertEqual(row_003["bbox_h"], 600.0)

    def test_unify_manifests_union(self):
        df_ms = load_ms_cxr(self.ms_cxr_csv)
        df_gold = load_chest_imagenome_gold(self.gold_txt)

        unified = unify_manifests(df_ms, df_gold, join_mode="union")
        self.assertEqual(len(unified), 4)

        required_cols = [
            "dicom_id",
            "subject_id",
            "study_id",
            "image_relative_path",
            "region_label",
            "bbox",
            "finding_text",
            "split",
        ]
        for c in required_cols:
            self.assertIn(c, unified.columns)

    def test_unify_manifests_inner(self):
        df_ms = load_ms_cxr(self.ms_cxr_csv)
        df_gold = load_chest_imagenome_gold(self.gold_txt)

        # Only dicom-001 is common to both
        unified_inner = unify_manifests(df_ms, df_gold, join_mode="inner")
        self.assertTrue(all(unified_inner["dicom_id"] == "dicom-001"))
        self.assertEqual(len(unified_inner), 2)  # 1 from MS-CXR, 1 from Gold

    def test_download_list_generation(self):
        df_ms = load_ms_cxr(self.ms_cxr_csv)
        df_gold = load_chest_imagenome_gold(self.gold_txt)
        unified = unify_manifests(df_ms, df_gold, join_mode="union")

        download_file = os.path.join(self.tmp_path, "download_list.txt")
        paths = generate_download_list(unified, download_file)

        # 3 unique images: dicom-001, dicom-002, dicom-003
        self.assertEqual(len(paths), 3)
        self.assertIn("files/p10/p10001/s50001/dicom-001.jpg", paths)
        self.assertIn("files/p12/p12002/s52002/dicom-002.jpg", paths)
        self.assertIn("files/p15/p15003/s55003/dicom-003.jpg", paths)

        # Verify file contents
        with open(download_file) as f:
            lines = [line.strip() for line in f if line.strip()]
        self.assertEqual(lines, paths)


if __name__ == "__main__":
    unittest.main()
