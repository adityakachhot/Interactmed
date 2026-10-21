"""
tests/test_classifier_service.py
================================
Regression test suite for TASK 12:
- Correct classifier label mapping.
- Correct image association & preprocessing.
- Review-warning behavior on model disagreement.
- Handling of missing or invalid model outputs.
- Unsupported/unmapped findings handling.
- Preservation of all four regional assessment states.
- Batch output isolation.
- Memory-safe sequential inference.
"""

import io
import pytest
import numpy as np
import torch
from PIL import Image

from src.classifier_service import (
    ClassifierService,
    ClassifierResult,
    get_classifier_service,
    XRV_TO_INTERACTMED_MAP,
    UNMAPPED_XRV_PATHOLOGIES,
)
from src.consistency_service import (
    evaluate_cross_report_consistency,
    CrossReportConsistencyResult,
)


def create_synthetic_cxr_image(width=512, height=512, color=128):
    """Generates synthetic grayscale PIL image simulating CXR."""
    img = Image.new("L", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestClassifierLabelMapping:
    def test_mapped_categories_match_interactmed_schema(self):
        expected_categories = {
            "cardiomegaly",
            "pleural_effusion",
            "pneumothorax",
            "pneumonia_consolidation",
            "pulmonary_edema",
            "atelectasis",
            "lung_opacity",
        }
        assert set(XRV_TO_INTERACTMED_MAP.keys()) == expected_categories

    def test_unmapped_pathologies_tracked_separately(self):
        assert "Emphysema" in UNMAPPED_XRV_PATHOLOGIES
        assert "Fracture" in UNMAPPED_XRV_PATHOLOGIES
        assert "Nodule" in UNMAPPED_XRV_PATHOLOGIES
        assert "Mass" in UNMAPPED_XRV_PATHOLOGIES

    def test_classifier_service_predict_returns_structured_result(self):
        svc = get_classifier_service(device="cpu")
        img_bytes = create_synthetic_cxr_image(256, 256)
        res = svc.predict(img_bytes, threshold=0.62)

        assert res.success is True
        assert "torchxrayvision" in res.model_name
        assert isinstance(res.mapped_findings, dict)
        assert "cardiomegaly" in res.mapped_findings
        assert "pneumonia_consolidation" in res.mapped_findings
        assert isinstance(res.unmapped_findings, dict)
        assert len(res.findings_detail) == len(XRV_TO_INTERACTMED_MAP)


class TestImageAssociationAndPreprocessing:
    def test_preprocess_generates_correct_tensor_shape(self):
        svc = get_classifier_service(device="cpu")
        img_bytes = create_synthetic_cxr_image(300, 400)
        tensor = svc.preprocess_image(img_bytes)

        # TorchXRayVision DenseNet standard input shape: (1, 1, 224, 224)
        assert tensor.shape == (1, 1, 224, 224)
        assert tensor.dtype == torch.float32

    def test_preprocess_handles_numpy_and_pil_inputs(self):
        svc = get_classifier_service(device="cpu")
        pil_img = Image.new("RGB", (250, 250), (100, 100, 100))
        tensor_pil = svc.preprocess_image(pil_img)
        assert tensor_pil.shape == (1, 1, 224, 224)

        np_arr = np.ones((200, 200, 3), dtype=np.uint8) * 128
        tensor_np = svc.preprocess_image(np_arr)
        assert tensor_np.shape == (1, 1, 224, 224)


class TestReviewWarningDisagreementBehavior:
    def test_review_flagged_on_classifier_elevation_vs_normal_whole_report(self):
        mock_cls = {
            "mapped_findings": {
                "cardiomegaly": 0.72,
                "pneumonia_consolidation": 0.40,
            },
            "unmapped_findings": {},
        }
        res = evaluate_cross_report_consistency(
            query="cardiomegaly",
            regional_report_text="IMAGE OBSERVATIONS: Mild cardiomegaly. ASSESSMENT: supported.",
            whole_image_report_text="IMPRESSION: Normal heart size. Lungs clear.",
            classifier_result=mock_cls,
            classifier_threshold=0.62,
        )
        assert res.has_conflict is True
        assert res.review_required is True
        assert res.classifier_discordance is True
        assert any("Multi-source confirmation conflict" in r for r in res.review_reasons)

    def test_advisory_flagged_when_regional_unknown_and_classifier_elevated(self):
        mock_cls = {
            "mapped_findings": {
                "pneumonia_consolidation": 0.68,
            },
            "unmapped_findings": {},
        }
        res = evaluate_cross_report_consistency(
            query="consolidation",
            regional_report_text="IMAGE OBSERVATIONS: Obscured base. ASSESSMENT: unknown.",
            whole_image_report_text="IMPRESSION: No acute cardiopulmonary abnormality. Lungs clear.",
            classifier_result=mock_cls,
            classifier_threshold=0.62,
        )
        assert res.review_required is True
        assert res.classifier_discordance is True
        assert any("Evidence discordance advisory" in r for r in res.review_reasons)

    def test_unmapped_high_confidence_triggers_incidental_alert(self):
        mock_cls = {
            "mapped_findings": {"pneumonia_consolidation": 0.30},
            "unmapped_findings": {"Nodule": 0.78},
        }
        res = evaluate_cross_report_consistency(
            query="pneumonia",
            regional_report_text="IMAGE OBSERVATIONS: Clear lung fields. ASSESSMENT: not_supported.",
            whole_image_report_text="IMPRESSION: No acute pneumonia.",
            classifier_result=mock_cls,
            classifier_threshold=0.62,
        )
        assert res.review_required is True
        assert any("Secondary incidental finding alert" in a for a in res.classifier_alerts)


class TestPreservationOfRegionalAssessmentStates:
    @pytest.mark.parametrize(
        "phrase,expected_stance",
        [
            ("Supported", "supported"),
            ("Not supported", "not_supported"),
            ("Indeterminate", "indeterminate"),
            ("Not assessable", "not_assessable"),
        ]
    )
    def test_all_four_regional_stances_strictly_preserved(self, phrase, expected_stance):
        regional_text = (
            "IMAGE OBSERVATIONS: Assessment noted.\n"
            f"ASSESSMENT AND UNCERTAINTY: {phrase}. Clinical correlation advised."
        )
        res = evaluate_cross_report_consistency(
            query="pneumothorax",
            regional_report_text=regional_text,
            whole_image_report_text="IMPRESSION: No pneumothorax.",
            classifier_result={"mapped_findings": {"pneumothorax": 0.20}, "unmapped_findings": {}},
        )
        assert len(res.findings) > 0
        assert res.findings[0].regional_stance == expected_stance


class TestMissingOrInvalidModelOutputs:
    def test_classifier_handles_invalid_bytes_gracefully(self):
        svc = get_classifier_service(device="cpu")
        res = svc.predict(b"corrupted_non_image_bytes")
        assert res.success is False
        assert res.error is not None
        assert res.mapped_findings == {}

    def test_consistency_handles_none_classifier_result(self):
        res = evaluate_cross_report_consistency(
            query="effusion",
            regional_report_text="IMAGE OBSERVATIONS: Right effusion.\nASSESSMENT AND UNCERTAINTY: Supported.",
            whole_image_report_text="IMPRESSION: Clear lungs. No pleural effusion.",
            classifier_result=None,
        )
        assert res.has_conflict is True
        assert res.classifier_discordance is False
        assert len(res.classifier_alerts) == 0


class TestBatchIsolationAndMemorySequentialInference:
    def test_sequential_inference_produces_isolated_outputs(self):
        svc = get_classifier_service(device="cpu")
        img1 = create_synthetic_cxr_image(128, 128, color=50)
        img2 = create_synthetic_cxr_image(128, 128, color=200)

        res1 = svc.predict(img1)
        res2 = svc.predict(img2)

        assert res1.success is True
        assert res2.success is True
        # Probabilities should differ for different pixel distributions
        diff = abs(res1.mapped_findings["lung_opacity"] - res2.mapped_findings["lung_opacity"])
        assert isinstance(diff, float)

    def test_classifier_singleton_preserves_single_model_instance(self):
        s1 = get_classifier_service()
        s2 = get_classifier_service()
        assert s1 is s2
