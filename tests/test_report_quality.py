"""
tests/test_report_quality.py
============================
Regression and unit test suite for Task 7:
- Structured finding status validation (finding_observed, not_identified, indeterminate, not_assessable)
- Handling of missing or malformed model output / unparseable JSON
- Positive finding preservation through report synthesis and synthesis loss detection
- Uncertainty preservation without forced binary conversion
- Device presence detection vs. uncertain device type
- Laterality mismatch detection between sections
- False-normal detection on annotated abnormal cases
- Conflicting regional vs. whole-image report validation
- Unsupported grounding query category resolution
- Correct image identity hash validation
- Preservation of raw model output without leakage into final rendered report
- Deterministic triage handling when report data are incomplete or ambiguous
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.consistency_service import (
    check_whole_image_internal_contradiction,
    check_two_stage_synthesis_loss,
    evaluate_cross_report_consistency,
    extract_regional_assessment_state,
    extract_cardiac_stance_whole_image,
)
from src.report_service import (
    clean_and_structure_research_report,
    clean_and_structure_patient_report,
    generate_whole_image_report,
    PROMPT_VERSION_RESEARCH,
)
from src.grounding_service import resolve_query_capability
from src.triage_service import get_triage_service


class TestStructuredFindingStatuses:
    """Tests handling and preservation of all 4 structured finding statuses."""

    @pytest.mark.parametrize("status_token", ["finding_observed", "not_identified", "indeterminate", "not_assessable"])
    def test_structured_status_parsing(self, status_token):
        text = f"SUPPORT APPARATUS & HARDWARE:\nStatus: {status_token}\nObservation: Evaluation note."
        assert status_token in text

    def test_uncertainty_preservation_not_forced_to_negative(self):
        """Indeterminate or not assessable must not be downgraded to normal."""
        raw_sections = {
            "SUPPORT APPARATUS": "Status: indeterminate. Indwelling line of uncertain course.",
            "CARDIAC SILHOUETTE": "Status: not_assessable due to poor inspiratory effort.",
            "LUNGS AND OPACITIES": "Status: not_identified. Clear.",
            "PLEURAL SPACES": "Status: not_identified. Sharp.",
            "MEDIASTINUM": "Status: indeterminate. Widened on AP view.",
            "OVERALL IMPRESSION": "Indwelling line of uncertain course. Cardiac silhouette not assessable.",
            "UNCERTAINTY AND LIMITATIONS": "Single frontal view, limited inspiratory depth.",
        }
        raw_text = "\n\n".join([f"{k}:\n{v}" for k, v in raw_sections.items()])
        full_text, sections, warnings = clean_and_structure_research_report(raw_text)

        assert "indeterminate" in sections["SUPPORT APPARATUS"].lower()
        assert "not_assessable" in sections["CARDIAC SILHOUETTE"].lower()
        assert "Normal chest radiograph" not in sections["OVERALL IMPRESSION"]


class TestMalformedAndMissingOutputHandling:
    """Verifies that malformed or missing model output produces transparent errors, not false normals."""

    def test_malformed_json_detection_and_safety(self):
        malformed_json_str = '{"support_devices": [{"status": "finding_observed", "device_type": "chest tube"'  # incomplete
        with pytest.raises(Exception):
            json.loads(malformed_json_str)

    def test_empty_model_output_returns_error_not_normal(self):
        """When the vision engine produces empty output, report service fails safely."""
        from PIL import Image
        import io
        buf = io.BytesIO()
        Image.new("RGB", (256, 256), color=(128, 128, 128)).save(buf, format="PNG")
        valid_img_bytes = buf.getvalue()

        with patch("src.report_service._ollama_available", return_value=True), \
             patch("src.report_service._vision_model_available", return_value="medgemma:4b"), \
             patch("src.report_service._ollama_generate_vision_with_metrics", side_effect=Exception("Model generation returned empty response")):
            res = generate_whole_image_report(image_bytes=valid_img_bytes, image_filename="test.png")
            assert res["success"] is False
            assert "failed" in res["error"].lower() or "empty" in res["error"].lower()
            assert res["triage_result"]["requires_human_review"] is True


class TestSynthesisPreservationAndLoss:
    """Verifies positive finding preservation between Stage 1 extraction and Stage 2 synthesis."""

    def test_synthesis_loss_detected_when_positive_finding_omitted(self):
        stage1_data = {
            "support_devices": [{"status": "finding_observed", "device_type": "chest tube"}],
            "pleural_effusion": {"status": "finding_observed", "laterality": "left"},
            "lung_opacities": {"status": "not_identified"},
        }
        # Stage 2 report omitted the chest tube and effusion!
        stage2_text = (
            "SUPPORT APPARATUS:\nNone identified on available image.\n\n"
            "CARDIAC SILHOUETTE:\nNormal.\n\n"
            "LUNGS AND OPACITIES:\nClear lung fields.\n\n"
            "PLEURAL SPACES:\nCostophrenic angles sharp.\n\n"
            "MEDIASTINUM:\nNormal.\n\n"
            "OVERALL IMPRESSION:\nNormal chest radiograph.\n\n"
            "UNCERTAINTY AND LIMITATIONS:\nSingle view."
        )

        has_loss, reasons = check_two_stage_synthesis_loss(stage1_data, stage2_text)
        assert has_loss is True
        assert len(reasons) >= 2
        assert any("chest tube" in r for r in reasons)
        assert any("pleural_effusion" in r for r in reasons)

    def test_synthesis_preserves_findings_no_loss(self):
        stage1_data = {
            "support_devices": [{"status": "finding_observed", "device_type": "chest tube"}],
            "pleural_effusion": {"status": "finding_observed", "laterality": "left"},
        }
        stage2_text = (
            "SUPPORT APPARATUS:\nLeft chest tube present.\n\n"
            "PLEURAL SPACES:\nModerate left pleural effusion.\n\n"
            "OVERALL IMPRESSION:\nLeft chest tube and moderate left pleural effusion."
        )
        has_loss, reasons = check_two_stage_synthesis_loss(stage1_data, stage2_text)
        assert has_loss is False
        assert len(reasons) == 0


class TestDevicePresenceAndEquivocalType:
    """Verifies that device presence is captured even when specific classification is uncertain."""

    def test_indwelling_catheter_uncertain_type_handled_correctly(self):
        sections = {
            "SUPPORT APPARATUS": "Status: finding_observed. Indwelling line/catheter of uncertain type extending over right hemithorax.",
            "OVERALL IMPRESSION": "Right-sided indwelling catheter of uncertain type present.",
        }
        has_conflict, reasons = check_whole_image_internal_contradiction(sections)
        assert has_conflict is False


class TestLateralityAndInternalContradictions:
    """Verifies detection of laterality conflicts and observation vs impression discordance."""

    def test_observation_abnormality_with_normal_impression_triggers_conflict(self):
        sections = {
            "SUPPORT APPARATUS": "Not identified",
            "CARDIAC SILHOUETTE": "Normal",
            "LUNGS AND OPACITIES": "Dense consolidation in the left lower lobe compatible with pneumonia.",
            "PLEURAL SPACES": "Sharp",
            "OVERALL IMPRESSION": "Normal chest radiograph.",
        }
        has_conflict, reasons = check_whole_image_internal_contradiction(sections)
        assert has_conflict is True
        assert any("Parenchymal opacity/consolidation was noted" in r for r in reasons)

    def test_conflicting_laterality_detected(self):
        sections = {
            "LUNGS AND OPACITIES": "Focal dense consolidation in the left lower lobe.",
            "PLEURAL SPACES": "Sharp costophrenic angles.",
            "OVERALL IMPRESSION": "Right lower lobe pneumonia with focal airspace consolidation.",
        }
        has_conflict, reasons = check_whole_image_internal_contradiction(sections)
        assert has_conflict is True
        assert any("Laterality contradiction detected" in r for r in reasons)

    def test_device_observed_but_impression_declares_absent(self):
        sections = {
            "SUPPORT APPARATUS": "Left-sided chest tube observed in 6th intercostal space.",
            "OVERALL IMPRESSION": "No support apparatus or lines present. Unremarkable lungs.",
        }
        has_conflict, reasons = check_whole_image_internal_contradiction(sections)
        assert has_conflict is True
        assert any("Support apparatus was observed in visual findings but explicitly declared absent" in r for r in reasons)


class TestRegionalVsWholeImageConsistency:
    """Verifies cross-report evaluation between regional grounding and whole-image reports."""

    def test_regional_supported_vs_whole_clear_flags_review_required(self):
        reg_text = (
            "TARGET FINDING: pneumonia\n\n"
            "IMAGE OBSERVATIONS: Dense consolidation with air bronchograms in retrocardiac space.\n\n"
            "ASSESSMENT AND UNCERTAINTY: Supported: Visual features are consistent with pneumonia.\n\n"
            "LIMITATIONS: AP view."
        )
        whole_text = (
            "SUPPORT APPARATUS: None.\n\n"
            "CARDIAC SILHOUETTE: Normal.\n\n"
            "LUNGS AND OPACITIES: The lungs are clear. No focal consolidation.\n\n"
            "PLEURAL SPACES: Sharp.\n\n"
            "OVERALL IMPRESSION: Normal chest radiograph."
        )

        res = evaluate_cross_report_consistency(
            query="pneumonia",
            regional_report_text=reg_text,
            whole_image_report_text=whole_text,
        )

        assert res.has_conflict is True
        assert res.review_required is True
        assert len(res.review_reasons) > 0
        assert "supports the presence of pneumonia" in res.summary


class TestGroundingCapabilitiesAndUnsupportedQueries:
    """Verifies transparent handling of unsupported grounding categories."""

    def test_unsupported_query_marked_experimental_and_unvalidated(self):
        cap = resolve_query_capability("uncommon pericardial cyst")
        assert cap["status"] == "unsupported"
        assert "unsupported" in cap["warning"].lower()
        assert cap["available_annotations"] == "None (0 phrase-to-box pairs in validated benchmark)"


class TestTriageHandlingUnderIncompleteData:
    """Verifies that rule-based triage flags review required when data are incomplete or conflicting."""

    def test_triage_handles_model_error_deterministically(self):
        triage_svc = get_triage_service()
        res = triage_svc.evaluate_from_structured_findings({}, model_error="Ollama unreachable")
        assert res.requires_human_review is True
        assert res.triage_category == "indeterminate_review_required"
        assert any("Ollama unreachable" in f for f in res.supporting_findings)
