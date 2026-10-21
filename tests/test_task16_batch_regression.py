"""
tests/test_task16_batch_regression.py
======================================
Comprehensive regression tests for Task 16:
1. Regex word boundary bugfix (preventing 'What is uncertaThis').
2. Deterministic report validation (detecting missing sections, truncations, malformed text).
3. Multi-evidence triage engine (8 states, matched rule IDs, review priority, review_required).
4. Batch item auxiliary classifier signals and cross-report consistency integration.
5. Batch CSV export schema integrity and column consistency.
"""

import io
import pytest
from unittest.mock import MagicMock, patch

from src.report_service import (
    clean_and_structure_patient_report,
    clean_and_structure_research_report,
    validate_generated_report,
)
from src.triage_service import get_triage_service, TriageEvaluationResult
from src.batch_service import BatchItem, BatchService


class TestRegexWordBoundaryRegression:
    """Verifies fix for 'What is uncertaThis' caused by unanchored 'in\\s+' pattern."""

    def test_what_is_uncertain_not_mangled(self):
        raw_text = (
            "What is uncertain\n"
            "This frontal view displays both the left and right sides of the chest, "
            "but as a single 2D projection, subtle findings cannot be confirmed."
        )
        cleaned_text, sections, warnings = clean_and_structure_patient_report(raw_text)

        assert "What is uncertaThis" not in cleaned_text
        assert "What is uncerta" not in cleaned_text or "What is uncertain" in cleaned_text
        assert sections["What is uncertain"].startswith("This frontal view")

    def test_in_at_start_of_word_handled_properly(self):
        raw_text = (
            "What is uncertain in this image, a single frontal radiograph only shows 2D projection."
        )
        cleaned_text, sections, warnings = clean_and_structure_patient_report(raw_text)
        assert "What is uncertaThis" not in cleaned_text


class TestReportValidation:
    """Tests deterministic structural and semantic validation for generated reports."""

    def test_valid_patient_report(self):
        text = (
            "**Your X-ray in simple words**\n"
            "The heart and lungs appear clear.\n\n"
            "**What the findings may mean**\n"
            "No active pneumonia or fluid is seen.\n\n"
            "**What is uncertain**\n"
            "Single frontal view cannot evaluate lateral depth.\n\n"
            "**What happens next**\n"
            "Discuss with your doctor."
        )
        sections = {
            "Your X-ray in simple words": "The heart and lungs appear clear.",
            "What the findings may mean": "No active pneumonia or fluid is seen.",
            "What is uncertain": "Single frontal view cannot evaluate lateral depth.",
            "What happens next": "Discuss with your doctor.",
        }
        is_valid, status, warns, review_req = validate_generated_report(text, sections, style="patient")
        assert is_valid is True
        assert status == "completed_valid"
        assert review_req is False
        assert len(warns) == 0

    def test_missing_section_flags_incomplete(self):
        text = (
            "**Your X-ray in simple words**\n"
            "The heart appears normal.\n\n"
            "**What happens next**\n"
            "Discuss with your doctor."
        )
        sections = {
            "Your X-ray in simple words": "The heart appears normal.",
            "What happens next": "Discuss with your doctor.",
        }
        is_valid, status, warns, review_req = validate_generated_report(text, sections, style="patient")
        assert is_valid is False
        assert status == "incomplete_report"
        assert review_req is True
        assert any("What the findings may mean" in w for w in warns)

    def test_truncated_sentence_flags_incomplete(self):
        text = (
            "**Your X-ray in simple words**\n"
            "The heart and lungs appear clear.\n\n"
            "**What the findings may mean**\n"
            "No active pneumonia is seen.\n\n"
            "**What is uncertain**\n"
            "However, it is a single two\n\n"
            "**What happens next**\n"
            "Please consult your ordering physician for qualified"
        )
        sections = {
            "Your X-ray in simple words": "The heart and lungs appear clear.",
            "What the findings may mean": "No active pneumonia is seen.",
            "What is uncertain": "However, it is a single two",
            "What happens next": "Please consult your ordering physician for qualified",
        }
        is_valid, status, warns, review_req = validate_generated_report(text, sections, style="patient")
        assert is_valid is False
        assert status == "incomplete_report"
        assert review_req is True
        assert any("truncated" in w.lower() for w in warns)

    def test_malformed_token_mangling_detected(self):
        text = (
            "**Your X-ray in simple words**\n"
            "Lungs clear.\n\n"
            "**What the findings may mean**\n"
            "Normal scan.\n\n"
            "What is uncertaThis frontal view displays both sides.\n\n"
            "**What happens next**\n"
            "Discuss with doctor."
        )
        sections = {
            "Your X-ray in simple words": "Lungs clear.",
            "What the findings may mean": "Normal scan.",
            "What is uncertain": "What is uncertaThis frontal view displays both sides.",
            "What happens next": "Discuss with doctor.",
        }
        is_valid, status, warns, review_req = validate_generated_report(text, sections, style="patient")
        assert is_valid is False
        assert status == "malformed_report"
        assert review_req is True
        assert any("What is uncertaThis" in w for w in warns)


class TestMultiEvidenceTriageEngine:
    """Tests the 8 research triage states and multi-evidence precedence rules."""

    def test_processing_error_state(self):
        svc = get_triage_service()
        res = svc.evaluate_multi_evidence(model_error="Ollama connection failed", is_valid_image=True)
        assert res.triage_state == "processing_error"
        assert res.requires_human_review is True
        assert res.human_review_priority == "urgent"

    def test_incomplete_or_malformed_report_state(self):
        svc = get_triage_service()
        res = svc.evaluate_multi_evidence(report_validity="incomplete_report")
        assert res.triage_state == "incomplete_or_malformed_report"
        assert res.matched_rule_id == "RULE_INDETERMINATE_MALFORMED_OR_INCOMPLETE_REPORT"
        assert res.requires_human_review is True

    def test_conflicting_ai_assessments_state(self):
        svc = get_triage_service()
        mock_cons = MagicMock()
        mock_cons.has_conflict = True
        mock_cons.classifier_discordance = True
        mock_cons.classifier_alerts = ["pneumothorax (score: 0.72 >= 0.62)"]
        mock_cons.summary = "Whole-image report noted lungs clear but classifier detected pneumothorax."

        res = svc.evaluate_multi_evidence(
            report_text="Lungs are clear.",
            consistency_result=mock_cons,
            report_validity="completed_valid",
        )
        assert res.triage_state == "conflicting_ai_assessments"
        assert res.matched_rule_id == "RULE_INDETERMINATE_CONFLICTING_AI_ASSESSMENTS"
        assert res.requires_human_review is True

    def test_potential_abnormality_from_classifier_when_report_misses_it(self):
        svc = get_triage_service()
        mock_cls = MagicMock()
        mock_cls.mapped_findings = {"pneumothorax": 0.74, "cardiomegaly": 0.20}

        res = svc.evaluate_multi_evidence(
            report_text="Lungs are clear. Heart size normal.",
            sections={"LUNGS AND OPACITIES": "Clear", "CARDIAC SILHOUETTE": "Normal"},
            classifier_result=mock_cls,
            report_validity="completed_valid",
        )
        assert res.triage_state == "potential_abnormality_signal"
        assert res.triage_category == "potentially_urgent"
        assert res.matched_rule_id == "RULE_POTENTIALLY_URGENT_CLASSIFIER_SIGNAL"
        assert res.human_review_priority == "urgent"

    def test_support_device_signal_state(self):
        svc = get_triage_service()
        report = (
            "SUPPORT APPARATUS:\nEndotracheal tube in mid trachea.\n\n"
            "CARDIAC SILHOUETTE:\nNormal cardiac size.\n\n"
            "LUNGS AND OPACITIES:\nClear.\n\n"
            "PLEURAL SPACES:\nSharp.\n\n"
            "MEDIASTINUM:\nNormal."
        )
        res = svc.evaluate_multi_evidence(report_text=report, report_validity="completed_valid")
        assert res.triage_state == "support_device_signal"
        assert res.matched_rule_id == "RULE_ABNORMAL_SUPPORT_APPARATUS_PRESENT"
        assert res.requires_human_review is True

    def test_no_configured_signal_state(self):
        svc = get_triage_service()
        report = (
            "SUPPORT APPARATUS:\nNo devices identified.\n\n"
            "CARDIAC SILHOUETTE:\nNormal heart size.\n\n"
            "LUNGS AND OPACITIES:\nLungs are clear without focal opacities.\n\n"
            "PLEURAL SPACES:\nCostophrenic angles sharp.\n\n"
            "MEDIASTINUM:\nNormal contour."
        )
        res = svc.evaluate_multi_evidence(report_text=report, report_validity="completed_valid")
        assert res.triage_state == "no_configured_signal"
        assert res.triage_category == "no_priority_finding_flagged"
        assert res.matched_rule_id == "RULE_NO_PRIORITY_FINDINGS_FLAGGED"
        # Must retain research review policy and not claim verified clinical normal
        assert res.requires_human_review is True


class TestBatchServiceIntegrationAndExport:
    """Verifies BatchItem fields, CSV export columns, and multi-evidence serialization."""

    def test_batch_item_default_fields(self):
        item = BatchItem(
            job_id="job_test01",
            batch_id="batch_test",
            filename="cxr_test.png",
            style="patient",
            bytes_sha256="abcd1234ef567890",
        )
        assert item.report_validity == "completed_valid"
        assert item.review_required is False
        assert item.classifier_alerts == []
        assert item.classifier_scores == {}
        assert item.consistency_result is None

    def test_csv_export_includes_quality_and_triage_columns(self):
        svc = BatchService()
        batch = svc.create_batch([("sample.png", b"dummy_bytes_here" * 10)], style="patient")
        item = batch.items[0]
        item.status = "completed"
        item.report_validity = "completed_valid"
        item.review_required = True
        item.classifier_alerts = ["pneumothorax (score: 0.68 >= 0.62)"]
        item.triage_result = {
            "triage_category": "potentially_urgent",
            "matched_rule_id": "RULE_POTENTIALLY_URGENT_CLASSIFIER_SIGNAL",
            "human_review_priority": "urgent",
            "triage_state": "potential_abnormality_signal",
        }

        csv_text = svc.export_summary(batch.batch_id, format="csv")
        header = csv_text.splitlines()[0].split(",")
        assert "Report Validity" in header
        assert "Review Required" in header
        assert "Classifier Alerts" in header
        assert "Triage State" in header

        row = csv_text.splitlines()[1]
        assert "completed_valid" in row
        assert "True" in row
        assert "potential_abnormality_signal" in row
