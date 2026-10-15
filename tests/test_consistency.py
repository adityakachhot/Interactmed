"""
tests/test_consistency.py
==========================
Regression and integration test suite for cross-report consistency evaluation.

Covers Task 5A requirements:
- Detection of observation describing enlargement while assessment says Not supported (internal contradiction).
- Regional report conflicting with whole-image assessment (cross-report conflict).
- Compatible report outputs that should NOT trigger a warning.
- Reports containing insufficient information to determine consistency.
- Unsupported queries and missing report outputs (single-report-only).
- Preservation of all four regional assessment states without keyword override.
- Assurance that consistency handling NEVER silently upgrades an uncertain assessment to Supported.
- Preservation of original reports and existing triage disclaimer.
- Correct image association when reports are generated for multiple images.
- Endpoint POST /report/consistency contracts and schema validation.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from src.consistency_service import (
    evaluate_cross_report_consistency,
    check_regional_internal_contradiction,
    extract_regional_assessment_state,
    extract_cardiac_stance_whole_image,
    extract_regional_sections,
    FindingConsistency,
    CrossReportConsistencyResult,
)
from src.report_service import clean_and_structure_regional_report
from src.triage_service import get_triage_service


class TestRegionalInternalContradiction:
    """Verifies detection of internal contradiction within a single regional report."""

    def test_enlargement_observations_with_not_supported_assessment(self):
        """Image observations describe enlargement while assessment is marked Not supported."""
        obs = "The cardiac silhouette appears enlarged relative to the thoracic width with prominent contours."
        assess = "Not supported: No definitive radiographic evidence of cardiomegaly is demonstrated on the evaluated projection."

        has_conflict, note = check_regional_internal_contradiction(obs, assess, query="cardiomegaly")
        assert has_conflict is True
        assert note is not None
        assert "Internal report contradiction" in note
        assert "cardiac enlargement" in note
        assert "Not supported" in note

    def test_clean_and_structure_regional_report_detects_internal_contradiction(self):
        """clean_and_structure_regional_report attaches internal contradiction warning without altering text."""
        raw = (
            "TARGET FINDING:\ncardiomegaly\n\n"
            "IMAGE OBSERVATIONS:\nThe cardiac silhouette appears enlarged relative to the thoracic width.\n\n"
            "ASSESSMENT AND UNCERTAINTY:\nNot supported: No definitive evidence of cardiomegaly.\n\n"
            "LIMITATIONS:\nSingle frontal projection."
        )
        cleaned, warnings = clean_and_structure_regional_report(raw, phrase="cardiomegaly", is_cardiac=True)
        assert "Not supported:" in cleaned
        assert "The cardiac silhouette appears enlarged" in cleaned
        assert any("Internal report contradiction" in w for w in warnings)

    def test_normal_observations_with_not_supported_assessment_no_contradiction(self):
        """Compatible observations and assessment must not be flagged as contradictory."""
        obs = "The cardiac silhouette demonstrates normal transverse diameter and crisp borders."
        assess = "Not supported: No definitive radiographic evidence of cardiomegaly is demonstrated."

        has_conflict, note = check_regional_internal_contradiction(obs, assess, query="cardiomegaly")
        assert has_conflict is False
        assert note is None


class TestCrossReportConsistencyEvaluation:
    """Verifies cross-report consistency checks across the four distinct states."""

    def test_conflicting_cardiomegaly_regional_enlarged_vs_whole_normal(self):
        """The regional report describes enlargement while whole-image report describes normal heart size."""
        reg_text = (
            "TARGET FINDING:\ncardiomegaly\n\n"
            "IMAGE OBSERVATIONS:\nThe cardiac silhouette appears enlarged relative to the thoracic width.\n\n"
            "ASSESSMENT AND UNCERTAINTY:\nSupported: Visible cardiac contours are enlarged.\n\n"
            "LIMITATIONS:\nPortable projection."
        )
        whole_text = (
            "**Your X-ray in simple words**\nThe heart size is normal and the lungs appear clear.\n\n"
            "**What the findings may mean**\nNo cardiac enlargement is noted."
        )

        res = evaluate_cross_report_consistency(
            query="cardiomegaly",
            regional_report_text=reg_text,
            whole_image_report_text=whole_text,
        )

        assert res.has_conflict is True
        assert res.warning_banner == "Conflicting AI assessments — review required"
        assert "The regional report describes possible cardiac enlargement, while the whole-image report describes normal heart size." in res.summary
        assert len(res.findings) == 1
        assert res.findings[0].status == "conflicting"
        assert res.findings[0].regional_stance == "supported"
        assert res.findings[0].whole_image_stance == "normal"

    def test_compatible_cardiomegaly_both_normal(self):
        """Both regional and whole-image reports agree heart is normal -> compatible, no warning banner."""
        reg_text = (
            "TARGET FINDING:\ncardiomegaly\n\n"
            "IMAGE OBSERVATIONS:\nThe cardiac silhouette is within normal transverse limits.\n\n"
            "ASSESSMENT AND UNCERTAINTY:\nNot supported: No definitive evidence of cardiomegaly.\n\n"
            "LIMITATIONS:\nNone."
        )
        whole_text = (
            "**Your X-ray in simple words**\nThe heart size is normal with clear lung fields.\n\n"
            "**What the findings may mean**\nHeart borders are normal."
        )

        res = evaluate_cross_report_consistency(
            query="cardiomegaly",
            regional_report_text=reg_text,
            whole_image_report_text=whole_text,
        )

        assert res.has_conflict is False
        assert res.warning_banner is None
        assert res.findings[0].status == "compatible"
        assert res.findings[0].regional_stance == "not_supported"
        assert res.findings[0].whole_image_stance == "normal"

    def test_compatible_cardiomegaly_both_enlarged(self):
        """Both regional and whole-image reports describe enlargement -> compatible, no conflict."""
        reg_text = (
            "TARGET FINDING:\ncardiomegaly\n\n"
            "IMAGE OBSERVATIONS:\nThe cardiac silhouette appears enlarged.\n\n"
            "ASSESSMENT AND UNCERTAINTY:\nSupported: Features consistent with cardiomegaly.\n\n"
            "LIMITATIONS:\nAP portable."
        )
        whole_text = (
            "**Your X-ray in simple words**\nThe heart appears enlarged in the chest cavity.\n\n"
            "**What the findings may mean**\nCardiomegaly is suggested."
        )

        res = evaluate_cross_report_consistency(
            query="cardiomegaly",
            regional_report_text=reg_text,
            whole_image_report_text=whole_text,
        )

        assert res.has_conflict is False
        assert res.warning_banner is None
        assert res.findings[0].status == "compatible"
        assert res.findings[0].regional_stance == "supported"
        assert res.findings[0].whole_image_stance == "enlarged"

    def test_insufficient_information_not_assessable(self):
        """Finding technically unassessable in regional analysis -> insufficient_information, no false conflict."""
        reg_text = (
            "TARGET FINDING:\ncardiomegaly\n\n"
            "IMAGE OBSERVATIONS:\nSevere patient rotation and suboptimal inspiratory effort.\n\n"
            "ASSESSMENT AND UNCERTAINTY:\nNot assessable: Radiographic features cannot be adequately assessed.\n\n"
            "LIMITATIONS:\nSeverely limited study."
        )
        whole_text = "**Your X-ray in simple words**\nThe heart size is normal."

        res = evaluate_cross_report_consistency(
            query="cardiomegaly",
            regional_report_text=reg_text,
            whole_image_report_text=whole_text,
        )

        assert res.has_conflict is False
        assert res.findings[0].status == "insufficient_information"
        assert "Results could not be compared reliably" in res.summary

    def test_single_report_only_when_whole_image_missing(self):
        """Only regional report generated -> single_report_only status."""
        reg_text = (
            "TARGET FINDING:\npneumonia\n\n"
            "IMAGE OBSERVATIONS:\nPatchy opacity in the right lower lobe.\n\n"
            "ASSESSMENT AND UNCERTAINTY:\nSupported: Findings consistent with pneumonia.\n\n"
            "LIMITATIONS:\nSingle view."
        )

        res = evaluate_cross_report_consistency(
            query="pneumonia",
            regional_report_text=reg_text,
            whole_image_report_text=None,
        )

        assert res.has_conflict is False
        assert len(res.findings) == 1
        assert res.findings[0].status == "single_report_only"
        assert "not yet generated" in res.summary

    def test_unsupported_query_single_report_only(self):
        """Unsupported query (e.g. rib fracture) only evaluated in regional view."""
        reg_text = (
            "TARGET FINDING:\nrib fracture\n\n"
            "IMAGE OBSERVATIONS:\nVisual inspection of posterior ribs.\n\n"
            "ASSESSMENT AND UNCERTAINTY:\nIndeterminate: Equivocal cortication.\n\n"
            "LIMITATIONS:\nDedicated rib series recommended."
        )
        whole_text = "**Your X-ray in simple words**\nHeart size is normal and lungs are clear."

        res = evaluate_cross_report_consistency(
            query="rib fracture",
            regional_report_text=reg_text,
            whole_image_report_text=whole_text,
        )

        assert res.has_conflict is False
        assert res.findings[0].status == "single_report_only"


class TestAssessmentStatePreservation:
    """Verifies that all 4 assessment states are strictly preserved without keyword upgrades."""

    @pytest.mark.parametrize("state,prefix", [
        ("supported", "Supported:"),
        ("not_supported", "Not supported:"),
        ("indeterminate", "Indeterminate:"),
        ("not_assessable", "Not assessable:"),
    ])
    def test_extract_regional_assessment_state_preservation(self, state, prefix):
        text = f"TARGET FINDING:\ncardiomegaly\n\nIMAGE OBSERVATIONS:\nSample obs.\n\nASSESSMENT AND UNCERTAINTY:\n{prefix} Sample reasoning.\n\nLIMITATIONS:\nNone."
        extracted = extract_regional_assessment_state(text)
        assert extracted == state

    def test_consistency_handling_never_silently_upgrades_uncertain_to_supported(self):
        """Even if whole image describes an enlarged heart, an Indeterminate regional report is NEVER upgraded."""
        reg_text = (
            "TARGET FINDING:\ncardiomegaly\n\n"
            "IMAGE OBSERVATIONS:\nCardiac contours visible but patient AP projection introduces magnification.\n\n"
            "ASSESSMENT AND UNCERTAINTY:\nIndeterminate: Imaging features remain equivocal without lateral view.\n\n"
            "LIMITATIONS:\nMagnification on AP view."
        )
        whole_text = "**Your X-ray in simple words**\nThe heart appears enlarged."

        res = evaluate_cross_report_consistency(
            query="cardiomegaly",
            regional_report_text=reg_text,
            whole_image_report_text=whole_text,
        )

        # Regional stance must remain indeterminate
        assert res.findings[0].regional_stance == "indeterminate"
        # It must not be forced to supported
        assert res.findings[0].regional_stance != "supported"
        assert res.has_conflict is False


class TestTriageSafetyAndDisclaimers:
    """Verifies that triage preserves safety semantics and disclaimers during conflicts."""

    def test_triage_detects_cross_report_conflict_as_indeterminate_review_required(self):
        """When reports conflict, triage flags indeterminate_review_required, NOT no_priority_finding_flagged."""
        svc = get_triage_service()
        report_text = (
            "**Your X-ray in simple words**\nThe heart size is normal.\n\n"
            "**What the findings may mean**\nNormal examination."
        )

        # Normal report alone gives no_priority_finding_flagged
        normal_eval = svc.evaluate_from_report_text(report_text)
        assert normal_eval.triage_category == "no_priority_finding_flagged"

        # With cross_report_conflict=True, must flag indeterminate_review_required
        conflict_eval = svc.evaluate_from_report_text(report_text, cross_report_conflict=True)
        assert conflict_eval.triage_category == "indeterminate_review_required"
        assert conflict_eval.matched_rule_id == "RULE_INDETERMINATE_CONTRADICTORY_FINDINGS"
        assert conflict_eval.human_review_priority == "urgent"

    def test_triage_preserves_research_disclaimer(self):
        """Triage evaluation output must always preserve the mandatory non-clinical research disclaimer."""
        svc = get_triage_service()
        eval_res = svc.evaluate_from_report_text("Heart size is normal.")
        assert "RESEARCH" in eval_res.research_disclaimer
        assert "NOT been evaluated" in eval_res.research_disclaimer or "Not clinically validated" in eval_res.research_disclaimer

    def test_image_association_across_multiple_images(self):
        """Distinct images must preserve independent consistency evaluations without cross-contamination."""
        # Image A: Cardiomegaly conflict
        reg_a = "TARGET FINDING:\ncardiomegaly\n\nIMAGE OBSERVATIONS:\nEnlarged heart.\n\nASSESSMENT AND UNCERTAINTY:\nSupported: Enlarged.\n\nLIMITATIONS:\nNone."
        whole_a = "**Your X-ray in simple words**\nThe heart size is normal."

        # Image B: Pneumonia compatible
        reg_b = "TARGET FINDING:\npneumonia\n\nIMAGE OBSERVATIONS:\nFocal right lower lobe opacity.\n\nASSESSMENT AND UNCERTAINTY:\nSupported: Consistent with pneumonia.\n\nLIMITATIONS:\nNone."
        whole_b = "**Your X-ray in simple words**\nThere is shading in the lower right lung suggesting possible pneumonia."

        res_a = evaluate_cross_report_consistency("cardiomegaly", reg_a, whole_a)
        res_b = evaluate_cross_report_consistency("pneumonia", reg_b, whole_b)

        assert res_a.has_conflict is True
        assert res_a.findings[0].finding_name == "cardiomegaly"

        assert res_b.has_conflict is False
        assert res_b.findings[0].finding_name == "pneumonia"
        assert res_b.findings[0].status == "compatible"


class TestConsistencyAPIEndpoint:
    """FastAPI endpoint contracts for POST /report/consistency."""

    def test_post_report_consistency_endpoint_conflict(self):
        client = TestClient(app)
        payload = {
            "query": "cardiomegaly",
            "regional_report_text": (
                "TARGET FINDING:\ncardiomegaly\n\n"
                "IMAGE OBSERVATIONS:\nThe cardiac silhouette appears enlarged relative to thoracic width.\n\n"
                "ASSESSMENT AND UNCERTAINTY:\nSupported: Cardiomegaly confirmed.\n\n"
                "LIMITATIONS:\nNone."
            ),
            "whole_image_report_text": (
                "**Your X-ray in simple words**\nThe heart size is normal."
            ),
        }
        resp = client.post("/report/consistency", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_conflict"] is True
        assert data["warning_banner"] == "Conflicting AI assessments — review required"
        assert len(data["findings"]) == 1
        assert data["findings"][0]["status"] == "conflicting"

    def test_post_report_consistency_endpoint_compatible(self):
        client = TestClient(app)
        payload = {
            "query": "cardiomegaly",
            "regional_report_text": (
                "TARGET FINDING:\ncardiomegaly\n\n"
                "IMAGE OBSERVATIONS:\nThe heart is normal in size.\n\n"
                "ASSESSMENT AND UNCERTAINTY:\nNot supported: No evidence of cardiomegaly.\n\n"
                "LIMITATIONS:\nNone."
            ),
            "whole_image_report_text": (
                "**Your X-ray in simple words**\nThe heart size is normal."
            ),
        }
        resp = client.post("/report/consistency", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_conflict"] is False
        assert data["warning_banner"] is None
        assert data["findings"][0]["status"] == "compatible"
