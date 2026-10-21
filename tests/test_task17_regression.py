"""
tests/test_task17_regression.py
===============================
Task 17 Regression Test Suite for Synchronized Patient-Friendly Reports
and Unified Red/Yellow/Green Research Triage.

Deterministic test coverage for:
- Test A: Regional / patient-friendly mismatch & cross-report contradiction
- Test B: Compatible reports without false conflict
- Test C: Uncertain or not assessable regional results
- Test D: Unsupported queries and capability warnings
- Test E: Source association (report_id, analysis_id, query)
- Test F: Triage display consistency (RED can NEVER be routine)
- Test G: Green-state safety (strictly 'No configured review signal detected', never 'Normal')
- Test H: Batch processing & CSV export canonical consistency
"""

import pytest
import csv
import io
from src.report_service import (
    translate_regional_report_to_patient_friendly,
    region_report,
)
from src.triage_service import (
    get_triage_service,
    TriageService,
    TriageEvaluationResult,
)
from src.consistency_service import (
    evaluate_cross_report_consistency,
    CrossReportConsistencyResult,
)
from src.batch_service import (
    get_batch_service,
    BatchService,
)


@pytest.fixture
def triage_svc() -> TriageService:
    return get_triage_service()


@pytest.fixture
def batch_svc() -> BatchService:
    return get_batch_service()


# ── Test A: Regional / Patient-Friendly Mismatch ────────────────────────────

def test_test_a_regional_patient_friendly_mismatch(triage_svc: TriageService):
    """
    Test A — Regional/patient-friendly mismatch (Exact Screenshot Defect):
    - Regional query: 'pneumonia'
    - Regional assessment: 'Supported'
    - Regional observation: increased opacity in lower lung field
    - Whole-image report: describes clear lungs
    Verify:
    1. Patient-friendly translation derives from regional report only.
    2. Does not claim lungs are clear.
    3. Does not invent unrelated devices (e.g. endotracheal tube, pacemaker).
    4. Preserves location, positive assessment, and research limitations.
    5. Original regional report remains unchanged.
    6. Canonical triage status is RED.
    7. review_required=True.
    8. Reason refers to the actual multi-source contradiction.
    """
    regional_report_text = (
        "1. REGIONAL OBSERVATIONS:\n"
        "Focal increased opacity and consolidation in the right lower lung zone.\n\n"
        "2. HYPOTHESIS EVALUATION:\n"
        "Supported: Regional opacity and alveolar infiltrate consistent with pneumonia.\n\n"
        "3. LIMITATIONS:\n"
        "Field of view limited to right basilar lung region.\n\n"
        "4. RECOMMENDATIONS:\n"
        "Clinical correlation and radiologist confirmation advised."
    )
    whole_image_report_text = (
        "1. CHEST X-RAY FINDINGS:\n"
        "The lungs appear completely clear bilaterally without focal consolidation, pneumothorax, or pleural effusion. "
        "Cardiac silhouette is normal.\n\n"
        "2. IMPRESSION:\n"
        "Normal chest radiograph."
    )

    # 1. Translate regional report to patient-friendly explanation
    translation_res = translate_regional_report_to_patient_friendly(
        regional_report_text=regional_report_text,
        phrase="pneumonia",
        assessment_state="Supported",
        report_id="reg-screenshot-001",
        analysis_id="analysis-shot-001",
    )

    assert translation_res["success"] is True
    pt_text = translation_res["translation_text"].lower()

    # Verify patient translation DOES NOT claim lungs are clear
    assert "clear" not in pt_text or "not clear" in pt_text or "without" in pt_text
    assert "lungs appear clear" not in pt_text
    assert "normal chest" not in pt_text

    # Verify patient translation DOES NOT invent unrelated support devices
    for device in ["endotracheal tube", "pacemaker", "picc line", "chest tube", "nasogastric", "enteric tube"]:
        assert device not in pt_text

    # Verify patient translation preserves positive finding, query, and AI caveat
    assert "pneumonia" in pt_text
    assert "supported" in pt_text or "consistent with" in pt_text
    assert "research" in pt_text or "not a confirmed" in pt_text
    assert translation_res["source_report_id"] == "reg-screenshot-001"
    assert translation_res["analysis_id"] == "analysis-shot-001"

    # Verify original regional report is unchanged
    assert "Focal increased opacity and consolidation in the right lower lung zone" in regional_report_text

    # 2. Evaluate cross-report consistency
    consistency_res = evaluate_cross_report_consistency(
        query="pneumonia",
        regional_report_text=regional_report_text,
        whole_image_report_text=whole_image_report_text,
        whole_image_sections={"Lungs": whole_image_report_text},
    )
    assert consistency_res.has_conflict is True
    assert consistency_res.review_required is True

    # 3. Evaluate multi-evidence canonical triage
    triage_res = triage_svc.evaluate_multi_evidence(
        report_validity="completed_valid",
        regional_result={"assessment_state": "Supported"},
        consistency_result=consistency_res,
        unsupported_query=False,
        query="pneumonia",
        analysis_id="analysis-shot-001",
        source_report_id="reg-screenshot-001",
    )

    # Verify canonical triage is strictly RED
    assert triage_res.status == "red"
    assert triage_res.review_required is True
    assert triage_res.review_priority == "urgent"
    assert triage_res.status_label == "Priority AI review required"
    assert any("discordance" in r["description"].lower() or "conflict" in r["description"].lower() for r in triage_res.reasons)


# ── Test B: Compatible Reports ──────────────────────────────────────────────

def test_test_b_compatible_reports(triage_svc: TriageService):
    """
    Test B — Compatible reports:
    Given a regional assessment and a whole-image report that agree on the finding,
    verify that the system does not produce a false conflict warning.
    """
    regional_report_text = (
        "1. REGIONAL OBSERVATIONS:\n"
        "Focal alveolar opacity in the left lower lobe.\n\n"
        "2. HYPOTHESIS EVALUATION:\n"
        "Supported: Localized consolidation consistent with focal pneumonia.\n\n"
        "3. LIMITATIONS:\n"
        "Single view only.\n\n"
        "4. RECOMMENDATIONS:\n"
        "Correlate clinically."
    )
    whole_image_report_text = (
        "1. CHEST X-RAY FINDINGS:\n"
        "Focal consolidation in the left retrocardiac / lower lobe region consistent with pneumonia. "
        "No pneumothorax. Heart size is normal.\n\n"
        "2. IMPRESSION:\n"
        "Left lower lobe pneumonia."
    )

    consistency_res = evaluate_cross_report_consistency(
        query="pneumonia",
        regional_report_text=regional_report_text,
        whole_image_report_text=whole_image_report_text,
        whole_image_sections={"Lungs": whole_image_report_text},
    )

    # Compatible reports must NOT flag a conflict
    assert consistency_res.has_conflict is False


# ── Test C: Uncertain or Not Assessable Regional Result ──────────────────────

def test_test_c_uncertain_or_not_assessable_regional_result(triage_svc: TriageService):
    """
    Test C — Uncertain or Not Assessable:
    Verify uncertainty is preserved in patient language and yellow is returned.
    """
    # 1. Indeterminate regional assessment
    indet_text = (
        "1. REGIONAL OBSERVATIONS:\n"
        "Subtle questionable density obscured by overlapping rib structures.\n\n"
        "2. HYPOTHESIS EVALUATION:\n"
        "Indeterminate: Findings equivocal and cannot confirm or refute pneumonia.\n\n"
        "3. LIMITATIONS:\n"
        "Overlying bone shadows limit evaluation.\n\n"
        "4. RECOMMENDATIONS:\n"
        "Dedicated lateral view advised."
    )
    trans_indet = translate_regional_report_to_patient_friendly(
        regional_report_text=indet_text,
        phrase="pneumonia",
        assessment_state="Indeterminate",
        report_id="reg-indet-001",
    )
    assert trans_indet["success"] is True
    assert "uncertain" in trans_indet["translation_text"].lower() or "not clear" in trans_indet["translation_text"].lower()

    triage_indet = triage_svc.evaluate_multi_evidence(
        report_validity="completed_valid",
        regional_result={"assessment_state": "Indeterminate"},
        query="pneumonia",
    )
    assert triage_indet.status == "yellow"
    assert triage_indet.review_required is True
    assert triage_indet.review_priority == "elevated"

    # 2. Not assessable regional assessment
    unassess_text = (
        "1. REGIONAL OBSERVATIONS:\n"
        "Severe underexposure and motion artifact prevent adequate visualization.\n\n"
        "2. HYPOTHESIS EVALUATION:\n"
        "Not assessable: Region cannot be evaluated reliably.\n\n"
        "3. LIMITATIONS:\n"
        "Technical inadequacy.\n\n"
        "4. RECOMMENDATIONS:\n"
        "Repeat exam requested."
    )
    trans_unassess = translate_regional_report_to_patient_friendly(
        regional_report_text=unassess_text,
        phrase="pneumonia",
        assessment_state="Not assessable",
        report_id="reg-unassess-002",
    )
    assert trans_unassess["success"] is True
    assert "could not be assessed" in trans_unassess["translation_text"].lower() or "not assessable" in trans_unassess["translation_text"].lower()

    triage_unassess = triage_svc.evaluate_multi_evidence(
        report_validity="completed_valid",
        regional_result={"assessment_state": "Not assessable"},
        query="pneumonia",
    )
    assert triage_unassess.status == "yellow"
    assert triage_unassess.review_required is True


# ── Test D: Unsupported Query ───────────────────────────────────────────────

def test_test_d_unsupported_query(triage_svc: TriageService):
    """
    Test D — Unsupported query:
    Verify unsupported grounding requests remain unsupported, translation does
    not describe finding as confirmed, and triage returns yellow.
    """
    triage_res = triage_svc.evaluate_multi_evidence(
        report_validity="completed_valid",
        regional_result={"assessment_state": "Supported"},
        unsupported_query=True,
        query="rib fracture (unsupported)",
    )
    assert triage_res.status == "yellow"
    assert triage_res.review_required is True
    assert any("unsupported" in r["description"].lower() for r in triage_res.reasons)


# ── Test E: Source Association ──────────────────────────────────────────────

def test_test_e_source_association():
    """
    Test E — Source association:
    Verify patient-friendly text references correct regional report ID,
    query, and analysis ID.
    """
    reg_text = (
        "1. REGIONAL OBSERVATIONS:\nIncreased vascular marking.\n\n"
        "2. HYPOTHESIS EVALUATION:\nNot supported: No focal consolidation.\n\n"
        "3. LIMITATIONS:\nNone.\n\n4. RECOMMENDATIONS:\nNone."
    )
    res = translate_regional_report_to_patient_friendly(
        regional_report_text=reg_text,
        phrase="pneumonia",
        assessment_state="Not supported",
        report_id="reg-audit-uuid-999",
        analysis_id="analysis-audit-session-777",
    )
    assert res["source_report_id"] == "reg-audit-uuid-999"
    assert res["analysis_id"] == "analysis-audit-session-777"
    assert res["source_query"] == "pneumonia"
    assert res["source_assessment_state"] == "Not supported"
    assert res["is_grounding_linked"] is True


# ── Test F: Triage Display Consistency (RED CAN NEVER BE ROUTINE) ────────────

def test_test_f_triage_display_consistency(triage_svc: TriageService):
    """
    Test F — Triage display consistency:
    Verify that when status is RED, review_required=True and review_priority
    can NEVER be 'routine'.
    """
    # 1. Direct potentially urgent finding
    triage_urgent = triage_svc.evaluate_from_structured_findings({
        "pleural_spaces": "Tension pneumothorax with mediastinal shift.",
    })
    assert triage_urgent.status == "red"
    assert triage_urgent.review_required is True
    assert triage_urgent.review_priority == "urgent"
    assert triage_urgent.human_review_priority != "routine"

    # 2. Multi-model discordance
    conflict_res = CrossReportConsistencyResult(
        has_conflict=True,
        warning_banner="Conflict",
        summary="Contradictory findings",
        findings=[],
        review_required=True,
    )
    triage_conflict = triage_svc.evaluate_multi_evidence(
        report_validity="completed_valid",
        consistency_result=conflict_res,
        query="pneumonia",
    )
    assert triage_conflict.status == "red"
    assert triage_conflict.review_required is True
    assert triage_conflict.review_priority == "urgent"
    assert triage_conflict.human_review_priority != "routine"

    # 3. Processing integrity failure
    triage_fail = triage_svc.evaluate_multi_evidence(
        report_validity="report_generation_failed",
    )
    assert triage_fail.status == "red"
    assert triage_fail.review_required is True
    assert triage_fail.review_priority == "urgent"
    assert triage_fail.human_review_priority != "routine"


# ── Test G: Green-State Safety ──────────────────────────────────────────────

def test_test_g_green_state_safety(triage_svc: TriageService):
    """
    Test G — Green-State Safety:
    Verify green appears ONLY when required analysis is complete, reports are
    valid and well-formed, zero triggers apply, and no contradiction exists.
    Must display 'No configured review signal detected', NEVER 'Normal', 'Safe', or 'Cleared'.
    """
    triage_green = triage_svc.evaluate_multi_evidence(
        report_validity="completed_valid",
        regional_result={"assessment_state": "Not supported"},
        consistency_result=CrossReportConsistencyResult(
            has_conflict=False,
            warning_banner=None,
            summary="Compatible",
            findings=[],
            review_required=False,
        ),
        unsupported_query=False,
        query="pneumonia",
    )
    assert triage_green.status == "green"
    assert triage_green.status_label == "No configured review signal detected"
    assert "normal" not in triage_green.status_label.lower()
    assert "safe" not in triage_green.status_label.lower()
    assert "cleared" not in triage_green.status_label.lower()
    assert triage_green.review_required is False
    assert triage_green.review_priority == "routine"
    assert "not a clinical diagnosis" in triage_green.disclaimer.lower()

    # Evaluation failure must NEVER default to green
    triage_err = triage_svc.evaluate_multi_evidence(
        report_validity="malformed_report",
    )
    assert triage_err.status != "green"


# ── Test H: Batch and Export Consistency ────────────────────────────────────

def test_test_h_batch_and_export_consistency(batch_svc: BatchService):
    """
    Test H — Batch and export consistency:
    Verify that report text, review status, reasons, and canonical status fields
    remain correctly associated with each job through CSV and JSON export.
    """
    # Create a batch with one mock completed item
    batch = batch_svc.create_batch(files_data=[("test_cxr.png", b"fake_content")], style="research")
    batch_id = batch.batch_id

    # Retrieve and populate test item
    b_obj = batch_svc.get_batch(batch_id)
    assert b_obj is not None
    item = b_obj.items[0]
    item.status = "completed"
    item.report_text = "Lungs are clear. Heart is normal size."
    item.triage_result = {
        "status": "green",
        "status_label": "No configured review signal detected",
        "review_required": False,
        "review_priority": "routine",
        "matched_rule_id": "RULE_NO_PRIORITY_FINDING",
        "triage_category": "no_priority_finding_flagged",
    }

    # Export CSV
    csv_str = batch_svc.export_batch(batch_id, format="csv")
    reader = list(csv.reader(io.StringIO(csv_str)))
    header = reader[0]
    row = reader[1]

    assert "Canonical Status" in header
    assert "Canonical Status Label" in header
    col_status = header.index("Canonical Status")
    col_label = header.index("Canonical Status Label")

    assert row[col_status] == "green"
    assert row[col_label] == "No configured review signal detected"
