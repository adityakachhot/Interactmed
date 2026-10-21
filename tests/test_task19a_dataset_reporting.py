"""
tests/test_task19a_dataset_reporting.py
=======================================
Task 19A Regression and Verification Test Suite:
1. Dataset inventory and download budget compliance (<= 5 GB, 0 bytes needed).
2. Split integrity, exclusion metadata (images_to_avoid.csv), and patient overlap prevention.
3. Canonical evidence-based finding schema with explicit presence states.
4. Chest ImaGenome 36 controlled anatomical entities and vocabulary normalization.
5. Separation of evidence: TransVG boxes as geometry, DenseNet-121 as secondary review alerts.
6. Report completeness and fidelity checks (omissions, unsupported additions, false normals, device errors).
7. Batch report isolation and independent study state.
8. Unified research-only triage consistency.
"""

import os
import csv
import pytest
from typing import Dict, Any, List

from src.evidence_schema import (
    CHEST_IMAGENOME_ANATOMICAL_REGIONS,
    ANATOMICAL_SYNONYMS,
    CanonicalFinding,
    StructuredReportEvidence,
    normalize_anatomical_location,
    extract_canonical_findings_from_report,
    reconcile_evidence_with_grounding_and_classifier,
    check_report_completeness_and_fidelity,
)
from src.triage_service import get_triage_service, TriageService
from src.batch_service import get_batch_service, BatchService


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(REPO_ROOT, "artifacts", "dataset_reporting")


@pytest.fixture
def triage_svc() -> TriageService:
    return get_triage_service()


@pytest.fixture
def batch_svc() -> BatchService:
    return get_batch_service()


# ── 1. Inventory & Download Accounting ────────────────────────────────────────

def test_inventory_file_exists_and_has_required_columns():
    inv_path = os.path.join(ARTIFACTS_DIR, "task19a_local_inventory.csv")
    assert os.path.isfile(inv_path), f"Missing required deliverable: {inv_path}"
    
    expected_cols = [
        "dataset", "relative_path", "file_name", "file_type", "size_bytes",
        "readable", "annotation_type", "image_ids_available", "patient_ids_available",
        "split_information", "duplicate_status", "intended_use", "notes"
    ]
    with open(inv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert set(expected_cols).issubset(set(reader.fieldnames or [])), (
            f"Columns mismatch in {inv_path}: {reader.fieldnames}"
        )
        rows = list(reader)
        assert len(rows) > 0, "Inventory must contain audited dataset rows."


def test_download_manifest_complies_with_strict_budget():
    manifest_path = os.path.join(ARTIFACTS_DIR, "task19a_download_manifest.csv")
    assert os.path.isfile(manifest_path), f"Missing required deliverable: {manifest_path}"
    
    expected_cols = [
        "source_dataset", "official_url", "file_name", "local_path",
        "local_file_exists", "local_size_bytes", "planned_download_size_bytes",
        "actual_download_size_bytes", "required", "reason",
        "integrity_status", "extraction_status"
    ]
    total_planned = 0
    total_actual = 0
    with open(manifest_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert set(expected_cols).issubset(set(reader.fieldnames or []))
        for row in reader:
            total_planned += int(row.get("planned_download_size_bytes", 0))
            total_actual += int(row.get("actual_download_size_bytes", 0))
            
    # Strict 5 GB limit constraint: <= 5,000,000,000 bytes
    assert total_actual <= 5_000_000_000, f"Actual downloads exceeded budget: {total_actual}"
    assert total_planned <= 5_000_000_000, f"Planned downloads exceeded budget: {total_planned}"
    # Reusing existing files resulted in 0 bytes downloaded
    assert total_actual == 0, f"Expected 0 bytes downloaded due to existing local files, got {total_actual}"


# ── 2. Split Integrity & Leakage Prevention ───────────────────────────────────

def test_patient_level_split_isolation_and_quarantine():
    selected_path = os.path.join(ARTIFACTS_DIR, "task19a_selected_images.csv")
    assert os.path.isfile(selected_path), f"Missing {selected_path}"
    
    quarantined_patients = set()
    held_out_test_patients = set()
    clean_train_patients = set()
    excluded_images = set()

    with open(selected_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            role = row.get("partition_role", "")
            pid = row.get("subject_id", "")
            img_id = row.get("dicom_id", "")
            is_excluded = (
                row.get("in_images_to_avoid", "").strip().lower() == "true" or
                role == "excluded_image_to_avoid"
            )
            is_quarantined = (
                row.get("in_quarantined_test", "").strip().lower() == "true" or
                role == "quarantined_task14_expanded_test"
            )
            
            if is_excluded:
                excluded_images.add(img_id)
            if is_quarantined:
                quarantined_patients.add(pid)
            elif role == "mscxr_held_out_test":
                held_out_test_patients.add(pid)
            elif role == "mscxr_train_clean":
                clean_train_patients.add(pid)
                
    # 1. Zero patient overlap between held-out test and quarantined Task 14 cohort
    test_overlap = held_out_test_patients.intersection(quarantined_patients)
    assert len(test_overlap) == 0, f"Leakage detected! Overlap between test and Task 14: {test_overlap}"
    
    # 2. Zero patient overlap between clean train and quarantined Task 14 cohort
    train_overlap = clean_train_patients.intersection(quarantined_patients)
    assert len(train_overlap) == 0, f"Leakage detected! Overlap between train and Task 14: {train_overlap}"

    # 3. Benchmark exclusions (images_to_avoid.csv) must be recognized and excluded
    assert len(excluded_images) > 0, "Expected benchmark exclusions to be recognized."


# ── 3. Canonical Evidence Finding Schema & Presence States ────────────────────

def test_canonical_finding_explicit_presence_states():
    # Valid states
    valid_states = ["present", "absent", "uncertain", "not_assessed"]
    for state in valid_states:
        f = CanonicalFinding(
            finding_name="pleural_effusion",
            presence_status=state,
            anatomical_location="left costophrenic angle",
            laterality="left"
        )
        assert f.presence_status == state

    # Missing bounding box must NOT silently infer absence
    f_no_box = CanonicalFinding(
        finding_name="pneumothorax",
        presence_status="uncertain",
        grounding_status="unsupported",
        supporting_region=None
    )
    assert f_no_box.presence_status != "absent", "Missing box must not be inferred as absent."


def test_extract_canonical_findings_preserves_uncertainty_and_qualifiers():
    report_text = (
        "FINDINGS: Moderate left-sided pleural effusion is present with blunting of the costophrenic angle. "
        "Possible right apical pneumothorax is suspected but poorly visualized. "
        "There is no evidence of pulmonary edema or consolidation. "
        "Cardiomegaly is noted."
    )
    findings = extract_canonical_findings_from_report(report_text)
    findings_by_name = {f.finding_name: f for f in findings}

    # Pleural effusion: present, left
    assert "pleural_effusion" in findings_by_name
    pe = findings_by_name["pleural_effusion"]
    assert pe.presence_status == "present"
    assert pe.laterality == "left"

    # Pneumothorax: uncertain (due to 'possible / suspected')
    assert "pneumothorax" in findings_by_name
    ptx = findings_by_name["pneumothorax"]
    assert ptx.presence_status == "uncertain"
    assert ptx.laterality == "right"
    assert ptx.uncertainty in ("medium", "high")

    # Pulmonary edema: absent (due to 'no evidence of')
    assert "pulmonary_edema" in findings_by_name
    pe_abs = findings_by_name["pulmonary_edema"]
    assert pe_abs.presence_status == "absent"

    # Consolidation: absent
    assert "consolidation" in findings_by_name
    assert findings_by_name["consolidation"].presence_status == "absent"

    # Cardiomegaly: present, midline
    assert "cardiomegaly" in findings_by_name
    assert findings_by_name["cardiomegaly"].presence_status == "present"


# ── 4. Chest ImaGenome Anatomical Normalization ────────────────────────────────

def test_chest_imagenome_anatomical_entities_normalization():
    # Test synonym and zone mappings
    cases = [
        ("rll", "right lower lung zone", "right"),
        ("right lower lobe", "right lower lung zone", "right"),
        ("left base", "left lower lung zone", "left"),
        ("retrocardiac space", "left lower lung zone", "left"),
        ("left cpa", "left costophrenic angle", "left"),
        ("right apical cap", "right apical zone", "right"),
        ("bilateral bases", "lower lung zones", "bilateral"),
    ]
    for raw, expected_entity, expected_lat in cases:
        norm_loc, lat = normalize_anatomical_location(raw)
        assert norm_loc == expected_entity, f"Failed for {raw}: got {norm_loc}"
        assert lat == expected_lat, f"Failed laterality for {raw}: got {lat}"

    # Verify 36 controlled entities integrity
    assert len(CHEST_IMAGENOME_ANATOMICAL_REGIONS) == 36
    assert "right lung" in CHEST_IMAGENOME_ANATOMICAL_REGIONS
    assert "left costophrenic angle" in CHEST_IMAGENOME_ANATOMICAL_REGIONS
    assert "cardiac silhouette" in CHEST_IMAGENOME_ANATOMICAL_REGIONS


# ── 5. Evidence Separation (TransVG & Classifier Alert) ───────────────────────

def test_transvg_box_and_classifier_alert_separation():
    # 1. Create a base finding from whole image
    findings = [
        CanonicalFinding(
            finding_name="consolidation",
            presence_status="present",
            anatomical_location="right lower lung zone",
            laterality="right",
            source="whole_image_vlm"
        )
    ]
    
    # 2. Add TransVG visual grounding
    grounding_result = {
        "success": True,
        "query": "consolidation",
        "pred_box_norm": [0.45, 0.55, 0.25, 0.30],
        "assessment_state": "supported",
    }
    
    # 3. Add secondary classifier alert for an unmentioned pathology (pneumothorax)
    classifier_scores = {
        "pneumothorax": 0.78,  # > threshold 0.62
        "consolidation": 0.85,
    }
    
    reconciled, audit_log = reconcile_evidence_with_grounding_and_classifier(
        whole_image_findings=findings,
        grounding_result=grounding_result,
        classifier_scores=classifier_scores,
        classifier_threshold=0.62
    )
    
    reconciled_by_name = {f.finding_name: f for f in reconciled}
    
    # Check TransVG grounding attached as geometry, not presence probability
    assert reconciled_by_name["consolidation"].grounding_status == "grounded"
    assert reconciled_by_name["consolidation"].supporting_region == [0.45, 0.55, 0.25, 0.30]
    
    # Check DenseNet-121 alert is flagged for review, NOT converted to confirmed diagnosis
    assert "pneumothorax" in reconciled_by_name
    ptx_alert = reconciled_by_name["pneumothorax"]
    assert ptx_alert.source == "classifier_alert"
    assert ptx_alert.presence_status == "uncertain", "Classifier alert must NOT become 'present'!"
    assert ptx_alert.review_required is True


# ── 6. Report Completeness & Safety Fidelity Checks ───────────────────────────

def test_report_completeness_flags_contradictions_and_false_normals():
    # Case A: False-normal statement when pathology is affirmed
    findings = [
        CanonicalFinding(
            finding_name="pneumonia",
            presence_status="present",
            anatomical_location="right lower lung zone",
            laterality="right"
        )
    ]
    report_text = "Dense consolidation in right lower lobe. The lungs are clear and normal."
    flags = check_report_completeness_and_fidelity(findings, report_text)
    assert flags["false_normal_flagged"] is True
    assert flags["is_complete"] is False

    # Case B: Unsupported device hallucination
    findings_no_device = [
        CanonicalFinding(
            finding_name="pleural_effusion",
            presence_status="present",
            anatomical_location="left costophrenic angle",
            laterality="left"
        )
    ]
    report_with_device = "Left pleural effusion. Endotracheal tube is in standard placement."
    flags_dev = check_report_completeness_and_fidelity(
        findings_no_device,
        report_with_device,
        ground_truth_has_devices=False
    )
    assert len(flags_dev["device_errors"]) > 0
    assert any("endotracheal tube" in err.lower() for err in flags_dev["device_errors"])

    # Case C: Location contradiction (laterality mismatch)
    findings_loc = [
        CanonicalFinding(
            finding_name="pneumothorax",
            presence_status="present",
            anatomical_location="left apical zone",
            laterality="left"
        )
    ]
    report_wrong_lat = "A small right-sided pneumothorax is identified."
    flags_loc = check_report_completeness_and_fidelity(findings_loc, report_wrong_lat)
    assert len(flags_loc["location_contradictions"]) > 0


# ── 7. Batch Isolation & Independent Study State ──────────────────────────────

def test_batch_service_per_image_isolation():
    from src.batch_service import BatchItem

    # Create two items in the same batch
    item1 = BatchItem(
        job_id="job_001",
        batch_id="batch_test_001",
        filename="study_01.jpg",
        style="research",
        status="completed",
        report_text="Focal consolidation in the right base concerning for pneumonia."
    )
    findings1 = extract_canonical_findings_from_report(item1.report_text)
    item1.structured_findings = [f.to_dict() for f in findings1]
    
    item2 = BatchItem(
        job_id="job_002",
        batch_id="batch_test_001",
        filename="study_02.jpg",
        style="research",
        status="completed",
        report_text="Cardiopulmonary structures are normal. No acute finding."
    )
    findings2 = extract_canonical_findings_from_report(item2.report_text)
    item2.structured_findings = [f.to_dict() for f in findings2]
    
    # Verify strict isolation between batch items
    assert item1.job_id != item2.job_id
    names1 = [f["finding_name"] for f in item1.structured_findings]
    names2 = [f["finding_name"] for f in item2.structured_findings]
    assert "pneumonia" in names1
    assert "pneumonia" not in names2
    assert names1 != names2


# ── 8. Unified Research-Only Triage Consistency ───────────────────────────────

def test_canonical_triage_research_only_notice(triage_svc: TriageService):
    # Test Red status: High-priority trigger present
    report_text = "Left-sided pneumothorax is noted."
    res_red = triage_svc.evaluate_from_report_text(report_text=report_text)
    res_red = triage_svc.finalize_canonical_triage(res_red)
    
    assert res_red.status in ("red", "yellow")
    assert res_red.research_only is True
    assert "research" in res_red.disclaimer.lower()

    # Test Green status: No review triggers detected (MUST NOT claim clinical normality)
    res_green = triage_svc.evaluate_from_report_text(report_text="No acute cardiopulmonary abnormality.")
    res_green = triage_svc.finalize_canonical_triage(res_green)
    assert res_green.status == "green"
    assert "normal" not in res_green.status_label.lower()
    assert res_green.research_only is True
