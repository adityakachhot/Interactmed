"""
tests/test_triage_service.py
============================
Automated unit tests for InteractMed-VLM Triage Service.

Verifies:
1. Priority mappings for potentially urgent findings (pneumothorax, severe edema).
2. Priority mappings for flagged abnormalities (cardiomegaly, atelectasis, effusion, devices).
3. Correct negation handling (e.g. 'no pneumothorax' does NOT trigger urgent).
4. Indeterminate safeguards for conflicting findings and unassessable features.
5. Failed inference and invalid image handling.
6. 'no_priority_finding_flagged' output is not converted to a 'normal' clinical declaration.
7. Output data contract integrity and human-review status flags.
"""

import pytest
from src.triage_service import TriageService, TriageEvaluationResult, get_triage_service


@pytest.fixture
def service() -> TriageService:
    return get_triage_service()


class TestTriageService:
    def test_singleton_retrieval(self, service: TriageService):
        assert service is not None
        assert service.version == "1.0.0"
        assert len(service.rules) > 0

    def test_potentially_urgent_pneumothorax(self, service: TriageService):
        findings = {
            "lungs": "Left lung demonstrates apical lucency and pleural line.",
            "pleural_spaces": "Small left apical pneumothorax is identified without tension.",
            "cardiac": "Heart is normal.",
        }
        res = service.evaluate_from_structured_findings(findings)
        assert res.triage_category == "potentially_urgent"
        assert res.matched_rule_id == "RULE_POTENTIALLY_URGENT_PNEUMOTHORAX"
        assert res.human_review_priority == "urgent"
        assert res.requires_human_review is True
        assert any("pneumothorax" in f.lower() for f in res.supporting_findings)

    def test_potentially_urgent_severe_edema(self, service: TriageService):
        findings = {
            "lungs": "Bilateral alveolar edema with prominent perihilar bat wing opacities and fluid overload.",
            "cardiac": "Enlarged cardiac silhouette.",
        }
        res = service.evaluate_from_structured_findings(findings)
        assert res.triage_category == "potentially_urgent"
        assert res.matched_rule_id == "RULE_POTENTIALLY_URGENT_SEVERE_PULMONARY_EDEMA"
        assert res.human_review_priority == "urgent"

    def test_negation_pneumothorax_does_not_trigger_urgent(self, service: TriageService):
        findings = {
            "lungs": "Lungs are clear bilaterally. No focal consolidation.",
            "pleural_spaces": "Costophrenic angles sharp. There is no pneumothorax or pleural effusion.",
            "cardiac": "Cardiac silhouette is within normal limits.",
        }
        res = service.evaluate_from_structured_findings(findings)
        # Should NOT trigger potentially urgent pneumothorax because it is negated!
        assert res.triage_category != "potentially_urgent"
        assert res.triage_category == "no_priority_finding_flagged"
        assert res.matched_rule_id == "RULE_NO_PRIORITY_FINDINGS_FLAGGED"
        assert res.requires_human_review is True
        assert "does not guarantee" in res.rule_reason.lower() or "does not mean" in res.rule_reason.lower()

    def test_abnormality_flagged_cardiomegaly(self, service: TriageService):
        findings = {
            "lungs": "Lungs are clear. No acute infiltration.",
            "pleural_spaces": "No effusion.",
            "cardiac": "Cardiomegaly is noted with CTR > 0.5.",
        }
        res = service.evaluate_from_structured_findings(findings)
        assert res.triage_category == "abnormality_flagged"
        assert res.matched_rule_id == "RULE_ABNORMAL_CARDIOMEGALY"
        assert res.human_review_priority == "routine"
        assert res.requires_human_review is True

    def test_abnormality_flagged_atelectasis(self, service: TriageService):
        findings = {
            "lungs": "Streaky retrocardiac linear density representing bibasilar atelectasis.",
            "pleural_spaces": "Costophrenic angles sharp.",
            "cardiac": "Normal contour.",
        }
        res = service.evaluate_from_structured_findings(findings)
        assert res.triage_category == "abnormality_flagged"
        assert res.matched_rule_id == "RULE_ABNORMAL_ATELECTASIS"
        assert res.human_review_priority == "routine"

    def test_abnormality_flagged_support_apparatus(self, service: TriageService):
        findings = {
            "lungs": "Lungs appear clear.",
            "support_apparatus": "Dual-chamber pacemaker leads terminate in the right atrium and ventricle.",
            "cardiac": "Normal size.",
        }
        res = service.evaluate_from_structured_findings(findings)
        assert res.triage_category == "abnormality_flagged"
        assert res.matched_rule_id == "RULE_ABNORMAL_SUPPORT_APPARATUS_PRESENT"

    def test_indeterminate_failed_inference(self, service: TriageService):
        res = service.evaluate_from_structured_findings(
            structured_findings={},
            model_error="Connection to Ollama timed out after 180s",
        )
        assert res.triage_category == "indeterminate_review_required"
        assert res.matched_rule_id == "RULE_INDETERMINATE_FAILED_INFERENCE"
        assert res.uncertainty_level == "high"
        assert res.requires_human_review is True

    def test_indeterminate_invalid_image(self, service: TriageService):
        res = service.evaluate_from_structured_findings(
            structured_findings={},
            is_valid_image=False,
        )
        assert res.triage_category == "indeterminate_review_required"
        assert res.matched_rule_id == "RULE_INDETERMINATE_INVALID_IMAGE"
        assert res.uncertainty_level == "high"

    def test_indeterminate_internal_contradiction(self, service: TriageService):
        res = service.evaluate_from_structured_findings(
            structured_findings={"lungs": "Dense focal consolidation in right base."},
            internal_contradiction=True,
        )
        assert res.triage_category == "indeterminate_review_required"
        assert res.matched_rule_id == "RULE_INDETERMINATE_CONTRADICTORY_FINDINGS"
        assert res.uncertainty_level == "high"

    def test_indeterminate_unassessable_cues(self, service: TriageService):
        findings = {
            "lungs": "The lung bases are not assessable due to severe motion artifact and suboptimal inspiration.",
        }
        res = service.evaluate_from_structured_findings(findings)
        assert res.triage_category == "indeterminate_review_required"
        assert res.matched_rule_id == "RULE_INDETERMINATE_UNASSESSABLE_OR_EQUIVOCAL"
        assert res.human_review_priority == "urgent"

    def test_no_priority_findings_flagged_contract(self, service: TriageService):
        findings = {
            "lungs": "Lungs are clear bilaterally without focal consolidation.",
            "cardiac": "Cardiac silhouette is normal.",
            "pleural_spaces": "Both costophrenic angles are sharp.",
            "support_apparatus": "No tubes, lines, or devices identified.",
        }
        res = service.evaluate_from_structured_findings(findings)
        assert res.triage_category == "no_priority_finding_flagged"
        assert res.matched_rule_id == "RULE_NO_PRIORITY_FINDINGS_FLAGGED"
        # Must still mandate human review
        assert res.requires_human_review is True
        # Must not state the patient is normal or disease-free
        assert "not guarantee" in res.rule_reason.lower() or "does not mean" in res.rule_reason.lower()

    def test_evaluate_from_report_text_convenience(self, service: TriageService):
        report = (
            "OBSERVATIONS:\n"
            "Cardiomegaly is present with enlargement of the heart.\n"
            "Lungs are clear.\n\n"
            "IMPRESSION:\n"
            "Cardiomegaly without acute infiltration."
        )
        res = service.evaluate_from_report_text(report)
        assert res.triage_category == "abnormality_flagged"
        assert res.matched_rule_id == "RULE_ABNORMAL_CARDIOMEGALY"

    def test_evaluate_from_contradictory_report_text(self, service: TriageService):
        report = (
            "OBSERVATIONS:\n"
            "Focal consolidation and air bronchograms noted in the right lower lobe.\n\n"
            "IMPRESSION:\n"
            "Normal chest radiograph. No acute cardiopulmonary abnormality."
        )
        res = service.evaluate_from_report_text(report)
        assert res.triage_category == "indeterminate_review_required"
        assert res.matched_rule_id == "RULE_INDETERMINATE_CONTRADICTORY_FINDINGS"

    def test_to_dict_schema_completeness(self, service: TriageService):
        res = service.evaluate_from_structured_findings({"lungs": "Normal"})
        d = res.to_dict()
        required_keys = [
            "triage_category",
            "matched_rule_id",
            "rule_reason",
            "supporting_findings",
            "uncertainty_level",
            "evidence_notes",
            "requires_human_review",
            "human_review_priority",
            "rule_engine_version",
            "research_disclaimer",
        ]
        for k in required_keys:
            assert k in d, f"Missing key {k} in triage result dict"

    def test_non_interference_grounding_unaffected(self):
        """Verifies grounding predictor interfaces remain completely decoupled from triage."""
        from src.grounding_service import GroundingService
        assert hasattr(GroundingService, "predict")
        assert not hasattr(GroundingService, "evaluate_from_structured_findings")


class TestTriageBatchIntegration:
    def test_batch_item_triage_association(self):
        from src.batch_service import BatchService, BatchItem
        from src.triage_service import get_triage_service
        svc = BatchService(max_workers=1)
        item = BatchItem(
            job_id="job_triage_test_01",
            batch_id="batch_01",
            filename="pneumo_sample.png",
            style="patient",
            _image_bytes=b"dummy_bytes",
        )
        # Simulate completed report with pneumothorax finding
        report_text = "OBSERVATIONS: Small right pneumothorax noted.\nIMPRESSION: Right apical pneumothorax."
        triage = get_triage_service().evaluate_from_report_text(report_text)
        item.triage_result = triage.to_dict()
        item.status = "completed"

        assert item.triage_result is not None
        assert item.triage_result["triage_category"] == "potentially_urgent"
        assert item.triage_result["matched_rule_id"] == "RULE_POTENTIALLY_URGENT_PNEUMOTHORAX"

    def test_batch_csv_summary_includes_triage_columns(self):
        from src.batch_service import BatchService, BatchItem, BatchJob
        svc = BatchService(max_workers=1)
        batch = BatchJob(
            batch_id="batch_triage_test",
            created_at=1234567.0,
            style="patient",
            items=[
                BatchItem(
                    job_id="j001",
                    batch_id="batch_triage_test",
                    filename="chest_xray.png",
                    style="patient",
                    status="completed",
                    report_text="No acute cardiopulmonary abnormality.",
                    model_used="medgemma:4b",
                    latency_ms=15000.0,
                    triage_result={
                        "triage_category": "no_priority_finding_flagged",
                        "matched_rule_id": "RULE_NO_PRIORITY_FINDINGS_FLAGGED",
                        "human_review_priority": "standard_review",
                    },
                )
            ]
        )
        svc._batches[batch.batch_id] = batch
        csv_out = svc.export_summary(batch.batch_id, format="csv")

        assert "Triage Category" in csv_out
        assert "Matched Rule ID" in csv_out
        assert "Review Priority" in csv_out
        assert "no_priority_finding_flagged" in csv_out
        assert "RULE_NO_PRIORITY_FINDINGS_FLAGGED" in csv_out

