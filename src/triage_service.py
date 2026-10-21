"""
src/triage_service.py
=====================
Independent, transparent rule-based triage evaluation service for InteractMed-VLM.

Scientific & Operational Policy:
--------------------------------
1. Decoupled from report generation text synthesis and TransVG-AGPT phrase grounding.
2. Operates on structured radiological findings with explicit negation detection.
3. Categorizes each radiograph into one of four provisional research categories:
   - potentially_urgent: Predefined serious finding flagged; urgent human review required.
   - abnormality_flagged: Possible abnormality or non-critical finding present.
   - no_priority_finding_flagged: No priority finding flagged (DOES NOT mean normal or disease-free).
   - indeterminate_review_required: Insufficient evidence, conflicting outputs, or technical failure.
4. Absence of findings does NOT equal a confirmed "normal" diagnosis.
5. All outputs are marked with 'requires_human_review: true'.
6. Configurable via configs/triage_rules.yaml.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "triage_rules.yaml"

log = logging.getLogger("triage_service")
if not log.handlers:
    log.addHandler(logging.NullHandler())

TriageCategory = Literal[
    "potentially_urgent",
    "abnormality_flagged",
    "no_priority_finding_flagged",
    "indeterminate_review_required",
]

PRE_NEG_CUES = (
    r'\b(?:no|without|negative for|free of|clear of|denies|ruled out for|absence of|'
    r'unremarkable for|no evidence of|no sign of|no gross|no definite|no acute|'
    r'neither|clean of|fails to show|not identify|not identified|not seen|absent)\b'
)
POST_NEG_CUES = r'\b(?:is ruled out|are ruled out|not identified|not seen|absent|unremarkable)\b'


@dataclass
class TriageEvaluationResult:
    """Standardized output data contract for InteractMed-VLM triage evaluation."""
    triage_category: TriageCategory
    matched_rule_id: str
    rule_reason: str
    supporting_findings: List[str] = field(default_factory=list)
    uncertainty_level: Literal["low", "medium", "high"] = "medium"
    evidence_notes: str = ""
    requires_human_review: bool = True
    human_review_priority: Literal["urgent", "routine", "standard_review", "priority_review", "device_verification", "none"] = "standard_review"
    rule_engine_version: str = "1.1.0"
    research_disclaimer: str = (
        "Research-only AI review status. This is not a clinical diagnosis or a substitute for professional interpretation."
    )
    triage_state: str = ""
    matched_rule_ids: List[str] = field(default_factory=list)
    review_reasons: List[str] = field(default_factory=list)
    evidence_sources: Dict[str, Any] = field(default_factory=dict)

    # Task 17 Canonical Triage Data Contract
    status: Literal["red", "yellow", "green"] = "yellow"
    status_label: str = "Review advised"
    review_required: bool = True
    reasons: List[Dict[str, str]] = field(default_factory=list)
    review_priority: str = "routine"
    uncertainty: str = "medium"
    research_only: bool = True
    policy_version: str = "1.1.0"
    disclaimer: str = (
        "Research-only AI review status. This is not a clinical diagnosis or a substitute for professional interpretation."
    )
    analysis_id: Optional[str] = None
    source_report_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["review_required"] = self.review_required
        d["review_priority"] = self.review_priority
        d["uncertainty"] = self.uncertainty
        d["research_only"] = self.research_only
        d["policy_version"] = self.policy_version
        d["disclaimer"] = self.disclaimer
        return d


class TriageService:
    """
    Transparent, rule-based clinical triage engine.
    Evaluates structured radiological observations against configurable precedence rules.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or CONFIG_PATH
        self._load_config()

    def _load_config(self) -> None:
        """Loads and parses triage rules from YAML configuration."""
        if not self.config_path.exists():
            log.warning(f"Triage configuration not found at {self.config_path}. Using safe fallback.")
            self.version = "1.0.0"
            self.rules = []
            self.disclaimer = (
                "RESEARCH NOTICE: Experimental heuristic triage output. "
                "Not clinically validated. Human review is mandatory."
            )
            return

        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.version = str(cfg.get("version", "1.0.0"))
        self.disclaimer = cfg.get("disclaimer", "")
        # Sort rules by precedence (lowest precedence number = evaluated first)
        raw_rules = cfg.get("rules", [])
        self.rules = sorted(raw_rules, key=lambda r: r.get("precedence", 999))
        log.info(f"Loaded {len(self.rules)} triage rules from {self.config_path} (version {self.version})")

    @staticmethod
    def _is_negated(sentence: str, match_start: int, match_end: int) -> bool:
        """Determines if a matched pattern is negated within its sentence context."""
        s_lower = sentence.lower()
        pre_slice = s_lower[:match_start]
        post_slice = s_lower[match_end:]
        if re.search(PRE_NEG_CUES, pre_slice):
            return True
        if re.search(POST_NEG_CUES, post_slice):
            return True
        return False

    def extract_affirmed_patterns(self, text: str, patterns: List[str]) -> List[str]:
        """
        Scans text sentence by sentence to find affirmed occurrences of patterns.
        Filters out negated mentions (e.g. 'no pneumothorax').
        """
        affirmed_matches = []
        sentences = re.split(r'[\.\n;]+', text)

        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            s_lower = s_clean.lower()
            for pat in patterns:
                pat_lower = pat.lower()
                for m in re.finditer(rf'\b{re.escape(pat_lower)}\b', s_lower):
                    if not self._is_negated(s_lower, m.start(), m.end()):
                        affirmed_matches.append(s_clean)
                        break
        return affirmed_matches

    def evaluate_from_structured_findings(
        self,
        structured_findings: Dict[str, str],
        model_error: Optional[str] = None,
        is_valid_image: bool = True,
        internal_contradiction: bool = False,
    ) -> TriageEvaluationResult:
        """
        Evaluates a chest radiograph's structured findings through the rule engine.

        Args:
            structured_findings: Dict mapping anatomical/finding domains (e.g. 'lungs',
                'pleural_spaces', 'cardiac', 'support_apparatus', 'mediastinum') to extracted text.
            model_error: If inference failed, the error string.
            is_valid_image: False if image decode or technical checks failed.
            internal_contradiction: True if report validator detected contradictory findings.
        """
        # 1. Guardrail: Failed inference
        if model_error:
            res = TriageEvaluationResult(
                triage_category="indeterminate_review_required",
                matched_rule_id="RULE_INDETERMINATE_FAILED_INFERENCE",
                rule_reason="Inference failed or timed out during visual feature extraction.",
                supporting_findings=[f"Model error: {model_error[:120]}"],
                uncertainty_level="high",
                evidence_notes="Cannot assess radiograph due to upstream service failure.",
                requires_human_review=True,
                human_review_priority="urgent",
                rule_engine_version=self.version,
                research_disclaimer=self.disclaimer,
                triage_state="processing_error",
            )
            return self.finalize_canonical_triage(res)

        # 2. Guardrail: Invalid image
        if not is_valid_image:
            res = TriageEvaluationResult(
                triage_category="indeterminate_review_required",
                matched_rule_id="RULE_INDETERMINATE_INVALID_IMAGE",
                rule_reason="Input image is corrupt, unreadable, or not a diagnostic radiograph.",
                supporting_findings=["Image validation failed."],
                uncertainty_level="high",
                evidence_notes="Technical image quality is insufficient for review.",
                requires_human_review=True,
                human_review_priority="urgent",
                rule_engine_version=self.version,
                research_disclaimer=self.disclaimer,
                triage_state="processing_error",
            )
            return self.finalize_canonical_triage(res)

        # 3. Guardrail: Internal contradiction
        if internal_contradiction:
            res = TriageEvaluationResult(
                triage_category="indeterminate_review_required",
                matched_rule_id="RULE_INDETERMINATE_CONTRADICTORY_FINDINGS",
                rule_reason="Model outputs contain conflicting diagnostic statements between sections.",
                supporting_findings=["Conflicting statements detected in report text."],
                uncertainty_level="high",
                evidence_notes="Internal contradictions indicate model hallucination or divergence.",
                requires_human_review=True,
                human_review_priority="urgent",
                rule_engine_version=self.version,
                research_disclaimer=self.disclaimer,
                triage_state="conflicting_ai_assessments",
            )
            return self.finalize_canonical_triage(res, has_conflict=True, conflict_msg=res.rule_reason)

        # Consolidate finding text across domains
        full_findings_text = "\n".join(structured_findings.values()).strip()
        if not full_findings_text:
            res = TriageEvaluationResult(
                triage_category="indeterminate_review_required",
                matched_rule_id="RULE_INDETERMINATE_UNASSESSABLE_OR_EQUIVOCAL",
                rule_reason="No structured observation text provided for triage evaluation.",
                supporting_findings=[],
                uncertainty_level="high",
                evidence_notes="Missing observation text.",
                requires_human_review=True,
                human_review_priority="urgent",
                rule_engine_version=self.version,
                research_disclaimer=self.disclaimer,
                triage_state="insufficient_or_indeterminate_evidence",
            )
            return self.finalize_canonical_triage(res)

        # 4. Iterate over precedence rules
        for rule in self.rules:
            rule_id = rule.get("id", "")
            cat = rule.get("category", "")
            patterns = rule.get("patterns", [])
            target_domains = rule.get("target_domains", [])
            reason = rule.get("reason", "")
            rec_priority = rule.get("human_review_priority", "routine")

            # Check if this rule is a default fallback
            if rule_id == "RULE_NO_PRIORITY_FINDINGS_FLAGGED" or not patterns:
                if rule_id.startswith("RULE_INDETERMINATE_"):
                    # Only triggered by explicit guardrails above
                    continue
                # Default match if no earlier rule fired
                res = TriageEvaluationResult(
                    triage_category="no_priority_finding_flagged",
                    matched_rule_id=rule_id,
                    rule_reason=reason,
                    supporting_findings=[],
                    uncertainty_level="low",
                    evidence_notes="Automated analysis did not detect any predefined priority features.",
                    requires_human_review=True,
                    human_review_priority=rec_priority,
                    rule_engine_version=self.version,
                    research_disclaimer=self.disclaimer,
                    triage_state="no_configured_signal",
                )
                return self.finalize_canonical_triage(res)

            # Determine search text: targeted domains or all text
            if target_domains:
                domain_texts = [structured_findings.get(d, "") for d in target_domains if d in structured_findings]
                search_text = "\n".join(domain_texts)
            else:
                search_text = full_findings_text

            # Scan for affirmed patterns
            affirmed_findings = self.extract_affirmed_patterns(search_text, patterns)
            if affirmed_findings:
                # Deduplicate and truncate findings for clean logging / presentation
                clean_findings = list(dict.fromkeys([f[:140] for f in affirmed_findings]))[:4]

                uncertainty = "medium"
                if cat == "potentially_urgent":
                    uncertainty = "low" if len(clean_findings) >= 2 else "medium"
                elif cat == "indeterminate_review_required":
                    uncertainty = "high"

                res = TriageEvaluationResult(
                    triage_category=cat,
                    matched_rule_id=rule_id,
                    rule_reason=reason,
                    supporting_findings=clean_findings,
                    uncertainty_level=uncertainty,
                    evidence_notes=f"Flagged by rule {rule_id} via {len(clean_findings)} supporting finding(s).",
                    requires_human_review=True,
                    human_review_priority=rec_priority,
                    rule_engine_version=self.version,
                    research_disclaimer=self.disclaimer,
                    triage_state="potential_abnormality_signal",
                )
                return self.finalize_canonical_triage(res)

        # 5. Default safety fallback
        res = TriageEvaluationResult(
            triage_category="no_priority_finding_flagged",
            matched_rule_id="RULE_NO_PRIORITY_FINDINGS_FLAGGED",
            rule_reason=(
                "No predefined priority findings were flagged. "
                "IMPORTANT: This does not mean the scan is normal or disease-free."
            ),
            supporting_findings=[],
            uncertainty_level="medium",
            evidence_notes="Completed scan with zero priority triggers.",
            requires_human_review=True,
            human_review_priority="standard_review",
            rule_engine_version=self.version,
            research_disclaimer=self.disclaimer,
            triage_state="no_configured_signal",
        )
        return self.finalize_canonical_triage(res)

    def evaluate_from_report_text(
        self,
        report_text: Optional[str],
        sections: Optional[Dict[str, str]] = None,
        error: Optional[str] = None,
        cross_report_conflict: bool = False,
        internal_contradiction: bool = False,
    ) -> TriageEvaluationResult:
        """
        Convenience evaluator from report draft text or section dictionary.
        Extracts observations and impression, checks contradictions, and evaluates triage.
        """
        if error or (not report_text and not sections):
            return self.evaluate_from_structured_findings(
                structured_findings={},
                model_error=error or "Report generation produced empty output",
                is_valid_image=True,
            )

        # Parse sections if not already provided
        sec_dict = sections or {}
        obs_text = (
            sec_dict.get("OBSERVATIONS")
            or sec_dict.get("IMAGE OBSERVATIONS")
            or sec_dict.get("Your X-ray in simple words")
            or sec_dict.get("1. Your X-ray in simple words")
            or ""
        )
        imp_text = (
            sec_dict.get("IMPRESSION")
            or sec_dict.get("OVERALL IMPRESSION")
            or sec_dict.get("ASSESSMENT AND UNCERTAINTY")
            or sec_dict.get("What the findings may mean")
            or sec_dict.get("2. What the findings may mean")
            or ""
        )

        # If sections not split, extract via regex
        if not obs_text and report_text:
            obs_m = re.search(
                r'(?:OBSERVATIONS|IMAGE OBSERVATIONS|Your X-ray in simple words|1\.\s*Your X-ray in simple words)\s*[:\n](.*?)(?=\b(?:IMPRESSION|OVERALL IMPRESSION|ASSESSMENT AND UNCERTAINTY|What the findings may mean|2\.\s*What the findings may mean)|$)',
                report_text,
                re.DOTALL | re.IGNORECASE,
            )
            if obs_m:
                obs_text = obs_m.group(1).strip()
            else:
                obs_text = report_text

        if not imp_text and report_text:
            imp_m = re.search(
                r'(?:IMPRESSION|OVERALL IMPRESSION|ASSESSMENT AND UNCERTAINTY|What the findings may mean|2\.\s*What the findings may mean)\s*[:\n](.*?)(?=\b(?:LIMITATIONS|UNCERTAINTY AND LIMITATIONS|What is uncertain|3\.\s*What is uncertain)|$)',
                report_text,
                re.DOTALL | re.IGNORECASE,
            )
            if imp_m:
                imp_text = imp_m.group(1).strip()

        # Check for contradictions: active pathology in observations but "normal" in impression
        contradiction = bool(internal_contradiction or cross_report_conflict)
        if obs_text and imp_text:
            has_pneumothorax = bool(re.search(r'\bpneumothorax\b', obs_text, re.IGNORECASE)) and not bool(re.search(PRE_NEG_CUES, obs_text, re.IGNORECASE))
            has_edema = bool(re.search(r'\bedema\b', obs_text, re.IGNORECASE)) and not bool(re.search(PRE_NEG_CUES, obs_text, re.IGNORECASE))
            has_consolidation = bool(re.search(r'\bconsolidation\b', obs_text, re.IGNORECASE)) and not bool(re.search(PRE_NEG_CUES, obs_text, re.IGNORECASE))
            has_cardiomegaly = bool(re.search(r'\b(?:cardiomegaly|enlarged\s+heart|enlarged\s+cardiac)\b', obs_text, re.IGNORECASE)) and not bool(re.search(PRE_NEG_CUES, obs_text, re.IGNORECASE))

            imp_claims_normal = bool(re.search(r'\b(?:normal|unremarkable|no acute|clear)\b', imp_text, re.IGNORECASE))
            if (has_pneumothorax or has_edema or has_consolidation or has_cardiomegaly) and imp_claims_normal:
                contradiction = True

        findings_map = {
            "lungs": obs_text,
            "pleural_spaces": obs_text,
            "cardiac": obs_text,
            "support_apparatus": obs_text,
            "mediastinum": obs_text,
            "impression": imp_text,
        }

        return self.evaluate_from_structured_findings(
            structured_findings=findings_map,
            internal_contradiction=contradiction,
        )

    def finalize_canonical_triage(
        self,
        res: TriageEvaluationResult,
        *,
        has_conflict: bool = False,
        conflict_msg: Optional[str] = None,
        unsupported_query: bool = False,
        query: Optional[str] = None,
        regional_assessment_state: Optional[str] = None,
        analysis_id: Optional[str] = None,
        source_report_id: Optional[str] = None,
    ) -> TriageEvaluationResult:
        """
        Enforces a single deterministic Red/Yellow/Green research review status
        with strict precedence rules:

        1. Processing / image integrity failure -> RED ("Priority AI review required")
        2. Cross-report discordance / multi-model contradiction -> RED ("Priority AI review required")
        3. Urgent findings / critical classifier alerts -> RED ("Priority AI review required")
        4. Equivocal, unassessable, unsupported, malformed, or advisory signals -> YELLOW ("Review advised")
        5. Valid complete evaluation with zero review triggers -> GREEN ("No configured review signal detected")
        """
        res.research_only = True
        res.policy_version = "1.1.0"
        res.disclaimer = (
            "Research-only AI review status. This is not a clinical diagnosis or a substitute for professional interpretation."
        )
        if analysis_id:
            res.analysis_id = analysis_id
        if source_report_id:
            res.source_report_id = source_report_id

        norm_reg_state = (regional_assessment_state or "").strip().lower().replace("_", " ")

        # Level 1: Processing / Image Integrity Failures (RED)
        if (
            res.matched_rule_id in ("RULE_INDETERMINATE_FAILED_INFERENCE", "RULE_INDETERMINATE_INVALID_IMAGE", "RULE_PROCESSING_ERROR")
            or res.triage_state == "processing_error"
        ):
            res.status = "red"
            res.status_label = "Priority AI review required"
            res.review_required = True
            res.human_review_priority = "urgent"
            res.review_priority = "urgent"
            res.uncertainty = "high"
            res.uncertainty_level = "high"
            res.triage_category = "indeterminate_review_required"
            res.reasons = [{"code": "CRITICAL_PROCESSING_FAILURE", "description": res.rule_reason}]
            res.review_reasons = [r["description"] for r in res.reasons]
            return res

        # Level 2: Cross-Report Contradiction / Multi-Model Discordance (RED)
        if (
            has_conflict
            or res.matched_rule_id in ("RULE_INDETERMINATE_CONFLICTING_AI_ASSESSMENTS", "RULE_INDETERMINATE_CONTRADICTORY_FINDINGS")
            or res.triage_state == "conflicting_ai_assessments"
        ):
            res.status = "red"
            res.status_label = "Priority AI review required"
            res.review_required = True
            res.human_review_priority = "urgent"
            res.review_priority = "urgent"
            res.uncertainty = "high"
            res.uncertainty_level = "high"
            res.triage_category = "indeterminate_review_required"
            reason_text = conflict_msg or res.rule_reason or "Material disagreement detected between AI analysis models."
            res.reasons = [{"code": "CROSS_REPORT_DISCORDANCE", "description": reason_text}]
            res.review_reasons = [reason_text]
            return res

        # Level 3: Urgent Findings / Urgent Classifier Signal (RED)
        is_urgent_rule = (
            res.triage_category == "potentially_urgent"
            or res.matched_rule_id in (
                "RULE_POTENTIALLY_URGENT_PNEUMOTHORAX",
                "RULE_POTENTIALLY_URGENT_SEVERE_PULMONARY_EDEMA",
                "RULE_POTENTIALLY_URGENT_CLASSIFIER_SIGNAL",
            )
        )
        if is_urgent_rule:
            res.status = "red"
            res.status_label = "Priority AI review required"
            res.review_required = True
            res.human_review_priority = "urgent"
            res.review_priority = "urgent"
            res.triage_category = "potentially_urgent"
            code = "URGENT_CLASSIFIER_SIGNAL" if "CLASSIFIER" in res.matched_rule_id else "POTENTIALLY_URGENT_FINDING"
            res.reasons = [{"code": code, "description": res.rule_reason}]
            res.review_reasons = [res.rule_reason]
            return res

        # Level 4: Incomplete or Malformed Report (YELLOW)
        if res.matched_rule_id == "RULE_INDETERMINATE_MALFORMED_OR_INCOMPLETE_REPORT":
            res.status = "yellow"
            res.status_label = "Review advised"
            res.review_required = True
            res.human_review_priority = "urgent"  # preserve existing requirement for malformed report
            res.review_priority = "elevated"
            res.uncertainty = "high"
            res.uncertainty_level = "high"
            res.reasons = [{"code": "INCOMPLETE_OR_MALFORMED_REPORT", "description": res.rule_reason}]
            res.review_reasons = [res.rule_reason]
            return res

        # Level 5: Regional Assessment Indeterminate or Not Assessable (YELLOW)
        if norm_reg_state in ("indeterminate", "not assessable"):
            res.status = "yellow"
            res.status_label = "Review advised"
            res.review_required = True
            res.human_review_priority = "routine"
            res.review_priority = "elevated"
            res.uncertainty = "high"
            res.uncertainty_level = "high"
            res.triage_category = "indeterminate_review_required"
            reason_text = f"Regional assessment is {regional_assessment_state}; automated findings require clinical verification."
            res.reasons = [{"code": "INDETERMINATE_REGIONAL_ASSESSMENT", "description": reason_text}]
            res.review_reasons = [reason_text]
            return res

        # Level 6: Unsupported Query Capability (YELLOW)
        if unsupported_query:
            res.status = "yellow"
            res.status_label = "Review advised"
            res.review_required = True
            res.human_review_priority = "routine"
            res.review_priority = "elevated"
            res.uncertainty = "high"
            res.uncertainty_level = "high"
            res.triage_category = "indeterminate_review_required"
            reason_text = f"Grounding query '{query or 'query'}' is unsupported by validated capability; output is experimental."
            res.reasons = [{"code": "UNSUPPORTED_QUERY_CAPABILITY", "description": reason_text}]
            res.review_reasons = [reason_text]
            return res

        # Level 7: Support Apparatus Present (YELLOW)
        if res.matched_rule_id == "RULE_ABNORMAL_SUPPORT_APPARATUS_PRESENT":
            res.status = "yellow"
            res.status_label = "Review advised"
            res.review_required = True
            res.human_review_priority = "routine"
            res.review_priority = "device_verification"
            res.triage_category = "abnormality_flagged"
            res.reasons = [{"code": "SUPPORT_APPARATUS_PRESENT", "description": res.rule_reason}]
            res.review_reasons = [res.rule_reason]
            return res

        # Level 8: Non-Urgent Abnormality Flagged (YELLOW)
        if res.triage_category == "abnormality_flagged":
            res.status = "yellow"
            res.status_label = "Review advised"
            res.review_required = True
            res.human_review_priority = "routine"
            res.review_priority = "elevated"
            code = "AUXILIARY_CLASSIFIER_ALERT" if "CLASSIFIER" in res.matched_rule_id else "ABNORMALITY_FLAGGED"
            res.reasons = [{"code": code, "description": res.rule_reason}]
            res.review_reasons = [res.rule_reason]
            return res

        # Level 9: Other Indeterminate Review Required (YELLOW)
        if res.triage_category == "indeterminate_review_required":
            res.status = "yellow"
            res.status_label = "Review advised"
            res.review_required = True
            if res.human_review_priority not in ("urgent", "routine"):
                res.human_review_priority = "routine"
            res.review_priority = "elevated"
            res.uncertainty = "high"
            res.uncertainty_level = "high"
            res.reasons = [{"code": "INDETERMINATE_EVIDENCE", "description": res.rule_reason}]
            res.review_reasons = [res.rule_reason]
            return res

        # Level 10: GREEN — No Configured Review Signal Detected (GREEN)
        if res.triage_category == "no_priority_finding_flagged":
            res.status = "green"
            res.status_label = "No configured review signal detected"
            res.review_required = False
            res.human_review_priority = "routine"
            res.review_priority = "routine"
            res.uncertainty = "low"
            res.uncertainty_level = "low"
            res.triage_category = "no_priority_finding_flagged"
            reason_text = "All configured research evaluations completed without triggering priority or advisory review rules."
            res.reasons = [{"code": "NO_CONFIGURED_REVIEW_SIGNAL", "description": reason_text}]
            res.review_reasons = [reason_text]
            return res

        # Default fallback (YELLOW)
        res.status = "yellow"
        res.status_label = "Review advised"
        res.review_required = True
        res.human_review_priority = "routine"
        res.review_priority = "elevated"
        res.reasons = [{"code": "ADVISORY_REVIEW_REQUIRED", "description": res.rule_reason}]
        res.review_reasons = [res.rule_reason]
        return res

    def evaluate_multi_evidence(
        self,
        report_text: Optional[str] = None,
        sections: Optional[Dict[str, str]] = None,
        structured_findings: Optional[Dict[str, str]] = None,
        classifier_result: Optional[Union[Dict[str, Any], Any]] = None,
        consistency_result: Optional[Union[Dict[str, Any], Any]] = None,
        grounding_result: Optional[Union[Dict[str, Any], Any]] = None,
        regional_result: Optional[Union[Dict[str, Any], Any]] = None,
        query: Optional[str] = None,
        report_validity: str = "completed_valid",
        model_error: Optional[str] = None,
        is_valid_image: bool = True,
        unsupported_query: bool = False,
        analysis_id: Optional[str] = None,
        source_report_id: Optional[str] = None,
    ) -> TriageEvaluationResult:
        """
        Unified multi-evidence triage evaluator integrating:
        1. MedGemma generative report and structured findings
        2. TorchXRayVision DenseNet-121 secondary review signals
        3. TransVG grounding outputs and query support
        4. Cross-report consistency & discordance results
        5. Report structural validity status
        6. Regional assessment state and query
        """
        # Extract regional assessment state if provided
        reg_state: Optional[str] = None
        if regional_result:
            if isinstance(regional_result, dict):
                reg_state = regional_result.get("assessment_state")
            else:
                reg_state = getattr(regional_result, "assessment_state", None)

        evidence: Dict[str, Any] = {
            "report_validity": report_validity,
            "has_classifier": classifier_result is not None,
            "has_consistency": consistency_result is not None,
            "has_grounding": grounding_result is not None,
            "has_regional": regional_result is not None,
            "regional_assessment_state": reg_state,
            "unsupported_query": unsupported_query,
            "query": query,
        }

        # 1. Processing Error (Level 1 RED)
        if model_error or not is_valid_image or report_validity == "report_generation_failed":
            res = self.evaluate_from_structured_findings(
                {},
                model_error=model_error or "Report generation failed",
                is_valid_image=is_valid_image,
            )
            res.triage_state = "processing_error"
            res.review_reasons = [res.rule_reason]
            res.matched_rule_ids = [res.matched_rule_id]
            res.evidence_sources = evidence
            return self.finalize_canonical_triage(
                res,
                analysis_id=analysis_id,
                source_report_id=source_report_id,
            )

        # 2. Incomplete or Malformed Report (Level 4 YELLOW)
        if report_validity in ("incomplete_report", "malformed_report"):
            rule_id = "RULE_INDETERMINATE_MALFORMED_OR_INCOMPLETE_REPORT"
            reason = f"Report validation flagged issues (status: {report_validity}). Unvalidated draft cannot be accepted without review."
            res = TriageEvaluationResult(
                triage_category="indeterminate_review_required",
                matched_rule_id=rule_id,
                rule_reason=reason,
                supporting_findings=[f"Validation status: {report_validity}"],
                uncertainty_level="high",
                evidence_notes="Structural or semantic defect in generated report.",
                requires_human_review=True,
                human_review_priority="urgent",
                rule_engine_version=self.version,
                research_disclaimer=self.disclaimer,
                triage_state="incomplete_or_malformed_report",
                matched_rule_ids=[rule_id],
                review_reasons=[reason],
                evidence_sources=evidence,
            )
            return self.finalize_canonical_triage(
                res,
                analysis_id=analysis_id,
                source_report_id=source_report_id,
            )

        # 3. Conflicting AI Assessments / Multi-Model Discordance
        has_conflict = False
        conflict_msg = ""
        if consistency_result:
            if isinstance(consistency_result, dict):
                has_conflict = bool(
                    consistency_result.get("has_conflict", False)
                    or consistency_result.get("classifier_discordance", False)
                    or len(consistency_result.get("classifier_alerts", [])) > 0
                )
                alerts = consistency_result.get("classifier_alerts", [])
                conflict_msg = "; ".join(alerts) if alerts else consistency_result.get("summary", "")
            else:
                has_conflict = bool(
                    getattr(consistency_result, "has_conflict", False)
                    or getattr(consistency_result, "classifier_discordance", False)
                    or len(getattr(consistency_result, "classifier_alerts", [])) > 0
                )
                alerts = getattr(consistency_result, "classifier_alerts", [])
                conflict_msg = "; ".join(alerts) if alerts else getattr(consistency_result, "summary", "")

        if has_conflict:
            rule_id = "RULE_INDETERMINATE_CONFLICTING_AI_ASSESSMENTS"
            reason = f"Material discordance detected between AI models: {conflict_msg[:160]}"
            res = TriageEvaluationResult(
                triage_category="indeterminate_review_required",
                matched_rule_id=rule_id,
                rule_reason=reason,
                supporting_findings=[conflict_msg[:140]],
                uncertainty_level="high",
                evidence_notes="Cross-model inconsistency mandates qualified human review.",
                requires_human_review=True,
                human_review_priority="urgent",
                rule_engine_version=self.version,
                research_disclaimer=self.disclaimer,
                triage_state="conflicting_ai_assessments",
                matched_rule_ids=[rule_id],
                review_reasons=[reason],
                evidence_sources=evidence,
            )
            return self.finalize_canonical_triage(
                res,
                has_conflict=True,
                conflict_msg=reason,
                analysis_id=analysis_id,
                source_report_id=source_report_id,
            )

        # 4. Evaluate report text & structured observations
        if report_text or sections or structured_findings:
            if structured_findings:
                base_eval = self.evaluate_from_structured_findings(structured_findings)
            else:
                base_eval = self.evaluate_from_report_text(report_text, sections)
        elif regional_result and isinstance(regional_result, dict) and regional_result.get("report_text"):
            base_eval = self.evaluate_from_report_text(regional_result["report_text"])
        elif reg_state and reg_state.strip().lower().replace("_", " ") in ("not supported", "unsupported", "negative", "absent"):
            base_eval = TriageEvaluationResult(
                triage_category="no_priority_finding_flagged",
                matched_rule_id="RULE_NO_PRIORITY_FINDINGS_FLAGGED",
                rule_reason="All configured research evaluations completed without triggering priority or advisory review rules.",
                supporting_findings=[],
                uncertainty_level="low",
                evidence_notes="Regional assessment negative for queried finding.",
                requires_human_review=False,
                human_review_priority="routine",
                rule_engine_version=self.version,
                research_disclaimer=self.disclaimer,
                triage_state="no_configured_signal",
            )
        else:
            base_eval = self.evaluate_from_structured_findings({})

        # 5. Check Auxiliary Classifier Alerts (DenseNet-121)
        cls_mapped: Dict[str, float] = {}
        if classifier_result:
            if isinstance(classifier_result, dict):
                cls_mapped = classifier_result.get("mapped_findings", {})
            else:
                cls_mapped = getattr(classifier_result, "mapped_findings", {})

        # Classifier alerts at tau=0.62 (Operational Review Target)
        cls_alerts = {k: v for k, v in cls_mapped.items() if v >= 0.62}
        evidence["classifier_alerts"] = cls_alerts

        # If base report was no_priority_finding_flagged but classifier detected an alert:
        if base_eval.triage_category == "no_priority_finding_flagged" and cls_alerts:
            # Check for potentially urgent classifier alert
            if "pneumothorax" in cls_alerts or "pulmonary_edema" in cls_alerts:
                cat = "potentially_urgent"
                rule_id = "RULE_POTENTIALLY_URGENT_CLASSIFIER_SIGNAL"
                reason = f"Auxiliary classifier detected urgent finding ({', '.join(cls_alerts.keys())}) omitted from report."
                priority = "urgent"
            else:
                cat = "abnormality_flagged"
                rule_id = "RULE_ABNORMAL_CLASSIFIER_SIGNAL"
                reason = f"Auxiliary classifier flagged secondary review alert for: {', '.join(cls_alerts.keys())}."
                priority = "routine"

            res = TriageEvaluationResult(
                triage_category=cat,
                matched_rule_id=rule_id,
                rule_reason=reason,
                supporting_findings=[f"{k}: {round(v, 2)}" for k, v in cls_alerts.items()],
                uncertainty_level="high",
                evidence_notes="Classifier discordance with generative text requires secondary review.",
                requires_human_review=True,
                human_review_priority=priority,
                rule_engine_version=self.version,
                research_disclaimer=self.disclaimer,
                triage_state="potential_abnormality_signal",
                matched_rule_ids=[rule_id],
                review_reasons=[reason],
                evidence_sources=evidence,
            )
            return self.finalize_canonical_triage(
                res,
                analysis_id=analysis_id,
                source_report_id=source_report_id,
            )

        # 6. Evaluate regional finding if provided
        if reg_state:
            norm_state = reg_state.strip().lower().replace("_", " ")
            if norm_state == "supported":
                # Check if query is urgent finding
                urgent_queries = ["pneumothorax", "pulmonary edema", "edema", "tension pneumothorax"]
                q_clean = (query or "").lower().strip()
                if any(u in q_clean for u in urgent_queries):
                    base_eval.triage_category = "potentially_urgent"
                    base_eval.matched_rule_id = "RULE_POTENTIALLY_URGENT_REGIONAL_SUPPORTED"
                    base_eval.rule_reason = f"Regional grounding supports presence of urgent finding '{query}'."
                    base_eval.human_review_priority = "urgent"
                else:
                    if base_eval.triage_category == "no_priority_finding_flagged":
                        base_eval.triage_category = "abnormality_flagged"
                        base_eval.matched_rule_id = "RULE_ABNORMAL_REGIONAL_SUPPORTED"
                        base_eval.rule_reason = f"Regional grounding supports presence of '{query}'."
                        base_eval.human_review_priority = "routine"

        # 7. Map base_eval into corresponding triage states
        if base_eval.triage_category == "potentially_urgent":
            state = "potential_abnormality_signal"
        elif base_eval.triage_category == "abnormality_flagged":
            if base_eval.matched_rule_id == "RULE_ABNORMAL_SUPPORT_APPARATUS_PRESENT":
                state = "support_device_signal"
            else:
                state = "potential_abnormality_signal"
        elif base_eval.triage_category == "indeterminate_review_required":
            state = "insufficient_or_indeterminate_evidence"
        elif unsupported_query:
            state = "unsupported_query"
        else:
            state = "no_configured_signal"

        base_eval.triage_state = state
        base_eval.matched_rule_ids = [base_eval.matched_rule_id]
        base_eval.review_reasons = [base_eval.rule_reason]
        base_eval.evidence_sources = evidence

        return self.finalize_canonical_triage(
            base_eval,
            unsupported_query=unsupported_query,
            query=query,
            regional_assessment_state=reg_state,
            analysis_id=analysis_id,
            source_report_id=source_report_id,
        )


_DEFAULT_TRIAGE_SERVICE: Optional[TriageService] = None


def get_triage_service() -> TriageService:
    """Returns singleton instance of TriageService."""
    global _DEFAULT_TRIAGE_SERVICE
    if _DEFAULT_TRIAGE_SERVICE is None:
        _DEFAULT_TRIAGE_SERVICE = TriageService()
    return _DEFAULT_TRIAGE_SERVICE
