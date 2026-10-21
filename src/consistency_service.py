"""
src/consistency_service.py
===========================
Cross-Report & Cross-Output Consistency Service for InteractMed-VLM.

Clinical & Scientific Policy:
-----------------------------
1. Transparent, testable consistency assessment for findings evaluated across
   both regional grounding reports and whole-image structured/patient-friendly reports.
2. Does NOT rewrite either report to force agreement.
3. Preserves original evidence and all four regional assessment states:
   - Supported
   - Not supported
   - Indeterminate
   - Not assessable
4. Distinguishes four distinct consistency states:
   - compatible: Both assessments align or have no material conflict.
   - conflicting: Explicit disagreement on an anatomical finding (e.g. regional
     describes cardiac enlargement while whole-image describes normal heart size).
   - insufficient_information: Finding is technically unassessable or outputs are missing.
   - single_report_only: Finding was evaluated in only one report pathway.
5. Detects internal contradiction within a regional report (e.g., affirmative
   imaging observations describing enlargement coupled with an assessment marked 'Not supported').
6. Produces prominent, neutral plain-language warnings for human review.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

log = logging.getLogger("consistency_service")
if not log.handlers:
    log.addHandler(logging.NullHandler())

ConsistencyStatus = Literal[
    "compatible",
    "conflicting",
    "insufficient_information",
    "single_report_only",
]

RegionalStance = Literal[
    "supported",
    "not_supported",
    "indeterminate",
    "not_assessable",
    "unknown",
]

WholeImageCardiacStance = Literal[
    "normal",
    "enlarged",
    "indeterminate",
    "not_assessable",
    "unmentioned",
]

PRE_NEG_CUES = (
    r'\b(?:no|without|negative for|free of|clear of|denies|ruled out for|absence of|'
    r'unremarkable for|no evidence of|no sign of|no gross|no definite|no acute|'
    r'neither|clean of|fails to show|not identify|not identified|not seen|absent)\b'
)


@dataclass
class FindingConsistency:
    """Consistency evaluation for a single clinical finding."""
    finding_name: str
    status: ConsistencyStatus
    regional_stance: Optional[str] = None
    whole_image_stance: Optional[str] = None
    conflict_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CrossReportConsistencyResult:
    """Overall cross-report consistency evaluation result."""
    has_conflict: bool
    warning_banner: Optional[str]
    summary: str
    findings: List[FindingConsistency]
    internal_regional_conflict: bool = False
    internal_regional_note: Optional[str] = None
    review_required: bool = False
    review_reasons: List[str] = field(default_factory=list)
    classifier_discordance: bool = False
    classifier_alerts: List[str] = field(default_factory=list)
    evaluated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_conflict": self.has_conflict,
            "warning_banner": self.warning_banner,
            "summary": self.summary,
            "findings": [f.to_dict() for f in self.findings],
            "internal_regional_conflict": self.internal_regional_conflict,
            "internal_regional_note": self.internal_regional_note,
            "review_required": self.review_required,
            "review_reasons": self.review_reasons,
            "classifier_discordance": self.classifier_discordance,
            "classifier_alerts": self.classifier_alerts,
            "evaluated_at": self.evaluated_at,
        }


def extract_regional_sections(report_text: Optional[str]) -> Dict[str, str]:
    """Parse a 4-section regional report text into its component sections."""
    if not report_text or not report_text.strip():
        return {}

    pattern = re.compile(
        r'(?:^|\n)\s*(?:\d+[\.\)]\s*)?(?:\*{1,3})?\s*(TARGET FINDING|IMAGE OBSERVATIONS|ASSESSMENT AND UNCERTAINTY|ASSESSMENT|LIMITATIONS)\s*(?:\*{1,3})?\s*:\s*',
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(report_text))
    sections: Dict[str, str] = {}
    if matches:
        for i, m in enumerate(matches):
            raw_header = m.group(1).upper()
            sec_name = "ASSESSMENT AND UNCERTAINTY" if "ASSESSMENT" in raw_header else raw_header
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(report_text)
            sections[sec_name] = report_text[start:end].strip()
    else:
        sections["IMAGE OBSERVATIONS"] = report_text.strip()
    return sections


def extract_regional_assessment_state(report_text: Optional[str]) -> RegionalStance:
    """
    Extract the explicit assessment stance from a regional report.
    Guarantees strict preservation of all four assessment states:
    Supported | Not supported | Indeterminate | Not assessable.
    """
    if not report_text:
        return "unknown"

    sections = extract_regional_sections(report_text)
    assess_text = sections.get("ASSESSMENT AND UNCERTAINTY", "")
    target = assess_text if assess_text else report_text
    lower = target.lower()

    # Look for explicit prefixes or stance statements
    if re.search(r'\b(?:not assessable|unassessable|cannot be assessed)\b', lower):
        return "not_assessable"
    if re.search(r'\b(?:not supported|no evidence|excluded|normal limits)\b', lower):
        return "not_supported"
    if re.search(r'\b(?:indeterminate|equivocal|uncertain|cannot definitively confirm)\b', lower):
        return "indeterminate"
    if re.search(r'\b(?:supported|consistent with|suggests|demonstrated)\b', lower):
        return "supported"

    return "unknown"


def check_regional_internal_contradiction(
    observations_text: str,
    assessment_text: str,
    query: str,
) -> Tuple[bool, Optional[str]]:
    """
    Detects internal contradictions within a single regional report:
    specifically when IMAGE OBSERVATIONS affirmatively describe visible pathology
    (e.g., cardiac enlargement relative to thoracic width), but ASSESSMENT AND
    UNCERTAINTY marks the hypothesis as 'Not supported'.
    """
    if not observations_text or not assessment_text:
        return False, None

    obs_lower = observations_text.lower()
    assess_lower = assessment_text.lower()
    is_cardiac = any(k in query.lower() for k in ["cardio", "heart", "cardiac"])

    if is_cardiac:
        # Check if observations affirm cardiac enlargement
        enlarge_patterns = [
            r'\b(?:cardiac silhouette|heart)\s+(?:appears\s+)?enlarged\b',
            r'\benlarged\s+relative\s+to\s+(?:the\s+)?(?:inner\s+)?thoracic\s+width\b',
            r'\bcardiomegaly\b',
            r'\benlarged\s+cardiac\s+silhouette\b',
            r'\bincreased\s+cardiothoracic\s+ratio\b',
            r'\bcardiac\s+enlargement\b',
            r'\bprominent\s+cardiac\s+silhouette\b',
            r'\bctr\s*(?:>|greater\s+than)\s*0\.5\b',
        ]
        has_enlarge_obs = any(re.search(p, obs_lower) for p in enlarge_patterns)
        # Verify it's not negated in the observation sentence
        if has_enlarge_obs:
            for s in re.split(r'[\.\n]+', obs_lower):
                if any(re.search(p, s) for p in enlarge_patterns):
                    if re.search(PRE_NEG_CUES, s):
                        has_enlarge_obs = False

        # Check if assessment is explicitly "Not supported"
        is_not_supported = bool(
            re.search(r'\b(?:not supported|no definitive radiographic evidence of cardiomegaly)\b', assess_lower)
        )

        if has_enlarge_obs and is_not_supported:
            return (
                True,
                "Internal report contradiction: Regional image observations describe cardiac enlargement, "
                "while the regional assessment marks the finding as 'Not supported'. "
                "Human radiologist review is required to resolve this internal discrepancy.",
            )

    # General check for other findings: affirmed observation but marked Not supported
    other_findings = ["pneumothorax", "pleural effusion", "consolidation", "pulmonary edema"]
    for finding in other_findings:
        if finding in query.lower():
            has_obs = bool(re.search(rf'\b{finding}\b', obs_lower)) and not bool(re.search(PRE_NEG_CUES, obs_lower))
            if has_obs and re.search(r'\bnot supported\b', assess_lower):
                return (
                    True,
                    f"Internal report contradiction: Regional image observations describe features of {finding}, "
                    f"while the regional assessment marks the hypothesis as 'Not supported'.",
                )

    return False, None


def extract_cardiac_stance_whole_image(
    report_text: Optional[str],
    sections: Optional[Dict[str, str]] = None,
) -> WholeImageCardiacStance:
    """
    Extracts the cardiac silhouette stance from the whole-image report narrative.
    Evaluates plain-language descriptions (e.g. 'heart size is normal', 'heart appears enlarged').
    """
    if not report_text and not sections:
        return "unmentioned"

    # Search in all available sections
    all_text = ""
    if sections:
        all_text = " ".join(sections.values())
    if report_text:
        all_text += " " + report_text
    lower = all_text.lower()

    if not any(k in lower for k in ["heart", "cardiac", "cardiomegaly", "ctr"]):
        return "unmentioned"

    # 1. Check for unassessable / indeterminate
    if re.search(r'\b(?:heart\s+size\s+is\s+(?:unclear|indeterminate|unassessable)|cannot\s+assess\s+heart)\b', lower):
        return "indeterminate"

    # 2. Check for enlarged heart
    enlarged_patterns = [
        r'\bheart\s+(?:size\s+)?(?:is\s+|appears\s+)?enlarged\b',
        r'\benlarged\s+heart\b',
        r'\bcardiomegaly\b',
        r'\bcardiac\s+enlargement\b',
        r'\benlarged\s+cardiac\s+silhouette\b',
        r'\bheart\s+is\s+larger\s+than\s+normal\b',
        r'\bprominent\s+heart\s+size\b',
    ]
    for p in enlarged_patterns:
        m = re.search(p, lower)
        if m:
            # Check negation
            sentence = ""
            for s in re.split(r'[\.\n;]+', lower):
                if re.search(p, s):
                    sentence = s
                    break
            if sentence and not re.search(PRE_NEG_CUES, sentence):
                return "enlarged"

    # 3. Check for normal heart
    normal_patterns = [
        r'\bheart\s+size\s+is\s+normal\b',
        r'\bnormal\s+heart\s+size\b',
        r'\bheart\s+appears\s+normal\b',
        r'\bheart\s+is\s+normal\b',
        r'\bnormal\s+cardiac\s+silhouette\b',
        r'\bcardiac\s+silhouette\s+is\s+normal\b',
        r'\bheart\s+borders\s+are\s+normal\b',
        r'\bheart\s+is\s+within\s+normal\s+limits\b',
        r'\bno\s+enlargement\s+of\s+the\s+heart\b',
        r'\bheart\s+looks\s+normal\b',
    ]
    for p in normal_patterns:
        if re.search(p, lower):
            return "normal"

    return "unmentioned"


def evaluate_cross_report_consistency(
    query: str,
    regional_report_text: Optional[str] = None,
    whole_image_report_text: Optional[str] = None,
    whole_image_sections: Optional[Dict[str, str]] = None,
    classifier_result: Optional[Union[Dict[str, Any], Any]] = None,
    classifier_threshold: float = 0.62,
) -> CrossReportConsistencyResult:
    """
    Evaluates consistency between regional phrase grounding report,
    whole-image report outputs, and optional auxiliary pretrained classifier outputs
    on the same radiograph.

    Follows the 4 strict cases:
      - compatible
      - conflicting
      - insufficient_information
      - single_report_only
    """
    findings_list: List[FindingConsistency] = []
    has_conflict = False
    warning_banner: Optional[str] = None
    summary_parts: List[str] = []

    # 1. Check existence of outputs
    has_regional = bool(regional_report_text and regional_report_text.strip())
    has_whole = bool(
        (whole_image_report_text and whole_image_report_text.strip())
        or (whole_image_sections and any(v.strip() for v in whole_image_sections.values()))
    )

    if not has_regional and not has_whole:
        return CrossReportConsistencyResult(
            has_conflict=False,
            warning_banner=None,
            summary="Neither regional nor whole-image reports are available for consistency comparison.",
            findings=[],
        )

    if has_regional and not has_whole:
        # Check internal consistency of regional report alone
        reg_sections = extract_regional_sections(regional_report_text)
        obs_text = reg_sections.get("IMAGE OBSERVATIONS", "")
        assess_text = reg_sections.get("ASSESSMENT AND UNCERTAINTY", "")
        internal_conflict, internal_note = check_regional_internal_contradiction(obs_text, assess_text, query)

        reg_stance = extract_regional_assessment_state(regional_report_text)
        finding_eval = FindingConsistency(
            finding_name=query,
            status="conflicting" if internal_conflict else "single_report_only",
            regional_stance=reg_stance,
            whole_image_stance="unmentioned",
            conflict_message=internal_note if internal_conflict else None,
            details={"note": "Evaluated in regional report only; whole-image report not yet generated."},
        )
        findings_list.append(finding_eval)

        summary = (
            internal_note
            if internal_conflict
            else f"Finding '{query}' evaluated in regional report only (assessment: {reg_stance}). Whole-image report is not yet generated."
        )
        return CrossReportConsistencyResult(
            has_conflict=internal_conflict,
            warning_banner="Conflicting AI assessments — review required" if internal_conflict else None,
            summary=summary,
            findings=findings_list,
            internal_regional_conflict=internal_conflict,
            internal_regional_note=internal_note,
        )

    if has_whole and not has_regional and not classifier_result:
        findings_list.append(
            FindingConsistency(
                finding_name=query or "general_assessment",
                status="single_report_only",
                regional_stance="unmentioned",
                whole_image_stance="evaluated",
                details={"note": "Whole-image report available without regional analysis."},
            )
        )
        return CrossReportConsistencyResult(
            has_conflict=False,
            warning_banner=None,
            summary="Whole-image report available. No regional phrase grounding report was generated for comparison.",
            findings=findings_list,
        )

    # 2. Both reports exist: parse regional report
    reg_sections = extract_regional_sections(regional_report_text)
    obs_text = reg_sections.get("IMAGE OBSERVATIONS", "")
    assess_text = reg_sections.get("ASSESSMENT AND UNCERTAINTY", "")
    reg_stance = extract_regional_assessment_state(regional_report_text)

    # Check internal regional contradiction
    internal_conflict, internal_note = check_regional_internal_contradiction(obs_text, assess_text, query)

    # 3. Clinical finding comparison: Cardiomegaly / Heart
    is_cardiac_query = any(k in query.lower() for k in ["cardio", "heart", "cardiac"])
    if is_cardiac_query:
        whole_cardiac = extract_cardiac_stance_whole_image(whole_image_report_text, whole_image_sections)

        # Regional evidence of enlargement: either affirmative observation or supported stance
        obs_lower = obs_text.lower()
        enlarge_in_obs = any(
            p in obs_lower
            for p in ["enlarged", "prominent cardiac", "increased cardiothoracic", "cardiomegaly"]
        ) and not bool(re.search(PRE_NEG_CUES, obs_lower))

        regional_claims_enlargement = (reg_stance == "supported") or enlarge_in_obs
        regional_claims_normal = (reg_stance == "not_supported") and not enlarge_in_obs

        cardiac_finding_status: ConsistencyStatus = "compatible"
        conflict_msg: Optional[str] = None

        if reg_stance == "not_assessable" or whole_cardiac == "not_assessable":
            cardiac_finding_status = "insufficient_information"
            conflict_msg = (
                "Technical limitations, patient positioning, or image exposure precluded reliable assessment "
                "of cardiac silhouette in one or both analyses. Results could not be compared reliably."
            )
        elif whole_cardiac == "unmentioned":
            cardiac_finding_status = "single_report_only"
            conflict_msg = "Cardiomegaly evaluated in regional analysis; whole-image report did not discuss cardiac dimensions."
        elif regional_claims_enlargement and whole_cardiac == "normal":
            # Explicit contradiction!
            cardiac_finding_status = "conflicting"
            has_conflict = True
            conflict_msg = (
                "The regional report describes possible cardiac enlargement, "
                "while the whole-image report describes normal heart size."
            )
        elif regional_claims_normal and whole_cardiac == "enlarged":
            # Explicit contradiction!
            cardiac_finding_status = "conflicting"
            has_conflict = True
            conflict_msg = (
                "The regional report found no evidence of cardiac enlargement, "
                "while the whole-image report describes an enlarged heart."
            )
        elif internal_conflict:
            # Regional report contradicts itself on cardiac observation vs stance
            cardiac_finding_status = "conflicting"
            has_conflict = True
            conflict_msg = internal_note
        elif reg_stance == "indeterminate" or whole_cardiac == "indeterminate":
            cardiac_finding_status = "compatible"
            conflict_msg = (
                "Equivocal or cautious assessment: one or both analyses noted uncertainty regarding cardiac silhouette, "
                "without asserting definitive enlargement or exclusion."
            )
        else:
            # Both agree on normal or both agree on enlarged
            cardiac_finding_status = "compatible"
            conflict_msg = (
                "Assessments are compatible: both analyses report consistent observations regarding the cardiac silhouette."
            )

        finding_eval = FindingConsistency(
            finding_name="cardiomegaly",
            status=cardiac_finding_status,
            regional_stance=reg_stance,
            whole_image_stance=whole_cardiac,
            conflict_message=conflict_msg,
            details={
                "regional_observations_affirm_enlargement": enlarge_in_obs,
                "internal_regional_conflict": internal_conflict,
            },
        )
        findings_list.append(finding_eval)
        if conflict_msg:
            summary_parts.append(conflict_msg)

    else:
        # Non-cardiac query comparison (e.g. pneumothorax, effusion, opacity)
        whole_text = ""
        if whole_image_sections:
            whole_text = " ".join(whole_image_sections.values())
        if whole_image_report_text:
            whole_text += " " + whole_image_report_text
        whole_lower = whole_text.lower()

        finding_key = query.lower().strip()
        synonyms = [finding_key]
        if any(w in finding_key for w in ["pneumonia", "consolidation", "opacity", "infiltrate"]):
            synonyms.extend(["pneumonia", "consolidation", "opacity", "infiltrat", "lung", "lungs"])
        elif any(w in finding_key for w in ["effusion", "fluid"]):
            synonyms.extend(["effusion", "pleural", "costophrenic"])
        elif any(w in finding_key for w in ["pneumothorax", "ptx"]):
            synonyms.extend(["pneumothorax", "ptx", "pleural"])
        elif any(w in finding_key for w in ["catheter", "tube", "line", "device", "apparatus"]):
            synonyms.extend(["catheter", "tube", "line", "device", "apparatus", "hardware", "support"])

        mentioned_in_whole = any(s in whole_lower for s in synonyms)

        if not mentioned_in_whole:
            finding_eval = FindingConsistency(
                finding_name=query,
                status="single_report_only",
                regional_stance=reg_stance,
                whole_image_stance="unmentioned",
                conflict_message=f"Finding '{query}' evaluated in regional report only; whole-image report did not explicitly address it.",
            )
            findings_list.append(finding_eval)
            summary_parts.append(f"Finding '{query}' was evaluated in regional analysis only.")
        elif reg_stance == "not_assessable":
            finding_eval = FindingConsistency(
                finding_name=query,
                status="insufficient_information",
                regional_stance=reg_stance,
                whole_image_stance="mentioned",
                conflict_message="Finding was technically unassessable in regional analysis. Results could not be compared reliably.",
            )
            findings_list.append(finding_eval)
            summary_parts.append("Results could not be compared reliably due to limited assessability.")
        else:
            # Check if whole-image report affirmatively describes the finding
            finding_terms = [s for s in synonyms if s not in ['lung', 'lungs']]
            finding_pattern = r'\b(?:' + '|'.join(re.escape(s) for s in finding_terms) + r')\b' if finding_terms else None
            affirmatively_reported = False
            if finding_pattern:
                for m in re.finditer(finding_pattern, whole_lower):
                    prefix = whole_lower[max(0, m.start() - 40):m.start()]
                    if not re.search(r'\b(?:no|without|negative\s+for|free\s+of|clear\s+of|denies|ruled\s+out|unremarkable\s+for|no\s+evidence\s+of|no\s+sign\s+of|fails\s+to\s+show)\b', prefix):
                        affirmatively_reported = True
                        break

            # Check if whole report claimed clear lungs or absent finding
            whole_claims_clear = bool(re.search(r'\b(?:lungs\s+(?:are\s+|appear\s+)?clear|clear\s+lungs|clear\s+bilaterally|no\s+(?:focal\s+)?(?:consolidation|opacity|infiltrate|pneumonia)|normal\s+chest\s+radiograph)\b', whole_lower))

            if reg_stance == "supported" and not affirmatively_reported and whole_claims_clear:
                has_conflict = True
                msg = f"The regional report supports the presence of {query}, while the whole-image report describes clear lungs."
                findings_list.append(
                    FindingConsistency(
                        finding_name=query,
                        status="conflicting",
                        regional_stance=reg_stance,
                        whole_image_stance="normal/clear",
                        conflict_message=msg,
                    )
                )
                summary_parts.append(msg)
            else:
                finding_eval = FindingConsistency(
                    finding_name=query,
                    status="compatible",
                    regional_stance=reg_stance,
                    whole_image_stance="compatible" if affirmatively_reported else "unmentioned_or_compatible",
                    conflict_message=f"Assessments for '{query}' are compatible without material contradiction.",
                )
                findings_list.append(finding_eval)
                summary_parts.append(f"Assessments for '{query}' are compatible.")

    # Check whole-image internal contradictions if sections or text available
    whole_internal_conflict, whole_conflict_reasons = check_whole_image_internal_contradiction(
        sections=whole_image_sections,
        report_text=whole_image_report_text,
    )

    # Check if internal contradiction was detected on any query
    if internal_conflict and not has_conflict:
        has_conflict = True
        summary_parts.insert(0, internal_note or "Internal contradiction detected within regional report.")

    if whole_internal_conflict and not has_conflict:
        has_conflict = True
        summary_parts.extend(whole_conflict_reasons)

    # ── 5. Auxiliary Pretrained Classifier Evidence Integration ───────────────
    classifier_discordance = False
    classifier_alerts: List[str] = []

    if classifier_result:
        mapped_findings = getattr(classifier_result, "mapped_findings", None)
        if mapped_findings is None and isinstance(classifier_result, dict):
            mapped_findings = classifier_result.get("mapped_findings", {})
        if not mapped_findings:
            mapped_findings = {}

        unmapped_findings = getattr(classifier_result, "unmapped_findings", None)
        if unmapped_findings is None and isinstance(classifier_result, dict):
            unmapped_findings = classifier_result.get("unmapped_findings", {})
        if not unmapped_findings:
            unmapped_findings = {}

        q_lower = query.lower().strip()
        query_cat = None
        if any(w in q_lower for w in ["heart", "cardio", "enlarge"]):
            query_cat = "cardiomegaly"
        elif any(w in q_lower for w in ["effusion", "fluid", "blunting"]):
            query_cat = "pleural_effusion"
        elif any(w in q_lower for w in ["pneumothorax", "ptx"]):
            query_cat = "pneumothorax"
        elif any(w in q_lower for w in ["pneumonia", "consolidation", "infiltrat"]):
            query_cat = "pneumonia_consolidation"
        elif any(w in q_lower for w in ["edema", "congestion"]):
            query_cat = "pulmonary_edema"
        elif any(w in q_lower for w in ["atelectasis", "collapse"]):
            query_cat = "atelectasis"
        elif any(w in q_lower for w in ["opacity", "opacities"]):
            query_cat = "lung_opacity"

        whole_lower = ((whole_image_report_text or "") + " " + " ".join((whole_image_sections or {}).values())).lower()
        whole_claims_normal = bool(re.search(r'\b(?:normal|clear|unremarkable|no\s+acute)\b', whole_lower))

        if query_cat and query_cat in mapped_findings:
            cls_score = float(mapped_findings[query_cat])
            reg_stance = extract_regional_assessment_state(regional_report_text) if has_regional else "unmentioned"

            if cls_score >= classifier_threshold:
                if reg_stance == "supported" and whole_claims_normal:
                    conf_msg = (
                        f"Multi-source confirmation conflict: Both regional review and auxiliary classifier indicate {query} "
                        f"(classifier score: {cls_score:.2f} >= {classifier_threshold:.2f}), directly contradicting whole-image normal/clear declaration."
                    )
                    has_conflict = True
                    classifier_discordance = True
                    classifier_alerts.append(conf_msg)
                    summary_parts.append(conf_msg)
                elif reg_stance in ["unknown", "not_supported", "unmentioned"] and whole_claims_normal:
                    advisory_msg = (
                        f"Evidence discordance advisory: Auxiliary pretrained classifier detected elevated evidence for {query} "
                        f"(score: {cls_score:.2f} >= {classifier_threshold:.2f}), while generative reports omitted or ruled out this finding."
                    )
                    classifier_discordance = True
                    classifier_alerts.append(advisory_msg)
                    summary_parts.append(advisory_msg)

        # Check unmapped secondary findings with high confidence (>= 0.70)
        for u_cat, u_score in unmapped_findings.items():
            if float(u_score) >= 0.70:
                u_alert = f"Secondary incidental finding alert: Auxiliary classifier detected high confidence for {u_cat} ({float(u_score):.2f} >= 0.70)."
                classifier_alerts.append(u_alert)

    all_review_reasons: List[str] = []
    if internal_conflict and internal_note:
        all_review_reasons.append(internal_note)
    if whole_internal_conflict:
        all_review_reasons.extend(whole_conflict_reasons)
    for f in findings_list:
        if f.status == "conflicting" and f.conflict_message:
            all_review_reasons.append(f.conflict_message)
    if classifier_alerts:
        all_review_reasons.extend(classifier_alerts)

    # Check for diagnostic uncertainty / indeterminate evidence
    indeterminate_evidence = False
    for f in findings_list:
        if f.regional_stance == "indeterminate" or f.whole_image_stance == "indeterminate":
            indeterminate_evidence = True
            indet_msg = f"Diagnostic uncertainty: Assessment for '{f.finding_name}' is indeterminate; clinical review required to resolve equivocal evidence."
            all_review_reasons.append(indet_msg)
            summary_parts.append(indet_msg)
            break

    review_required = has_conflict or internal_conflict or whole_internal_conflict or classifier_discordance or (len(classifier_alerts) > 0) or indeterminate_evidence

    if has_conflict or review_required:
        warning_banner = "Conflicting AI assessments — review required"
        summary = " ".join(summary_parts) if summary_parts else "Conflicting AI assessments detected between report pathways."
    else:
        summary = " ".join(summary_parts) if summary_parts else "Report outputs are compatible without detected contradictions."

    return CrossReportConsistencyResult(
        has_conflict=has_conflict,
        warning_banner=warning_banner,
        summary=summary,
        findings=findings_list,
        internal_regional_conflict=internal_conflict,
        internal_regional_note=internal_note,
        review_required=review_required,
        review_reasons=all_review_reasons,
        classifier_discordance=classifier_discordance,
        classifier_alerts=classifier_alerts,
    )


def check_whole_image_internal_contradiction(
    sections: Optional[Dict[str, str]] = None,
    report_text: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """
    Detects internal contradictions within a whole-image structured or narrative report:
    1. Positive observations vs. overall impression declaring normal/unremarkable.
    2. Support apparatus/hardware explicitly observed but declared absent in impression.
    3. Conflicting laterality across report sections for the same pathology.
    Does NOT rewrite the report text; returns (has_contradiction, list_of_reasons).
    """
    reasons: List[str] = []
    if not sections and not report_text:
        return False, reasons

    sec = dict(sections) if sections else {}
    if not sec and report_text:
        # Simple extraction of key sections if not pre-parsed
        for line in report_text.split("\n\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                sec[k.strip().upper()] = v.strip()

    imp = sec.get("OVERALL IMPRESSION") or sec.get("IMPRESSION") or ""
    if not imp and report_text:
        imp_m = re.search(r'(?:overall\s+(?:clinical\s+)?impression|impression)\s*:\s*(.*?)(?:\n\s*[A-Z0-9_\s]{3,}\s*:|$)', report_text, re.IGNORECASE | re.DOTALL)
        if imp_m:
            imp = imp_m.group(1).strip()
    imp_lower = imp.lower()

    is_normal_imp = bool(
        re.search(
            r'\b(?:normal\s+chest\s+radiograph|no\s+acute\s+(?:cardiopulmonary\s+)?abnormality|'
            r'no\s+acute\s+findings|clear\s+lung\s+fields|unremarkable\s+chest|no\s+significant\s+findings)\b',
            imp_lower,
        )
    )

    # 1. Support apparatus observation vs. impression
    supp_text = (sec.get("SUPPORT APPARATUS") or "").lower()
    has_device_obs = False
    if "observed" in supp_text or "finding_observed" in supp_text:
        if not re.search(r'\b(?:not identified|not seen|none|absent)\b', supp_text):
            has_device_obs = True
    elif any(d in supp_text for d in ["chest tube", "pigtail", "catheter", "endotracheal tube", "picc", "port-a-cath", "sternotomy", "wire", "pacemaker"]):
        if not re.search(r'\b(?:not identified|no|without|none)\b', supp_text):
            has_device_obs = True

    if has_device_obs:
        if re.search(r'\b(?:no\s+support\s+apparatus|no\s+tubes\s+or\s+lines|no\s+devices)\b', imp_lower):
            reasons.append("Support apparatus was observed in visual findings but explicitly declared absent in overall impression.")
        elif is_normal_imp and "tube" not in imp_lower and "catheter" not in imp_lower and "line" not in imp_lower and "wire" not in imp_lower:
            reasons.append("Support apparatus was identified in visual observations, but overall impression describes examination as normal/unremarkable.")

    # 2. Cardiac silhouette observation vs. impression
    card_text = (sec.get("CARDIAC SILHOUETTE") or "").lower()
    has_cardiac_obs = ("finding_observed" in card_text or "enlarged" in card_text or "cardiomegaly" in card_text) and not ("not identified" in card_text or "normal" in card_text)
    if has_cardiac_obs and is_normal_imp and "cardiomegaly" not in imp_lower and "enlarged" not in imp_lower:
        reasons.append("Cardiac enlargement was observed in cardiac section, but overall impression declares examination normal/unremarkable.")

    # 3. Lungs & opacities observation vs. impression
    lung_text = (sec.get("LUNGS AND OPACITIES") or "").lower()
    has_lung_obs = any(p in lung_text for p in ["consolidation", "pneumonia", "opacity", "infiltrat", "atelectasis", "edema"]) and not any(neg in lung_text for neg in ["not identified", "clear", "no focal", "no acute"])
    if has_lung_obs and is_normal_imp:
        reasons.append("Parenchymal opacity/consolidation was noted in lung observations, but overall impression declares examination normal/unremarkable.")

    # 4. Pleural space observation vs. impression
    pleural_text = (sec.get("PLEURAL SPACES") or "").lower()
    has_pleural_obs = any(p in pleural_text for p in ["effusion", "pneumothorax", "blunt"]) and not any(
        neg in pleural_text for neg in [
            "not identified", "sharp", "no effusion", "no pneumothorax", "no pleural effusion",
            "no pleural", "without effusion", "without pneumothorax", "clear", "unremarkable", "normal",
        ]
    )
    if has_pleural_obs and is_normal_imp:
        reasons.append("Pleural abnormality (effusion or pneumothorax) was described in pleural section, but overall impression declares examination normal/unremarkable.")

    # 5. Laterality conflict check between observations and impression
    for finding in ["pneumothorax", "effusion", "consolidation", "atelectasis", "opacity"]:
        if finding in lung_text or finding in pleural_text:
            obs_context = lung_text if finding in lung_text else pleural_text
            obs_left = "left" in obs_context and "right" not in obs_context
            obs_right = "right" in obs_context and "left" not in obs_context
            imp_left = "left" in imp_lower and "right" not in imp_lower
            imp_right = "right" in imp_lower and "left" not in imp_lower

            if (obs_left and imp_right) or (obs_right and imp_left):
                reasons.append(f"Laterality contradiction detected for {finding}: observations describe one side while impression describes the opposite side.")

    has_conflict = len(reasons) > 0
    return has_conflict, reasons


def check_two_stage_synthesis_loss(
    stage1_json: Dict[str, Any],
    stage2_text: str,
) -> Tuple[bool, List[str]]:
    """
    Checks if positive findings validated in Stage 1 structured visual extraction
    were dropped or contradicted during Stage 2 report synthesis.
    """
    dropped: List[str] = []
    text_lower = stage2_text.lower()

    # Check support devices
    devs = stage1_json.get("support_devices", [])
    if isinstance(devs, list):
        for d in devs:
            if isinstance(d, dict) and d.get("status") == "finding_observed":
                dev_type = d.get("device_type", "support device").lower()
                if dev_type != "not_applicable" and dev_type not in text_lower and "tube" not in text_lower and "line" not in text_lower and "catheter" not in text_lower:
                    dropped.append(f"support_device ({dev_type})")

    # Check key pathologies
    pathology_keys = [
        ("cardiac_silhouette", ["cardiomegaly", "enlarged", "heart"]),
        ("lung_opacities", ["consolidation", "pneumonia", "opacity", "infiltrat"]),
        ("atelectatic_changes", ["atelectasis", "volume loss"]),
        ("pulmonary_edema", ["edema", "congestion"]),
        ("pleural_effusion", ["effusion", "fluid"]),
        ("pneumothorax", ["pneumothorax", "ptx"]),
    ]

    for key, search_terms in pathology_keys:
        item = stage1_json.get(key, {})
        if isinstance(item, dict) and item.get("status") == "finding_observed":
            # Finding was observed in Stage 1
            if not any(t in text_lower for t in search_terms):
                dropped.append(key)

    has_loss = len(dropped) > 0
    reasons = [f"Stage 1 extraction observed '{d}' but it was omitted from Stage 2 synthesized report." for d in dropped]
    return has_loss, reasons
