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
    human_review_priority: Literal["urgent", "routine", "standard_review"] = "standard_review"
    rule_engine_version: str = "1.0.0"
    research_disclaimer: str = (
        "RESEARCH PROTOTYPE NOTICE: Experimental heuristic classification. "
        "Not clinically validated. Human medical review is required."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
            return TriageEvaluationResult(
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
            )

        # 2. Guardrail: Invalid image
        if not is_valid_image:
            return TriageEvaluationResult(
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
            )

        # 3. Guardrail: Internal contradiction
        if internal_contradiction:
            return TriageEvaluationResult(
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
            )

        # Consolidate finding text across domains
        full_findings_text = "\n".join(structured_findings.values()).strip()
        if not full_findings_text:
            return TriageEvaluationResult(
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
            )

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
                return TriageEvaluationResult(
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
                )

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

                return TriageEvaluationResult(
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
                )

        # 5. Default safety fallback
        return TriageEvaluationResult(
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
        )

    def evaluate_from_report_text(
        self,
        report_text: Optional[str],
        sections: Optional[Dict[str, str]] = None,
        error: Optional[str] = None,
    ) -> TriageEvaluationResult:
        """
        Convenience evaluator from report draft text or section dictionary.
        Extracts observations and impression, checks contradictions, and evaluates triage.
        """
        if error or not report_text:
            return self.evaluate_from_structured_findings(
                structured_findings={},
                model_error=error or "Report generation produced empty output",
                is_valid_image=True,
            )

        # Parse sections if not already provided
        sec_dict = sections or {}
        obs_text = sec_dict.get("OBSERVATIONS") or sec_dict.get("1. Your X-ray in simple words") or ""
        imp_text = sec_dict.get("IMPRESSION") or sec_dict.get("2. What the findings may mean") or ""

        # If sections not split, extract via regex
        if not obs_text and report_text:
            obs_m = re.search(r'OBSERVATIONS\s*[:\n](.*?)(?=\bIMPRESSION|$)', report_text, re.DOTALL | re.IGNORECASE)
            if obs_m:
                obs_text = obs_m.group(1).strip()
            else:
                obs_text = report_text

        if not imp_text and report_text:
            imp_m = re.search(r'IMPRESSION\s*[:\n](.*?)(?=\bLIMITATIONS|$)', report_text, re.DOTALL | re.IGNORECASE)
            if imp_m:
                imp_text = imp_m.group(1).strip()

        # Check for contradictions: active pathology in observations but "normal" in impression
        contradiction = False
        if obs_text and imp_text:
            has_pneumothorax = bool(re.search(r'\bpneumothorax\b', obs_text, re.IGNORECASE)) and not bool(re.search(PRE_NEG_CUES, obs_text, re.IGNORECASE))
            has_edema = bool(re.search(r'\bedema\b', obs_text, re.IGNORECASE)) and not bool(re.search(PRE_NEG_CUES, obs_text, re.IGNORECASE))
            has_consolidation = bool(re.search(r'\bconsolidation\b', obs_text, re.IGNORECASE)) and not bool(re.search(PRE_NEG_CUES, obs_text, re.IGNORECASE))

            imp_claims_normal = bool(re.search(r'\b(?:normal|unremarkable|no acute|clear)\b', imp_text, re.IGNORECASE))
            if (has_pneumothorax or has_edema or has_consolidation) and imp_claims_normal:
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


_DEFAULT_TRIAGE_SERVICE: Optional[TriageService] = None


def get_triage_service() -> TriageService:
    """Returns singleton instance of TriageService."""
    global _DEFAULT_TRIAGE_SERVICE
    if _DEFAULT_TRIAGE_SERVICE is None:
        _DEFAULT_TRIAGE_SERVICE = TriageService()
    return _DEFAULT_TRIAGE_SERVICE
