"""
triage/classifier.py
====================
Independent clinical urgency triage classifier for InteractMed-VLM.

IMPORTANT SAFETY NOTE:
This module is deliberately decoupled from the report generation model to prevent
hallucination leakage and provide an independent safety monitoring layer.

Input/Output Contract:
----------------------
Input:  clinical_finding (str)
Output: TriageResult(category in {'Green', 'Yellow', 'Red'}, confidence in [0.0, 1.0], rationale, emergency_flags)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
import re


@dataclass
class TriageResult:
    """Represents a standardized triage urgency evaluation."""
    category: Literal["Green", "Yellow", "Red"]
    confidence: float
    rationale: str
    urgency_level: str
    emergency_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "confidence": round(self.confidence, 4),
            "rationale": self.rationale,
            "urgency_level": self.urgency_level,
            "emergency_flags": self.emergency_flags,
        }


class TriageClassifier:
    """
    Independent triage classification layer evaluating the severity and urgency
    of clinical findings.

    Triage categories:
      - Green:  Normal or non-acute baseline findings. Routine follow-up.
      - Yellow: Non-emergent pathology (e.g. mild cardiomegaly, chronic atelectasis). Next-day follow-up.
      - Red:    Critical emergent pathology (e.g. pneumothorax, pulmonary edema, consolidation). Immediate triage.

    TODO: When selecting a triage backbone, consider:
      1. 'emilyalsentzer/Bio_ClinicalBERT' (Trained on MIMIC-III clinical notes)
      2. 'microsoft/deberta-v3-small' (Fine-tuned on rad-lex emergency label taxonomy)
      3. Hybrid neuro-symbolic layer (Rule-based safety overrides on top of classifier)
    """

    CRITICAL_KEYWORDS = [
        "pneumothorax",
        "tension",
        "pulmonary edema",
        "acute airspace",
        "consolidation",
        "perforation",
        "massive effusion",
        "respiratory distress",
    ]

    MODERATE_KEYWORDS = [
        "cardiomegaly",
        "effusion",
        "atelectasis",
        "opacity",
        "infiltrate",
        "hazy",
        "enlarged",
        "blunting",
    ]

    NORMAL_KEYWORDS = [
        "normal",
        "clear",
        "unremarkable",
        "within normal limits",
        "no acute",
        "sharp",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model_name: str = self.config.get("classifier_model_name_or_path", "placeholder_triage_classifier")
        self.categories = ["Green", "Yellow", "Red"]

    def classify(self, finding_text: str) -> TriageResult:
        """
        Classifies clinical finding text into Green, Yellow, or Red urgency.

        Args:
            finding_text: Text string of clinical findings.

        Returns:
            TriageResult with category, confidence, rationale, and emergency flags.
        """
        text = str(finding_text).lower()

        # TODO: In production, run tokenization and BERT forward pass:
        # inputs = self.tokenizer(finding_text, return_tensors='pt')
        # logits = self.classifier_model(**inputs).logits
        # probs = torch.softmax(logits, dim=-1)

        # Baseline safety rule-guided evaluation
        # Detect negations with a flexible window (e.g. "no pneumothorax or consolidation")
        negation_prefix = r"(?:no|without|denies|negative for|free of|resolved)\s+(?:[\w\s,/-]{0,60}?)\b"

        true_criticals = []
        for kw in self.CRITICAL_KEYWORDS:
            if re.search(r"\b" + re.escape(kw) + r"\b", text):
                pattern = negation_prefix + re.escape(kw) + r"\b"
                if not re.search(pattern, text):
                    true_criticals.append(kw)

        if true_criticals:
            return TriageResult(
                category="Red",
                confidence=0.96,
                rationale=f"Critical finding identified: {', '.join(true_criticals)}. Requires urgent clinical review.",
                urgency_level="Immediate (Stat)",
                emergency_flags=true_criticals,
            )

        # Check moderate abnormality
        moderate_flags = []
        for kw in self.MODERATE_KEYWORDS:
            if re.search(r"\b" + re.escape(kw) + r"\b", text):
                pattern = negation_prefix + re.escape(kw) + r"\b"
                if not re.search(pattern, text):
                    moderate_flags.append(kw)

        if moderate_flags:
            return TriageResult(
                category="Yellow",
                confidence=0.88,
                rationale=f"Abnormal finding detected: {', '.join(moderate_flags)}. Recommended for non-urgent evaluation.",
                urgency_level="Priority (24 hours)",
                emergency_flags=moderate_flags,
            )

        # Normal / Routine baseline
        return TriageResult(
            category="Green",
            confidence=0.95,
            rationale="No acute cardiopulmonary abnormalities detected. Routine follow-up.",
            urgency_level="Routine",
            emergency_flags=[],
        )

    def __call__(self, finding_text: str) -> TriageResult:
        """Alias for classify()."""
        return self.classify(finding_text)
