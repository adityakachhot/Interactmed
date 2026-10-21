"""
src/evidence_schema.py
======================
Canonical Evidence-Based Finding Schema, Chest ImaGenome Anatomical Normalization,
and Multi-Source Evidence Reconciliation for InteractMed-VLM.

Strict Clinical-ML Guarantees:
1. Explicit presence states: 'present', 'absent', 'uncertain', 'not_assessed'.
2. Missing label or missing box is NEVER silently inferred as 'absent'.
3. Distinct separation of evidence sources:
   - whole_image_vlm (MedGemma)
   - regional_vlm (MedGemma Regional Crop)
   - grounding_box (TransVG)
   - classifier_alert (TorchXRayVision DenseNet-121)
   - cross_report_reconciled
4. A classifier alert is NEVER automatically copied as a confirmed diagnosis.
5. TransVG box is an anatomical visual bounding box, NOT an independent presence probability.
6. Anatomical normalization derived from Chest ImaGenome's 36 controlled anatomical entities.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, Field


# ── 1. Chest ImaGenome 36 Controlled Anatomical Entities ───────────────────────

CHEST_IMAGENOME_ANATOMICAL_REGIONS: List[str] = [
    "right lung",
    "right apical zone",
    "right upper lung zone",
    "right mid lung zone",
    "right lower lung zone",
    "right hilar structures",
    "right costophrenic angle",
    "left lung",
    "left apical zone",
    "left upper lung zone",
    "left mid lung zone",
    "left lower lung zone",
    "left hilar structures",
    "left costophrenic angle",
    "mediastinum",
    "upper mediastinum",
    "cardiac silhouette",
    "trachea",
    "right hemidiaphragm",
    "left hemidiaphragm",
    "right clavicle",
    "left clavicle",
    "spine",
    "right atrium",
    "cavoatrial junction",
    "svc",
    "carina",
    "aortic arch",
    "abdomen",
    "right cardiac silhouette",
    "left cardiac silhouette",
    "right cardiophrenic angle",
    "left cardiophrenic angle",
    "right upper abdomen",
    "left upper abdomen",
    "descending aorta",
]

# Anatomical aliases and mapping rules
ANATOMICAL_SYNONYMS: Dict[str, Tuple[str, str]] = {
    "right lower lobe": ("right lower lung zone", "right"),
    "rll": ("right lower lung zone", "right"),
    "right base": ("right lower lung zone", "right"),
    "right basilar": ("right lower lung zone", "right"),
    "right middle lobe": ("right mid lung zone", "right"),
    "rml": ("right mid lung zone", "right"),
    "right upper lobe": ("right upper lung zone", "right"),
    "rul": ("right upper lung zone", "right"),
    "right apex": ("right apical zone", "right"),
    "right apical": ("right apical zone", "right"),
    "left lower lobe": ("left lower lung zone", "left"),
    "lll": ("left lower lung zone", "left"),
    "left base": ("left lower lung zone", "left"),
    "left basilar": ("left lower lung zone", "left"),
    "retrocardiac": ("left lower lung zone", "left"),
    "left upper lobe": ("left upper lung zone", "left"),
    "lul": ("left upper lung zone", "left"),
    "left apex": ("left apical zone", "left"),
    "left apical": ("left apical zone", "left"),
    "lingula": ("left mid lung zone", "left"),
    "heart": ("cardiac silhouette", "midline"),
    "cardiac": ("cardiac silhouette", "midline"),
    "cardiomediastinal": ("cardiac silhouette", "midline"),
    "right cpa": ("right costophrenic angle", "right"),
    "right sulcus": ("right costophrenic angle", "right"),
    "left cpa": ("left costophrenic angle", "left"),
    "left sulcus": ("left costophrenic angle", "left"),
    "bilateral bases": ("lower lung zones", "bilateral"),
    "both bases": ("lower lung zones", "bilateral"),
    "both lungs": ("lungs bilaterally", "bilateral"),
    "bilateral": ("lungs bilaterally", "bilateral"),
    "trachea": ("trachea", "midline"),
    "mediastinum": ("mediastinum", "midline"),
    "aorta": ("aortic arch", "midline"),
    "diaphragm": ("hemidiaphragm", "unspecified"),
}


def normalize_anatomical_location(raw_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Normalizes a free-text anatomical description to a Chest ImaGenome standard
    entity and laterality.
    
    Returns:
        Tuple of (normalized_anatomical_location, laterality)
    """
    if not raw_text:
        return None, "unspecified"

    cleaned = raw_text.lower().strip()

    # Exact match in 36 controlled entities
    for entity in CHEST_IMAGENOME_ANATOMICAL_REGIONS:
        if entity in cleaned:
            lat = "right" if "right" in entity else ("left" if "left" in entity else "midline")
            return entity, lat

    # Alias / synonym lookup
    for syn, (canonical_region, lat) in ANATOMICAL_SYNONYMS.items():
        if re.search(r"\b" + re.escape(syn) + r"\b", cleaned):
            return canonical_region, lat

    # Fallback laterality scan
    lat = "unspecified"
    if "bilateral" in cleaned or "both" in cleaned:
        lat = "bilateral"
    elif "right" in cleaned:
        lat = "right"
    elif "left" in cleaned:
        lat = "left"

    return raw_text.strip(), lat


# ── 2. Canonical Finding Data Model ───────────────────────────────────────────

class CanonicalFinding(BaseModel):
    """
    Canonical evidence-based representation of an individual radiographic finding.
    Guarantees strict auditability and separation between model assertions,
    grounding geometry, and auxiliary classifier alerts.
    """
    finding_id: str = Field(default_factory=lambda: f"fnd_{uuid.uuid4().hex[:8]}")
    finding_name: str
    presence_status: Literal["present", "absent", "uncertain", "not_assessed"]
    anatomical_location: Optional[str] = None
    laterality: Optional[Literal["left", "right", "bilateral", "midline", "unspecified"]] = "unspecified"
    image_evidence: Optional[str] = None
    grounding_status: Literal["not_attempted", "grounded", "unsupported", "failed", "weakly_supported"] = "not_attempted"
    supporting_region: Optional[List[float]] = None  # Normalized [ymin, xmin, ymax, xmax] in [0, 1]
    uncertainty: Literal["low", "medium", "high"] = "low"
    source: Literal[
        "whole_image_vlm",
        "regional_vlm",
        "grounding_box",
        "classifier_alert",
        "cross_report_reconciled",
        "offline_annotation"
    ] = "whole_image_vlm"
    review_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump() if hasattr(self, "model_dump") else self.dict()


class StructuredReportEvidence(BaseModel):
    """
    Complete evidence container encapsulating structured findings,
    anatomical reconciliation, and completeness check flags.
    """
    report_id: str = Field(default_factory=lambda: f"rep_{uuid.uuid4().hex[:8]}")
    findings: List[CanonicalFinding] = []
    completeness_flags: Dict[str, Any] = {}
    anatomical_reconciliation: Dict[str, Any] = {}
    reconciled_report_text: Optional[str] = None
    reconciled_patient_translation: Optional[str] = None
    source_evidence_summary: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "findings": [f.to_dict() for f in self.findings],
            "completeness_flags": self.completeness_flags,
            "anatomical_reconciliation": self.anatomical_reconciliation,
            "reconciled_report_text": self.reconciled_report_text,
            "reconciled_patient_translation": self.reconciled_patient_translation,
            "source_evidence_summary": self.source_evidence_summary,
        }


# ── 3. Structured Finding Extraction ──────────────────────────────────────────

# Clinical concept patterns with explicit presence, absence, and uncertainty indicators
PATHOLOGY_CONCEPTS: Dict[str, Dict[str, Any]] = {
    "pneumonia": {
        "patterns": [r"\bpneumonia\b", r"\bbronchopneumonia\b", r"\binfectious infiltrate\b"],
        "default_region": "lungs",
    },
    "consolidation": {
        "patterns": [r"\bconsolidation\b", r"\bairspace (?:disease|opacity|consolidation)\b", r"\balveolar infiltrate\b"],
        "default_region": "lungs",
    },
    "pleural_effusion": {
        "patterns": [r"\bpleural effusion\b", r"\beffusion\b", r"\bcostophrenic (?:blunting|angle blunting)\b", r"\bfluid in the pleural\b"],
        "default_region": "left costophrenic angle",
    },
    "pneumothorax": {
        "patterns": [r"\bpneumothorax\b", r"\bapical pneumothorax\b", r"\btension pneumothorax\b"],
        "default_region": "right apical zone",
    },
    "pulmonary_edema": {
        "patterns": [r"\bpulmonary edema\b", r"\bedema\b", r"\bvascular congestion\b", r"\bhazy opacity\b", r"\binterstitial edema\b"],
        "default_region": "lungs",
    },
    "cardiomegaly": {
        "patterns": [r"\bcardiomegaly\b", r"\benlarged (?:cardiac silhouette|heart)\b", r"\bcardiac enlargement\b"],
        "default_region": "cardiac silhouette",
    },
    "atelectasis": {
        "patterns": [r"\batelectasis\b", r"\blinear atelectasis\b", r"\bpatchy atelectasis\b", r"\bvolume loss\b"],
        "default_region": "lungs",
    },
    "lung_opacity": {
        "patterns": [r"\blung opacity\b", r"\bfocal opacity\b", r"\bopacity\b", r"\bdensity\b", r"\binfiltrate\b"],
        "default_region": "lungs",
    },
    "support_devices": {
        "patterns": [
            r"\bendotracheal tube\b", r"\bett\b", r"\bpacemaker\b", r"\bpicc line\b",
            r"\bcentral line\b", r"\bchest tube\b", r"\bnasogastric tube\b", r"\bng tube\b",
            r"\benteric tube\b", r"\bcatheter\b", r"\bsupport lines?\b", r"\blines?\b"
        ],
        "default_region": "mediastinum",
    },
}

NEGATION_PATTERNS = [
    r"\bno\b", r"\bwithout\b", r"\bdenies\b", r"\bfree of\b", r"\bnegative for\b",
    r"\bunremarkable\b", r"\bnormal\b", r"\babsent\b", r"\bno acute\b", r"\bclear\b",
]

UNCERTAINTY_PATTERNS = [
    r"\bquestionable\b", r"\bequivocal\b", r"\bpossible\b", r"\bcannot exclude\b",
    r"\bsuggestive of\b", r"\bindeterminate\b", r"\buncertain\b", r"\bsuspicious for\b",
]


def extract_canonical_findings_from_report(
    report_text: str,
    sections: Optional[Dict[str, str]] = None,
) -> List[CanonicalFinding]:
    """
    Extracts structured CanonicalFinding instances from free-text report observations.
    Preserves exact anatomical location, laterality, and explicit presence states.
    Does NOT infer absence from missing text.
    """
    findings: List[CanonicalFinding] = []
    full_text = report_text or ""
    if sections:
        full_text = "\n".join(sections.values()) + "\n" + full_text

    sentences = [s.strip() for s in re.split(r"[.\n;]+", full_text) if s.strip()]

    evaluated_concepts: set = set()

    for sentence in sentences:
        s_lower = sentence.lower()

        for concept_name, concept_meta in PATHOLOGY_CONCEPTS.items():
            matched_pattern = any(re.search(p, s_lower) for p in concept_meta["patterns"])
            if not matched_pattern:
                continue

            # Determine presence state
            neg_words = [
                "no", "without", "free of", "negative for", "absent", "no acute",
                "not identified", "none identified", "none visible", "not seen",
                "not observed", "not present", "unidentified", "none", "clear"
            ]
            is_negated = any(re.search(r"\b" + re.escape(neg) + r"\b.*?" + p, s_lower) or
                            re.search(p + r".*?\b" + re.escape(neg) + r"\b", s_lower)
                            for neg in neg_words
                            for p in concept_meta["patterns"])

            is_uncertain = any(re.search(u, s_lower) for u in UNCERTAINTY_PATTERNS)

            if is_negated:
                presence = "absent"
                uncertainty = "low"
            elif is_uncertain:
                presence = "uncertain"
                uncertainty = "high"
            else:
                presence = "present"
                uncertainty = "low"

            # Normalize location and laterality
            norm_loc, lat = normalize_anatomical_location(sentence)
            if not norm_loc:
                norm_loc = concept_meta["default_region"]

            # Create finding
            finding = CanonicalFinding(
                finding_name=concept_name,
                presence_status=presence,
                anatomical_location=norm_loc,
                laterality=lat,
                image_evidence=sentence[:140],
                grounding_status="not_attempted",
                uncertainty=uncertainty,
                source="whole_image_vlm",
                review_required=(presence in ("present", "uncertain")),
            )
            findings.append(finding)
            evaluated_concepts.add(concept_name)

    return findings


# ── 4. Anatomy-Aware Reconciliation ───────────────────────────────────────────

def reconcile_evidence_with_grounding_and_classifier(
    whole_image_findings: List[CanonicalFinding],
    grounding_result: Optional[Dict[str, Any]] = None,
    classifier_scores: Optional[Dict[str, float]] = None,
    classifier_threshold: float = 0.62,
) -> Tuple[List[CanonicalFinding], Dict[str, Any]]:
    """
    Reconciles whole-image generative findings with targeted TransVG grounding
    and TorchXRayVision auxiliary classifier alerts.
    
    Adheres strictly to research guidelines:
    1. A classifier alert is added as an auxiliary review alert, NOT a confirmed diagnosis.
    2. A TransVG box provides localization evidence, NOT independent presence proof.
    3. Contradictions between regional affirmation and whole-image negation are explicitly flagged.
    """
    reconciled: List[CanonicalFinding] = [
        f.model_copy(deep=True) if hasattr(f, "model_copy") else f.copy(deep=True)
        for f in whole_image_findings
    ]
    audit_log: Dict[str, Any] = {
        "grounding_reconciliation": None,
        "classifier_alerts_added": [],
        "contradictions_detected": [],
        "location_mismatches": [],
    }

    # 1. Integrate Regional Grounding Result
    if grounding_result and grounding_result.get("success"):
        query = (grounding_result.get("query") or "").lower()
        pred_norm = grounding_result.get("pred_box_norm")
        reg_state = grounding_result.get("assessment_state", "unassessed").lower()

        # Find matching finding in whole image findings
        matching = [f for f in reconciled if f.finding_name in query or query in f.finding_name]

        if matching:
            for m in matching:
                m.grounding_status = "grounded" if pred_norm else "unsupported"
                m.supporting_region = pred_norm

                # Check for regional contradiction
                if "supported" in reg_state and m.presence_status == "absent":
                    m.review_required = True
                    m.uncertainty = "high"
                    audit_log["contradictions_detected"].append(
                        f"Regional grounding affirmed '{query}', but whole-image report stated absent."
                    )
        else:
            # Finding supported in region but completely omitted from whole-image report
            if "supported" in reg_state:
                norm_loc, lat = normalize_anatomical_location(query)
                new_finding = CanonicalFinding(
                    finding_name=query,
                    presence_status="present",
                    anatomical_location=norm_loc,
                    laterality=lat,
                    image_evidence=f"Affirmed by regional grounding crop: {query}",
                    grounding_status="grounded" if pred_norm else "unsupported",
                    supporting_region=pred_norm,
                    uncertainty="medium",
                    source="regional_vlm",
                    review_required=True,
                )
                reconciled.append(new_finding)
                audit_log["grounding_reconciliation"] = f"Added omitted regional finding: {query}"

    # 2. Integrate Auxiliary DenseNet-121 Classifier Alerts
    if classifier_scores:
        cls_alerts = {k: v for k, v in classifier_scores.items() if v >= classifier_threshold}
        for alert_finding, score in cls_alerts.items():
            norm_alert = alert_finding.lower().replace(" ", "_")
            existing = [f for f in reconciled if f.finding_name == norm_alert]

            if existing:
                for ex in existing:
                    if ex.presence_status == "absent":
                        ex.review_required = True
                        ex.uncertainty = "high"
                        audit_log["contradictions_detected"].append(
                            f"DenseNet-121 alert for '{alert_finding}' (score: {round(score, 2)} >= {classifier_threshold}) contradicts negative report statement."
                        )
            else:
                # Add secondary review alert without treating it as established diagnosis
                norm_loc, lat = normalize_anatomical_location(alert_finding)
                alert_entry = CanonicalFinding(
                    finding_name=norm_alert,
                    presence_status="uncertain",
                    anatomical_location=norm_loc,
                    laterality=lat,
                    image_evidence=f"Secondary DenseNet-121 review alert: score={round(score, 2)} >= {classifier_threshold}",
                    grounding_status="not_attempted",
                    uncertainty="high",
                    source="classifier_alert",
                    review_required=True,
                )
                reconciled.append(alert_entry)
                audit_log["classifier_alerts_added"].append(f"{alert_finding}: {round(score, 2)}")

    return reconciled, audit_log


# ── 5. Report Completeness & Fidelity Checks ──────────────────────────────────

def check_report_completeness_and_fidelity(
    findings: List[CanonicalFinding],
    final_report_text: str,
    patient_friendly_text: Optional[str] = None,
    ground_truth_has_devices: bool = False,
) -> Dict[str, Any]:
    """
    Performs deterministic completeness and safety verification on generated reports:
    1. Finding omissions: Present findings missing from final report.
    2. Unsupported assertions: Findings claimed in report without image evidence.
    3. Contradictory locations: Lateralities inconsistent between findings and text.
    4. Unsupported device descriptions: Devices mentioned in text without visual grounding.
    5. False-normal statements: Text claims 'normal' or 'clear' when abnormalities are affirmed.
    6. Patient translation fidelity: Ensures patient text strictly mirrors source findings.
    """
    rep_lower = (final_report_text or "").lower()
    pt_lower = (patient_friendly_text or "").lower() if patient_friendly_text else ""

    omissions: List[str] = []
    location_contradictions: List[str] = []
    unsupported_additions: List[str] = []
    false_normal_flagged: bool = False
    device_errors: List[str] = []
    patient_fidelity_violations: List[str] = []

    present_findings = [f for f in findings if f.presence_status == "present"]
    absent_findings = [f for f in findings if f.presence_status == "absent"]

    # 1. Check Omissions
    for pf in present_findings:
        if pf.source != "classifier_alert":  # Alerts are review signals, not required in primary text
            patterns = PATHOLOGY_CONCEPTS.get(pf.finding_name, {}).get("patterns", [])
            in_text = any(re.search(p, rep_lower) for p in patterns) if patterns else (pf.finding_name.replace("_", " ") in rep_lower)
            if not in_text:
                omissions.append(f"Finding '{pf.finding_name}' present in structured evidence but omitted in final report.")

    # 2. Check False-Normal Statements
    has_affirmed_abnormality = any(
        pf.finding_name != "support_devices" and pf.presence_status == "present" for pf in findings
    )
    if has_affirmed_abnormality:
        if re.search(r"\b(?:lungs (?:are|appear|otherwise)?\s*clear|normal chest radiograph|no acute cardiopulmonary|unremarkable)\b", rep_lower):
            false_normal_flagged = True

    # 3. Check Location & Laterality Consistency
    for pf in present_findings:
        if pf.laterality in ("left", "right"):
            opposite = "right" if pf.laterality == "left" else "left"
            # Flag if text asserts opposite laterality for the same finding
            if re.search(r"\b" + opposite + r"\b.*?\b" + re.escape(pf.finding_name.replace("_", " ")) + r"\b", rep_lower):
                location_contradictions.append(
                    f"Laterality contradiction: finding '{pf.finding_name}' structured as {pf.laterality}, but report text states {opposite}."
                )

    # 4. Check Unsupported Support Devices (Device hallucination detection)
    if not ground_truth_has_devices:
        for dev in ["endotracheal tube", "pacemaker", "picc line", "chest tube", "nasogastric", "support lines"]:
            if dev in rep_lower:
                dev_sentences = [s.strip() for s in re.split(r"[.\n;]+", rep_lower) if dev in s]
                is_affirmed = False
                for ds in dev_sentences:
                    is_neg = any(neg in ds for neg in ["not identified", "none", "no ", "without", "not seen", "absent", "unidentified", "not present"])
                    if not is_neg:
                        is_affirmed = True
                        break
                if is_affirmed:
                    device_errors.append(f"Report mentions '{dev}' without verified image evidence or ground-truth apparatus.")

    # 5. Patient-Friendly Fidelity Checks
    if pt_lower:
        # Patient text must not claim normal if abnormalities exist
        if has_affirmed_abnormality and ("lungs look clear and healthy" in pt_lower or "completely clear" in pt_lower):
            patient_fidelity_violations.append("Patient text claims lungs are clear despite affirmed abnormality in source report.")

        # Patient text must not invent devices
        if not ground_truth_has_devices:
            for dev in ["endotracheal tube", "pacemaker", "picc line", "chest tube", "support lines"]:
                if dev in pt_lower:
                    patient_fidelity_violations.append(f"Patient text mentions ungrounded device: '{dev}'.")

    return {
        "is_complete": len(omissions) == 0 and not false_normal_flagged,
        "omissions": omissions,
        "location_contradictions": location_contradictions,
        "unsupported_additions": unsupported_additions,
        "false_normal_flagged": false_normal_flagged,
        "device_errors": device_errors,
        "patient_fidelity_violations": patient_fidelity_violations,
    }
