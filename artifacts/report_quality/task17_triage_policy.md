# Task 17: Unified Research Triage Policy & Precedence Specification

## 1. Scope and Core Directives

This specification defines the canonical triage evaluation logic for the InteractMed-VLM system. All triage assessments across single-image grounding, whole-image generative reporting, auxiliary DenseNet-121 classification, and multi-file batch processing adhere strictly to this deterministic 10-level precedence engine.

### Strict Safety Requirements:
1. **Red Status Invariant:** Any Red assessment MUST have `review_required = true` and `review_priority = "urgent"`. It can **NEVER** display or emit `"routine"`.
2. **Green Status Invariant:** Green assessment MUST display `No configured review signal detected`. It can **NEVER** display `"Normal"`, `"Disease absent"`, `"Safe"`, or `"Cleared"`.
3. **Fail-Safe Invariant:** Any technical processing failure, invalid image, unvalidated draft, malformed report, or missing evidence can **NEVER** default or evaluate to Green.

---

## 2. Deterministic Precedence Hierarchy

The evaluation pipeline processes signals in strict ascending order of priority (Levels 1 through 10). The first matching rule determines the canonical status:

```
+-----------------------------------------------------------------------------------------+
| Level 1: Critical Processing / Image Integrity Failures                                 |
| - Condition: model_error, invalid_image, report_generation_failed                       |
| - Status: RED | Priority: urgent | Review Required: True                                |
+-----------------------------------------------------------------------------------------+
                                             | No
                                             v
+-----------------------------------------------------------------------------------------+
| Level 2: Cross-Report Contradiction / Multi-Model Discordance                           |
| - Condition: consistency_result.has_conflict == True or classifier_discordance == True  |
| - Status: RED | Priority: urgent | Review Required: True                                |
+-----------------------------------------------------------------------------------------+
                                             | No
                                             v
+-----------------------------------------------------------------------------------------+
| Level 3: Urgent Clinical Findings / Urgent Auxiliary Classifier Alerts                  |
| - Condition: Pneumothorax, severe pulmonary edema, or DenseNet-121 score >= 0.62         |
| - Status: RED | Priority: urgent | Review Required: True                                |
+-----------------------------------------------------------------------------------------+
                                             | No
                                             v
+-----------------------------------------------------------------------------------------+
| Level 4: Incomplete or Malformed Report                                                 |
| - Condition: report_validity in ("incomplete_report", "malformed_report")               |
| - Status: YELLOW | Priority: elevated | Review Required: True                          |
+-----------------------------------------------------------------------------------------+
                                             | No
                                             v
+-----------------------------------------------------------------------------------------+
| Level 5: Regional Assessment Indeterminate or Not Assessable                            |
| - Condition: regional_assessment_state in ("indeterminate", "not assessable")           |
| - Status: YELLOW | Priority: elevated | Review Required: True                          |
+-----------------------------------------------------------------------------------------+
                                             | No
                                             v
+-----------------------------------------------------------------------------------------+
| Level 6: Unsupported Grounding Query Capability                                         |
| - Condition: unsupported_query == True                                                  |
| - Status: YELLOW | Priority: elevated | Review Required: True                          |
+-----------------------------------------------------------------------------------------+
                                             | No
                                             v
+-----------------------------------------------------------------------------------------+
| Level 7: Support Apparatus / Line Placement Present                                     |
| - Condition: Endotracheal tube, pacemaker, PICC line, chest tube, or enteric tube       |
| - Status: YELLOW | Priority: device_verification | Review Required: True               |
+-----------------------------------------------------------------------------------------+
                                             | No
                                             v
+-----------------------------------------------------------------------------------------+
| Level 8: Non-Urgent Abnormality Flagged                                                  |
| - Condition: Pneumonia, consolidation, effusion, cardiomegaly, nodule, atelectasis      |
| - Status: YELLOW | Priority: elevated | Review Required: True                          |
+-----------------------------------------------------------------------------------------+
                                             | No
                                             v
+-----------------------------------------------------------------------------------------+
| Level 9: Insufficient or Indeterminate Evidence Fallback                                 |
| - Condition: Missing required sections, equivocal text                                  |
| - Status: YELLOW | Priority: elevated | Review Required: True                          |
+-----------------------------------------------------------------------------------------+
                                             | No
                                             v
+-----------------------------------------------------------------------------------------+
| Level 10: Concordant Negative (Zero Configured Signals Detected)                        |
| - Condition: Valid report, negative findings, zero classifier alerts, zero conflicts   |
| - Status: GREEN | Priority: routine | Review Required: False                            |
| - Label: "No configured review signal detected"                                         |
+-----------------------------------------------------------------------------------------+
```

---

## 3. Detailed Level Definitions

### Level 1: Critical Processing Failures (RED)
- **Status:** `red`
- **Status Label:** `Priority AI review required`
- **Review Required:** `true`
- **Review Priority:** `urgent`
- **Triggers:**
  - `model_error is not None` (Ollama timeout, connection drop, CUDA OOM).
  - `is_valid_image == False` (Corrupt PNG/JPEG, non-radiographic input).
  - `report_validity == "report_generation_failed"`.
- **Reason Code:** `CRITICAL_PROCESSING_FAILURE`

### Level 2: Multi-Model Contradiction & Discordance (RED)
- **Status:** `red`
- **Status Label:** `Priority AI review required`
- **Review Required:** `true`
- **Review Priority:** `urgent`
- **Triggers:**
  - Cross-report consistency service identifies a logical or semantic contradiction between the whole-image report and regional grounding hypothesis (e.g., regional report affirms focal pneumonia, whole-image report asserts clear lungs).
  - DenseNet-121 auxiliary classifier detects an acute abnormality ($\tau \ge 0.62$) that is explicitly denied or omitted from generative text.
- **Reason Code:** `CROSS_REPORT_DISCORDANCE`

### Level 3: Urgent Clinical Findings (RED)
- **Status:** `red`
- **Status Label:** `Priority AI review required`
- **Review Required:** `true`
- **Review Priority:** `urgent`
- **Triggers:**
  - Affirmed pneumothorax or tension pneumothorax.
  - Severe acute pulmonary edema with vascular congestion.
  - Auxiliary classifier pneumothorax score $\ge 0.62$.
- **Reason Code:** `POTENTIALLY_URGENT_FINDING` / `URGENT_CLASSIFIER_SIGNAL`

### Level 4: Incomplete or Malformed Report (YELLOW)
- **Status:** `yellow`
- **Status Label:** `Review advised`
- **Review Required:** `true`
- **Review Priority:** `elevated`
- **Triggers:**
  - `report_validity == "incomplete_report"` (missing required numbered sections).
  - `report_validity == "malformed_report"` (corrupted formatting or prompt leakage).
- **Reason Code:** `INCOMPLETE_OR_MALFORMED_REPORT`

### Level 5: Indeterminate Regional Assessment (YELLOW)
- **Status:** `yellow`
- **Status Label:** `Review advised`
- **Review Required:** `true`
- **Review Priority:** `elevated`
- **Triggers:**
  - Regional grounding assessment evaluates to `Indeterminate` or `Not assessable`.
- **Reason Code:** `INDETERMINATE_REGIONAL_ASSESSMENT`

### Level 6: Unsupported Grounding Query Capability (YELLOW)
- **Status:** `yellow`
- **Status Label:** `Review advised`
- **Review Required:** `true`
- **Review Priority:** `elevated`
- **Triggers:**
  - Query submitted is outside the validated capability vocabulary (e.g., rib fractures, surgical clips).
- **Reason Code:** `UNSUPPORTED_QUERY_CAPABILITY`

### Level 7: Support Apparatus Present (YELLOW)
- **Status:** `yellow`
- **Status Label:** `Review advised`
- **Review Required:** `true`
- **Review Priority:** `device_verification`
- **Triggers:**
  - Support lines, endotracheal tubes, pacemakers, or enteric catheters identified in image text.
- **Reason Code:** `SUPPORT_APPARATUS_PRESENT`

### Level 8: Non-Urgent Abnormality Flagged (YELLOW)
- **Status:** `yellow`
- **Status Label:** `Review advised`
- **Review Required:** `true`
- **Review Priority:** `elevated`
- **Triggers:**
  - Focal consolidation, non-acute effusion, cardiomegaly, lung nodule, atelectasis.
- **Reason Code:** `ABNORMALITY_FLAGGED` / `AUXILIARY_CLASSIFIER_ALERT`

### Level 9: Indeterminate Evidence Fallback (YELLOW)
- **Status:** `yellow`
- **Status Label:** `Review advised`
- **Review Required:** `true`
- **Review Priority:** `elevated`
- **Triggers:**
  - Equivocal observations, missing observation fields.
- **Reason Code:** `INDETERMINATE_EVIDENCE`

### Level 10: Concordant Negative (GREEN)
- **Status:** `green`
- **Status Label:** `No configured review signal detected`
- **Review Required:** `false`
- **Review Priority:** `routine`
- **Triggers:**
  - All automated analyses completed successfully.
  - Zero priority or non-priority abnormalities identified.
  - Zero classifier alerts ($\tau < 0.62$).
  - Regional hypothesis evaluated as not supported/negative.
  - No cross-report contradictions detected.
- **Reason Code:** `NO_CONFIGURED_REVIEW_SIGNAL`
- **Mandatory Safety Disclaimer:**
  > *"Research-only AI review status. This is not a clinical diagnosis or a substitute for professional interpretation."*

---

## 4. UI Representation Standards

| Canonical Status | Color Theme | Border / Badge Styling | Header Icon | Primary Title | Review Priority Pill |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RED** | Red (`#ef4444`) | Red border, red-tinted backdrop | 🚨 | `Priority AI review required` | `Priority: URGENT` |
| **YELLOW**| Amber (`#f59e0b`)| Amber border, amber-tinted backdrop | ⚠️ | `Review advised` | `Priority: ELEVATED` / `DEVICE_VERIFICATION` |
| **GREEN** | Emerald (`#10b981`)| Green border, green-tinted backdrop | 🛡️ | `No configured review signal detected` | `Priority: ROUTINE` |
