# Task 17: API and Contract Schema Changes

## 1. Overview

This document specifies the exact schema and interface modifications introduced in **Task 17**. These changes synchronize patient-friendly explanations with phrase grounding results, establish unified traffic-light triage models, and provide complete auditability across single-inference and batch workflows.

---

## 2. Backend FastAPI Endpoint Additions

### `POST /report/translate-regional`
Translates a grounding-linked regional report into a deterministic, patient-accessible plain-language explanation without secondary generative VLM inference.

#### Request Body (`RegionalTranslationRequest`):
```json
{
  "regional_report_text": "string (Required. Raw or structured text from region_report)",
  "phrase": "string (Optional. Grounding query text)",
  "assessment_state": "string (Optional. 'Supported' | 'Not supported' | 'Indeterminate' | 'Not assessable')",
  "report_id": "string (Optional. Unique identifier of source regional report)",
  "analysis_id": "string (Optional. Active analysis session identifier)",
  "crop_pixel_sha256": "string (Optional. SHA-256 digest of regional image crop)",
  "image_bytes_base64": "string (Optional. For multi-evidence canonical triage evaluation)"
}
```

#### Response Body (`RegionalTranslationResponse`):
```json
{
  "success": true,
  "translation_text": "string (Deterministic plain-language explanation)",
  "is_grounding_linked": true,
  "source_report_id": "string",
  "analysis_id": "string",
  "crop_pixel_sha256": "string",
  "source_query": "string",
  "source_assessment_state": "string",
  "canonical_triage": {
    "status": "red | yellow | green",
    "status_label": "string",
    "review_required": true,
    "review_priority": "urgent | elevated | device_verification | routine",
    "reasons": [
      {
        "code": "string",
        "description": "string"
      }
    ],
    "disclaimer": "string"
  }
}
```

---

## 3. Backend Pydantic Model Modifications

### 3.1 `UnifiedTriage` Sub-Model
```python
class UnifiedTriage(BaseModel):
    status: Literal["red", "yellow", "green"]
    status_label: str
    review_required: bool
    review_priority: Literal["urgent", "elevated", "device_verification", "routine"]
    reasons: List[Dict[str, str]] = []
    disclaimer: str
    uncertainty: Optional[str] = "low"
    triage_category: Optional[str] = None
    matched_rule_id: Optional[str] = None
    analysis_id: Optional[str] = None
    source_report_id: Optional[str] = None
```

### 3.2 `RegionReportResponse` (Extended)
```python
class RegionReportResponse(BaseModel):
    # Existing fields:
    success: bool
    report_text: str
    sections: Dict[str, str] = {}
    assessment_state: str
    finding_statement: str
    latency_ms: Optional[float] = None
    # Task 17 New Fields:
    report_id: Optional[str] = None
    analysis_id: Optional[str] = None
    crop_pixel_sha256: Optional[str] = None
    patient_translation: Optional[str] = None
    canonical_triage: Optional[UnifiedTriage] = None
```

### 3.3 `PatientFriendlyReportResponse` (Extended)
```python
class PatientFriendlyReportResponse(BaseModel):
    # Existing fields:
    success: bool
    report_text: str
    reading_level: str
    reading_ease_score: Optional[float] = None
    key_findings: List[str] = []
    disclaimer: str
    # Task 17 New Fields:
    is_grounding_linked: bool = False
    source_report_id: Optional[str] = None
    analysis_id: Optional[str] = None
    crop_pixel_sha256: Optional[str] = None
    source_query: Optional[str] = None
    canonical_triage: Optional[UnifiedTriage] = None
```

### 3.4 `ConsistencyCheckResponse` (Extended)
```python
class ConsistencyCheckResponse(BaseModel):
    # Existing fields:
    has_conflict: bool
    review_required: bool
    classifier_discordance: bool = False
    warning_banner: Optional[str] = None
    summary: str
    findings: List[Dict[str, Any]] = []
    classifier_alerts: List[str] = []
    # Task 17 New Fields:
    canonical_triage: Optional[UnifiedTriage] = None
```

---

## 4. Frontend TypeScript Contracts (`frontend/src/api.ts`)

```typescript
export interface UnifiedTriageReason {
  code: string;
  description: string;
}

export interface UnifiedTriage {
  status: 'red' | 'yellow' | 'green';
  status_label: string;
  review_required: boolean;
  review_priority: 'urgent' | 'elevated' | 'device_verification' | 'routine';
  reasons: UnifiedTriageReason[];
  disclaimer: string;
  uncertainty?: string;
  triage_category?: string;
  matched_rule_id?: string;
  analysis_id?: string;
  source_report_id?: string;
}

export interface RegionalTranslationResponse {
  success: boolean;
  translation_text: string;
  is_grounding_linked: boolean;
  source_report_id?: string;
  analysis_id?: string;
  crop_pixel_sha256?: string;
  source_query?: string;
  source_assessment_state?: string;
  canonical_triage?: UnifiedTriage;
}

export async function translateRegionalReport(
  regionalReportText: string,
  phrase?: string,
  assessmentState?: string,
  reportId?: string,
  analysisId?: string,
  cropPixelSha256?: string
): Promise<RegionalTranslationResponse> {
  const res = await fetch(`${API_BASE}/report/translate-regional`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      regional_report_text: regionalReportText,
      phrase,
      assessment_state: assessmentState,
      report_id: reportId,
      analysis_id: analysisId,
      crop_pixel_sha256: cropPixelSha256,
    }),
  });
  if (!res.ok) {
    throw new Error(`Translation failed with status: ${res.status}`);
  }
  return res.json();
}
```

---

## 5. Batch Service CSV & JSON Export Additions

### CSV Export Header:
```csv
Job ID,Filename,Style,Status,Image SHA256,Bytes SHA256,Pixel SHA256,Orig Dims,Prep Dims,Prompt Version,Model,Triage Category,Matched Rule ID,Review Priority,Latency (ms),Tokens/sec,Error,Report Text,Report Validity,Review Required,Classifier Alerts,Triage State,Canonical Status,Canonical Status Label
```

New Columns:
1. `Canonical Status`: `"red"` | `"yellow"` | `"green"`
2. `Canonical Status Label`: `"Priority AI review required"` | `"Review advised"` | `"No configured review signal detected"`

---

## 6. Backward Compatibility & Migration Note
- All existing endpoints (`/report/region`, `/report/patient-friendly`, `/consistency/check`, `/batch/*`) remain 100% backward-compatible.
- New fields are optional in requests and additive in responses.
- The previous MedGemma whole-image patient report endpoint remains intact for users explicitly opting into an independent whole-image overview.
