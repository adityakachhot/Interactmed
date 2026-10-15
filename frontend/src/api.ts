// api.ts — InteractMed-VLM backend client

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

export interface HealthResponse {
  status: string;
  grounding_model_loaded: boolean;
  grounding_model_device: string | null;
  grounding_model_error: string | null;
  grounding_model_version?: string | null;
  ollama_available: boolean;
  vision_model: string | null;   // e.g. "llava:7b" if installed
  text_model: string | null;     // e.g. "llama3:latest"
  timestamp: string;
}

export interface VisionModelStatus {
  ollama_available: boolean;
  vision_model: string | null;
  text_model: string | null;
  vision_model_ready: boolean;
  install_command: string | null;
}

export interface GroundingCapabilityEvidence {
  mean_iou: number;
  recall_50: number;
  recall_75: number;
  n_validation_queries: number;
}

export interface GroundingCapability {
  canonical_query: string;
  category: string;
  aliases: string[];
  available_annotations: string;
  selected_checkpoint: string;
  status: 'validated' | 'experimental' | 'unsupported';
  validation_evidence?: GroundingCapabilityEvidence | null;
  description?: string;
  warning?: string;
}

export interface GroundingCapabilitiesResponse {
  capabilities: Record<string, GroundingCapability>;
  unsupported_defaults: {
    status: string;
    warning: string;
    checkpoint: string;
  };
}

export interface GroundingResponse {
  success: boolean;
  query: string;
  checkpoint: string;
  model_version?: string | null;
  canonical_query?: string | null;
  category?: string | null;
  capability_status?: 'validated' | 'experimental' | 'unsupported' | null;
  validation_evidence?: GroundingCapabilityEvidence | null;
  warning?: string | null;
  pred_box_norm: number[] | null;     // [cx, cy, w, h] in [0,1]
  pred_box_pixel: number[] | null;    // [x1, y1, x2, y2] original pixels
  pred_box_canvas: number[] | null;   // [x1, y1, x2, y2] 640×640 canvas
  orig_h: number | null;
  orig_w: number | null;
  latency_ms: number;
  overlay_image_b64: string | null;   // annotated image with predicted box
  crop_image_b64: string | null;      // cropped region around predicted box
  checkpoint_advisory?: string | null;
  error: string | null;
}

export interface RegionReportResponse {
  success: boolean;
  report_text: string | null;
  assessment_state?: string | null;
  internal_contradiction?: boolean;
  internal_contradiction_warning?: string | null;
  crop_info: Record<string, unknown> | null;
  latency_ms: number;
  timing?: {
    prep_ms: number;
    inference_ms: number;
    post_ms: number;
    total_ms: number;
  } | null;
  model_used: string | null;
  image_input_used: boolean;
  n_images_sent: number;
  validation_warnings: string[];
  validation_flagged: boolean;
  error: string | null;
}

export interface TriageResult {
  triage_category: 'potentially_urgent' | 'abnormality_flagged' | 'no_priority_finding_flagged' | 'indeterminate_review_required';
  matched_rule_id: string;
  rule_reason: string;
  supporting_findings: string[];
  uncertainty_level: 'low' | 'medium' | 'high';
  evidence_notes: string;
  requires_human_review: boolean;
  human_review_priority: 'urgent' | 'routine' | 'standard_review';
  rule_engine_version: string;
  research_disclaimer: string;
}

export interface PatientFriendlyReportResponse {
  success: boolean;
  report_text: string | null;
  sections: Record<string, string> | null;
  cardiac_observation?: string | null;
  triage_result?: TriageResult | null;
  bytes_sha256?: string | null;
  pixel_sha256?: string | null;
  orig_dims?: [number, number] | null;
  prep_dims?: [number, number] | null;
  prompt_version?: string | null;
  latency_ms: number;
  timing?: {
    prep_ms: number;
    inference_ms: number;
    post_ms: number;
    total_ms: number;
  } | null;
  model_used: string | null;
  image_input_used: boolean;
  validation_warnings: string[];
  validation_flagged: boolean;
  error: string | null;
}

export type StudyReportResponse = PatientFriendlyReportResponse;

export interface FindingConsistency {
  finding_name: string;
  status: 'compatible' | 'conflicting' | 'insufficient_information' | 'single_report_only';
  regional_stance?: string | null;
  whole_image_stance?: string | null;
  conflict_message?: string | null;
  details?: Record<string, unknown> | null;
}

export interface ConsistencyCheckResponse {
  has_conflict: boolean;
  warning_banner?: string | null;
  summary: string;
  findings: FindingConsistency[];
  internal_regional_conflict: boolean;
  internal_regional_note?: string | null;
  evaluated_at: string;
}

export async function checkReportConsistency(
  query: string,
  regionalReportText?: string | null,
  wholeImageReportText?: string | null,
  wholeImageSections?: Record<string, string> | null,
): Promise<ConsistencyCheckResponse> {
  const resp = await fetch(`${BASE_URL}/report/consistency`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      regional_report_text: regionalReportText || null,
      whole_image_report_text: wholeImageReportText || null,
      whole_image_sections: wholeImageSections || null,
    }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail ?? `Consistency check failed: ${resp.status}`);
  }
  return resp.json();
}

export async function fetchHealth(): Promise<HealthResponse> {
  const resp = await fetch(`${BASE_URL}/health`);
  if (!resp.ok) throw new Error(`Health check failed: ${resp.status}`);
  return resp.json();
}

export async function fetchVisionModelStatus(): Promise<VisionModelStatus> {
  const resp = await fetch(`${BASE_URL}/vision-model-status`);
  if (!resp.ok) throw new Error(`Vision model status failed: ${resp.status}`);
  return resp.json();
}

export type CheckpointOption = 'auto' | 'multipathology' | 'finetuned' | 'baseline' | 'task2';

export async function fetchGroundingCapabilities(): Promise<GroundingCapabilitiesResponse> {
  const resp = await fetch(`${BASE_URL}/grounding/capabilities`);
  if (!resp.ok) throw new Error(`Grounding capabilities failed: ${resp.status}`);
  return resp.json();
}

export async function groundPhrase(
  file: File,
  query: string,
  checkpoint: CheckpointOption = 'auto',
  returnOverlay = true,
  returnCrop = true,
): Promise<GroundingResponse> {
  const fd = new FormData();
  fd.append('file', file);
  fd.append('query', query);
  fd.append('checkpoint', checkpoint);
  fd.append('return_overlay', String(returnOverlay));
  fd.append('return_crop', String(returnCrop));
  const resp = await fetch(`${BASE_URL}/ground`, { method: 'POST', body: fd });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail ?? `Ground request failed: ${resp.status}`);
  }
  return resp.json();
}

export async function generateRegionReport(
  file: File,
  query: string,
  boxPixelXyxy: number[],
  origH: number,
  origW: number,
  marginPx = 40,
): Promise<RegionReportResponse> {
  const fd = new FormData();
  fd.append('file', file);
  fd.append('query', query);
  fd.append('box_pixel_xyxy', JSON.stringify(boxPixelXyxy));
  fd.append('orig_h', String(origH));
  fd.append('orig_w', String(origW));
  fd.append('margin_px', String(marginPx));
  const resp = await fetch(`${BASE_URL}/report/region`, { method: 'POST', body: fd });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail ?? `Region report failed: ${resp.status}`);
  }
  return resp.json();
}

export async function generatePatientFriendlyReport(
  file: File,
  reasonForXray?: string,
  relevantHistory?: string,
): Promise<PatientFriendlyReportResponse> {
  const fd = new FormData();
  fd.append('file', file);
  if (reasonForXray) fd.append('reason_for_xray', reasonForXray);
  if (relevantHistory) fd.append('relevant_history', relevantHistory);
  const resp = await fetch(`${BASE_URL}/report/patient-friendly`, { method: 'POST', body: fd });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail ?? `Patient report failed: ${resp.status}`);
  }
  return resp.json();
}

export async function generateStudyReport(
  file: File,
  clinicalIndication: string,
  relevantHistory?: string,
  _priorReport?: string,
  _groundingResults?: GroundingResponse[],
): Promise<StudyReportResponse> {
  return generatePatientFriendlyReport(file, clinicalIndication, relevantHistory);
}

export async function generateResearchReport(
  file: File,
  clinicalIndication?: string,
  relevantHistory?: string,
): Promise<PatientFriendlyReportResponse> {
  const fd = new FormData();
  fd.append('file', file);
  if (clinicalIndication) fd.append('clinical_indication', clinicalIndication);
  if (relevantHistory) fd.append('relevant_history', relevantHistory);
  const resp = await fetch(`${BASE_URL}/report/research`, { method: 'POST', body: fd });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail ?? `Research report failed: ${resp.status}`);
  }
  return resp.json();
}

// ── Batch Reports API ────────────────────────────────────────────────────────

export interface BatchItemResponse {
  job_id: string;
  batch_id: string;
  filename: string;
  image_sha256?: string | null;
  bytes_sha256?: string | null;
  pixel_sha256?: string | null;
  orig_dims?: [number, number] | null;
  prep_dims?: [number, number] | null;
  prompt_version?: string | null;
  cardiac_observation?: string | null;
  style: 'patient' | 'research';
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  preview_b64?: string | null;
  report_text?: string | null;
  sections?: Record<string, string> | null;
  triage_result?: TriageResult | null;
  error?: string | null;
  created_at: number;
  started_at?: number | null;
  completed_at?: number | null;
  latency_ms?: number | null;
  timing?: {
    prep_ms?: number;
    inference_ms?: number;
    post_ms?: number;
    load_duration_ms?: number;
    prompt_eval_ms?: number;
    eval_duration_ms?: number;
    total_ms?: number;
  } | null;
  eval_count?: number | null;
  tokens_per_second?: number | null;
  model_used?: string | null;
  validation_warnings?: string[];
  validation_flagged?: boolean;
}

export interface BatchStatusResponse {
  batch_id: string;
  style: 'patient' | 'research';
  created_at: number;
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  cancelled: boolean;
  total_items: number;
  completed_count: number;
  failed_count: number;
  processing_count: number;
  queued_count: number;
  items: BatchItemResponse[];
}

export async function createBatch(
  files: File[],
  style: 'patient' | 'research' = 'patient',
): Promise<BatchStatusResponse> {
  const fd = new FormData();
  files.forEach(f => fd.append('files', f));
  fd.append('style', style);
  const resp = await fetch(`${BASE_URL}/batch/create`, { method: 'POST', body: fd });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail ?? `Batch creation failed: ${resp.status}`);
  }
  return resp.json();
}

export async function fetchBatchStatus(batchId: string): Promise<BatchStatusResponse> {
  const resp = await fetch(`${BASE_URL}/batch/${batchId}`);
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail ?? `Fetch batch failed: ${resp.status}`);
  }
  return resp.json();
}

export async function cancelBatch(batchId: string): Promise<BatchStatusResponse> {
  const resp = await fetch(`${BASE_URL}/batch/${batchId}/cancel`, { method: 'POST' });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail ?? `Cancel batch failed: ${resp.status}`);
  }
  return resp.json();
}

export async function retryBatch(batchId: string, jobIds?: string[]): Promise<BatchStatusResponse> {
  const resp = await fetch(`${BASE_URL}/batch/${batchId}/retry`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_ids: jobIds }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail ?? `Retry batch failed: ${resp.status}`);
  }
  return resp.json();
}

export function getBatchDownloadUrl(batchId: string, format: 'csv' | 'json' = 'json'): string {
  return `${BASE_URL}/batch/${batchId}/download?format=${format}`;
}
