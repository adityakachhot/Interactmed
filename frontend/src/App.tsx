import { useState, useEffect, useRef, useCallback, DragEvent } from 'react';
import './index.css';
import {
  fetchHealth, fetchVisionModelStatus, groundPhrase, generateRegionReport,
  generatePatientFriendlyReport, generateStudyReport,
  createBatch, fetchBatchStatus, cancelBatch, retryBatch, getBatchDownloadUrl,
  fetchGroundingCapabilities, checkReportConsistency, translateRegionalReport,
  type CheckpointOption, type GroundingCapability,
  type HealthResponse, type VisionModelStatus, type GroundingResponse,
  type RegionReportResponse, type PatientFriendlyReportResponse, type StudyReportResponse,
  type BatchItemResponse, type BatchStatusResponse, type TriageResult,
  type ConsistencyCheckResponse, type UnifiedTriage, type RegionalTranslationResponse,
} from './api';

const EXAMPLE_QUERIES = [
  'pneumonia',
  'cardiomegaly',
  'pleural effusion',
  'pneumothorax',
  'consolidation',
  'atelectasis',
  'pulmonary edema',
  'lung opacity',
  'rib fracture (unsupported)',
];

// ── Sub-components ─────────────────────────────────────────────────────────────

function StatusBar({ health, visionStatus }: { health: HealthResponse | null; visionStatus: VisionModelStatus | null }) {
  const modelOk  = health?.grounding_model_loaded;
  const ollamaOk = health?.ollama_available;
  const visionOk = visionStatus?.vision_model_ready;
  return (
    <div className="status-bar">
      <span className="status-pill">
        <span className={`status-dot ${health ? 'ok' : 'idle'}`} />
        Backend {health ? 'connected' : 'connecting…'}
      </span>
      <span className="status-pill">
        <span className={`status-dot ${modelOk ? 'ok' : health ? 'err' : 'idle'}`} />
        TransVG-AGPT {modelOk ? `(${health!.grounding_model_device ?? 'cpu'})` : 'not loaded'}
      </span>
      <span className="status-pill">
        <span className={`status-dot ${ollamaOk ? 'ok' : health ? 'warn' : 'idle'}`} />
        Ollama {ollamaOk ? 'ready' : 'offline'}
      </span>
      <span className="status-pill">
        <span className={`status-dot ${visionOk ? 'ok' : ollamaOk ? 'warn' : 'idle'}`} />
        {visionOk ? `Vision: ${visionStatus!.vision_model}` : ollamaOk ? 'No vision model' : 'Vision: —'}
      </span>
    </div>
  );
}

function Spinner() { return <div className="spinner" />; }

function LatencyBadge({
  ms,
  timing,
}: {
  ms: number;
  timing?: { prep_ms: number; inference_ms: number; post_ms: number; total_ms: number } | null;
}) {
  const title = timing
    ? `Total: ${timing.total_ms}ms (Prep: ${timing.prep_ms}ms, Inference: ${timing.inference_ms}ms, Post: ${timing.post_ms}ms)`
    : undefined;
  return (
    <span className="latency-badge" title={title}>
      ⏱ {ms.toFixed(0)} ms{timing ? ` (prep ${timing.prep_ms.toFixed(0)} / infer ${timing.inference_ms.toFixed(0)} / post ${timing.post_ms.toFixed(0)})` : ''}
    </span>
  );
}

function Alert({ type, message }: { type: 'error' | 'success' | 'info' | 'warn'; message: string }) {
  const icons: Record<string, string> = { error: '⚠️', success: '✓', info: 'ℹ', warn: '⚡' };
  return (
    <div className={`alert alert-${type === 'warn' ? 'error' : type}`}>
      <span>{icons[type]}</span>
      <span style={{ whiteSpace: 'pre-wrap' }}>{message}</span>
    </div>
  );
}

function Disclaimer() {
  return (
    <div className="disclaimer-banner">
      ⚠️ <strong>Experimental AI-generated report draft — not for clinical use without qualified professional review.</strong>
    </div>
  );
}

function PatientDisclaimer() {
  return (
    <div className="disclaimer-banner" style={{ marginTop: 14 }}>
      ⚠️ <strong>AI-generated explanation for educational purposes only. It may contain errors and cannot diagnose a condition or replace professional medical advice. Please discuss your X-ray results with a qualified healthcare professional.</strong>
    </div>
  );
}

function StatusPillBadge({ status }: { status: string }) {
  const icons: Record<string, string> = {
    queued: '⏳',
    processing: '⚡',
    completed: '✓',
    failed: '✕',
    cancelled: '⏹',
  };
  return (
    <span className={`status-pill-badge status-${status}`}>
      <span>{icons[status] || '•'}</span>
      <span>{status}</span>
    </span>
  );
}

function BatchItemStatusBadge({ item }: { item: BatchItemResponse }) {
  if (item.status === 'completed') {
    if (item.report_validity === 'malformed_report') {
      return (
        <span className="status-pill-badge status-malformed" title="Report parsing detected malformed structure or syntax cutoffs">
          <span>✕</span>
          <span>Report parsing failed</span>
        </span>
      );
    }
    if (item.report_validity === 'incomplete_report') {
      return (
        <span className="status-pill-badge status-incomplete" title="Report is missing required sections or truncated">
          <span>⚠️</span>
          <span>Incomplete report</span>
        </span>
      );
    }
    if (item.report_validity === 'report_generation_failed') {
      return (
        <span className="status-pill-badge status-failed" title="Report generation encountered an error">
          <span>✕</span>
          <span>Processing failed</span>
        </span>
      );
    }
    if (item.review_required) {
      return (
        <span className="status-pill-badge status-completed-review" title="Completed inference; multi-evidence check flagged review required">
          <span>⚖️</span>
          <span>Completed — review required</span>
        </span>
      );
    }
    return (
      <span className="status-pill-badge status-completed-validated" title="Report completed and validated structurally">
        <span>✓</span>
        <span>Completed — report validated</span>
      </span>
    );
  }
  if (item.status === 'failed') {
    return (
      <span className="status-pill-badge status-failed">
        <span>✕</span>
        <span>Processing failed</span>
      </span>
    );
  }
  if (item.status === 'processing') {
    return (
      <span className="status-pill-badge status-processing">
        <span>⚡</span>
        <span>Processing</span>
      </span>
    );
  }
  if (item.status === 'cancelled') {
    return (
      <span className="status-pill-badge status-cancelled">
        <span>⏹</span>
        <span>Cancelled</span>
      </span>
    );
  }
  return (
    <span className="status-pill-badge status-queued">
      <span>⏳</span>
      <span>Queued</span>
    </span>
  );
}

function TriageCard({
  triage,
  canonicalTriage,
  crossConflict = false,
  title = "Research-Only AI Review Status"
}: {
  triage?: TriageResult | null;
  canonicalTriage?: UnifiedTriage | null;
  crossConflict?: boolean;
  title?: string;
}) {
  const unified: (Partial<UnifiedTriage> & Partial<TriageResult>) | undefined = (canonicalTriage || triage) || undefined;
  if (!unified && !crossConflict) return null;

  // Canonical status determination
  let status: 'red' | 'yellow' | 'green' = 'yellow';
  if (crossConflict) {
    status = 'red';
  } else if (unified?.status) {
    status = unified.status;
  } else if (unified?.triage_category === 'potentially_urgent') {
    status = 'red';
  } else if (unified?.triage_category === 'no_priority_finding_flagged') {
    status = 'green';
  } else {
    status = 'yellow';
  }

  // Canonical human-readable status labels
  const statusLabel = status === 'red'
    ? '🚨 Priority AI Review Required'
    : status === 'yellow'
    ? '⚠️ Review Advised'
    : '🛡️ No Configured Review Signal Detected';

  // Strict review required logic
  const reviewRequired = status === 'red' || (status === 'yellow' && unified?.review_required !== false) || crossConflict;

  // Strict priority mapping: RED is NEVER routine
  let priority = unified?.review_priority || (status === 'red' ? 'urgent' : status === 'yellow' ? 'elevated' : 'routine');
  if (status === 'red' && (priority === 'routine' || priority === 'standard_review')) {
    priority = 'urgent';
  }

  // Compile structured reasons
  const structuredReasons: { code?: string; description: string }[] = [];
  if (crossConflict) {
    structuredReasons.push({
      code: 'RULE_MULTI_MODEL_DISCORDANCE',
      description: 'Conflicting AI assessments detected between regional grounding and whole-image reports. Independent professional review is required.'
    });
  }
  if (unified?.reasons && unified.reasons.length > 0) {
    unified.reasons.forEach(r => {
      if (!structuredReasons.some(existing => existing.description === r.description)) {
        structuredReasons.push({ code: r.code, description: r.description });
      }
    });
  } else if (triage?.rule_reason) {
    structuredReasons.push({
      code: triage.matched_rule_id,
      description: triage.rule_reason,
    });
  }

  return (
    <div className={`triage-card triage-status-${status}`} id="canonical-triage-card">
      <div className="triage-header-row">
        <div>
          <span style={{ fontSize: '0.74rem', textTransform: 'uppercase', letterSpacing: 0.5, fontWeight: 700, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>
            {title}
          </span>
          <span className={`triage-badge triage-${status}`}>
            {statusLabel}
          </span>
        </div>
        <div className="triage-meta-row">
          <span className="triage-pill">Review Required: <strong>{reviewRequired ? 'YES' : 'NO'}</strong></span>
          <span className="triage-pill">Priority: <strong>{priority.toUpperCase()}</strong></span>
          <span className="triage-pill">Policy: v{unified?.policy_version || '1.1.0'}</span>
          {unified?.uncertainty && (
            <span className="triage-pill">Uncertainty: {unified.uncertainty}</span>
          )}
        </div>
      </div>

      {structuredReasons.length > 0 && (
        <div className="triage-reason" style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <strong>Decision Evidence &amp; Triggers:</strong>
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {structuredReasons.map((r, idx) => (
              <li key={idx}>
                {r.code && <code style={{ marginRight: 6, fontSize: '0.74rem' }}>{r.code}</code>}
                {r.description}
              </li>
            ))}
          </ul>
        </div>
      )}

      {status === 'green' && (
        <div style={{ marginTop: 4, fontSize: '0.75rem', color: '#6ee7b7', background: 'rgba(16, 185, 129, 0.08)', padding: '6px 10px', borderRadius: 4, border: '1px solid rgba(16, 185, 129, 0.25)' }}>
          🛡️ <strong>No configured review signal detected:</strong> All automated evaluation steps completed successfully, reports are well-formed, and no configured review triggers are present. This does <strong>NOT</strong> confirm that the chest radiograph is normal, safe, or disease-free.
        </div>
      )}

      {unified?.supporting_findings && (unified.supporting_findings as string[]).length > 0 && (
        <div className="triage-findings-box">
          <div className="triage-findings-title">Supporting Clinical Findings</div>
          <ul className="triage-findings-list">
            {(unified.supporting_findings as string[]).map((finding, idx) => (
              <li key={idx}>{finding}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="triage-disclaimer-box">
        ⚖️ <strong>Research Disclaimer:</strong> {unified?.disclaimer || (unified as any)?.research_disclaimer || "Research-only AI review status. This is not a clinical diagnosis or a substitute for professional interpretation."}
      </div>
    </div>
  );
}

function ConsistencyBanner({ consistency }: { consistency: ConsistencyCheckResponse | null }) {
  if (!consistency) return null;

  if (consistency.has_conflict || consistency.review_required || (consistency.classifier_alerts && consistency.classifier_alerts.length > 0)) {
    return (
      <div className="consistency-box consistency-conflict" id="consistency-conflict-banner">
        <div className="consistency-header">
          <span>⚖️</span>
          <span>{consistency.warning_banner ?? 'Multi-Source AI Assessment Review Required'}</span>
        </div>
        <p className="consistency-summary">{consistency.summary}</p>
        {consistency.internal_regional_note && (
          <div className="consistency-note">
            <strong>Internal finding note:</strong> {consistency.internal_regional_note}
          </div>
        )}
        {consistency.classifier_alerts && consistency.classifier_alerts.length > 0 && (
          <div className="consistency-note" style={{ marginTop: '0.5rem', borderLeft: '3px solid #f59e0b' }}>
            <strong>Auxiliary Classifier Evidence Signals:</strong>
            <ul style={{ margin: '0.25rem 0 0 1.25rem', padding: 0 }}>
              {consistency.classifier_alerts.map((alert, idx) => (
                <li key={idx}>{alert}</li>
              ))}
            </ul>
          </div>
        )}
        <div className="consistency-disclaimer">
          Note: All model outputs (generative VLM, regional grounding, and pretrained classifier) are experimental research signals. Neither report is presumed correct. Professional radiologist evaluation is required.
        </div>
      </div>
    );
  }

  if (consistency.findings.some(f => f.status === 'insufficient_information')) {
    return (
      <div className="consistency-box consistency-insufficient">
        <div className="consistency-header">
          <span>ℹ️</span>
          <span>AI Assessment Comparison: Insufficient Information</span>
        </div>
        <p className="consistency-summary">{consistency.summary}</p>
      </div>
    );
  }

  return null;
}

function TriageCompactBadge({
  triage,
  canonicalTriage
}: {
  triage?: TriageResult | null;
  canonicalTriage?: UnifiedTriage | null;
}) {
  const unified = canonicalTriage || triage;
  if (!unified) return null;

  let status: 'red' | 'yellow' | 'green' = 'yellow';
  if (unified.status) {
    status = unified.status;
  } else if (unified.triage_category === 'potentially_urgent') {
    status = 'red';
  } else if (unified.triage_category === 'no_priority_finding_flagged') {
    status = 'green';
  }

  const badgeMap = {
    red: { label: '🚨 Priority Review', bg: 'rgba(239, 68, 68, 0.22)', color: '#fca5a5', border: '#ef4444' },
    yellow: { label: '⚠️ Review Advised', bg: 'rgba(245, 158, 11, 0.22)', color: '#fde68a', border: '#f59e0b' },
    green: { label: '🛡️ No Signal', bg: 'rgba(16, 185, 129, 0.20)', color: '#6ee7b7', border: '#10b981' },
  }[status];

  return (
    <span
      className={`triage-badge triage-${status}`}
      style={{
        padding: '2px 8px',
        fontSize: '0.70rem',
        fontWeight: 650,
        background: badgeMap.bg,
        color: badgeMap.color,
        border: `1px solid ${badgeMap.border}`,
        borderRadius: 4,
      }}
      title={`${unified.status_label || status.toUpperCase()}: Review Required = ${unified.review_required ? 'Yes' : 'No'}`}
    >
      {badgeMap.label}
    </span>
  );
}

function BatchDetailModal({
  item,
  onClose,
  onCopy,
  onDownload,
  copied,
}: {
  item: BatchItemResponse;
  onClose: () => void;
  onCopy: (text: string, jobId: string) => void;
  onDownload: (item: BatchItemResponse) => void;
  copied: boolean;
}) {
  const preview = item.preview_b64
    ? `data:image/jpeg;base64,${item.preview_b64}`
    : undefined;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-window" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <div className="modal-title">{item.filename}</div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'monospace', marginTop: 2 }}>
              Job ID: {item.job_id} · Style: {item.style}
            </div>
          </div>
          <button className="modal-close-btn" onClick={onClose} aria-label="Close modal">×</button>
        </div>

        <div className="modal-body">
          <div className="modal-grid">
            <div className="modal-img-container">
              {preview ? (
                <img src={preview} alt={item.filename} />
              ) : (
                <div className="empty-state" style={{ height: 200 }}>No preview available</div>
              )}
              <div style={{ marginTop: 12 }}>
                <BatchItemStatusBadge item={item} />
              </div>

              {item.classifier_alerts && item.classifier_alerts.length > 0 && (
                <div style={{ marginTop: 12, padding: 10, borderRadius: 6, background: 'rgba(245, 158, 11, 0.08)', border: '1px solid rgba(245, 158, 11, 0.3)', fontSize: '0.72rem' }}>
                  <div style={{ fontWeight: 600, color: '#fbbf24', marginBottom: 4 }}>
                    🔍 Auxiliary Classifier Signals (DenseNet-121)
                  </div>
                  <ul style={{ margin: '4px 0 0 14px', padding: 0 }}>
                    {item.classifier_alerts.map((al, i) => (
                      <li key={i}>{al}</li>
                    ))}
                  </ul>
                  <div style={{ fontSize: '0.67rem', color: 'var(--text-muted)', marginTop: 4 }}>
                    Research review signals; not calibrated probabilities or diagnoses.
                  </div>
                </div>
              )}

              {item.consistency_warnings && item.consistency_warnings.length > 0 && (
                <div style={{ marginTop: 10, padding: 10, borderRadius: 6, background: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.3)', fontSize: '0.72rem' }}>
                  <div style={{ fontWeight: 600, color: '#fca5a5', marginBottom: 4 }}>
                    ⚖️ Multi-Source Review Warning
                  </div>
                  <ul style={{ margin: '4px 0 0 14px', padding: 0 }}>
                    {item.consistency_warnings.map((cw, i) => (
                      <li key={i}>{cw}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Provenance and Identity Audit Card */}
              <div style={{ marginTop: 12, padding: 12, borderRadius: 6, background: 'var(--bg-card)', border: '1px solid var(--border-color)', fontSize: '0.72rem' }}>
                <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  🔒 Inference Identity Audit
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '4px 8px', wordBreak: 'break-all' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Bytes SHA-256:</span>
                  <span style={{ fontFamily: 'monospace' }}>{item.bytes_sha256 ? item.bytes_sha256.slice(0, 16) + '…' : (item.image_sha256 || '—')}</span>
                  
                  <span style={{ color: 'var(--text-muted)' }}>Pixel SHA-256:</span>
                  <span style={{ fontFamily: 'monospace' }}>{item.pixel_sha256 ? item.pixel_sha256.slice(0, 16) + '…' : '—'}</span>

                  <span style={{ color: 'var(--text-muted)' }}>Original Dims:</span>
                  <span>{item.orig_dims ? `${item.orig_dims[1]} × ${item.orig_dims[0]} px` : '—'}</span>

                  <span style={{ color: 'var(--text-muted)' }}>Model Prep Dims:</span>
                  <span>{item.prep_dims ? `${item.prep_dims[1]} × ${item.prep_dims[0]} px` : '—'}</span>

                  <span style={{ color: 'var(--text-muted)' }}>Prompt Schema:</span>
                  <span style={{ fontFamily: 'monospace', color: 'var(--accent)' }}>{item.prompt_version || '—'}</span>

                  <span style={{ color: 'var(--text-muted)' }}>Model:</span>
                  <span>{item.model_used || '—'}</span>
                </div>
              </div>
            </div>

            <div>
              {item.status === 'completed' && item.report_text ? (
                <>
                  {item.sections ? (
                    <div className="patient-sections-container">
                      {Object.entries(item.sections).map(([title, content]) => (
                        <div key={title} className="patient-section-block">
                          <div className="patient-section-title">{title}</div>
                          <div className="patient-section-body">{content.replace(/\*\*/g, '')}</div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="report-text-content" style={{ whiteSpace: 'pre-line' }}>
                      {item.report_text.replace(/\*\*/g, '')}
                    </div>
                  )}



                  {/* Timing & Performance audit */}
                  <div className="metrics-grid" style={{ marginTop: 16 }}>
                    <div className="metric-card">
                      <span className="metric-val">{item.latency_ms ? (item.latency_ms / 1000).toFixed(1) + 's' : '—'}</span>
                      <span className="metric-label">Total Latency</span>
                    </div>
                    {item.tokens_per_second && (
                      <div className="metric-card">
                        <span className="metric-val">{item.tokens_per_second} t/s</span>
                        <span className="metric-label">Generation Speed</span>
                      </div>
                    )}
                    {item.eval_count && (
                      <div className="metric-card">
                        <span className="metric-val">{item.eval_count}</span>
                        <span className="metric-label">Tokens Generated</span>
                      </div>
                    )}
                    {item.timing?.prompt_eval_ms && (
                      <div className="metric-card">
                        <span className="metric-val">{item.timing.prompt_eval_ms.toFixed(0)} ms</span>
                        <span className="metric-label">Prompt & Vision Eval</span>
                      </div>
                    )}
                  </div>

                  <div className="report-actions" style={{ marginTop: 16 }}>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      onClick={() => onCopy(item.report_text!, item.job_id)}
                    >
                      {copied ? '✓ Copied!' : '📋 Copy Report'}
                    </button>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      onClick={() => onDownload(item)}
                    >
                      💾 Download (.txt)
                    </button>
                  </div>
                </>
              ) : item.status === 'failed' ? (
                <>
                  <Alert type="error" message={item.error ?? 'Generation failed'} />
                </>
              ) : item.status === 'cancelled' ? (
                <Alert type="warn" message="This job was cancelled before processing started." />
              ) : (
                <div className="report-loading">
                  <Spinner />
                  <span>Processing radiograph with MedGemma 4B…</span>
                </div>
              )}
            </div>
          </div>
          <PatientDisclaimer />
        </div>
      </div>
    </div>
  );
}

function ValidationWarnings({ warnings, flagged }: { warnings: string[]; flagged: boolean }) {
  if (!warnings.length) return null;
  return (
    <div className={`alert ${flagged ? 'alert-error' : 'alert-info'}`} style={{ marginTop: 8 }}>
      <span>{flagged ? '⚠️' : 'ℹ'}</span>
      <div>
        <strong>Validation notices:</strong>
        <ul style={{ margin: '4px 0 0 16px', padding: 0 }}>
          {warnings.map((w, i) => <li key={i}>{w}</li>)}
        </ul>
      </div>
    </div>
  );
}

function CoordinatePanel({ result }: { result: GroundingResponse }) {
  const px = result.pred_box_pixel;
  const nm = result.pred_box_norm;
  const oh = result.orig_h ?? 0;
  const ow = result.orig_w ?? 0;

  // Derive area % from pixel box (not from normalized values)
  const areaPct = px && oh && ow
    ? (((px[2] - px[0]) * (px[3] - px[1])) / (ow * oh) * 100).toFixed(1)
    : null;

  return (
    <div className="metrics-grid">
      <div className="metric-card">
        <span className="metric-val">{ow} × {oh}</span>
        <span className="metric-label">Image W × H (px)</span>
      </div>
      <div className="metric-card">
        <span className="metric-val">{px?.map(v => Math.round(v)).join(', ')}</span>
        <span className="metric-label">Box [x1, y1, x2, y2] (px)</span>
      </div>
      <div className="metric-card">
        <span className="metric-val">{nm?.map(v => v.toFixed(4)).join(', ')}</span>
        <span className="metric-label">Model output [cx, cy, w, h] (norm)</span>
      </div>
      {areaPct && (
        <div className="metric-card">
          <span className="metric-val">{areaPct}%</span>
          <span className="metric-label">Box area (% of image pixels)</span>
        </div>
      )}
    </div>
  );
}

function ReportCard({
  report,
  loading,
  error,
  visionStatus,
  query,
  consistencyResult
}: {
  report: RegionReportResponse | null;
  loading: boolean;
  error: string | null;
  visionStatus: VisionModelStatus | null;
  query?: string;
  consistencyResult?: ConsistencyCheckResponse | null;
}) {
  if (loading) {
    return (
      <div className="report-loading">
        <Spinner />
        <span>Generating region report with {visionStatus?.vision_model ?? 'vision model'}…</span>
        <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
          First run may take 30–60 s while the model loads.
        </span>
      </div>
    );
  }
  if (error) return <Alert type="error" message={error} />;
  if (!report) return null;

  const modelBadge = report.model_used
    ? `${report.model_used}${report.image_input_used ? ` · ${report.n_images_sent} image(s) sent` : ' · text-only (no image)'}`
    : 'unknown model';

  let assessBadge: { text: string; bg: string; color: string; border: string } | null = null;
  if (report?.report_text) {
    const textLower = report.report_text.toLowerCase();
    if (textLower.includes('not assessable:') || textLower.includes('not assessable')) {
      assessBadge = { text: 'Hypothesis: Not Assessable', bg: 'rgba(239, 68, 68, 0.15)', color: '#f87171', border: 'rgba(239, 68, 68, 0.4)' };
    } else if (textLower.includes('not supported:') || textLower.includes('not supported')) {
      assessBadge = { text: 'Hypothesis: Not Supported', bg: 'rgba(59, 130, 246, 0.15)', color: '#60a5fa', border: 'rgba(59, 130, 246, 0.4)' };
    } else if (textLower.includes('indeterminate:') || textLower.includes('indeterminate')) {
      assessBadge = { text: 'Hypothesis: Indeterminate', bg: 'rgba(245, 158, 11, 0.15)', color: '#fbbf24', border: 'rgba(245, 158, 11, 0.4)' };
    } else if (textLower.includes('supported:') || textLower.includes('supported')) {
      assessBadge = { text: 'Hypothesis: Supported', bg: 'rgba(16, 185, 129, 0.15)', color: '#34d399', border: 'rgba(16, 185, 129, 0.4)' };
    }
  }

  return (
    <div className="region-report-card">
      <div className="region-report-header">
        <span className="region-report-title">📋 Grounding-Linked Regional Report</span>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          {assessBadge && (
            <span
              style={{
                padding: '3px 9px',
                borderRadius: 4,
                fontSize: '0.75rem',
                fontWeight: 600,
                background: assessBadge.bg,
                color: assessBadge.color,
                border: `1px solid ${assessBadge.border}`,
              }}
            >
              {assessBadge.text}
            </span>
          )}
          <LatencyBadge ms={report.latency_ms} timing={report.timing} />
          <span className="latency-badge">model: {modelBadge}</span>
          {report.image_input_used
            ? <span className="badge-vision">🖼 Image analysed</span>
            : <span className="badge-noimage">⚠ No image input</span>}
        </div>
      </div>

      {report.success ? (
        <>
          {report.internal_contradiction_warning && (
            <div style={{ marginBottom: 12, padding: '10px 14px', background: 'rgba(245, 158, 11, 0.1)', border: '1px solid rgba(245, 158, 11, 0.4)', borderRadius: 6, color: '#fbbf24', fontSize: '0.8rem' }}>
              ⚠️ <strong>Internal Report Discrepancy:</strong> {report.internal_contradiction_warning}
            </div>
          )}
          <div className="report-text-content">{report.report_text}</div>

          {/* Task 17 Grounding-Linked Patient Translation */}
          {report.patient_translation && report.patient_translation.success && (
            <div className="grounding-translation-box" id="grounding-patient-translation">
              <div className="grounding-translation-header">
                <div className="grounding-translation-title">
                  <span>🎯 Synchronized Patient-Friendly Translation (Grounding-Linked)</span>
                </div>
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  <span className="translation-meta-pill">Query: <strong>{report.patient_translation.source_query || query || 'Finding'}</strong></span>
                  <span className="translation-meta-pill">Assessment: <strong>{report.patient_translation.source_assessment_state || report.assessment_state || 'Supported'}</strong></span>
                  {report.report_id && <span className="translation-meta-pill">Report ID: <code>{report.report_id}</code></span>}
                </div>
              </div>

              {report.patient_translation.sections ? (
                <div className="translation-sections-grid">
                  {Object.entries(report.patient_translation.sections).map(([secTitle, secContent]) => (
                    <div key={secTitle} className="translation-section-card">
                      <div className="translation-section-name">{secTitle}</div>
                      <div className="translation-section-text">{secContent}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="translation-section-text" style={{ whiteSpace: 'pre-line' }}>
                  {report.patient_translation.translation_text}
                </div>
              )}
            </div>
          )}

          {/* Unified Red/Yellow/Green Triage Card */}
          {(report.canonical_triage || report.triage_result) && (
            <TriageCard
              triage={report.triage_result}
              canonicalTriage={report.canonical_triage || consistencyResult?.canonical_triage}
              crossConflict={consistencyResult?.has_conflict}
              title="Regional Research Review Status"
            />
          )}

          <ValidationWarnings warnings={report.validation_warnings ?? []} flagged={report.validation_flagged ?? false} />
          <Disclaimer />
        </>
      ) : (
        <Alert
          type="error"
          message={report.error ?? 'Report generation failed.'}
        />
      )}
    </div>
  );
}

function ComparisonTable() {
  const rows = [
    { metric: 'Mean Best-Match IoU', baseVal: '0.3124', baseTest: '0.3130', fineVal: '0.4494', fineTest: '0.4530', delta: '+44.7%' },
    { metric: 'Recall @ IoU ≥ 0.50', baseVal: '0.2331', baseTest: '0.2397', fineVal: '0.4928', fineTest: '0.4972', delta: '+107.5%' },
    { metric: 'Recall @ IoU ≥ 0.75', baseVal: '0.0433', baseTest: '0.0555', fineVal: '0.0688', fineTest: '0.0810', delta: '+45.9%' },
    { metric: 'Inference Success',    baseVal: '100%',   baseTest: '100%',   fineVal: '100%',   fineTest: '100%',   delta: '—' },
    { metric: 'Mean Latency',         baseVal: '139 ms', baseTest: '146 ms', fineVal: '134 ms', fineTest: '140 ms', delta: '—' },
  ];
  return (
    <table className="comparison-table">
      <thead>
        <tr>
          <th>Metric</th><th>Base (Val)</th><th>Fine (Val)</th>
          <th>Base (Test)</th><th>Fine (Test)</th><th>Test Rel Δ</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(r => (
          <tr key={r.metric}>
            <td>{r.metric}</td>
            <td>{r.baseVal}</td>
            <td className="highlight">{r.fineVal}</td>
            <td>{r.baseTest}</td>
            <td className="highlight">{r.fineTest}</td>
            <td className={r.delta !== '—' ? 'improve' : ''}>{r.delta}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ── Main App ───────────────────────────────────────────────────────────────────

export default function App() {
  const [health, setHealth]             = useState<HealthResponse | null>(null);
  const [visionStatus, setVisionStatus] = useState<VisionModelStatus | null>(null);
  const [activeTab, setActiveTab]       = useState<'ground' | 'report' | 'batch' | 'metrics'>('ground');

  // Upload state
  const [file, setFile]                 = useState<File | null>(null);
  const [previewUrl, setPreviewUrl]     = useState<string | null>(null);
  const [isDragOver, setIsDragOver]     = useState(false);
  const fileInputRef                    = useRef<HTMLInputElement>(null);

  // Grounding state
  const [query, setQuery]               = useState('pneumonia');
  const [checkpoint, setCheckpoint]     = useState<CheckpointOption>('auto');
  const [capabilities, setCapabilities] = useState<Record<string, GroundingCapability>>({});
  const [groundingResult, setGroundingResult] = useState<GroundingResponse | null>(null);
  const [groundingLoading, setGroundingLoading] = useState(false);
  const [groundingError, setGroundingError]     = useState<string | null>(null);

  // Region report state (on Grounding tab)
  const [regionReport, setRegionReport] = useState<RegionReportResponse | null>(null);
  const [regionLoading, setRegionLoading] = useState(false);
  const [regionError, setRegionError]     = useState<string | null>(null);

  // Patient-friendly report state (Patient-Friendly Report tab)
  const [reasonForXray, setReasonForXray]   = useState('');
  const [medicalHistory, setMedicalHistory] = useState('');
  const [patientReport, setPatientReport]   = useState<PatientFriendlyReportResponse | null>(null);
  const [patientLoading, setPatientLoading] = useState(false);
  const [patientError, setPatientError]     = useState<string | null>(null);
  const [copied, setCopied]                 = useState(false);
  const [showRegionalModal, setShowRegionalModal] = useState(false);
  const patientFileInputRef                 = useRef<HTMLInputElement>(null);

  // Cross-report consistency state
  const [consistencyResult, setConsistencyResult] = useState<ConsistencyCheckResponse | null>(null);

  const runConsistencyCheck = useCallback(async (
    regRep: RegionReportResponse | null,
    patRep: PatientFriendlyReportResponse | null,
    currentQuery: string,
  ) => {
    if (!regRep?.report_text || !patRep?.report_text) {
      if (regRep?.report_text && regRep.internal_contradiction) {
        try {
          const res = await checkReportConsistency(currentQuery, regRep.report_text, null, null);
          setConsistencyResult(res);
        } catch (e) {
          console.error("Consistency check error:", e);
        }
      } else {
        setConsistencyResult(null);
      }
      return;
    }

    try {
      const res = await checkReportConsistency(
        currentQuery,
        regRep.report_text,
        patRep.report_text,
        patRep.sections || undefined,
      );
      setConsistencyResult(res);
    } catch (e) {
      console.error("Cross-report consistency evaluation error:", e);
    }
  }, []);

  // Batch Reports state
  const [batchFiles, setBatchFiles]         = useState<File[]>([]);
  const [batchPreviews, setBatchPreviews]   = useState<{ file: File; url: string }[]>([]);
  const [batchStyle, setBatchStyle]         = useState<'patient' | 'research'>('patient');
  const [currentBatch, setCurrentBatch]     = useState<BatchStatusResponse | null>(null);
  const [batchSubmitting, setBatchSubmitting] = useState(false);
  const [batchError, setBatchError]         = useState<string | null>(null);
  const [selectedDetailItem, setSelectedDetailItem] = useState<BatchItemResponse | null>(null);
  const [copiedJobId, setCopiedJobId]       = useState<string | null>(null);
  const batchFileInputRef                   = useRef<HTMLInputElement>(null);

  // Poll health + vision status + load capabilities
  useEffect(() => {
    const poll = () => {
      fetchHealth().then(setHealth).catch(() => setHealth(null));
      fetchVisionModelStatus().then(setVisionStatus).catch(() => setVisionStatus(null));
    };
    poll();
    fetchGroundingCapabilities().then(res => setCapabilities(res.capabilities || {})).catch(() => {});
    const t = setInterval(poll, 15000);
    return () => clearInterval(t);
  }, []);

  // File handling
  const handleFile = useCallback((f: File) => {
    setFile(f);
    setGroundingResult(null);
    setRegionReport(null);
    setPatientReport(null);
    setConsistencyResult(null);
    setGroundingError(null);
    setRegionError(null);
    setPatientError(null);
    setPreviewUrl(URL.createObjectURL(f));
  }, []);

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  };

  // Grounding
  const handleGround = async () => {
    if (!file || !query.trim()) return;
    setGroundingLoading(true);
    setGroundingError(null);
    setGroundingResult(null);
    setRegionReport(null);
    setPatientReport(null);
    setRegionError(null);
    setConsistencyResult(null);
    try {
      const result = await groundPhrase(file, query.trim(), checkpoint, true, true);
      setGroundingResult(result);
      if (!result.success) setGroundingError(result.error ?? 'Grounding failed');
    } catch (err) {
      setGroundingError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setGroundingLoading(false);
    }
  };

  // Region report (on Grounding tab)
  const handleRegionReport = async () => {
    if (!file || !groundingResult?.pred_box_pixel) return;
    setRegionLoading(true);
    setRegionError(null);
    try {
      const result = await generateRegionReport(
        file, query,
        groundingResult.pred_box_pixel,
        groundingResult.orig_h ?? 0,
        groundingResult.orig_w ?? 0,
      );
      setRegionReport(result);
      if (!result.success) {
        setRegionError(result.error ?? 'Region report failed');
      } else {
        // Automatically synchronize the patient-friendly report with the grounding translation
        if (result.patient_translation?.success) {
          const pt = result.patient_translation;
          setPatientReport({
            success: true,
            report_text: pt.translation_text,
            sections: pt.sections,
            is_grounding_linked: true,
            source_report_id: pt.source_report_id || result.report_id,
            source_query: pt.source_query || query,
            source_assessment_state: pt.source_assessment_state || result.assessment_state,
            analysis_id: pt.analysis_id || result.analysis_id,
            canonical_triage: pt.canonical_triage || result.canonical_triage,
            triage_result: result.triage_result,
            latency_ms: 1,
            model_used: 'Deterministic Translation Engine v1.0',
            image_input_used: true,
            validation_warnings: [],
            validation_flagged: false,
            error: null,
          });
        }
        runConsistencyCheck(result, patientReport, query);
      }
    } catch (err) {
      setRegionError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setRegionLoading(false);
    }
  };

  // Translate regional report to patient-friendly explanation
  const handleTranslateRegional = async () => {
    if (!regionReport?.report_text) return;
    if (regionReport.patient_translation?.success) {
      const pt = regionReport.patient_translation;
      setPatientReport({
        success: true,
        report_text: pt.translation_text,
        sections: pt.sections,
        is_grounding_linked: true,
        source_report_id: pt.source_report_id || regionReport.report_id,
        source_query: pt.source_query || query,
        source_assessment_state: pt.source_assessment_state || regionReport.assessment_state,
        analysis_id: pt.analysis_id || regionReport.analysis_id,
        canonical_triage: pt.canonical_triage || regionReport.canonical_triage,
        triage_result: regionReport.triage_result,
        latency_ms: 1,
        model_used: 'Deterministic Translation Engine v1.0',
        image_input_used: true,
        validation_warnings: [],
        validation_flagged: false,
        error: null,
      });
      return;
    }
    setPatientLoading(true);
    setPatientError(null);
    try {
      const trans = await translateRegionalReport(
        regionReport.report_text,
        query,
        regionReport.assessment_state || 'Supported',
        regionReport.crop_info || undefined,
        regionReport.report_id || undefined,
        regionReport.analysis_id || undefined,
      );
      if (trans.success) {
        setPatientReport({
          success: true,
          report_text: trans.translation_text,
          sections: trans.sections,
          is_grounding_linked: true,
          source_report_id: trans.source_report_id,
          source_query: trans.source_query,
          source_assessment_state: trans.source_assessment_state,
          analysis_id: trans.analysis_id,
          canonical_triage: trans.canonical_triage || regionReport.canonical_triage,
          triage_result: regionReport.triage_result,
          latency_ms: 1,
          model_used: 'Deterministic Translation Engine v1.0',
          image_input_used: true,
          validation_warnings: [],
          validation_flagged: false,
          error: null,
        });
      } else {
        setPatientError(trans.error || 'Regional translation failed');
      }
    } catch (err) {
      setPatientError(err instanceof Error ? err.message : 'Regional translation failed');
    } finally {
      setPatientLoading(false);
    }
  };

  // Patient-friendly report (Patient-Friendly Report tab)
  const handlePatientReport = async () => {
    if (!file) return;
    setPatientLoading(true);
    setPatientError(null);
    try {
      const result = await generatePatientFriendlyReport(
        file,
        reasonForXray || undefined,
        medicalHistory || undefined,
      );
      setPatientReport(result);
      if (!result.success) {
        setPatientError(result.error ?? 'Report generation failed');
      } else {
        runConsistencyCheck(regionReport, result, query);
      }
    } catch (err) {
      setPatientError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setPatientLoading(false);
    }
  };

  const handleCopy = () => {
    if (!patientReport?.report_text) return;
    navigator.clipboard.writeText(patientReport.report_text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  const handleDownload = () => {
    if (!patientReport?.report_text) return;
    const blob = new Blob([patientReport.report_text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `patient_chest_xray_explanation.txt`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Polling for ongoing batch
  useEffect(() => {
    if (!currentBatch) return;
    const isOngoing = currentBatch.status === 'queued' || currentBatch.status === 'processing';
    if (!isOngoing) return;

    const timer = setInterval(async () => {
      try {
        const updated = await fetchBatchStatus(currentBatch.batch_id);
        setCurrentBatch(updated);
        if (selectedDetailItem) {
          const matched = updated.items.find(i => i.job_id === selectedDetailItem.job_id);
          if (matched) setSelectedDetailItem(matched);
        }
      } catch (err) {
        console.error('Error polling batch status:', err);
      }
    }, 1500);

    return () => clearInterval(timer);
  }, [currentBatch, selectedDetailItem]);

  const handleAddBatchFiles = (newFiles: FileList | File[]) => {
    const validExts = ['.dcm', '.png', '.jpg', '.jpeg', '.tiff', '.tif'];
    const added: File[] = [];
    Array.from(newFiles).forEach(f => {
      const ext = '.' + f.name.split('.').pop()?.toLowerCase();
      if (validExts.includes(ext)) {
        added.push(f);
      }
    });

    const combined = [...batchFiles, ...added];
    if (combined.length > 10) {
      setBatchError('Maximum 10 images allowed per batch. Excess files were ignored.');
    } else {
      setBatchError(null);
    }
    const capped = combined.slice(0, 10);
    setBatchFiles(capped);
    setBatchPreviews(capped.map(f => ({ file: f, url: URL.createObjectURL(f) })));
  };

  const handleRemoveBatchFile = (index: number) => {
    const updated = batchFiles.filter((_, i) => i !== index);
    setBatchFiles(updated);
    setBatchPreviews(updated.map(f => ({ file: f, url: URL.createObjectURL(f) })));
    if (updated.length <= 10) setBatchError(null);
  };

  const handleClearBatch = () => {
    setBatchFiles([]);
    setBatchPreviews([]);
    setCurrentBatch(null);
    setBatchError(null);
    setSelectedDetailItem(null);
  };

  const handleGenerateBatch = async () => {
    if (batchFiles.length === 0) return;
    setBatchSubmitting(true);
    setBatchError(null);
    try {
      const result = await createBatch(batchFiles, batchStyle);
      setCurrentBatch(result);
    } catch (err) {
      setBatchError(err instanceof Error ? err.message : 'Batch creation failed');
    } finally {
      setBatchSubmitting(false);
    }
  };

  const handleCancelBatch = async () => {
    if (!currentBatch) return;
    try {
      const updated = await cancelBatch(currentBatch.batch_id);
      setCurrentBatch(updated);
    } catch (err) {
      setBatchError(err instanceof Error ? err.message : 'Cancel failed');
    }
  };

  const handleRetryFailed = async (jobIds?: string[]) => {
    if (!currentBatch) return;
    try {
      const updated = await retryBatch(currentBatch.batch_id, jobIds);
      setCurrentBatch(updated);
    } catch (err) {
      setBatchError(err instanceof Error ? err.message : 'Retry failed');
    }
  };

  const handleCopyReport = (text: string, jobId: string) => {
    navigator.clipboard.writeText(text);
    setCopiedJobId(jobId);
    setTimeout(() => setCopiedJobId(null), 2500);
  };

  const handleDownloadSingleReport = (item: BatchItemResponse) => {
    if (!item.report_text) return;
    const blob = new Blob([item.report_text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${item.filename.replace(/\.[^/.]+$/, '')}_report.txt`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const overlayImage = groundingResult?.overlay_image_b64
    ? `data:image/jpeg;base64,${groundingResult.overlay_image_b64}`
    : null;
  const cropImage = groundingResult?.crop_image_b64
    ? `data:image/jpeg;base64,${groundingResult.crop_image_b64}`
    : null;

  const canGenerateReport = !!groundingResult?.pred_box_pixel && groundingResult.success;

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🩺 InteractMed-VLM</h1>
        <p>Phrase-Grounded Chest Radiograph Localization &amp; AI Report Drafting</p>
        <StatusBar health={health} visionStatus={visionStatus} />
      </header>

      <div className="tabs">
        {(['ground', 'report', 'batch', 'metrics'] as const).map(t => (
          <button key={t} className={`tab ${activeTab === t ? 'active' : ''}`} onClick={() => setActiveTab(t)}>
            {t === 'ground' ? '🎯 Grounding' : t === 'report' ? '🧑‍⚕️ Patient-Friendly Report' : t === 'batch' ? '📑 Batch Reports' : '📊 Metrics'}
          </button>
        ))}
      </div>

      {/* ── Grounding tab ─────────────────────────────────────────── */}
      {activeTab === 'ground' && (
        <div className="ground-layout">

          {/* Column 1: Upload + controls */}
          <div className="card">
            <div className="card-title">1 · Upload Chest Radiograph</div>
            <div
              className={`upload-zone ${isDragOver ? 'drag-over' : ''}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={e => { e.preventDefault(); setIsDragOver(true); }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={onDrop}
            >
              <svg width="36" height="36" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <p>{file ? `✓ ${file.name}` : 'Drop DICOM / PNG / JPEG here'}</p>
              <span>Max 50 MB · PA or AP view</span>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept=".dcm,.png,.jpg,.jpeg,.tiff,.tif"
              style={{ display: 'none' }}
              onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
            />

            {previewUrl && (
              <div className="image-preview" style={{ marginTop: 16 }}>
                <img src={previewUrl} alt="Uploaded radiograph" />
                <span className="image-label">Original — {file?.name}</span>
              </div>
            )}

            <div className="divider" />

            <div className="card-title">2 · Ground a Clinical Phrase</div>
            <div className="form-group">
              <label className="form-label">Clinical Phrase</label>
              <input
                id="phrase-input"
                className="form-input"
                value={query}
                onChange={e => setQuery(e.target.value)}
                placeholder="e.g. pneumonia"
              />
              <div className="example-pills">
                {EXAMPLE_QUERIES.map(q => {
                  const isUnsupported = q.includes('unsupported');
                  const cleanQ = q.replace(' (unsupported)', '');
                  return (
                    <button
                      key={q}
                      className="example-pill"
                      style={isUnsupported ? { borderColor: 'rgba(239, 68, 68, 0.4)', color: '#f87171' } : undefined}
                      onClick={() => setQuery(cleanQ)}
                    >
                      {q}
                    </button>
                  );
                })}
              </div>
            </div>
            <div className="form-group">
              <label className="form-label">Grounding Checkpoint & Routing</label>
              <select
                id="checkpoint-select"
                className="form-select"
                value={checkpoint}
                onChange={e => setCheckpoint(e.target.value as CheckpointOption)}
              >
                <option value="auto">Auto (Task 3 Epoch 5 Multi-Pathology, Val IoU 0.6550) — Recommended Default</option>
                <option value="multipathology">Multi-Pathology Unified (Task 3 Epoch 5, Val IoU 0.6550)</option>
                <option value="task2">Historical Task 2 Unified (Epoch 3, Val IoU 0.6510)</option>
                <option value="baseline">Pretrained Baseline (MS-CXR, Val IoU 0.6385)</option>
                <option value="finetuned">Fine-Tuned RSNA (Pneumonia Specialized, Val IoU 0.371 on multi-finding)</option>
              </select>
              {checkpoint === 'finetuned' && (query.toLowerCase().includes('cardio') || query.toLowerCase().includes('heart') || query.toLowerCase().includes('effusion') || query.toLowerCase().includes('pneumothorax')) && (
                <div style={{ marginTop: 6, fontSize: '0.76rem', color: '#eab308', lineHeight: 1.4 }}>
                  ⚠️ The RSNA fine-tuned checkpoint was specialized exclusively on pneumonia opacities and has reduced accuracy on other findings. The Unified or Baseline model is strongly recommended for this finding.
                </div>
              )}
            </div>
            <button
              id="ground-btn"
              className="btn btn-primary btn-full"
              onClick={handleGround}
              disabled={!file || !query.trim() || groundingLoading}
            >
              {groundingLoading ? <><Spinner /> Grounding…</> : '🎯 Ground Phrase'}
            </button>
            {groundingError && <Alert type="error" message={groundingError} />}
          </div>

          {/* Column 2: Prediction + crop */}
          <div className="card">
            <div className="card-title">3 · Predicted Region</div>
            {overlayImage ? (
              <>
                <div className="image-row">
                  <div className="image-preview">
                    <img src={overlayImage} alt="Annotated prediction" />
                    <span className="image-label">Predicted box (cyan)</span>
                  </div>
                  {cropImage && (
                    <div className="image-preview crop-preview">
                      <img src={cropImage} alt="Region crop" />
                      <span className="image-label">Region crop (+40 px margin)</span>
                    </div>
                  )}
                </div>

                <div style={{ marginTop: 10, display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                  <LatencyBadge ms={groundingResult!.latency_ms} />
                  <span className="latency-badge">ckpt: {groundingResult!.checkpoint}</span>
                  {groundingResult?.model_version && (
                    <span className="latency-badge" style={{ borderColor: 'rgba(56, 189, 248, 0.4)', color: '#38bdf8' }}>
                      {groundingResult.model_version}
                    </span>
                  )}
                  <span className="latency-badge">phrase: "{groundingResult!.query}"</span>
                  {groundingResult?.capability_status && (
                    <span
                      style={{
                        padding: '2px 8px',
                        borderRadius: 4,
                        fontSize: '0.74rem',
                        fontWeight: 600,
                        background: groundingResult.capability_status === 'validated' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                        color: groundingResult.capability_status === 'validated' ? '#10b981' : '#f87171',
                        border: `1px solid ${groundingResult.capability_status === 'validated' ? 'rgba(16, 185, 129, 0.4)' : 'rgba(239, 68, 68, 0.4)'}`
                      }}
                    >
                      {groundingResult.capability_status === 'validated' ? '✓ Supported category — benchmark evaluated' : '⚠️ Unsupported category — unvalidated finding'}
                    </span>
                  )}
                </div>

                {groundingResult?.capability_status === 'validated' && groundingResult.validation_evidence && (
                  <div style={{ marginTop: 8, padding: '8px 12px', background: 'rgba(16, 185, 129, 0.08)', border: '1px solid rgba(16, 185, 129, 0.25)', borderRadius: 6, fontSize: '0.78rem' }}>
                    <div style={{ fontWeight: 600, color: '#10b981' }}>
                      Category Benchmark Coverage: <span style={{ textTransform: 'capitalize' }}>{groundingResult.canonical_query}</span> ({groundingResult.category})
                    </div>
                    <div style={{ marginTop: 2, color: '#94a3b8', fontSize: '0.74rem' }}>
                      Aggregate Benchmark Performance: Mean IoU <strong>{(groundingResult.validation_evidence.mean_iou * 100).toFixed(1)}%</strong> | Recall@50 <strong>{groundingResult.validation_evidence.recall_50.toFixed(1)}%</strong> | Recall@75 <strong>{groundingResult.validation_evidence.recall_75.toFixed(1)}%</strong> (N={groundingResult.validation_evidence.n_validation_queries} dev queries)
                    </div>
                    <div style={{ marginTop: 4, color: 'var(--text-muted)', fontSize: '0.71rem', fontStyle: 'italic' }}>
                      Note: These metrics reflect aggregate benchmark evaluation on the development cohort and do NOT represent confidence or disease presence for this specific patient image.
                    </div>
                  </div>
                )}

                {groundingResult?.warning && (
                  <div style={{ marginTop: 8 }}>
                    <Alert type="warn" message={groundingResult.warning} />
                  </div>
                )}

                {groundingResult?.checkpoint_advisory && !groundingResult?.warning && (
                  <div style={{ marginTop: 8 }}>
                    <Alert type="warn" message={groundingResult.checkpoint_advisory} />
                  </div>
                )}

                <div className="divider" />
                <CoordinatePanel result={groundingResult!} />

                <div className="divider" />
                <div className="card-title" style={{ fontSize: '0.95rem' }}>4 · Region Report</div>

                {!visionStatus?.vision_model_ready && (
                  <Alert
                    type="warn"
                    message={
                      `No vision model detected in Ollama. Region reports require an image-capable vision-language model (e.g., medgemma:4b or llava:7b).\n` +
                      (visionStatus?.install_command ? `Install command: ${visionStatus.install_command}` : 'Install with: ollama pull medgemma:4b')
                    }
                  />
                )}

                <button
                  id="region-report-btn"
                  className="btn btn-accent btn-full"
                  onClick={handleRegionReport}
                  disabled={!canGenerateReport || regionLoading}
                  style={{ marginTop: 10 }}
                >
                  {regionLoading
                    ? <><Spinner /> Generating Region Report…</>
                    : visionStatus?.vision_model_ready
                      ? `📋 Generate Region Report (${visionStatus.vision_model})`
                      : '📋 Generate Region Report'}
                </button>

                <ReportCard
                  report={regionReport}
                  loading={regionLoading}
                  error={regionError}
                  visionStatus={visionStatus}
                  query={query}
                  consistencyResult={consistencyResult}
                />
                <ConsistencyBanner consistency={consistencyResult} />
              </>
            ) : (
              <div className="empty-state">
                <svg width="48" height="48" fill="none" viewBox="0 0 24 24" stroke="currentColor" opacity={0.3}>
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1}
                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <span>Upload an image and click "Ground Phrase"</span>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                  Predicted region and report will appear here
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Patient-Friendly Report tab ─────────────────────────────── */}
      {activeTab === 'report' && (
        <div className="main-grid">
          <div className="card">
            <div className="card-title">Understand Your Chest X-ray</div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: 16, lineHeight: 1.6 }}>
              Get a simple explanation of your chest X-ray using an AI-generated draft. This explanation is intended to help you understand medical terminology and is not a substitute for a clinician's interpretation.
            </p>

            {/* Image upload or reuse */}
            {file ? (
              <div className="reusable-image-bar">
                <div>
                  <div style={{ fontSize: '0.84rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                    📁 Using current X-ray: <span style={{ color: 'var(--accent)' }}>{file.name}</span>
                  </div>
                  <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginTop: 2 }}>
                    Reusing the image uploaded on the Grounding tab
                  </div>
                </div>
                <button
                  type="button"
                  className="btn btn-secondary"
                  style={{ padding: '5px 12px', fontSize: '0.76rem' }}
                  onClick={() => patientFileInputRef.current?.click()}
                >
                  Upload Different Image
                </button>
              </div>
            ) : (
              <div
                className={`upload-zone ${isDragOver ? 'drag-over' : ''}`}
                onClick={() => patientFileInputRef.current?.click()}
                onDragOver={e => { e.preventDefault(); setIsDragOver(true); }}
                onDragLeave={() => setIsDragOver(false)}
                onDrop={onDrop}
                style={{ marginBottom: 16 }}
              >
                <svg width="36" height="36" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <p>Upload Chest X-ray (DICOM / PNG / JPEG)</p>
                <span>Click to browse or drag & drop</span>
              </div>
            )}
            <input
              ref={patientFileInputRef}
              type="file"
              accept=".dcm,.png,.jpg,.jpeg,.tiff,.tif"
              style={{ display: 'none' }}
              onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
            />

            {/* Grounding-Linked Regional Report Availability Status */}
            {regionReport?.report_text ? (
              <div style={{ marginBottom: 18, padding: '14px 16px', background: 'rgba(56, 189, 248, 0.08)', border: '1.5px solid rgba(56, 189, 248, 0.4)', borderRadius: 8 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
                  <div>
                    <div style={{ fontSize: '0.94rem', fontWeight: 700, color: '#38bdf8' }}>
                      🎯 Grounding-Linked Regional Report Available
                    </div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: 3 }}>
                      Query: <strong>"{query}"</strong> · Regional Assessment: <strong>{regionReport.assessment_state || 'Supported'}</strong> · Report ID: <code>{regionReport.report_id || 'reg-report'}</code>
                    </div>
                    <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginTop: 2 }}>
                      Translates the grounding-linked regional report faithfully without independent VLM hallucination.
                    </div>
                  </div>
                  <button
                    id="translate-regional-btn"
                    type="button"
                    className="btn btn-primary"
                    onClick={handleTranslateRegional}
                    disabled={patientLoading}
                    style={{ padding: '8px 16px', fontWeight: 650 }}
                  >
                    {patientLoading ? <><Spinner /> Translating…</> : '🎯 Translate Grounding-Linked Report (Instant)'}
                  </button>
                </div>
              </div>
            ) : (
              <div style={{ marginBottom: 18, padding: '14px 16px', background: 'rgba(245, 158, 11, 0.08)', border: '1.5px solid rgba(245, 158, 11, 0.4)', borderRadius: 8 }} id="regional-unavailable-notice">
                <div style={{ fontSize: '0.90rem', fontWeight: 700, color: '#fbbf24' }}>
                  ⚠️ Regional report unavailable — patient-friendly translation cannot be generated
                </div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: 4, lineHeight: 1.5 }}>
                  A faithful patient-friendly explanation requires an active grounding-linked regional report. To generate one, navigate to the <strong>🎯 Grounding</strong> tab, ground a clinical phrase (e.g. <em>pneumonia</em>), and generate a regional report. The system will not silently invent a normal report or fall back to an unrelated whole-image interpretation.
                </div>
              </div>
            )}

            {patientLoading && (
              <div className="report-loading">
                <Spinner />
                <span>Processing translation / CXR analysis with {visionStatus?.vision_model ?? 'MedGemma 4B'}…</span>
                <span style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>
                  Synthesizing observations in plain, accessible language…
                </span>
              </div>
            )}

            {patientError && <Alert type="error" message={patientError} />}

            {patientReport && (
              <div className="patient-report-card">
                <div className="patient-report-header">
                  <div>
                    <span className="patient-report-title">
                      {patientReport.is_grounding_linked
                        ? '🎯 Grounding-Linked Patient-Friendly Translation'
                        : '🖼️ Whole-Image CXR Overview (Independent — Not Grounding-Linked)'}
                    </span>
                    {patientReport.is_grounding_linked ? (
                      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 4 }}>
                        <span className="translation-meta-pill">Query: <strong>{patientReport.source_query || query}</strong></span>
                        <span className="translation-meta-pill">Assessment: <strong>{patientReport.source_assessment_state || regionReport?.assessment_state || 'Supported'}</strong></span>
                        {patientReport.source_report_id && (
                          <span className="translation-meta-pill">Source ID: <code>{patientReport.source_report_id}</code></span>
                        )}
                      </div>
                    ) : (
                      <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginTop: 4 }}>
                        Independent whole-image interpretation. Does not translate or reflect regional phrase grounding.
                      </div>
                    )}
                  </div>

                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                    {patientReport.is_grounding_linked && regionReport?.report_text && (
                      <button
                        type="button"
                        className="btn btn-secondary"
                        style={{ padding: '4px 10px', fontSize: '0.74rem' }}
                        onClick={() => setShowRegionalModal(true)}
                        id="view-regional-report-btn"
                      >
                        🔍 View Original Regional Report
                      </button>
                    )}
                    <LatencyBadge ms={patientReport.latency_ms} timing={patientReport.timing} />
                    {patientReport.model_used && (
                      <span className="latency-badge">engine: {patientReport.model_used}</span>
                    )}
                    {patientReport.image_input_used ? (
                      <span className="badge-vision">🖼 Image verified</span>
                    ) : (
                      <span className="badge-noimage">⚠ No image input</span>
                    )}
                  </div>
                </div>

                {patientReport.success ? (
                  <>
                    <ConsistencyBanner consistency={consistencyResult} />

                    {patientReport.sections ? (
                      <div className="patient-sections-container">
                        {Object.entries(patientReport.sections).map(([title, content]) => (
                          <div key={title} className="patient-section-block">
                            <div className="patient-section-title">{title}</div>
                            <div className="patient-section-body">{content}</div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="report-text-content">{patientReport.report_text}</div>
                    )}

                    {/* Unified Canonical Triage Card */}
                    {(patientReport.canonical_triage || patientReport.triage_result) && (
                      <TriageCard
                        triage={patientReport.triage_result}
                        canonicalTriage={patientReport.canonical_triage || consistencyResult?.canonical_triage}
                        crossConflict={consistencyResult?.has_conflict}
                        title="Canonical AI Review Status"
                      />
                    )}

                    <div className="report-actions">
                      <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={handleCopy}
                        id="copy-report-btn"
                      >
                        {copied ? '✓ Copied to Clipboard!' : '📋 Copy Explanation'}
                      </button>
                      <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={handleDownload}
                        id="download-report-btn"
                      >
                        💾 Download Report (.txt)
                      </button>
                    </div>

                    <ValidationWarnings
                      warnings={patientReport.validation_warnings ?? []}
                      flagged={patientReport.validation_flagged ?? false}
                    />
                    <PatientDisclaimer />
                  </>
                ) : (
                  <>
                    <Alert type="error" message={patientReport.error ?? 'Generation failed'} />
                    {patientReport.triage_result && (
                      <TriageCard triage={patientReport.triage_result} />
                    )}
                  </>
                )}
              </div>
            )}

            {/* Separate Workflow: Independent Whole-Image CXR Overview */}
            <div className="independent-overview-callout">
              <div style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: 4 }}>
                🖼️ Independent Whole-Image CXR Overview (Separate Feature)
              </div>
              <p style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
                Generate an independent whole-image reading with MedGemma 4B across the full radiograph. This operates independently and is <strong>not</strong> a translation of regional phrase grounding.
              </p>

              <div className="form-group">
                <label className="form-label" htmlFor="reason-xray-input">
                  Reason the X-ray was taken <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>(optional)</span>
                </label>
                <input
                  id="reason-xray-input"
                  className="form-input"
                  value={reasonForXray}
                  onChange={e => setReasonForXray(e.target.value)}
                  placeholder="e.g. Cough for two weeks, routine checkup, or shortness of breath"
                />
                <span style={{ fontSize: '0.73rem', color: 'var(--text-muted)' }}>
                  Provides clinical context for the explanation. Do not include names, birth dates, or identifiers.
                </span>
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="history-xray-input">
                  Relevant medical history <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>(optional)</span>
                </label>
                <textarea
                  id="history-xray-input"
                  className="form-textarea"
                  value={medicalHistory}
                  onChange={e => setMedicalHistory(e.target.value)}
                  placeholder="e.g. Mild asthma, former smoker, no prior heart conditions"
                  rows={2}
                />
                <span style={{ fontSize: '0.73rem', color: 'var(--text-muted)' }}>
                  Only general health history. Never include patient name, address, or date of birth.
                </span>
              </div>

              <button
                id="patient-report-btn"
                className="btn btn-secondary btn-full"
                onClick={handlePatientReport}
                disabled={!file || patientLoading}
              >
                {patientLoading ? (
                  <><Spinner /> Analysing Full Chest X-ray…</>
                ) : (
                  'Generate Whole-Image CXR Overview (Independent)'
                )}
              </button>
            </div>

            {/* Source Regional Report Audit Modal */}
            {showRegionalModal && regionReport && (
              <div className="modal-overlay" onClick={() => setShowRegionalModal(false)}>
                <div className="modal-window" onClick={e => e.stopPropagation()} style={{ maxWidth: 650 }}>
                  <div className="modal-header">
                    <div>
                      <div className="modal-title">Source Grounding-Linked Regional Report</div>
                      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'monospace', marginTop: 2 }}>
                        Report ID: {regionReport.report_id || 'regional-report'} · Analysis ID: {regionReport.analysis_id || 'analysis-001'}
                      </div>
                    </div>
                    <button type="button" className="btn btn-secondary" onClick={() => setShowRegionalModal(false)}>✕ Close</button>
                  </div>
                  <div className="modal-body" style={{ maxHeight: '65vh', overflowY: 'auto' }}>
                    <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
                      <span className="triage-pill">Query: <strong>{query}</strong></span>
                      <span className="triage-pill">Assessment: <strong>{regionReport.assessment_state || 'Supported'}</strong></span>
                    </div>
                    <div className="report-text-content" style={{ whiteSpace: 'pre-line' }}>
                      {regionReport.report_text}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Right: image preview */}
          <div className="card">
            <div className="card-title">Current Chest X-ray</div>
            {previewUrl ? (
              <div className="image-preview">
                <img src={previewUrl} alt="Uploaded radiograph" />
                <span className="image-label">{file?.name}</span>
              </div>
            ) : (
              <div className="empty-state">
                <span>No image uploaded yet</span>
              </div>
            )}
            {overlayImage && (
              <>
                <div className="divider" />
                <div className="card-title" style={{ fontSize: '0.9rem' }}>Grounding Reference Overlay</div>
                <div className="image-preview">
                  <img src={overlayImage} alt="Grounding overlay" />
                  <span className="image-label">"{groundingResult?.query}" — {groundingResult?.checkpoint}</span>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* ── Batch Reports tab ─────────────────────────────────────── */}
      {activeTab === 'batch' && (
        <div>
          <div className="card batch-header-box">
            <div className="card-title">📑 Batch Chest X-Ray Report Generation</div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: 1.6, marginBottom: 16 }}>
              Upload up to 10 chest X-ray images for sequential whole-image report drafting using MedGemma 4B.
              Each radiograph is processed independently without region grounding or bounding boxes.
            </p>

            {/* Dropzone */}
            <div
              className={`upload-zone ${isDragOver ? 'drag-over' : ''}`}
              onClick={() => batchFileInputRef.current?.click()}
              onDragOver={e => { e.preventDefault(); setIsDragOver(true); }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={e => {
                e.preventDefault();
                setIsDragOver(false);
                if (e.dataTransfer.files) handleAddBatchFiles(e.dataTransfer.files);
              }}
              style={{ marginBottom: 16 }}
            >
              <svg width="40" height="40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                  d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
              <p>Drag &amp; drop chest X-rays or click to browse</p>
              <span>Up to 10 images · DICOM (.dcm), PNG, JPEG, TIFF · Max 50 MB each</span>
            </div>

            <input
              ref={batchFileInputRef}
              type="file"
              accept=".dcm,.png,.jpg,.jpeg,.tiff,.tif"
              multiple
              style={{ display: 'none' }}
              onChange={e => {
                if (e.target.files) handleAddBatchFiles(e.target.files);
                e.target.value = '';
              }}
            />

            {batchError && <Alert type="error" message={batchError} />}

            {/* Selected files preview grid */}
            {batchPreviews.length > 0 && (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                    Selected Images ({batchPreviews.length} / 10)
                  </span>
                  <button
                    type="button"
                    className="btn btn-secondary"
                    style={{ padding: '3px 8px', fontSize: '0.74rem' }}
                    onClick={handleClearBatch}
                  >
                    Clear All
                  </button>
                </div>
                <div className="batch-preview-grid">
                  {batchPreviews.map((p, idx) => (
                    <div key={idx} className="batch-preview-card">
                      <img src={p.url} alt={p.file.name} />
                      <div className="file-tag" title={p.file.name}>{p.file.name}</div>
                      <button
                        type="button"
                        className="batch-preview-remove"
                        onClick={() => handleRemoveBatchFile(idx)}
                        title="Remove image"
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Batch Toolbar */}
          <div className="batch-toolbar">
            <div className="batch-toolbar-left">
              <label htmlFor="batch-style-select" style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', fontWeight: 500 }}>
                Report Style:
              </label>
              <select
                id="batch-style-select"
                className="batch-style-select"
                value={batchStyle}
                onChange={e => setBatchStyle(e.target.value as 'patient' | 'research')}
                disabled={currentBatch?.status === 'processing'}
              >
                <option value="patient">🧑‍⚕️ Patient-Friendly Explanation (Default)</option>
                <option value="research">🔬 Structured Research-Style Report</option>
              </select>

              <button
                type="button"
                className="btn btn-primary"
                id="batch-generate-btn"
                onClick={handleGenerateBatch}
                disabled={batchFiles.length === 0 || batchSubmitting || currentBatch?.status === 'processing'}
              >
                {batchSubmitting ? (
                  <><Spinner /> Submitting Batch…</>
                ) : currentBatch?.status === 'processing' ? (
                  <><Spinner /> Processing Batch…</>
                ) : (
                  `✨ Generate Reports (${batchFiles.length} images)`
                )}
              </button>

              {currentBatch && (currentBatch.status === 'processing' || currentBatch.status === 'queued') && (
                <button
                  type="button"
                  className="btn btn-secondary"
                  id="batch-cancel-btn"
                  onClick={handleCancelBatch}
                  style={{ borderColor: 'var(--orange)', color: 'var(--orange)' }}
                >
                  ⏹ Cancel Remaining Jobs
                </button>
              )}

              {currentBatch && (currentBatch.failed_count > 0 || currentBatch.items.some(i => i.status === 'cancelled')) && (
                <button
                  type="button"
                  className="btn btn-secondary"
                  id="batch-retry-btn"
                  onClick={() => handleRetryFailed()}
                >
                  🔄 Retry Failed Jobs ({currentBatch.failed_count + currentBatch.items.filter(i => i.status === 'cancelled').length})
                </button>
              )}
            </div>

            <div className="batch-toolbar-right">
              {currentBatch && (
                <>
                  <a
                    href={getBatchDownloadUrl(currentBatch.batch_id, 'csv')}
                    download={`${currentBatch.batch_id}_summary.csv`}
                    className="btn btn-secondary"
                    id="download-csv-btn"
                  >
                    📥 Download CSV Summary
                  </a>
                  <a
                    href={getBatchDownloadUrl(currentBatch.batch_id, 'json')}
                    download={`${currentBatch.batch_id}_summary.json`}
                    className="btn btn-secondary"
                    id="download-json-btn"
                  >
                    📥 Download JSON Summary
                  </a>
                </>
              )}
            </div>
          </div>

          {/* Progress Bar */}
          {currentBatch && (
            <div className="batch-progress-wrapper">
              <div className="batch-progress-header">
                <span>
                  Batch <code>{currentBatch.batch_id}</code>: {currentBatch.completed_count} of {currentBatch.total_items} completed
                  {currentBatch.processing_count > 0 && ` · ${currentBatch.processing_count} processing`}
                  {currentBatch.failed_count > 0 && ` · ${currentBatch.failed_count} failed`}
                  {currentBatch.items.some(i => i.status === 'cancelled') &&
                    ` · ${currentBatch.items.filter(i => i.status === 'cancelled').length} cancelled`}
                </span>
                <span>
                  {Math.round(((currentBatch.completed_count + currentBatch.failed_count) / currentBatch.total_items) * 100)}%
                </span>
              </div>
              <div className="batch-progress-track">
                <div
                  className="batch-progress-fill"
                  style={{
                    width: `${((currentBatch.completed_count + currentBatch.failed_count) / currentBatch.total_items) * 100}%`,
                  }}
                />
              </div>
            </div>
          )}

          {/* Results List */}
          {currentBatch && currentBatch.items.length > 0 && (
            <div className="batch-items-list">
              {currentBatch.items.map((item, idx) => {
                const previewSrc = item.preview_b64
                  ? `data:image/jpeg;base64,${item.preview_b64}`
                  : (batchPreviews[idx]?.url || batchPreviews.find(p => p.file.name === item.filename)?.url);

                return (
                  <div key={item.job_id} className="batch-item-card" id={`batch-card-${item.job_id}`}>
                    <div className="batch-item-left">
                      {previewSrc ? (
                        <img
                          src={previewSrc}
                          alt={item.filename}
                          className="batch-item-thumb"
                          onClick={() => setSelectedDetailItem(item)}
                          style={{ cursor: 'pointer' }}
                          title="Click to view full image and report"
                        />
                      ) : (
                        <div className="batch-item-thumb empty-state" style={{ height: 160 }}>No Image</div>
                      )}
                      <div className="batch-item-filename">{item.filename}</div>
                      <div className="batch-item-jobid">{item.job_id}</div>
                      <BatchItemStatusBadge item={item} />
                    </div>

                    <div className="batch-item-right">
                      <div className="batch-item-header">
                        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                          <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                            {item.style === 'research' ? '🔬 Research Report' : '🧑‍⚕️ Patient Explanation'}
                          </span>
                          {item.model_used && (
                            <span className="latency-badge">model: {item.model_used}</span>
                          )}
                          {item.latency_ms && (
                            <LatencyBadge ms={item.latency_ms} />
                          )}
                          {item.tokens_per_second && (
                            <span className="latency-badge">{item.tokens_per_second} t/s</span>
                          )}
                        </div>
                        {item.classifier_alerts && item.classifier_alerts.length > 0 && (
                          <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginTop: 6, flexWrap: 'wrap' }}>
                            <span style={{ fontSize: '0.70rem', color: 'var(--text-muted)' }}>Classifier Signals:</span>
                            {item.classifier_alerts.map((alert, aIdx) => (
                              <span key={aIdx} className="classifier-alert-chip" title="Pretrained DenseNet-121 research signal (threshold 0.62)">
                                <span>🔍</span> {alert}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>

                      <div className="batch-item-body">
                        {item.status === 'completed' && item.report_text ? (
                          <div
                            className="report-text-content"
                            style={{ maxHeight: 180, cursor: 'pointer', overflowY: 'auto' }}
                            onClick={() => setSelectedDetailItem(item)}
                            title="Click to view full report"
                          >
                            {item.sections ? (
                              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                                {Object.entries(item.sections).map(([title, content]) => (
                                  <div key={title} style={{ fontSize: '0.82rem', lineHeight: 1.4 }}>
                                    <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{title}: </span>
                                    <span style={{ color: 'var(--text-secondary)' }}>{content.replace(/\*\*/g, '')}</span>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div style={{ whiteSpace: 'pre-line' }}>{item.report_text.replace(/\*\*/g, '')}</div>
                            )}
                          </div>
                        ) : item.status === 'processing' ? (
                          <div className="report-loading" style={{ padding: 20 }}>
                            <Spinner />
                            <span>MedGemma 4B analysing full radiograph…</span>
                          </div>
                        ) : item.status === 'queued' ? (
                          <div style={{ padding: 20, color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                            ⏳ Queued in batch worker pool…
                          </div>
                        ) : item.status === 'failed' ? (
                          <Alert type="error" message={item.error ?? 'Generation failed'} />
                        ) : (
                          <div style={{ padding: 20, color: 'var(--orange)', fontSize: '0.85rem' }}>
                            ⏹ Cancelled by user before processing started.
                          </div>
                        )}
                      </div>

                      <div className="batch-item-footer">
                        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                          {item.status === 'completed' && (
                            <>
                              <button
                                type="button"
                                className="btn btn-secondary"
                                style={{ padding: '4px 10px', fontSize: '0.76rem' }}
                                onClick={() => setSelectedDetailItem(item)}
                              >
                                🔍 Full Details
                              </button>
                              <button
                                type="button"
                                className="btn btn-secondary"
                                style={{ padding: '4px 10px', fontSize: '0.76rem' }}
                                onClick={() => handleCopyReport(item.report_text!, item.job_id)}
                              >
                                {copiedJobId === item.job_id ? '✓ Copied' : '📋 Copy'}
                              </button>
                              <button
                                type="button"
                                className="btn btn-secondary"
                                style={{ padding: '4px 10px', fontSize: '0.76rem' }}
                                onClick={() => handleDownloadSingleReport(item)}
                              >
                                💾 Download .txt
                              </button>
                            </>
                          )}
                          {(item.status === 'failed' || item.status === 'cancelled') && (
                            <button
                              type="button"
                              className="btn btn-secondary"
                              style={{ padding: '4px 10px', fontSize: '0.76rem' }}
                              onClick={() => handleRetryFailed([item.job_id])}
                            >
                              🔄 Retry this image
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Modal */}
          {selectedDetailItem && (
            <BatchDetailModal
              item={selectedDetailItem}
              onClose={() => setSelectedDetailItem(null)}
              onCopy={handleCopyReport}
              onDownload={handleDownloadSingleReport}
              copied={copiedJobId === selectedDetailItem.job_id}
            />
          )}

          <div style={{ marginTop: 24 }}>
            <PatientDisclaimer />
          </div>
        </div>
      )}

      {/* ── Metrics tab ───────────────────────────────────────────── */}
      {activeTab === 'metrics' && (
        <div className="card">
          <div className="card-title">RSNA Pneumonia Localization — Final Evaluation Results</div>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: 16 }}>
            Query: <code style={{ color: 'var(--accent)' }}>"pneumonia"</code> · N = 901 held-out test images ·
            Matching: best-overlap (max IoU over all GT boxes per image)
          </p>
          <ComparisonTable />
          <div className="divider" />
          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.7 }}>
            <strong style={{ color: 'var(--text-secondary)' }}>Protocol:</strong> Train/val/test splits are patient-disjoint (seed=42).
            Test IDs were never used for training, checkpoint selection, or query tuning.
            Fine-tuned checkpoint selected on val IoU only.
            All metrics computed with identical preprocessing and matching rules on both checkpoints.
          </div>
        </div>
      )}
    </div>
  );
}
