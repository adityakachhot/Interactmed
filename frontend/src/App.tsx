import { useState, useEffect, useRef, useCallback, DragEvent } from 'react';
import './index.css';
import {
  fetchHealth, fetchVisionModelStatus, groundPhrase, generateRegionReport,
  generatePatientFriendlyReport, generateStudyReport,
  createBatch, fetchBatchStatus, cancelBatch, retryBatch, getBatchDownloadUrl,
  type HealthResponse, type VisionModelStatus, type GroundingResponse,
  type RegionReportResponse, type PatientFriendlyReportResponse, type StudyReportResponse,
  type BatchItemResponse, type BatchStatusResponse, type TriageResult,
} from './api';

const EXAMPLE_QUERIES = ['pneumonia', 'airspace opacity', 'pleural effusion', 'consolidation', 'cardiomegaly'];

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

function TriageCard({ triage }: { triage?: TriageResult | null }) {
  if (!triage) return null;

  const categoryLabelMap: Record<string, string> = {
    potentially_urgent: '🚨 Potentially Urgent',
    abnormality_flagged: '⚠️ Abnormality Flagged',
    no_priority_finding_flagged: 'ℹ️ No Priority Finding Flagged',
    indeterminate_review_required: '❓ Indeterminate (Review Required)',
  };

  const label = categoryLabelMap[triage.triage_category] || triage.triage_category;

  return (
    <div className="triage-card">
      <div className="triage-header-row">
        <div>
          <span style={{ fontSize: '0.74rem', textTransform: 'uppercase', letterSpacing: 0.5, fontWeight: 700, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>
            Research-Only Triage Suggestion
          </span>
          <span className={`triage-badge triage-${triage.triage_category}`}>
            {label}
          </span>
        </div>
        <div className="triage-meta-row">
          <span className="triage-pill">Rule: <code>{triage.matched_rule_id}</code></span>
          <span className="triage-pill">Review: <strong>{triage.human_review_priority}</strong></span>
          <span className="triage-pill">Uncertainty: {triage.uncertainty_level}</span>
          <span className="triage-pill">v{triage.rule_engine_version}</span>
        </div>
      </div>

      <div className="triage-reason">
        <strong>Reason:</strong> {triage.rule_reason}
      </div>

      {triage.supporting_findings && triage.supporting_findings.length > 0 && (
        <div className="triage-findings-box">
          <div className="triage-findings-title">Supporting Structured Findings</div>
          <ul className="triage-findings-list">
            {triage.supporting_findings.map((finding, idx) => (
              <li key={idx}>{finding}</li>
            ))}
          </ul>
        </div>
      )}

      {triage.evidence_notes && (
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
          Evidence notes: {triage.evidence_notes}
        </div>
      )}

      <div className="triage-disclaimer-box">
        ⚖️ <strong>Research Notice:</strong> {triage.research_disclaimer || "Experimental triage priority suggestion. Not clinically validated. Does not establish medical urgency or replace clinical review."}
      </div>
    </div>
  );
}

function TriageCompactBadge({ triage }: { triage?: TriageResult | null }) {
  if (!triage) return null;
  const categoryLabelMap: Record<string, string> = {
    potentially_urgent: '🚨 Urgent Flag',
    abnormality_flagged: '⚠️ Abnormality',
    no_priority_finding_flagged: 'ℹ️ No Priority Finding',
    indeterminate_review_required: '❓ Review Required',
  };
  const label = categoryLabelMap[triage.triage_category] || triage.triage_category;
  return (
    <span
      className={`triage-badge triage-${triage.triage_category}`}
      style={{ padding: '2px 8px', fontSize: '0.70rem' }}
      title={`Rule: ${triage.matched_rule_id} — ${triage.rule_reason}`}
    >
      {label}
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
                <StatusPillBadge status={item.status} />
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
                          <div className="patient-section-body">{content}</div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="report-text-content">{item.report_text}</div>
                  )}

                  {/* Research-only Triage Priority Suggestion */}
                  {item.triage_result && (
                    <TriageCard triage={item.triage_result} />
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
                  {item.triage_result && (
                    <TriageCard triage={item.triage_result} />
                  )}
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

function ReportCard({ report, loading, error, visionStatus }: {
  report: RegionReportResponse | null;
  loading: boolean;
  error: string | null;
  visionStatus: VisionModelStatus | null;
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

  return (
    <div className="region-report-card">
      <div className="region-report-header">
        <span className="region-report-title">📋 Region Report</span>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          <LatencyBadge ms={report.latency_ms} timing={report.timing} />
          <span className="latency-badge">model: {modelBadge}</span>
          {report.image_input_used
            ? <span className="badge-vision">🖼 Image analysed</span>
            : <span className="badge-noimage">⚠ No image input</span>}
        </div>
      </div>

      {report.success ? (
        <>
          <div className="report-text-content">{report.report_text}</div>
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
  const [checkpoint, setCheckpoint]     = useState<'finetuned' | 'baseline'>('finetuned');
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
  const patientFileInputRef                 = useRef<HTMLInputElement>(null);

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

  // Poll health + vision status
  useEffect(() => {
    const poll = () => {
      fetchHealth().then(setHealth).catch(() => setHealth(null));
      fetchVisionModelStatus().then(setVisionStatus).catch(() => setVisionStatus(null));
    };
    poll();
    const t = setInterval(poll, 15000);
    return () => clearInterval(t);
  }, []);

  // File handling
  const handleFile = useCallback((f: File) => {
    setFile(f);
    setGroundingResult(null);
    setRegionReport(null);
    setPatientReport(null);
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
    setRegionError(null);
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
      if (!result.success) setRegionError(result.error ?? 'Region report failed');
    } catch (err) {
      setRegionError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setRegionLoading(false);
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
      if (!result.success) setPatientError(result.error ?? 'Report generation failed');
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
                {EXAMPLE_QUERIES.map(q => (
                  <button key={q} className="example-pill" onClick={() => setQuery(q)}>{q}</button>
                ))}
              </div>
            </div>
            <div className="form-group">
              <label className="form-label">Checkpoint</label>
              <select
                id="checkpoint-select"
                className="form-select"
                value={checkpoint}
                onChange={e => setCheckpoint(e.target.value as 'finetuned' | 'baseline')}
              >
                <option value="finetuned">Fine-Tuned (Run 01, 3 ep) — recommended</option>
                <option value="baseline">Pretrained Baseline — research comparison</option>
              </select>
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
                    <div className="image-preview">
                      <img src={cropImage} alt="Region crop" />
                      <span className="image-label">Region crop (+40 px margin)</span>
                    </div>
                  )}
                </div>

                <div style={{ marginTop: 10, display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                  <LatencyBadge ms={groundingResult!.latency_ms} />
                  <span className="latency-badge">ckpt: {groundingResult!.checkpoint}</span>
                  <span className="latency-badge">phrase: "{groundingResult!.query}"</span>
                </div>

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
                />
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
              className="btn btn-primary btn-full"
              onClick={handlePatientReport}
              disabled={!file || patientLoading}
            >
              {patientLoading ? (
                <><Spinner /> Analysing Chest X-ray…</>
              ) : (
                'Generate Patient-Friendly Report'
              )}
            </button>

            {patientLoading && (
              <div className="report-loading">
                <Spinner />
                <span>Analysing full chest X-ray with {visionStatus?.vision_model ?? 'MedGemma 4B'}…</span>
                <span style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>
                  Synthesizing observations in plain, accessible language (approx. 15–30s)
                </span>
              </div>
            )}

            {patientError && <Alert type="error" message={patientError} />}

            {patientReport && (
              <div className="patient-report-card">
                <div className="patient-report-header">
                  <span className="patient-report-title">🧑‍⚕️ Patient-Friendly Explanation</span>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                    <LatencyBadge ms={patientReport.latency_ms} timing={patientReport.timing} />
                    {patientReport.model_used && (
                      <span className="latency-badge">model: {patientReport.model_used}</span>
                    )}
                    {patientReport.image_input_used ? (
                      <span className="badge-vision">🖼 Full X-ray analysed</span>
                    ) : (
                      <span className="badge-noimage">⚠ No image input</span>
                    )}
                  </div>
                </div>

                {patientReport.success ? (
                  <>
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

                    {/* Research-only Triage Priority Suggestion */}
                    {patientReport.triage_result && (
                      <TriageCard triage={patientReport.triage_result} />
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
                  : batchPreviews[idx]?.url;

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
                      <StatusPillBadge status={item.status} />
                    </div>

                    <div className="batch-item-right">
                      <div className="batch-item-header">
                        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                          <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                            {item.style === 'research' ? '🔬 Research Report' : '🧑‍⚕️ Patient Explanation'}
                          </span>
                          {item.triage_result && (
                            <TriageCompactBadge triage={item.triage_result} />
                          )}
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
                      </div>

                      <div className="batch-item-body">
                        {item.status === 'completed' && item.report_text ? (
                          <div
                            className="report-text-content"
                            style={{ maxHeight: 180, cursor: 'pointer' }}
                            onClick={() => setSelectedDetailItem(item)}
                            title="Click to view full report"
                          >
                            {item.report_text}
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
