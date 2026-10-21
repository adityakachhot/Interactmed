"""
src/batch_service.py
====================
Batch Whole-Chest-X-Ray Report Generation Service for InteractMed-VLM.

Guarantees & Constraints:
  - Operates on full chest radiographs independently (one image per request).
  - Absolutely NO dependency on TransVG-AGPT, region grounding, predicted bounding boxes, or crops.
  - Reuses local MedGemma 4B via Ollama with persistent model keep_alive ("30m").
  - Bounded concurrency (single inference worker) for Apple Silicon unified memory safety.
  - Profiles model load, image prep, prompt/vision eval, token eval, and tokens/sec.
  - Supports both Patient-Friendly and Structured Research-Style report styles.
  - Independent per-image status: queued, processing, completed, failed, cancelled.
  - Granular retry of failed jobs without re-running successful jobs.
  - In-flight cancellation of remaining queued jobs.
  - Export to CSV and JSON formats.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import textwrap
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.report_service import (
    OLLAMA_BASE_URL,
    _ollama_available,
    _vision_model_available,
    _load_to_rgb_array,
    _array_to_base64_jpeg,
    clean_and_structure_patient_report,
    clean_and_structure_research_report,
    generate_whole_image_report,
    PROMPT_VERSION_RESEARCH,
    PROMPT_VERSION_PATIENT,
    _UNSUPPORTED_MEASUREMENTS,
)
from src.triage_service import get_triage_service


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class BatchItem:
    job_id: str
    batch_id: str
    filename: str
    style: str
    status: str = "queued"          # queued | processing | completed | failed | cancelled
    image_sha256: Optional[str] = None
    bytes_sha256: str = ""
    pixel_sha256: str = ""
    orig_dims: Optional[List[int]] = None
    prep_dims: Optional[List[int]] = None
    prompt_version: str = ""
    preview_b64: Optional[str] = None
    report_text: Optional[str] = None
    sections: Optional[Dict[str, str]] = None
    cardiac_observation: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    latency_ms: Optional[float] = None
    timing: Optional[Dict[str, float]] = None
    eval_count: Optional[int] = None
    tokens_per_second: Optional[float] = None
    model_used: Optional[str] = None
    validation_warnings: List[str] = field(default_factory=list)
    validation_flagged: bool = False
    triage_result: Optional[Dict[str, Any]] = None
    report_validity: str = "completed_valid"
    review_required: bool = False
    classifier_alerts: List[str] = field(default_factory=list)
    classifier_scores: Dict[str, float] = field(default_factory=dict)
    consistency_result: Optional[Dict[str, Any]] = None
    structured_findings: List[Dict[str, Any]] = field(default_factory=list)
    completeness_flags: Dict[str, Any] = field(default_factory=dict)

    # Stored image bytes for potential retry and boundary verification
    _image_bytes: bytes = field(default=b"", repr=False)

    def __post_init__(self):
        if not self.bytes_sha256 and self.image_sha256:
            self.bytes_sha256 = self.image_sha256
        elif self.bytes_sha256 and not self.image_sha256:
            self.image_sha256 = self.bytes_sha256[:16]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("_image_bytes", None)
        return d


@dataclass
class BatchJob:
    batch_id: str
    style: str
    created_at: float = field(default_factory=time.time)
    status: str = "queued"          # queued | processing | completed | failed | cancelled
    cancelled: bool = False
    items: List[BatchItem] = field(default_factory=list)

    @property
    def total_items(self) -> int:
        return len(self.items)

    @property
    def completed_count(self) -> int:
        return sum(1 for it in self.items if it.status == "completed")

    @property
    def failed_count(self) -> int:
        return sum(1 for it in self.items if it.status == "failed")

    @property
    def processing_count(self) -> int:
        return sum(1 for it in self.items if it.status == "processing")

    @property
    def queued_count(self) -> int:
        return sum(1 for it in self.items if it.status == "queued")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "style": self.style,
            "created_at": self.created_at,
            "status": self.status,
            "cancelled": self.cancelled,
            "total_items": self.total_items,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "processing_count": self.processing_count,
            "queued_count": self.queued_count,
            "items": [it.to_dict() for it in self.items],
        }


# ── Batch Service Implementation ─────────────────────────────────────────────

class BatchService:
    def __init__(self, max_workers: int = 1):
        self._batches: Dict[str, BatchJob] = {}
        self._lock = threading.Lock()
        # Single inference worker preserves stability on 16GB Mac
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="BatchWorker")

    def create_batch(
        self,
        files_data: List[Tuple[str, bytes]],
        style: str = "patient",
    ) -> BatchJob:
        """
        Create a new batch, prepare preview thumbnails, and submit items to queue.
        Validates maximum batch size <= 10.
        """
        if len(files_data) > 10:
            raise ValueError(f"Maximum batch size is 10 images. Received {len(files_data)}.")

        if not files_data:
            raise ValueError("No images provided for batch processing.")

        batch_id = f"batch_{uuid.uuid4().hex[:8]}"
        batch = BatchJob(batch_id=batch_id, style=style)

        for filename, data in files_data:
            job_id = f"job_{uuid.uuid4().hex[:8]}"

            # Generate lightweight preview thumbnail & canonical pixel hash & dims
            preview_b64 = None
            orig_dims = None
            prep_dims = None
            pixel_sha256 = ""
            bytes_sha256 = hashlib.sha256(data).hexdigest()
            prompt_version = (
                PROMPT_VERSION_RESEARCH if style == "research" else PROMPT_VERSION_PATIENT
            )

            try:
                arr = _load_to_rgb_array(data, filename)
                if arr is not None and arr.size > 0:
                    orig_dims = [int(arr.shape[0]), int(arr.shape[1])]
                    pixel_sha256 = hashlib.sha256(arr.tobytes()).hexdigest()
                    preview_b64 = _array_to_base64_jpeg(arr, max_side=256)
                    h, w = arr.shape[:2]
                    if max(h, w) > 1024:
                        scale = 1024.0 / max(h, w)
                        prep_dims = [int(round(h * scale)), int(round(w * scale))]
                    else:
                        prep_dims = [int(h), int(w)]
            except Exception:
                # If preview fails at this stage, it will fail during processing with full error
                pass

            item = BatchItem(
                job_id=job_id,
                batch_id=batch_id,
                filename=filename,
                style=style,
                status="queued",
                image_sha256=bytes_sha256[:16],
                bytes_sha256=bytes_sha256,
                pixel_sha256=pixel_sha256,
                orig_dims=orig_dims,
                prep_dims=prep_dims,
                prompt_version=prompt_version,
                preview_b64=preview_b64,
                _image_bytes=data,
            )
            batch.items.append(item)

        with self._lock:
            self._batches[batch_id] = batch

        # Submit jobs to worker pool
        for item in batch.items:
            self._executor.submit(self._process_single_item, item)

        return batch

    def get_batch(self, batch_id: str) -> Optional[BatchJob]:
        with self._lock:
            return self._batches.get(batch_id)

    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel remaining queued items in the batch."""
        with self._lock:
            batch = self._batches.get(batch_id)
            if not batch:
                return False
            batch.cancelled = True
            for it in batch.items:
                if it.status == "queued":
                    it.status = "cancelled"
                    it.error = "Cancelled by user"
                    it.completed_at = time.time()
            self._update_batch_status(batch)
            return True

    def retry_jobs(self, batch_id: str, job_ids: Optional[List[str]] = None) -> int:
        """
        Retry failed or cancelled jobs in a batch without re-running completed ones.
        """
        to_retry: List[BatchItem] = []
        with self._lock:
            batch = self._batches.get(batch_id)
            if not batch:
                return 0
            # Reset batch-level cancelled flag if retrying
            batch.cancelled = False
            batch.status = "processing"

            target_ids = set(job_ids) if job_ids else None
            for it in batch.items:
                if it.status in ("failed", "cancelled"):
                    if target_ids is None or it.job_id in target_ids:
                        it.status = "queued"
                        it.error = None
                        it.report_text = None
                        it.sections = None
                        it.started_at = None
                        it.completed_at = None
                        it.latency_ms = None
                        it.timing = None
                        to_retry.append(it)

        for it in to_retry:
            self._executor.submit(self._process_single_item, it)

        return len(to_retry)

    def export_summary(self, batch_id: str, format: str = "json") -> str:
        """Export batch reports as CSV or JSON with per-job identity audit."""
        batch = self.get_batch(batch_id)
        if not batch:
            raise KeyError(f"Batch {batch_id} not found.")

        if format.lower() == "csv":
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([
                "Job ID", "Filename", "Style", "Status", "Image SHA256", "Bytes SHA256", "Pixel SHA256",
                "Orig Dims", "Prep Dims", "Prompt Version", "Model",
                "Triage Category", "Matched Rule ID", "Review Priority",
                "Latency (ms)", "Tokens/sec", "Error", "Report Text",
                "Report Validity", "Review Required", "Classifier Alerts", "Triage State",
                "Canonical Status", "Canonical Status Label"
            ])
            for it in batch.items:
                t_cat = (it.triage_result or {}).get("triage_category", "")
                t_rule = (it.triage_result or {}).get("matched_rule_id", "")
                t_prio = (it.triage_result or {}).get("human_review_priority", "")
                t_state = (it.triage_result or {}).get("triage_state", "")
                c_status = (it.triage_result or {}).get("status", "")
                c_label = (it.triage_result or {}).get("status_label", "")
                orig_d = f"{it.orig_dims[0]}x{it.orig_dims[1]}" if it.orig_dims else ""
                prep_d = f"{it.prep_dims[0]}x{it.prep_dims[1]}" if it.prep_dims else ""
                cls_str = "; ".join(it.classifier_alerts) if it.classifier_alerts else "none"
                writer.writerow([
                    it.job_id,
                    it.filename,
                    it.style,
                    it.status,
                    it.image_sha256 or "",
                    it.bytes_sha256 or "",
                    it.pixel_sha256 or "",
                    orig_d,
                    prep_d,
                    it.prompt_version or "",
                    it.model_used or "",
                    t_cat,
                    t_rule,
                    t_prio,
                    f"{it.latency_ms:.1f}" if it.latency_ms else "",
                    f"{it.tokens_per_second:.1f}" if it.tokens_per_second else "",
                    it.error or "",
                    it.report_text or "",
                    it.report_validity,
                    str(it.review_required),
                    cls_str,
                    t_state,
                    c_status,
                    c_label,
                ])
            return output.getvalue()
        else:
            return json.dumps(batch.to_dict(), indent=2)

    export_batch = export_summary

    # ── Worker Processing ────────────────────────────────────────────────────

    def _process_single_item(self, item: BatchItem) -> None:
        """Process a single image item through unified whole-image report generator."""
        batch = self.get_batch(item.batch_id)
        if not batch or batch.cancelled or item.status == "cancelled":
            with self._lock:
                item.status = "cancelled"
                item.error = "Cancelled by user"
                item.completed_at = time.time()
                if batch:
                    self._update_batch_status(batch)
            return

        with self._lock:
            item.status = "processing"
            item.started_at = time.time()
            self._update_batch_status(batch)

        t0 = time.perf_counter()

        # 1. Image verification at inference boundary
        if not item._image_bytes or len(item._image_bytes) < 100:
            err = f"Inference boundary failure ({item.filename}): Image file is empty or missing data."
            with self._lock:
                item.status = "failed"
                item.error = err
                item.completed_at = time.time()
                item.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                item.report_validity = "report_generation_failed"
                item.review_required = True
                item.triage_result = get_triage_service().evaluate_multi_evidence(
                    model_error=item.error, is_valid_image=False, report_validity="report_generation_failed"
                ).to_dict()
                self._update_batch_status(batch)
            return

        actual_bytes_hash = hashlib.sha256(item._image_bytes).hexdigest()
        if item.bytes_sha256 and actual_bytes_hash != item.bytes_sha256:
            err = f"Inference boundary identity mismatch ({item.filename}): expected {item.bytes_sha256[:16]}, got {actual_bytes_hash[:16]}"
            with self._lock:
                item.status = "failed"
                item.error = err
                item.completed_at = time.time()
                item.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                item.report_validity = "report_generation_failed"
                item.review_required = True
                item.triage_result = get_triage_service().evaluate_multi_evidence(
                    model_error=item.error, is_valid_image=False, report_validity="report_generation_failed"
                ).to_dict()
                self._update_batch_status(batch)
            return

        # 2. Invoke unified whole-image report generator
        try:
            rep = generate_whole_image_report(
                image_bytes=item._image_bytes,
                image_filename=item.filename,
                style=item.style,
            )
        except Exception as exc:
            err = f"Model generation error ({item.filename}): {exc}"
            with self._lock:
                item.status = "failed"
                item.error = err
                item.completed_at = time.time()
                item.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                item.report_validity = "report_generation_failed"
                item.review_required = True
                item.triage_result = get_triage_service().evaluate_multi_evidence(
                    model_error=item.error, is_valid_image=True, report_validity="report_generation_failed"
                ).to_dict()
                self._update_batch_status(batch)
            return

        # 3. Handle generator failure
        if not rep.get("success", False):
            with self._lock:
                item.status = "failed"
                item.error = rep.get("error", "Unknown report generation failure")
                item.completed_at = time.time()
                item.latency_ms = rep.get("latency_ms", round((time.perf_counter() - t0) * 1000, 2))
                item.report_validity = "report_generation_failed"
                item.review_required = True
                item.triage_result = rep.get("triage_result") or get_triage_service().evaluate_multi_evidence(
                    model_error=item.error, is_valid_image=True, report_validity="report_generation_failed"
                ).to_dict()
                self._update_batch_status(batch)
            return

        # 4. Auxiliary Pretrained Classifier & Cross-Report Consistency Evaluation
        cls_res = None
        cls_scores: Dict[str, float] = {}
        cls_alerts: List[str] = []
        try:
            from src.classifier_service import get_classifier_service
            cls_svc = get_classifier_service()
            if cls_svc.ensure_loaded():
                cls_res = cls_svc.predict(item._image_bytes, threshold=0.50)
                cls_scores = cls_res.mapped_findings
                cls_alerts = [
                    f"{k} (score: {round(v, 2)} >= 0.62)"
                    for k, v in cls_scores.items() if v >= 0.62
                ]
        except Exception:
            # Fallback gracefully if classifier fails, treating as unavailable evidence
            pass

        cons_res = None
        cons_alerts: List[str] = []
        cons_review_required = False
        try:
            from src.consistency_service import evaluate_cross_report_consistency
            target_queries = ["pneumothorax", "pleural_effusion", "pneumonia_consolidation", "pulmonary_edema", "cardiomegaly"]
            for q in target_queries:
                c_eval = evaluate_cross_report_consistency(
                    query=q,
                    whole_image_report_text=rep.get("report_text"),
                    whole_image_sections=rep.get("sections"),
                    classifier_result=cls_res,
                    classifier_threshold=0.62,
                )
                if c_eval.has_conflict or c_eval.classifier_discordance:
                    cons_review_required = True
                    cons_alerts.extend(c_eval.classifier_alerts)
                    cons_res = c_eval
                    break
        except Exception:
            pass

        # 5. Multi-evidence triage evaluation
        multi_triage = get_triage_service().evaluate_multi_evidence(
            report_text=rep.get("report_text"),
            sections=rep.get("sections"),
            classifier_result=cls_res,
            consistency_result=cons_res,
            report_validity=rep.get("report_validity", "completed_valid"),
            is_valid_image=True,
        )

        with self._lock:
            item.status = "completed"
            item.report_text = rep.get("report_text")
            item.sections = rep.get("sections")
            item.cardiac_observation = rep.get("cardiac_observation")
            item.triage_result = multi_triage.to_dict()
            item.report_validity = rep.get("report_validity", "completed_valid")
            item.review_required = bool(
                rep.get("review_required", False)
                or cons_review_required
                or multi_triage.requires_human_review
                or (multi_triage.human_review_priority in ("urgent", "priority_review"))
            )
            item.classifier_scores = cls_scores
            item.classifier_alerts = cls_alerts
            item.consistency_result = cons_res.to_dict() if cons_res else None
            item.completed_at = time.time()
            item.latency_ms = rep.get("latency_ms")
            item.timing = rep.get("timing")
            item.eval_count = rep.get("eval_count")
            item.tokens_per_second = rep.get("tokens_per_second")
            item.model_used = rep.get("model_used")
            all_warns = list(rep.get("validation_warnings", [])) + cons_alerts
            item.validation_warnings = list(dict.fromkeys(all_warns))
            item.structured_findings = rep.get("structured_findings") or []
            item.completeness_flags = rep.get("completeness_flags") or {}
            item.bytes_sha256 = rep.get("bytes_sha256", item.bytes_sha256)
            item.pixel_sha256 = rep.get("pixel_sha256", item.pixel_sha256)
            item.orig_dims = rep.get("orig_dims", item.orig_dims)
            item.prep_dims = rep.get("prep_dims", item.prep_dims)
            item.prompt_version = rep.get("prompt_version", item.prompt_version)
            self._update_batch_status(batch)

    def _update_batch_status(self, batch: BatchJob) -> None:
        """Update aggregate batch status based on item statuses."""
        if batch.cancelled:
            batch.status = "cancelled"
            return
        statuses = [it.status for it in batch.items]
        if all(s == "completed" for s in statuses):
            batch.status = "completed"
        elif all(s in ("completed", "failed", "cancelled") for s in statuses):
            batch.status = "completed" if any(s == "completed" for s in statuses) else "failed"
        elif any(s == "processing" for s in statuses):
            batch.status = "processing"
        else:
            batch.status = "queued"


# Global singleton instance
_BATCH_SERVICE: Optional[BatchService] = None

def get_batch_service() -> BatchService:
    global _BATCH_SERVICE
    if _BATCH_SERVICE is None:
        _BATCH_SERVICE = BatchService(max_workers=1)
    return _BATCH_SERVICE
