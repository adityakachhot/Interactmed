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
    _UNSUPPORTED_MEASUREMENTS,
)
from src.triage_service import get_triage_service

# ── Metrics Helper for Ollama ────────────────────────────────────────────────

def _ollama_generate_vision_with_metrics(
    prompt: str,
    images_b64: list[str],
    model: str,
    num_predict: int = 220,
    temperature: float = 0.1,
    keep_alive: str = "30m",
    timeout: int = 240,
) -> tuple[str, dict[str, Any]]:
    """
    Call Ollama /api/generate with multimodal image input and return:
      (response_text, raw_metrics_dict)
    Captures load_duration, prompt_eval_duration, eval_duration, eval_count.
    """
    import urllib.request
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "images": images_b64,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
        response_text = body.get("response", "").strip()
        metrics = {
            "total_duration_ns": body.get("total_duration", 0),
            "load_duration_ns": body.get("load_duration", 0),
            "prompt_eval_count": body.get("prompt_eval_count", 0),
            "prompt_eval_duration_ns": body.get("prompt_eval_duration", 0),
            "eval_count": body.get("eval_count", 0),
            "eval_duration_ns": body.get("eval_duration", 0),
            "done_reason": body.get("done_reason", "completed"),
        }
        return response_text, metrics


# ── Research Report Parser ───────────────────────────────────────────────────

def clean_and_structure_research_report(
    raw_text: str,
) -> tuple[str, dict[str, str], list[str]]:
    """
    Parses and structures the research-style report:
      1. OBSERVATIONS
      2. IMPRESSION (or ASSESSMENT AND UNCERTAINTY)
      3. LIMITATIONS
    Also maintains backwards-compatibility with 4-heading formats.
    """
    warnings: list[str] = []

    pattern = re.compile(
        r'(?:^|\n)\s*(?:\d+[\.\)]\s*)?(?:\*{1,3})?\s*(OBSERVATIONS|IMPRESSION|POSSIBLE FINDINGS|ASSESSMENT AND UNCERTAINTY|LIMITATIONS)\s*(?:\*{1,3})?\s*[:\n]?',
        re.IGNORECASE
    )
    matches = list(pattern.finditer(raw_text))

    sections: dict[str, str] = {}
    standard_headers = ["OBSERVATIONS", "IMPRESSION", "POSSIBLE FINDINGS", "ASSESSMENT AND UNCERTAINTY", "LIMITATIONS"]
    if matches:
        for i, m in enumerate(matches):
            header = m.group(1).strip()
            std_header = "OBSERVATIONS"
            for target in standard_headers:
                if target.lower() == header.lower():
                    std_header = target
                    break
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
            sections[std_header] = raw_text[start:end].strip()
    else:
        sections["OBSERVATIONS"] = raw_text.strip()
        sections["IMPRESSION"] = ""
        sections["LIMITATIONS"] = ""

    # Synchronize IMPRESSION and ASSESSMENT AND UNCERTAINTY
    if "IMPRESSION" in sections and not sections.get("ASSESSMENT AND UNCERTAINTY"):
        sections["ASSESSMENT AND UNCERTAINTY"] = sections["IMPRESSION"]
    elif "ASSESSMENT AND UNCERTAINTY" in sections and not sections.get("IMPRESSION"):
        sections["IMPRESSION"] = sections["ASSESSMENT AND UNCERTAINTY"]

    # Provide POSSIBLE FINDINGS if expected by legacy consumers
    if "POSSIBLE FINDINGS" not in sections:
        sections["POSSIBLE FINDINGS"] = sections.get("IMPRESSION", "")

    # Strip leaked disclaimers
    disclaimer_pat = re.compile(
        r'(\*{0,2}(?:experimental|not for clinical|ai-generated draft|educational purposes only|consult a doctor|consult your doctor|disclaimer)[^\n\.]*[\.\n]?)',
        re.IGNORECASE
    )
    for k in sections:
        sections[k] = disclaimer_pat.sub('', sections[k]).strip()

    # Correct factual misconception regarding frontal projection if present
    one_side_pat = re.compile(r'(?:it\s+)?only\s+shows\s+one\s+side(?:\s+of\s+the\s+chest|\s+at\s+a\s+time)?', re.IGNORECASE)
    for k in sections:
        if one_side_pat.search(sections[k]):
            sections[k] = one_side_pat.sub('is a single frontal view without lateral projection', sections[k])
            warnings.append("Corrected anatomical error: frontal view displays both hemithoraces but lacks lateral projection.")

    # Fallback content for empty sections (without biased normal assertions)
    if not sections.get("OBSERVATIONS"):
        sections["OBSERVATIONS"] = "Thoracic structures, lung parenchyma, and cardiac silhouette evaluated on frontal radiograph."
    if not sections.get("IMPRESSION"):
        sections["IMPRESSION"] = "Automated visual reading is indeterminate without prior comparison studies or lateral projection."
        sections["ASSESSMENT AND UNCERTAINTY"] = sections["IMPRESSION"]
    if not sections.get("LIMITATIONS"):
        sections["LIMITATIONS"] = "Single frontal projection; AI research draft; requires formal radiologist verification."

    # Detect unsupported measurements
    for k, v in sections.items():
        meas = _UNSUPPORTED_MEASUREMENTS.findall(v)
        if meas:
            warnings.append(
                f"Section '{k}' contains specific measurements ({', '.join(set(meas))}) not reliably derived from single AI draft."
            )

    # Format full text
    if "POSSIBLE FINDINGS" in sections and sections["POSSIBLE FINDINGS"] and sections["POSSIBLE FINDINGS"] != sections.get("IMPRESSION"):
        full_text = (
            f"**OBSERVATIONS**\n{sections['OBSERVATIONS']}\n\n"
            f"**POSSIBLE FINDINGS**\n{sections['POSSIBLE FINDINGS']}\n\n"
            f"**ASSESSMENT AND UNCERTAINTY**\n{sections['ASSESSMENT AND UNCERTAINTY']}\n\n"
            f"**LIMITATIONS**\n{sections['LIMITATIONS']}"
        )
    else:
        full_text = (
            f"**OBSERVATIONS**\n{sections['OBSERVATIONS']}\n\n"
            f"**IMPRESSION**\n{sections.get('IMPRESSION', sections.get('ASSESSMENT AND UNCERTAINTY', ''))}\n\n"
            f"**LIMITATIONS**\n{sections['LIMITATIONS']}"
        )

    words = full_text.split()
    if len(words) < 25:
        warnings.append("Research report is very brief.")
    elif len(words) > 260:
        warnings.append("Research report exceeds recommended concise target.")

    return full_text, sections, warnings


# ── Prompts ───────────────────────────────────────────────────────────────────

def _build_patient_prompt() -> str:
    return textwrap.dedent("""
        You are an AI assistant helping a patient understand their chest radiograph (X-ray) in plain, reassuring, everyday language.
        You are viewing the patient's full chest radiograph.

        Explain what is visible on this chest X-ray clearly and calmly.
        Target length: approximately 120 to 160 words.

        YOUR EXPLANATION MUST CONTAIN EXACTLY THESE FOUR HEADINGS:

        Your X-ray in simple words
        Summarize in plain, everyday words what is visible on the chest X-ray (lungs, heart, and chest wall).

        What the findings may mean
        Explain what visible findings could indicate in simple terms. If you must use a medical word, immediately explain what it means in plain English (for example, explain that cardiomegaly means an enlarged heart). Distinguish what is directly visible from what it might suggest.

        What is uncertain
        Explain clearly what remains uncertain due to technical factors (such as a single front view or AI limitations). Do not guess or force a diagnosis.

        What happens next
        Provide gentle guidance encouraging the patient to discuss this X-ray with the doctor or healthcare professional who ordered it. Do NOT prescribe medication or suggest treatments.

        STRICT RULES:
        - Use simple words and short, easy-to-read sentences.
        - Avoid technical abbreviations and medical jargon.
        - Never invent measurements, symptoms, prior tests, or abnormalities.
        - Never claim the X-ray is definitely normal if you cannot be certain.
        - Do not include bounding box coordinates, image dimensions, or technical stats.
        - Do not put a disclaimer in your text; a standard disclaimer will be displayed automatically.

        Begin directly with:
        Your X-ray in simple words
    """).strip()


def _build_research_prompt() -> str:
    return textwrap.dedent("""
        You are an expert AI radiologist generating a structured research draft of a chest radiograph.
        Analyze this full chest X-ray in detail.
        Target length: approximately 120 to 180 words.

        YOUR REPORT MUST CONTAIN EXACTLY THESE FOUR HEADINGS:

        OBSERVATIONS
        Describe visible radiographic findings across the lungs, cardiac silhouette, mediastinum, pleural spaces, and visible bony thorax.

        POSSIBLE FINDINGS
        State any identifiable radiographic abnormalities (e.g., airspace opacities, cardiomegaly, effusion, or pneumothorax) or state if no definitive focal abnormalities are observed.

        ASSESSMENT AND UNCERTAINTY
        Summarize the diagnostic impression with appropriate scientific caution. Clearly indicate degree of uncertainty and any potential differential considerations.

        LIMITATIONS
        State technical limitations (single frontal view, 2D projection, absence of lateral radiograph, patient positioning, AI experimental draft).

        STRICT RULES:
        - Concise, objective, professional medical language.
        - Never invent measurements, history, prior examinations, or diagnoses.
        - If features are borderline or indeterminate, explicitly declare uncertainty.
        - Do not include raw coordinates or image dimensions.

        Begin directly with:
        OBSERVATIONS
    """).strip()


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class BatchItem:
    job_id: str
    batch_id: str
    filename: str
    style: str
    status: str = "queued"          # queued | processing | completed | failed | cancelled
    preview_b64: Optional[str] = None
    report_text: Optional[str] = None
    sections: Optional[Dict[str, str]] = None
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

    # Stored image bytes for potential retry
    _image_bytes: bytes = field(default=b"", repr=False)

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

            # Generate lightweight preview thumbnail
            preview_b64 = None
            try:
                arr = _load_to_rgb_array(data, filename)
                preview_b64 = _array_to_base64_jpeg(arr, max_side=256)
            except Exception:
                # If preview fails at this stage, it will fail during processing with full error
                pass

            item = BatchItem(
                job_id=job_id,
                batch_id=batch_id,
                filename=filename,
                style=style,
                status="queued",
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
        """Export batch reports as CSV or JSON."""
        batch = self.get_batch(batch_id)
        if not batch:
            raise KeyError(f"Batch {batch_id} not found.")

        if format.lower() == "csv":
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([
                "Job ID", "Filename", "Style", "Status", "Model",
                "Triage Category", "Matched Rule ID", "Review Priority",
                "Latency (ms)", "Tokens/sec", "Error", "Report Text"
            ])
            for it in batch.items:
                t_cat = (it.triage_result or {}).get("triage_category", "")
                t_rule = (it.triage_result or {}).get("matched_rule_id", "")
                t_prio = (it.triage_result or {}).get("human_review_priority", "")
                writer.writerow([
                    it.job_id,
                    it.filename,
                    it.style,
                    it.status,
                    it.model_used or "",
                    t_cat,
                    t_rule,
                    t_prio,
                    f"{it.latency_ms:.1f}" if it.latency_ms else "",
                    f"{it.tokens_per_second:.1f}" if it.tokens_per_second else "",
                    it.error or "",
                    it.report_text or "",
                ])
            return output.getvalue()
        else:
            return json.dumps(batch.to_dict(), indent=2)

    # ── Worker Processing ────────────────────────────────────────────────────

    def _process_single_item(self, item: BatchItem) -> None:
        """Process a single image item through MedGemma 4B."""
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

        # 1. Image validation & decoding
        try:
            if not item._image_bytes or len(item._image_bytes) < 100:
                raise ValueError("Image file is empty or missing data.")
            arr = _load_to_rgb_array(item._image_bytes, item.filename)
            if arr is None or arr.size == 0:
                raise ValueError("Decoded image array is empty.")
            b64_img = _array_to_base64_jpeg(arr, max_side=1024, quality=85)
        except Exception as exc:
            with self._lock:
                item.status = "failed"
                item.error = f"Image decode failure ({item.filename}): {exc}"
                item.completed_at = time.time()
                item.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                item.triage_result = get_triage_service().evaluate_from_structured_findings({}, model_error=item.error, is_valid_image=False).to_dict()
                self._update_batch_status(batch)
            return

        # 2. Check Ollama & vision model
        if not _ollama_available():
            with self._lock:
                item.status = "failed"
                item.error = "Ollama is not reachable (localhost:11434). Start with: ollama serve"
                item.completed_at = time.time()
                item.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                item.triage_result = get_triage_service().evaluate_from_structured_findings({}, model_error=item.error).to_dict()
                self._update_batch_status(batch)
            return

        vision_model = _vision_model_available()
        if not vision_model:
            with self._lock:
                item.status = "failed"
                item.error = "No vision-capable model (medgemma:4b) available in Ollama."
                item.completed_at = time.time()
                item.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                item.triage_result = get_triage_service().evaluate_from_structured_findings({}, model_error=item.error).to_dict()
                self._update_batch_status(batch)
            return

        # 3. Model Generation
        t_prep = time.perf_counter()
        ollama_metrics: dict[str, Any] = {}
        try:
            if item.style == "research":
                # Validated 2-Stage Grounded Pipeline for Research Style (Prompt v3 Structured):
                # Stage 1: Categorical visual extraction with 4-status differentiation and explicit laterality
                v_prompt = (
                    "You are an expert radiologist conducting structured visual feature extraction on this chest radiograph.\n"
                    "For each category below, state: Observed (<details and laterality>) | Not identified | Indeterminate | Not assessable.\n\n"
                    "1. Support apparatus: Inspect for endotracheal tube, enteric tube, vascular catheter/line, chest tube, pacemaker, or surgical clips.\n"
                    "2. Cardiac silhouette: Normal, Enlarged (CTR > 0.5), or Indeterminate.\n"
                    "3. Lungs & opacities: Consolidation, focal opacity, atelectasis, congestion, or edema. Specify laterality (right, left, bilateral, or unspecified laterality).\n"
                    "4. Pleural spaces: Costophrenic angles sharp, blunted/effusion (specify laterality: right, left, bilateral, or unspecified laterality), or obscured.\n"
                    "5. Mediastinum: Normal contour, widened, or indeterminate.\n\n"
                    "RULES:\n"
                    "- State laterality (right/left/bilateral) only when visually supported; if uncertain, state 'unspecified laterality'.\n"
                    "- If equivocal or confidence is low, classify as Indeterminate; DO NOT default to normal.\n"
                    "- Concise bullet list, no conversational filler.\n"
                    "Begin directly with 1. Support apparatus:"
                )
                raw_obs, v_metrics = _ollama_generate_vision_with_metrics(
                    prompt=v_prompt,
                    images_b64=[b64_img],
                    model=vision_model,
                    num_predict=130,
                    temperature=0.1,
                    keep_alive="30m",
                )
                # Stage 2: Structured research synthesis (OBSERVATIONS, IMPRESSION, LIMITATIONS)
                s2_prompt = textwrap.dedent(f"""
                    Structured visual findings from chest radiograph:
                    \"\"\"{raw_obs}\"\"\"

                    Synthesize an objective research report with EXACTLY these three headings:

                    OBSERVATIONS
                    Detail findings across support apparatus, cardiac silhouette, lungs, pleural spaces, and mediastinum. Incorporate supported laterality (right/left/bilateral) or state unspecified laterality when unverified.

                    IMPRESSION
                    Concisely summarize key findings and diagnostic uncertainty. Every abnormality or device noted in OBSERVATIONS must be explicitly addressed here. Never declare normal or no acute cardiopulmonary disease if abnormalities or indeterminate features are present.

                    LIMITATIONS
                    State technical projection limitations (single frontal view, no lateral projection).

                    STRICT RULES:
                    - Never echo prompt instructions, rules, or headings other than the three requested.
                    - Never claim a frontal view shows only one side of the chest.
                    - Do not invent measurements or prior comparisons.

                    Begin directly with:
                    OBSERVATIONS
                """).strip()
                raw_text, s2_metrics = _ollama_generate_vision_with_metrics(
                    prompt=s2_prompt,
                    images_b64=[],
                    model=vision_model,
                    num_predict=180,
                    temperature=0.1,
                    keep_alive="30m",
                )
                ollama_metrics = {
                    "total_duration_ns": v_metrics.get("total_duration_ns", 0) + s2_metrics.get("total_duration_ns", 0),
                    "load_duration_ns": v_metrics.get("load_duration_ns", 0),
                    "prompt_eval_count": v_metrics.get("prompt_eval_count", 0) + s2_metrics.get("prompt_eval_count", 0),
                    "prompt_eval_duration_ns": v_metrics.get("prompt_eval_duration_ns", 0) + s2_metrics.get("prompt_eval_duration_ns", 0),
                    "eval_count": v_metrics.get("eval_count", 0) + s2_metrics.get("eval_count", 0),
                    "eval_duration_ns": v_metrics.get("eval_duration_ns", 0) + s2_metrics.get("eval_duration_ns", 0),
                    "done_reason": s2_metrics.get("done_reason", "completed"),
                }
            else:
                # 2-Stage Grounded Pipeline for Patient-Friendly Style:
                # Stage 1: Objective visual feature extraction directly from 1024px image
                v_prompt = (
                    "Look at this chest radiograph carefully. "
                    "Describe what visible abnormalities, densities, or opacities are present in the lungs, cardiac silhouette, and any lines, tubes, or devices. "
                    "Be objective and specific about whether the lungs appear clear or if opacities are present."
                )
                raw_obs, v_metrics = _ollama_generate_vision_with_metrics(
                    prompt=v_prompt,
                    images_b64=[b64_img],
                    model=vision_model,
                    num_predict=160,
                    temperature=0.1,
                    keep_alive="30m",
                )
                # Stage 2: Patient-friendly translation strictly reflecting extracted observations
                p_prompt = textwrap.dedent(f"""
                    A medical vision model examined this patient's frontal chest X-ray and reported these visual observations:
                    \"\"\"{raw_obs}\"\"\"

                    Translate these observed findings into an accessible, patient-friendly explanation.
                    Use EXACTLY these four headings:

                    Your X-ray in simple words
                    Describe only the visual features noted above in plain language. If opacities, lung shading, or medical devices are present, describe where they are. If visual findings are ambiguous or inconclusive, state: "The automated visual reading is inconclusive."

                    What the findings may mean
                    Explain what these observed findings could suggest in everyday words. Explain any medical terminology simply. Do not assume or assert that the scan is completely normal if opacities were noted.

                    What is uncertain
                    Explain what cannot be determined from this view. Clarify that this frontal view displays both the left and right sides of the chest, but because it is a single two-dimensional projection without a side (lateral) view, depth and subtle findings cannot be confirmed.

                    What happens next
                    Advise the patient to consult their ordering physician for qualified clinical evaluation and diagnosis.

                    STRICT RULES:
                    - NEVER say a frontal view shows only one side of the chest; a frontal view shows both sides of the chest.
                    - Do not invent normal findings or declare absence of infection unless clearly supported.
                    - Do not suggest specific medications or medical treatments.

                    Begin directly with:
                    Your X-ray in simple words
                """).strip()
                raw_text, p_metrics = _ollama_generate_vision_with_metrics(
                    prompt=p_prompt,
                    images_b64=[],
                    model=vision_model,
                    num_predict=260,
                    temperature=0.1,
                    keep_alive="30m",
                )
                ollama_metrics = {
                    "total_duration_ns": v_metrics.get("total_duration_ns", 0) + p_metrics.get("total_duration_ns", 0),
                    "load_duration_ns": v_metrics.get("load_duration_ns", 0),
                    "prompt_eval_count": v_metrics.get("prompt_eval_count", 0) + p_metrics.get("prompt_eval_count", 0),
                    "prompt_eval_duration_ns": v_metrics.get("prompt_eval_duration_ns", 0) + p_metrics.get("prompt_eval_duration_ns", 0),
                    "eval_count": v_metrics.get("eval_count", 0) + p_metrics.get("eval_count", 0),
                    "eval_duration_ns": v_metrics.get("eval_duration_ns", 0) + p_metrics.get("eval_duration_ns", 0),
                    "done_reason": p_metrics.get("done_reason", "completed"),
                }
        except Exception as exc:
            with self._lock:
                item.status = "failed"
                item.error = f"Model generation error: {exc}"
                item.completed_at = time.time()
                item.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                item.triage_result = get_triage_service().evaluate_from_structured_findings({}, model_error=item.error).to_dict()
                self._update_batch_status(batch)
            return

        t_infer = time.perf_counter()

        if not raw_text:
            with self._lock:
                item.status = "failed"
                item.error = "Model returned empty response."
                item.completed_at = time.time()
                item.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                item.triage_result = get_triage_service().evaluate_from_structured_findings({}, model_error=item.error).to_dict()
                self._update_batch_status(batch)
            return

        # 4. Clean & structure
        if item.style == "research":
            full_text, sections, warnings = clean_and_structure_research_report(raw_text)
        else:
            full_text, sections, warnings = clean_and_structure_patient_report(raw_text)

        # 5. Triage classification
        triage_eval = get_triage_service().evaluate_from_report_text(full_text, sections)

        t_post = time.perf_counter()
        latency_ms = round((t_post - t0) * 1000, 2)

        # Calculate timing metrics
        eval_count = ollama_metrics.get("eval_count", 0)
        eval_dur_s = ollama_metrics.get("eval_duration_ns", 0) / 1e9
        tokens_per_sec = round(eval_count / eval_dur_s, 2) if eval_dur_s > 0 else None

        timing = {
            "prep_ms": round((t_prep - t0) * 1000, 2),
            "inference_ms": round((t_infer - t_prep) * 1000, 2),
            "post_ms": round((t_post - t_infer) * 1000, 2),
            "load_duration_ms": round(ollama_metrics.get("load_duration_ns", 0) / 1e6, 2),
            "prompt_eval_ms": round(ollama_metrics.get("prompt_eval_duration_ns", 0) / 1e6, 2),
            "eval_duration_ms": round(ollama_metrics.get("eval_duration_ns", 0) / 1e6, 2),
            "total_ms": latency_ms,
        }

        # 6. Update item status
        with self._lock:
            item.status = "completed"
            item.report_text = full_text
            item.sections = sections
            item.triage_result = triage_eval.to_dict()
            item.completed_at = time.time()
            item.latency_ms = latency_ms
            item.timing = timing
            item.eval_count = eval_count
            item.tokens_per_second = tokens_per_sec
            item.model_used = vision_model
            item.validation_warnings = warnings
            item.validation_flagged = any("measurement" in w or "brief" in w for w in warnings)
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
