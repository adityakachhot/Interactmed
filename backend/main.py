"""
backend/main.py
===============
InteractMed-VLM FastAPI backend.

Endpoints:
  GET  /health               — Service health, model status, vision model availability
  GET  /vision-model-status  — Whether a vision-capable model is installed in Ollama
  POST /ground               — Phrase grounding → predicted bounding box + overlay + crop
  POST /report/region        — Region-specific report (passes actual image to vision model)
  POST /report/study         — Study-level report from clinical indication + context

Model lifecycle:
  - Grounding model (TransVG-AGPT) loaded ONCE at startup via lifespan.
  - Report generation uses Ollama (ollama serve) separately.
  - Baseline model selectable via checkpoint form field.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, PlainTextResponse
from pydantic import BaseModel, Field

# ── Path setup ────────────────────────────────────────────────────────────────
BACKEND_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.grounding_service import get_grounding_service, GroundingService
from src.report_service import (
    region_report, patient_friendly_report, study_report,
    _ollama_available, _vision_model_available, _text_model_available,
    extract_crop, _array_to_base64_jpeg, _load_to_rgb_array,
)
from src.overlay_service import draw_overlay, image_to_base64_jpeg
from src.batch_service import get_batch_service

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
ALLOWED_SUFFIXES = {".dcm", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Pre-loading fine-tuned grounding model …")
    svc = get_grounding_service("finetuned")
    try:
        svc.ensure_loaded()
        print("[Startup] Fine-tuned model ready.")
    except Exception as exc:
        print(f"[Startup WARNING] Fine-tuned model failed to load: {exc}")
        try:
            get_grounding_service("baseline").ensure_loaded()
            print("[Startup] Baseline model loaded as fallback.")
        except Exception as exc2:
            print(f"[Startup WARNING] Baseline model also failed: {exc2}")
    yield
    print("[Shutdown] Grounding service stopped.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="InteractMed-VLM API",
    description="Phrase grounding and report generation for chest radiographs.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Response models ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    grounding_model_loaded: bool
    grounding_model_device: Optional[str]
    grounding_model_error: Optional[str]
    ollama_available: bool
    vision_model: Optional[str]        # name of vision model if installed
    text_model: Optional[str]          # name of best text model if installed
    timestamp: str


class VisionModelStatusResponse(BaseModel):
    ollama_available: bool
    vision_model: Optional[str]        # e.g. "llava:7b" or None
    text_model: Optional[str]          # e.g. "llama3:latest" or None
    vision_model_ready: bool
    install_command: Optional[str]     # shown when vision model is missing


class GroundingResponse(BaseModel):
    success: bool
    query: str
    checkpoint: str
    pred_box_norm: Optional[List[float]]   = Field(None, description="[cx,cy,w,h] in [0,1]")
    pred_box_pixel: Optional[List[float]]  = Field(None, description="[x1,y1,x2,y2] original pixels")
    pred_box_canvas: Optional[List[float]] = Field(None, description="[x1,y1,x2,y2] 640×640 canvas")
    orig_h: Optional[int]
    orig_w: Optional[int]
    latency_ms: float
    overlay_image_b64: Optional[str]       = Field(None, description="Base64 JPEG annotated image")
    crop_image_b64: Optional[str]          = Field(None, description="Base64 JPEG of predicted region crop")
    error: Optional[str]


class RegionReportResponse(BaseModel):
    success: bool
    report_text: Optional[str]
    crop_info: Optional[dict]
    latency_ms: float
    timing: Optional[dict] = None
    model_used: Optional[str]
    image_input_used: bool = False
    n_images_sent: int = 0
    validation_warnings: List[str] = []
    validation_flagged: bool = False
    error: Optional[str]


class PatientFriendlyReportResponse(BaseModel):
    success: bool
    report_text: Optional[str]
    sections: Optional[dict] = None
    triage_result: Optional[dict] = None
    latency_ms: float
    timing: Optional[dict] = None
    model_used: Optional[str] = None
    image_input_used: bool = False
    validation_warnings: List[str] = []
    validation_flagged: bool = False
    error: Optional[str] = None


StudyReportResponse = PatientFriendlyReportResponse


class BatchItemResponse(BaseModel):
    job_id: str
    batch_id: str
    filename: str
    style: str
    status: str
    preview_b64: Optional[str] = None
    report_text: Optional[str] = None
    sections: Optional[dict] = None
    triage_result: Optional[dict] = None
    error: Optional[str] = None
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    latency_ms: Optional[float] = None
    timing: Optional[dict] = None
    eval_count: Optional[int] = None
    tokens_per_second: Optional[float] = None
    model_used: Optional[str] = None
    validation_warnings: List[str] = []
    validation_flagged: bool = False


class BatchStatusResponse(BaseModel):
    batch_id: str
    style: str
    created_at: float
    status: str
    cancelled: bool
    total_items: int
    completed_count: int
    failed_count: int
    processing_count: int
    queued_count: int
    items: List[BatchItemResponse]


class BatchRetryRequest(BaseModel):
    job_ids: Optional[List[str]] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_upload(file: UploadFile) -> None:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{suffix}'. Accepted: {sorted(ALLOWED_SUFFIXES)}",
        )


async def _read_upload(file: UploadFile) -> bytes:
    _validate_upload(file)
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit.",
        )
    return data


def _check_grounding_service(checkpoint: str) -> GroundingService:
    svc = get_grounding_service(checkpoint)
    if not svc._loaded and svc._load_error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Grounding model failed to load: {svc._load_error}",
        )
    return svc


def _get_image_dims(image_bytes: bytes, filename: str) -> tuple[int, int]:
    """Return (orig_h, orig_w) or (0, 0) on failure."""
    try:
        suffix = Path(filename).suffix.lower()
        if suffix == ".dcm":
            import io
            import pydicom
            ds = pydicom.dcmread(io.BytesIO(image_bytes))
            h, w = ds.pixel_array.shape[:2]
        else:
            import io
            from PIL import Image as PILImage
            img = PILImage.open(io.BytesIO(image_bytes))
            w, h = img.size
        return int(h), int(w)
    except Exception:
        return 0, 0


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Return service health and model status."""
    svc = get_grounding_service("finetuned")
    vm = _vision_model_available() if _ollama_available() else None
    tm = _text_model_available()   if _ollama_available() else None
    return HealthResponse(
        status="ok",
        grounding_model_loaded=svc._loaded,
        grounding_model_device=svc.status.get("device"),
        grounding_model_error=svc.status.get("error"),
        ollama_available=_ollama_available(),
        vision_model=vm,
        text_model=tm,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


@app.get("/vision-model-status", response_model=VisionModelStatusResponse)
async def vision_model_status():
    """Check whether a vision-capable model is available in Ollama."""
    available = _ollama_available()
    vm = _vision_model_available() if available else None
    tm = _text_model_available()   if available else None
    return VisionModelStatusResponse(
        ollama_available=available,
        vision_model=vm,
        text_model=tm,
        vision_model_ready=(vm is not None),
        install_command="ollama pull medgemma:4b" if vm is None else None,
    )


@app.post("/ground", response_model=GroundingResponse)
async def ground(
    file: Annotated[UploadFile, File(description="Chest radiograph (DICOM, PNG, JPEG, TIFF)")],
    query: Annotated[str, Form(description="Clinical phrase to ground")] = "pneumonia",
    checkpoint: Annotated[str, Form(description="'finetuned' or 'baseline'")] = "finetuned",
    return_overlay: Annotated[bool, Form(description="Include base64 annotated image")] = True,
    return_crop: Annotated[bool, Form(description="Include base64 region crop")] = True,
):
    """
    Ground a clinical phrase. Returns predicted box, overlay, and region crop.
    Crop uses the SAME pixel coordinates as the overlay renderer.
    """
    query = query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty.")
    if checkpoint not in ("finetuned", "baseline"):
        raise HTTPException(status_code=400, detail="checkpoint must be 'finetuned' or 'baseline'.")

    image_bytes = await _read_upload(file)
    svc = _check_grounding_service(checkpoint)
    result = svc.predict(image_bytes, query, image_filename=file.filename or "upload.png")

    overlay_b64 = None
    crop_b64 = None

    if result["success"]:
        box = result["pred_box_pixel"]
        orig_h = result["orig_h"]
        orig_w = result["orig_w"]

        if return_overlay:
            try:
                img_annotated = draw_overlay(
                    image_bytes=image_bytes,
                    image_filename=file.filename or "upload.png",
                    pred_box_pixel_xyxy=box,
                    query=query,
                    model_label=f"TransVG-AGPT ({checkpoint})",
                )
                overlay_b64 = image_to_base64_jpeg(img_annotated)
            except Exception:
                overlay_b64 = None

        if return_crop and box:
            try:
                crop_arr = extract_crop(
                    image_bytes=image_bytes,
                    image_filename=file.filename or "upload.png",
                    box_pixel_xyxy=box,
                    margin_px=40,
                )
                if crop_arr is not None:
                    crop_b64 = _array_to_base64_jpeg(crop_arr, max_side=512)
            except Exception:
                crop_b64 = None

    return GroundingResponse(
        success=result["success"],
        query=query,
        checkpoint=result["checkpoint"],
        pred_box_norm=result["pred_box_norm"],
        pred_box_pixel=result["pred_box_pixel"],
        pred_box_canvas=result["pred_box_canvas"],
        orig_h=result["orig_h"],
        orig_w=result["orig_w"],
        latency_ms=result["latency_ms"],
        overlay_image_b64=overlay_b64,
        crop_image_b64=crop_b64,
        error=result["error"],
    )


@app.post("/report/region", response_model=RegionReportResponse)
async def report_region(
    file: Annotated[UploadFile, File(description="Chest radiograph")],
    query: Annotated[str, Form(description="Clinical phrase that was grounded")] = "pneumonia",
    box_pixel_xyxy: Annotated[str, Form(description="JSON [x1,y1,x2,y2] original pixel coords")] = "[]",
    orig_h: Annotated[int, Form()] = 0,
    orig_w: Annotated[int, Form()] = 0,
    margin_px: Annotated[int, Form()] = 40,
):
    """
    Generate a region-specific radiology report draft.

    Requires a vision-capable model (llava:7b) to be installed in Ollama.
    Passes the actual crop image and full image to the model.
    Returns success=False with a clear error if no vision model is available.
    """
    try:
        box = json.loads(box_pixel_xyxy)
        if len(box) != 4:
            raise ValueError("Expected 4 elements")
        box = [float(v) for v in box]
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="box_pixel_xyxy must be a JSON array of 4 floats [x1,y1,x2,y2].",
        )

    if orig_h <= 0 or orig_w <= 0:
        raise HTTPException(status_code=400, detail="orig_h and orig_w must be positive integers.")

    image_bytes = await _read_upload(file)
    result = region_report(
        image_bytes=image_bytes,
        image_filename=file.filename or "upload.png",
        phrase=query.strip(),
        box_pixel_xyxy=box,
        orig_h=orig_h,
        orig_w=orig_w,
        margin_px=margin_px,
    )
    filtered = {k: v for k, v in result.items() if k in RegionReportResponse.model_fields}
    return RegionReportResponse(**filtered)


@app.post("/report/patient-friendly", response_model=PatientFriendlyReportResponse)
@app.post("/report/study", response_model=PatientFriendlyReportResponse)
async def report_patient_friendly_endpoint(
    file: Annotated[UploadFile, File(description="Chest radiograph")],
    reason_for_xray: Annotated[Optional[str], Form(description="Reason for X-ray (optional)")] = None,
    clinical_indication: Annotated[Optional[str], Form(description="Legacy alias for reason_for_xray")] = None,
    relevant_history: Annotated[Optional[str], Form(description="Relevant medical history (optional)")] = None,
    prior_report: Annotated[Optional[str], Form()] = None,
    grounding_json: Annotated[Optional[str], Form()] = None,
):
    """
    Generate an accessible, patient-friendly explanation of a chest radiograph.
    Passes the full image to MedGemma 4B or best available vision model.
    """
    image_bytes = await _read_upload(file)
    filename = file.filename or "upload.png"

    reason = reason_for_xray or clinical_indication or ""

    result = patient_friendly_report(
        image_bytes=image_bytes,
        image_filename=filename,
        reason_for_xray=reason.strip() if reason else None,
        relevant_history=relevant_history.strip() if relevant_history else None,
    )
    filtered = {k: v for k, v in result.items() if k in PatientFriendlyReportResponse.model_fields}
    return PatientFriendlyReportResponse(**filtered)


# ── Batch Report Endpoints ───────────────────────────────────────────────────

@app.post("/batch/create", response_model=BatchStatusResponse)
async def create_batch_endpoint(
    files: Annotated[List[UploadFile], File(description="Up to 10 chest X-ray images")],
    style: Annotated[str, Form(description="Report style: patient or research")] = "patient",
):
    """
    Process up to 10 chest X-rays in a batch independently using MedGemma 4B.
    Does NOT depend on TransVG-AGPT or region grounding.
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum batch size is 10 images. Received {len(files)}.",
        )
    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required.")

    valid_styles = {"patient", "research"}
    if style not in valid_styles:
        style = "patient"

    files_data: list[tuple[str, bytes]] = []
    for f in files:
        data = await _read_upload(f)
        files_data.append((f.filename or "image.png", data))

    batch_svc = get_batch_service()
    batch = batch_svc.create_batch(files_data, style=style)
    return BatchStatusResponse(**batch.to_dict())


@app.get("/batch/{batch_id}", response_model=BatchStatusResponse)
async def get_batch_status_endpoint(batch_id: str):
    """Return status and reports for all jobs in a batch."""
    batch_svc = get_batch_service()
    batch = batch_svc.get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found.")
    return BatchStatusResponse(**batch.to_dict())


@app.post("/batch/{batch_id}/cancel", response_model=BatchStatusResponse)
async def cancel_batch_endpoint(batch_id: str):
    """Cancel remaining queued items in this batch."""
    batch_svc = get_batch_service()
    ok = batch_svc.cancel_batch(batch_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found.")
    batch = batch_svc.get_batch(batch_id)
    return BatchStatusResponse(**batch.to_dict())


@app.post("/batch/{batch_id}/retry", response_model=BatchStatusResponse)
async def retry_batch_endpoint(batch_id: str, req: Optional[BatchRetryRequest] = None):
    """Retry failed or cancelled jobs without re-running completed ones."""
    batch_svc = get_batch_service()
    job_ids = req.job_ids if req else None
    batch_svc.retry_jobs(batch_id, job_ids)
    batch = batch_svc.get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found.")
    return BatchStatusResponse(**batch.to_dict())


@app.get("/batch/{batch_id}/download")
async def download_batch_endpoint(batch_id: str, format: str = "json"):
    """Download batch summary as CSV or JSON."""
    batch_svc = get_batch_service()
    try:
        content = batch_svc.export_summary(batch_id, format=format)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found.")

    media_type = "text/csv" if format.lower() == "csv" else "application/json"
    filename = f"{batch_id}_summary.{format.lower()}"
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
