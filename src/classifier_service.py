"""
src/classifier_service.py
=========================
Auxiliary Chest X-Ray Pretrained Classification Service for InteractMed-VLM.
Uses TorchXRayVision DenseNet-121 (pretrained on multiple CXR datasets:
NIH, PC, CheXpert, MIMIC-CXR, OpenI, Kaggle, etc.).

Clinical & Safety Guardrails:
-----------------------------
1. Treats classifier outputs strictly as secondary review signals.
2. Does NOT represent classifier scores as calibrated probabilities or definitive diagnoses.
3. Maps only compatible TorchXRayVision pathologies to InteractMed-VLM categories.
4. Preserves unmapped findings and model provenance.
5. Operates in inference-only mode on CPU/MPS without creating excessive memory pressure.
6. Does NOT convert missing classifier output into evidence of normality.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

log = logging.getLogger("classifier_service")
if not log.handlers:
    log.addHandler(logging.NullHandler())

# ── Mapping from TorchXRayVision Pathologies to InteractMed-VLM Categories ───
# TorchXRayVision DenseNet-121 output labels (18):
# ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
#  'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
#  'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'Lung Lesion', 'Fracture',
#  'Lung Opacity', 'Enlarged Cardiomediastinum']

XRV_TO_INTERACTMED_MAP: Dict[str, List[str]] = {
    "cardiomegaly": ["Cardiomegaly", "Enlarged Cardiomediastinum"],
    "pleural_effusion": ["Effusion"],
    "pneumothorax": ["Pneumothorax"],
    "pneumonia_consolidation": ["Pneumonia", "Consolidation", "Infiltration"],
    "pulmonary_edema": ["Edema"],
    "atelectasis": ["Atelectasis"],
    "lung_opacity": ["Lung Opacity"],
}

UNMAPPED_XRV_PATHOLOGIES: List[str] = [
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Nodule",
    "Mass",
    "Hernia",
    "Lung Lesion",
    "Fracture",
]

DEFAULT_CONFIDENCE_THRESHOLD = 0.50


@dataclass
class ClassifierFinding:
    """Represents a single finding mapped to an InteractMed-VLM category."""
    category: str
    confidence: float
    contributing_pathologies: Dict[str, float]
    exceeds_threshold: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClassifierResult:
    """Full inference result from the auxiliary pretrained chest X-ray classifier."""
    success: bool
    model_name: str
    device: str
    latency_ms: float
    mapped_findings: Dict[str, float]
    findings_detail: List[ClassifierFinding]
    unmapped_findings: Dict[str, float]
    raw_probabilities: Dict[str, float]
    review_alerts: List[str] = field(default_factory=list)
    threshold_applied: float = DEFAULT_CONFIDENCE_THRESHOLD
    error: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "model_name": self.model_name,
            "device": self.device,
            "latency_ms": self.latency_ms,
            "mapped_findings": self.mapped_findings,
            "findings_detail": [f.to_dict() for f in self.findings_detail],
            "unmapped_findings": self.unmapped_findings,
            "raw_probabilities": self.raw_probabilities,
            "review_alerts": self.review_alerts,
            "threshold_applied": self.threshold_applied,
            "error": self.error,
            "provenance": self.provenance,
        }


class ClassifierService:
    """Pretrained TorchXRayVision DenseNet-121 classification service."""

    def __init__(self, weights: str = "densenet121-res224-all", device: Optional[str] = None):
        self.weights = weights
        self._requested_device = device
        self._device: Optional[torch.device] = None
        self._model: Optional[Any] = None
        self._pathologies: List[str] = []
        self._loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device_name(self) -> str:
        return str(self._device) if self._device else "unloaded"

    def ensure_loaded(self) -> None:
        """Loads TorchXRayVision DenseNet-121 into memory on demand."""
        if self._loaded and self._model is not None:
            return

        try:
            import torchxrayvision as xrv
        except ImportError as e:
            log.error(f"torchxrayvision not installed: {e}")
            raise RuntimeError(
                "torchxrayvision is not installed in the active environment. "
                "Install via 'pip install torchxrayvision'."
            ) from e

        # Determine target device (prefer MPS on Apple Silicon, fallback to CPU)
        if self._requested_device:
            self._device = torch.device(self._requested_device)
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        log.info(f"Loading TorchXRayVision DenseNet ({self.weights}) on {self._device}...")
        t0 = time.perf_counter()

        try:
            # Instantiate DenseNet model
            self._model = xrv.models.DenseNet(weights=self.weights)
            self._model = self._model.to(self._device)
            self._model.eval()
            self._pathologies = list(self._model.pathologies)
            self._loaded = True
            elapsed = (time.perf_counter() - t0) * 1000
            log.info(f"Classifier loaded in {elapsed:.1f}ms on {self._device}. {len(self._pathologies)} pathologies.")
        except Exception as e:
            log.error(f"Failed to load TorchXRayVision model: {e}")
            self._model = None
            self._loaded = False
            raise RuntimeError(f"Failed to initialize classifier: {e}") from e

    def preprocess_image(self, image_input: Union[str, Path, bytes, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocesses image strictly matching TorchXRayVision requirements:
        1. Convert image to single-channel 2D grayscale array.
        2. Normalize values to [-1024, 1024] using xrv.datasets.normalize.
        3. Apply center-crop and resize to 224x224.
        4. Shape into (1, 1, 224, 224) torch tensor.
        """
        import torchxrayvision as xrv

        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input)).convert("L")
            img_arr = np.array(pil_img)
        elif isinstance(image_input, bytes):
            import io
            pil_img = Image.open(io.BytesIO(image_input)).convert("L")
            img_arr = np.array(pil_img)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("L")
            img_arr = np.array(pil_img)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 3:
                # RGB to Grayscale
                if image_input.shape[2] == 3:
                    img_arr = np.dot(image_input[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
                elif image_input.shape[0] == 3:
                    img_arr = np.dot(image_input.transpose(1, 2, 0)[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
                else:
                    img_arr = image_input.squeeze()
            else:
                img_arr = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Rescale image to [-1024, 1024] per torchxrayvision standards
        img_arr = img_arr.astype(np.float32)
        img_norm = xrv.datasets.normalize(img_arr, maxval=255.0, reshape=True)

        # Transforms: CenterCrop and Resize 224
        transform_crop = xrv.datasets.XRayCenterCrop()
        img_cropped = transform_crop(img_norm)
        transform_resize = xrv.datasets.XRayResizer(224)
        img_resized = transform_resize(img_cropped)

        # Convert to tensor: shape (1, 1, 224, 224)
        tensor = torch.from_numpy(img_resized).unsqueeze(0).to(torch.float32)
        if self._device:
            tensor = tensor.to(self._device)
        return tensor

    def predict(
        self,
        image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
        threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> ClassifierResult:
        """
        Executes classification on input image and returns structured findings.
        """
        self.ensure_loaded()
        t0 = time.perf_counter()

        try:
            tensor = self.preprocess_image(image_input)

            with torch.no_grad():
                logits = self._model(tensor)
                probs = torch.sigmoid(logits)[0].cpu().numpy()

            latency_ms = round((time.perf_counter() - t0) * 1000, 2)

            raw_probs: Dict[str, float] = {
                pathology: round(float(probs[i]), 4)
                for i, pathology in enumerate(self._pathologies)
            }

            # Map to InteractMed-VLM categories
            mapped_findings: Dict[str, float] = {}
            findings_detail: List[ClassifierFinding] = []

            for cat, xrv_keys in XRV_TO_INTERACTMED_MAP.items():
                contrib = {k: raw_probs[k] for k in xrv_keys if k in raw_probs}
                # Max aggregation across synonymous XRV categories
                max_score = max(contrib.values()) if contrib else 0.0
                mapped_findings[cat] = max_score
                findings_detail.append(
                    ClassifierFinding(
                        category=cat,
                        confidence=max_score,
                        contributing_pathologies=contrib,
                        exceeds_threshold=(max_score >= threshold),
                    )
                )

            # Unmapped pathologies
            unmapped_findings: Dict[str, float] = {
                k: raw_probs[k] for k in UNMAPPED_XRV_PATHOLOGIES if k in raw_probs
            }

            # Review alerts for notable high-confidence signals
            alerts: List[str] = []
            for f in findings_detail:
                if f.exceeds_threshold:
                    alerts.append(f"Classifier elevated score: {f.category} ({f.confidence:.2f} >= {threshold:.2f})")

            return ClassifierResult(
                success=True,
                model_name=f"torchxrayvision-{self.weights}",
                device=str(self._device),
                latency_ms=latency_ms,
                mapped_findings=mapped_findings,
                findings_detail=findings_detail,
                unmapped_findings=unmapped_findings,
                raw_probabilities=raw_probs,
                review_alerts=alerts,
                threshold_applied=threshold,
                error=None,
                provenance={
                    "weights": self.weights,
                    "resolution": 224,
                    "normalization": "[-1024, 1024]",
                    "library_version": "1.5.5",
                    "disclaimer": "Experimental review signal; not a primary diagnostic tool.",
                },
            )

        except Exception as e:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            log.error(f"Classifier inference failed: {e}")
            return ClassifierResult(
                success=False,
                model_name=f"torchxrayvision-{self.weights}",
                device=str(self._device) if self._device else "none",
                latency_ms=latency_ms,
                mapped_findings={},
                findings_detail=[],
                unmapped_findings={},
                raw_probabilities={},
                review_alerts=[],
                threshold_applied=threshold,
                error=str(e),
                provenance={},
            )


# ── Global Singleton Accessor ─────────────────────────────────────────────────
_classifier_service_instance: Optional[ClassifierService] = None


def get_classifier_service(device: Optional[str] = None) -> ClassifierService:
    """Returns singleton instance of ClassifierService."""
    global _classifier_service_instance
    if _classifier_service_instance is None:
        _classifier_service_instance = ClassifierService(device=device)
    return _classifier_service_instance
