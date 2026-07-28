from __future__ import annotations

"""OCR-engine adapters used by controlled field-level benchmarks.

PaddleOCR is imported lazily so the core repository and Tesseract pipeline remain
usable when the optional PaddlePaddle runtime is not installed.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Protocol

import cv2
import numpy as np

from regdoc_ai.ocr.tesseract_engine import recognize as recognize_tesseract


class EngineUnavailableError(RuntimeError):
    """Raised when an optional OCR runtime cannot be initialized."""


@dataclass(frozen=True)
class EngineOCRResult:
    text: str
    mean_confidence: float


class FieldOCREngine(Protocol):
    name: str
    model_name: str

    def recognize(self, image: np.ndarray, *, multiline: bool) -> EngineOCRResult:
        ...

    def metadata(self) -> dict[str, Any]:
        ...


def _package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


class TesseractFieldEngine:
    name = "tesseract"
    model_name = "tesseract-eng"

    def __init__(self, *, language: str = "eng", psm: int = 6) -> None:
        self.language = language
        self.psm = psm

    def recognize(self, image: np.ndarray, *, multiline: bool) -> EngineOCRResult:
        del multiline  # Keep PSM identical to the Day 2/3 controlled benchmark.
        result = recognize_tesseract(image, psm=self.psm, language=self.language)
        return EngineOCRResult(text=result.text.strip(), mean_confidence=result.mean_confidence)

    def metadata(self) -> dict[str, Any]:
        import pytesseract

        try:
            binary_version = str(pytesseract.get_tesseract_version())
        except Exception:  # noqa: BLE001  # pragma: no cover - diagnostic only
            binary_version = None
        return {
            "engine": self.name,
            "model_name": self.model_name,
            "language": self.language,
            "page_segmentation_mode": self.psm,
            "pytesseract_version": _package_version("pytesseract"),
            "tesseract_binary_version": binary_version,
        }


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    for attribute in ("json", "res"):
        candidate = getattr(value, attribute, None)
        if callable(candidate):
            try:
                candidate = candidate()
            except TypeError:
                pass
        if isinstance(candidate, Mapping):
            return candidate
    return None


def parse_paddle_recognition_result(value: Any) -> EngineOCRResult:
    """Normalize PaddleOCR 3.x TextRecognition result variants.

    PaddleOCR result objects have changed shape across minor versions. This
    parser accepts dictionaries, result objects exposing ``json``/``res``, and
    nested ``{"res": ...}`` payloads while failing loudly for unknown shapes.
    """
    mapping = _as_mapping(value)
    if mapping is None:
        raise ValueError(f"Unsupported PaddleOCR result type: {type(value)!r}")

    payload: Mapping[str, Any] = mapping
    while "res" in payload and isinstance(payload["res"], Mapping):
        payload = payload["res"]

    text = payload.get("rec_text")
    score = payload.get("rec_score")

    # Some wrappers expose plural values even for one input crop.
    if text is None:
        text = payload.get("rec_texts")
    if score is None:
        score = payload.get("rec_scores")

    if isinstance(text, Sequence) and not isinstance(text, (str, bytes)):
        text_values = [str(item) for item in text]
        text = "\n".join(item for item in text_values if item.strip())
    if isinstance(score, Sequence) and not isinstance(score, (str, bytes)):
        score_values = [float(item) for item in score]
        score = sum(score_values) / len(score_values) if score_values else 0.0

    if text is None:
        raise ValueError(f"PaddleOCR result did not contain recognized text: {mapping!r}")
    confidence = float(score) if score is not None else 0.0
    # Paddle returns scores in [0, 1]; repository summaries use percentages.
    if confidence <= 1.0:
        confidence *= 100.0
    return EngineOCRResult(text=str(text).strip(), mean_confidence=confidence)


def segment_text_lines(image: np.ndarray) -> list[np.ndarray]:
    """Split a multi-line field crop into line images using row projections."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Suppress thin horizontal form rules before calculating active text rows.
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, gray.shape[1] // 4), 1))
    rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, horizontal_kernel)
    ink = cv2.subtract(ink, rules)
    row_counts = (ink > 0).sum(axis=1)
    active = row_counts >= max(2, int(gray.shape[1] * 0.0025))

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, enabled in enumerate(active.tolist() + [False]):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            if index - start >= 3:
                runs.append((start, index))
            start = None

    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] <= 5:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    if len(merged) <= 1:
        return [image]
    padding = 4
    return [image[max(0, start - padding) : min(image.shape[0], end + padding)] for start, end in merged]


class PaddleOCRFieldEngine:
    name = "paddleocr"

    def __init__(
        self,
        *,
        model_name: str = "en_PP-OCRv5_mobile_rec",
        device: str = "cpu",
        batch_size: int = 1,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        try:
            from paddleocr import TextRecognition
        except Exception as exc:  # pragma: no cover - optional runtime
            raise EngineUnavailableError(
                "PaddleOCR is unavailable. Install requirements-paddle.txt in Python 3.10-3.13 "
                "or run notebooks/04_tesseract_vs_paddleocr.ipynb in Google Colab."
            ) from exc
        try:
            self._model = TextRecognition(model_name=model_name, device=device)
        except TypeError:
            # Compatibility with PaddleOCR releases that do not expose device here.
            self._model = TextRecognition(model_name=model_name)
        except Exception as exc:  # pragma: no cover - model download/runtime
            raise EngineUnavailableError(f"Unable to initialize PaddleOCR model {model_name!r}: {exc}") from exc

    def _recognize_one(self, image: np.ndarray) -> EngineOCRResult:
        try:
            outputs = self._model.predict(input=image, batch_size=self.batch_size)
        except TypeError:
            outputs = self._model.predict(image, batch_size=self.batch_size)
        outputs = list(outputs)
        if not outputs:
            return EngineOCRResult(text="", mean_confidence=0.0)
        return parse_paddle_recognition_result(outputs[0])

    def recognize(self, image: np.ndarray, *, multiline: bool) -> EngineOCRResult:
        crops = segment_text_lines(image) if multiline else [image]
        results = [self._recognize_one(crop) for crop in crops]
        texts = [result.text for result in results if result.text]
        confidences = [result.mean_confidence for result in results if result.text]
        return EngineOCRResult(
            text="\n".join(texts).strip(),
            mean_confidence=(sum(confidences) / len(confidences)) if confidences else 0.0,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "engine": self.name,
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "paddleocr_version": _package_version("paddleocr"),
            "paddlepaddle_version": _package_version("paddlepaddle"),
        }


def create_engine(name: str, **kwargs: Any) -> FieldOCREngine:
    normalized = name.strip().casefold()
    if normalized == "tesseract":
        return TesseractFieldEngine(**kwargs)
    if normalized == "paddleocr":
        return PaddleOCRFieldEngine(**kwargs)
    raise ValueError(f"Unknown OCR engine {name!r}; expected 'tesseract' or 'paddleocr'")
