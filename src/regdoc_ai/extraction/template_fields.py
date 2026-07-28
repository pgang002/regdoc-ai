from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from regdoc_ai.evaluation.text_metrics import normalize_text
from regdoc_ai.ocr.engines import FieldOCREngine, TesseractFieldEngine
from regdoc_ai.preprocessing.image import preprocess_image


@dataclass(frozen=True)
class FieldOCRPrediction:
    text: str
    normalized_text: str
    mean_confidence: float


def pdf_rect_to_pixels(
    rect: tuple[float, float, float, float],
    *,
    dpi: int,
    padding_points: float = 1.5,
) -> tuple[int, int, int, int]:
    scale = dpi / 72.0
    x0, y0, x1, y1 = rect
    return (
        max(0, int(round((x0 - padding_points) * scale))),
        max(0, int(round((y0 - padding_points) * scale))),
        int(round((x1 + padding_points) * scale)),
        int(round((y1 + padding_points) * scale)),
    )


def crop_image(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    height, width = image.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid crop after clipping: {box}")
    return image[y0:y1, x0:x1]


def prepare_field_crop(
    image: np.ndarray,
    rect_pdf: tuple[float, float, float, float],
    *,
    dpi: int,
    preprocessing: str,
) -> np.ndarray:
    """Crop and normalize a form field before passing it to any OCR engine."""
    box = pdf_rect_to_pixels(rect_pdf, dpi=dpi, padding_points=0.0)
    crop = crop_image(image, box)
    processed = preprocess_image(crop, preprocessing)  # type: ignore[arg-type]
    if processed.shape[0] < 100:
        processed = cv2.resize(processed, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return processed


def recognize_field_with_engine(
    image: np.ndarray,
    rect_pdf: tuple[float, float, float, float],
    *,
    dpi: int,
    multiline: bool,
    preprocessing: str,
    engine: FieldOCREngine,
) -> FieldOCRPrediction:
    processed = prepare_field_crop(
        image, rect_pdf, dpi=dpi, preprocessing=preprocessing
    )
    result = engine.recognize(processed, multiline=multiline)
    return FieldOCRPrediction(
        text=result.text.strip(),
        normalized_text=normalize_text(result.text),
        mean_confidence=result.mean_confidence,
    )


def recognize_field(
    image: np.ndarray,
    rect_pdf: tuple[float, float, float, float],
    *,
    dpi: int,
    multiline: bool,
    preprocessing: str,
) -> FieldOCRPrediction:
    """Backward-compatible Tesseract field recognition used by Day 2/3."""
    return recognize_field_with_engine(
        image,
        rect_pdf,
        dpi=dpi,
        multiline=multiline,
        preprocessing=preprocessing,
        engine=TesseractFieldEngine(),
    )
