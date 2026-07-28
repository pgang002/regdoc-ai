from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from regdoc_ai.extraction.template_fields import crop_image, pdf_rect_to_pixels


@dataclass(frozen=True)
class CheckboxPrediction:
    checked: bool
    inner_dark_ratio: float
    threshold: float


def detect_checkbox(
    image: np.ndarray,
    rect_pdf: tuple[float, float, float, float],
    *,
    dpi: int,
    threshold: float = 0.08,
) -> CheckboxPrediction:
    """Classify a checkbox using ink density inside the border.

    The outer 25% border is removed so the printed square itself does not dominate
    the measurement. This deterministic baseline works well for clean fixed forms.
    """
    box = pdf_rect_to_pixels(rect_pdf, dpi=dpi, padding_points=0.5)
    crop = crop_image(image, box)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    height, width = gray.shape
    margin_y = max(1, int(round(height * 0.25)))
    margin_x = max(1, int(round(width * 0.25)))
    inner = gray[margin_y : height - margin_y, margin_x : width - margin_x]
    if inner.size == 0:
        inner = gray
    # Normalize the ink threshold to the local checkbox background. A fixed global
    # threshold can classify a uniformly shadowed but empty box as fully checked.
    local_background = float(np.percentile(inner, 85))
    ink_cutoff = max(0.0, min(160.0, local_background - 35.0))
    dark_ratio = float(np.mean(inner < ink_cutoff))
    return CheckboxPrediction(
        checked=dark_ratio >= threshold,
        inner_dark_ratio=dark_ratio,
        threshold=threshold,
    )
