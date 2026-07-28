from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DocumentEnhancementResult:
    image: np.ndarray
    estimated_skew_degrees: float


def estimate_skew_degrees(image: np.ndarray, *, max_abs_angle: float = 7.0) -> float:
    """Estimate skew using long near-horizontal form lines."""
    if image is None:
        raise ValueError("image cannot be None")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    height, width = gray.shape
    scale = min(1.0, 1400.0 / max(height, width))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    edges = cv2.Canny(normalized, 50, 150, apertureSize=3)
    h, w = edges.shape
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 720.0,
        threshold=max(60, int(w * 0.05)),
        minLineLength=max(100, int(w * 0.18)),
        maxLineGap=max(10, int(w * 0.02)),
    )
    if lines is None:
        return 0.0

    angles: list[float] = []
    weights: list[float] = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx, dy = float(x2 - x1), float(y2 - y1)
        if abs(dx) < 1e-6:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        if abs(angle) <= max_abs_angle:
            angles.append(angle)
            weights.append(float(np.hypot(dx, dy)))
    if not angles:
        return 0.0

    order = np.argsort(angles)
    sorted_angles = np.asarray(angles)[order]
    sorted_weights = np.asarray(weights)[order]
    cumulative = np.cumsum(sorted_weights)
    index = int(np.searchsorted(cumulative, cumulative[-1] / 2.0))
    return float(sorted_angles[min(index, len(sorted_angles) - 1)])


def rotate_keep_size(image: np.ndarray, degrees: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), degrees, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def enhance_document_page(image: np.ndarray) -> DocumentEnhancementResult:
    """Deskew and restore a scanned form before coordinate-based extraction."""
    angle = estimate_skew_degrees(image)
    aligned = rotate_keep_size(image, angle) if abs(angle) >= 0.05 else image.copy()
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY) if aligned.ndim == 3 else aligned
    denoised = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(denoised)
    blurred = cv2.GaussianBlur(clahe, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(clahe, 1.45, blurred, -0.45, 0)
    return DocumentEnhancementResult(image=sharpened, estimated_skew_degrees=angle)
