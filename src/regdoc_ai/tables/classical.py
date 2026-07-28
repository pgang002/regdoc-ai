from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import adjacent_cells, merge_nearby


@dataclass(frozen=True)
class GridPrediction:
    x_boundaries: list[int]
    y_boundaries: list[int]
    cells: list[tuple[int, int, int, int]]
    horizontal_mask: np.ndarray
    vertical_mask: np.ndarray


def _line_centers(mask: np.ndarray, axis: str, threshold_fraction: float, tolerance: int) -> list[int]:
    if axis == "horizontal":
        projection = np.count_nonzero(mask, axis=1)
        threshold = max(5, int(mask.shape[1] * threshold_fraction))
    elif axis == "vertical":
        projection = np.count_nonzero(mask, axis=0)
        threshold = max(5, int(mask.shape[0] * threshold_fraction))
    else:
        raise ValueError(f"Unknown axis: {axis}")

    indices = np.flatnonzero(projection >= threshold).tolist()
    return merge_nearby(indices, tolerance=tolerance)


def _ensure_outer_boundaries(values: list[int], length: int, margin: int = 12) -> list[int]:
    values = sorted(values)
    if not values:
        return values
    if values[0] > margin:
        values.insert(0, 0)
    if length - 1 - values[-1] > margin:
        values.append(length - 1)
    return values


def detect_ruled_table_grid(
    image: np.ndarray,
    *,
    horizontal_kernel_fraction: float = 0.12,
    vertical_kernel_fraction: float = 0.12,
    projection_threshold_fraction: float = 0.08,
    merge_tolerance_px: int = 5,
) -> GridPrediction:
    """Detect row/column rules in a cropped, ruled table image.

    The detector is deliberately classical and explainable. It is used as an
    executable local baseline while the optional deep table models run in a
    separate model-enabled environment.
    """
    if image is None or image.size == 0:
        raise ValueError("A non-empty image is required")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 13
    )
    height, width = gray.shape
    horizontal_length = max(12, int(width * horizontal_kernel_fraction))
    vertical_length = max(12, int(height * vertical_kernel_fraction))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_length, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_length))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    x_boundaries = _line_centers(
        vertical, "vertical", projection_threshold_fraction, merge_tolerance_px
    )
    y_boundaries = _line_centers(
        horizontal, "horizontal", projection_threshold_fraction, merge_tolerance_px
    )
    x_boundaries = _ensure_outer_boundaries(x_boundaries, width)
    y_boundaries = _ensure_outer_boundaries(y_boundaries, height)
    cells = adjacent_cells(x_boundaries, y_boundaries) if len(x_boundaries) > 1 and len(y_boundaries) > 1 else []
    return GridPrediction(x_boundaries, y_boundaries, cells, horizontal, vertical)
