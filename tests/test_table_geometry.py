from __future__ import annotations

import cv2
import numpy as np

from regdoc_ai.tables.classical import detect_ruled_table_grid
from regdoc_ai.tables.geometry import box_iou
from regdoc_ai.tables.metrics import boundary_metrics, cell_box_metrics


def test_box_iou_identity_and_disjoint() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_boundary_metrics_with_tolerance() -> None:
    result = boundary_metrics([1, 101, 201], [0, 100, 200], tolerance=2)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_cell_box_metrics() -> None:
    result = cell_box_metrics([(0, 0, 10, 10)], [(0, 0, 10, 10)], iou_threshold=0.9)
    assert result.f1 == 1.0


def test_classical_grid_detector_on_ruled_table() -> None:
    image = np.full((220, 320, 3), 255, dtype=np.uint8)
    for x in (10, 110, 210, 310):
        cv2.line(image, (x, 10), (x, 210), (0, 0, 0), 2)
    for y in (10, 110, 210):
        cv2.line(image, (10, y), (310, y), (0, 0, 0), 2)
    result = detect_ruled_table_grid(image, projection_threshold_fraction=0.2)
    assert len(result.x_boundaries) == 4
    assert len(result.y_boundaries) == 3
    assert len(result.cells) == 6
