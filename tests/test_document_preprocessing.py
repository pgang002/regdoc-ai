from __future__ import annotations

import cv2
import numpy as np

from regdoc_ai.preprocessing.document import estimate_skew_degrees, rotate_keep_size


def test_estimate_skew_on_rotated_form_lines() -> None:
    image = np.full((700, 900, 3), 255, dtype=np.uint8)
    for y in range(100, 650, 100):
        cv2.line(image, (80, y), (820, y), (0, 0, 0), 3)
    rotated = rotate_keep_size(image, 2.0)
    estimate = estimate_skew_degrees(rotated)
    assert -2.5 < estimate < -1.5
