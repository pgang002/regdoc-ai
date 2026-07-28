from __future__ import annotations

import cv2
import numpy as np

from regdoc_ai.checkboxes.classical import detect_checkbox


def _page_with_checkbox(*, checked: bool, background: int) -> np.ndarray:
    image = np.full((200, 200, 3), background, dtype=np.uint8)
    cv2.rectangle(image, (60, 60), (140, 140), (20, 20, 20), 4)
    if checked:
        cv2.line(image, (78, 102), (98, 122), (10, 10, 10), 8)
        cv2.line(image, (98, 122), (126, 78), (10, 10, 10), 8)
    return image


def test_shadowed_empty_checkbox_is_not_checked() -> None:
    image = _page_with_checkbox(checked=False, background=125)
    result = detect_checkbox(image, (21.6, 21.6, 50.4, 50.4), dpi=200, threshold=0.08)
    assert result.checked is False


def test_shadowed_marked_checkbox_is_checked() -> None:
    image = _page_with_checkbox(checked=True, background=125)
    result = detect_checkbox(image, (21.6, 21.6, 50.4, 50.4), dpi=200, threshold=0.08)
    assert result.checked is True
