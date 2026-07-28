from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

PreprocessMode = Literal["raw", "grayscale", "adaptive", "clahe"]


def preprocess_image(image: np.ndarray, mode: PreprocessMode) -> np.ndarray:
    """Apply deterministic preprocessing suitable for scanned document OCR."""
    if image is None:
        raise ValueError("image cannot be None")
    if mode == "raw":
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if mode == "grayscale":
        return gray
    if mode == "adaptive":
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35,
            11,
        )
    if mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    raise ValueError(f"Unsupported preprocessing mode: {mode}")
