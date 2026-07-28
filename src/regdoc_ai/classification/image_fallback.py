from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC


@dataclass(frozen=True)
class ImageClassification:
    label: str
    confidence: float
    margin: float
    scores: dict[str, float]


def _letterbox_grayscale(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    h, w = gray.shape
    scale = min(width / w, height / h)
    resized = cv2.resize(
        gray,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    canvas = np.full((height, width), 255, dtype=np.uint8)
    y0 = (height - resized.shape[0]) // 2
    x0 = (width - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def extract_hog_features(
    image: np.ndarray,
    *,
    width: int = 384,
    height: int = 512,
) -> np.ndarray:
    prepared = _letterbox_grayscale(image, width=width, height=height)
    features = hog(
        prepared,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )
    return features.astype(np.float32)


class HOGLinearSVCClassifier:
    def __init__(self, *, width: int = 384, height: int = 512, random_state: int = 0):
        self.width = width
        self.height = height
        self.model = LinearSVC(C=1.0, class_weight="balanced", random_state=random_state, max_iter=20000)

    def fit(self, images: list[np.ndarray], labels: list[str]) -> "HOGLinearSVCClassifier":
        if len(images) != len(labels) or not images:
            raise ValueError("images and labels must be non-empty and aligned")
        matrix = np.vstack(
            [extract_hog_features(img, width=self.width, height=self.height) for img in images]
        )
        self.model.fit(matrix, labels)
        return self

    def predict(self, image: np.ndarray) -> ImageClassification:
        features = extract_hog_features(image, width=self.width, height=self.height)[None, :]
        raw = np.asarray(self.model.decision_function(features))
        classes = [str(value) for value in self.model.classes_]
        if raw.ndim == 1:
            raw = np.column_stack([-raw, raw])
        row = raw[0]
        order = np.argsort(row)[::-1]
        best = int(order[0])
        second = int(order[1]) if len(order) > 1 else best
        shifted = row - np.max(row)
        probs = np.exp(shifted) / max(float(np.exp(shifted).sum()), 1e-12)
        return ImageClassification(
            label=classes[best],
            confidence=float(probs[best]),
            margin=float(row[best] - row[second]),
            scores={label: float(score) for label, score in zip(classes, row)},
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"width": self.width, "height": self.height, "model": self.model}, path
        )

    @classmethod
    def load(cls, path: Path) -> "HOGLinearSVCClassifier":
        payload = joblib.load(path)
        instance = cls(width=int(payload["width"]), height=int(payload["height"]))
        instance.model = payload["model"]
        return instance
