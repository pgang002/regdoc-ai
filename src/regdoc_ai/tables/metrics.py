from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .geometry import box_iou


@dataclass(frozen=True)
class PRF1:
    precision: float
    recall: float
    f1: float
    true_positives: int
    predicted: int
    reference: int


def _prf1(tp: int, predicted: int, reference: int) -> PRF1:
    precision = tp / predicted if predicted else (1.0 if reference == 0 else 0.0)
    recall = tp / reference if reference else (1.0 if predicted == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PRF1(precision, recall, f1, tp, predicted, reference)


def boundary_metrics(predicted: Iterable[int], reference: Iterable[int], tolerance: int = 8) -> PRF1:
    pred = list(predicted)
    ref = list(reference)
    candidates = sorted(
        (abs(p - r), pi, ri)
        for pi, p in enumerate(pred)
        for ri, r in enumerate(ref)
        if abs(p - r) <= tolerance
    )
    used_pred: set[int] = set()
    used_ref: set[int] = set()
    for _, pi, ri in candidates:
        if pi not in used_pred and ri not in used_ref:
            used_pred.add(pi)
            used_ref.add(ri)
    return _prf1(len(used_pred), len(pred), len(ref))


def cell_box_metrics(
    predicted: Iterable[Iterable[float]],
    reference: Iterable[Iterable[float]],
    iou_threshold: float = 0.7,
) -> PRF1:
    pred = [tuple(box) for box in predicted]
    ref = [tuple(box) for box in reference]
    candidates = sorted(
        [
            (box_iou(p, r), pi, ri)
            for pi, p in enumerate(pred)
            for ri, r in enumerate(ref)
            if box_iou(p, r) >= iou_threshold
        ],
        reverse=True,
    )
    used_pred: set[int] = set()
    used_ref: set[int] = set()
    for _, pi, ri in candidates:
        if pi not in used_pred and ri not in used_ref:
            used_pred.add(pi)
            used_ref.add(ri)
    return _prf1(len(used_pred), len(pred), len(ref))
