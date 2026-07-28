from __future__ import annotations

from collections.abc import Iterable


def box_area(box: Iterable[float]) -> float:
    x0, y0, x1, y1 = [float(v) for v in box]
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def box_iou(first: Iterable[float], second: Iterable[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in first]
    bx0, by0, bx1, by1 = [float(v) for v in second]
    inter = (
        max(0.0, min(ax1, bx1) - max(ax0, bx0))
        * max(0.0, min(ay1, by1) - max(ay0, by0))
    )
    union = box_area((ax0, ay0, ax1, ay1)) + box_area((bx0, by0, bx1, by1)) - inter
    return inter / union if union else 0.0


def merge_nearby(values: Iterable[int], tolerance: int = 4) -> list[int]:
    ordered = sorted(int(v) for v in values)
    if not ordered:
        return []
    groups: list[list[int]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [int(round(sum(group) / len(group))) for group in groups]


def adjacent_cells(x_boundaries: list[int], y_boundaries: list[int]) -> list[tuple[int, int, int, int]]:
    return [
        (x_boundaries[col], y_boundaries[row], x_boundaries[col + 1], y_boundaries[row + 1])
        for row in range(len(y_boundaries) - 1)
        for col in range(len(x_boundaries) - 1)
    ]
