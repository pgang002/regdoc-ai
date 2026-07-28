from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import pytesseract


def ocr_words(image: np.ndarray, psm: int = 6, timeout: float | None = None) -> pd.DataFrame:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
    frame = pytesseract.image_to_data(
        rgb,
        config=f"--oem 3 --psm {psm}",
        output_type=pytesseract.Output.DATAFRAME,
        timeout=timeout,
    )
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["text", "left", "top", "width", "height", "conf"])
    frame = frame.dropna(subset=["text"]).copy()
    frame["text"] = frame["text"].astype(str)
    frame = frame[frame["text"].str.strip().ne("")]
    frame["conf"] = pd.to_numeric(frame["conf"], errors="coerce").fillna(-1)
    return frame[frame["conf"] >= 0].copy()


def assign_words_to_grid(
    words: pd.DataFrame,
    x_boundaries: list[int],
    y_boundaries: list[int],
) -> list[list[str]]:
    rows = max(0, len(y_boundaries) - 1)
    cols = max(0, len(x_boundaries) - 1)
    buckets: dict[tuple[int, int], list[tuple[int, int, str]]] = defaultdict(list)
    for record in words.to_dict(orient="records"):
        cx = float(record["left"]) + float(record["width"]) / 2
        cy = float(record["top"]) + float(record["height"]) / 2
        col = next((i for i in range(cols) if x_boundaries[i] <= cx <= x_boundaries[i + 1]), None)
        row = next((i for i in range(rows) if y_boundaries[i] <= cy <= y_boundaries[i + 1]), None)
        if row is not None and col is not None:
            buckets[(row, col)].append((int(record["top"]), int(record["left"]), str(record["text"])))
    matrix = [["" for _ in range(cols)] for _ in range(rows)]
    for (row, col), tokens in buckets.items():
        matrix[row][col] = " ".join(token for _, _, token in sorted(tokens))
    return matrix


def matrix_to_dataframe(matrix: list[list[str]]) -> pd.DataFrame:
    width = max((len(row) for row in matrix), default=0)
    padded = [row + [""] * (width - len(row)) for row in matrix]
    return pd.DataFrame(padded)
