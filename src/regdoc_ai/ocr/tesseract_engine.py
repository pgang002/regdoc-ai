from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytesseract
from pytesseract import Output


@dataclass(frozen=True)
class OCRResult:
    text: str
    words: pd.DataFrame
    mean_confidence: float


def recognize(image: np.ndarray, *, psm: int = 6, language: str = "eng") -> OCRResult:
    """Run Tesseract and retain word-level boxes and confidence values."""
    config = f"--oem 3 --psm {psm}"
    frame = pytesseract.image_to_data(
        image,
        lang=language,
        config=config,
        output_type=Output.DATAFRAME,
        pandas_config={"dtype": str},
    )
    frame = frame.dropna(subset=["text"]).copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame[frame["text"] != ""]
    frame["conf"] = pd.to_numeric(frame["conf"], errors="coerce")
    valid_conf = frame.loc[frame["conf"] >= 0, "conf"]
    mean_confidence = float(valid_conf.mean()) if not valid_conf.empty else 0.0

    # Reconstruct reading order from physical line coordinates rather than
    # block IDs, which can be non-sequential in long multi-line form fields.
    numeric_columns = [
        "page_num", "block_num", "par_num", "line_num", "word_num",
        "left", "top", "width", "height",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    lines: list[tuple[float, float, str]] = []
    grouping = ["page_num", "block_num", "par_num", "line_num"]
    for _, group in frame.groupby(grouping, sort=False):
        ordered = group.sort_values(["left", "word_num"])
        line = " ".join(ordered["text"].tolist()).strip()
        if line:
            lines.append((float(group["top"].min()), float(group["left"].min()), line))
    lines.sort(key=lambda item: (item[0], item[1]))
    text = "\n".join(line for _, _, line in lines)
    return OCRResult(text=text, words=frame, mean_confidence=mean_confidence)
