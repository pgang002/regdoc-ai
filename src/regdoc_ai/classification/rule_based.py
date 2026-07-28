from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract

from regdoc_ai.tables.classical import detect_ruled_table_grid


@dataclass(frozen=True)
class RuleClassification:
    label: str
    confidence: float
    reason: str
    ocr_text: str
    table_line_score: float


def normalize_page_text(text: str) -> str:
    text = text.replace("\x0c", " ")
    return re.sub(r"\s+", " ", text).strip()


def ocr_page_text(image: np.ndarray, *, max_dimension: int = 1800) -> str:
    if image is None or image.size == 0:
        raise ValueError("A non-empty page image is required")
    height, width = image.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    working = image
    if scale < 1.0:
        working = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY) if working.ndim == 3 else working
    text = pytesseract.image_to_string(gray, config="--oem 3 --psm 6")
    return normalize_page_text(text)


def ruled_table_score(image: np.ndarray) -> float:
    height, width = image.shape[:2]
    scale = min(1.0, 1200.0 / max(height, width))
    working = image
    if scale < 1.0:
        working = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    try:
        grid = detect_ruled_table_grid(
            working,
            horizontal_kernel_fraction=0.16,
            vertical_kernel_fraction=0.10,
            projection_threshold_fraction=0.12,
            merge_tolerance_px=4,
        )
    except Exception:  # noqa: BLE001
        return 0.0
    row_rules = max(0, len(grid.y_boundaries) - 2)
    col_rules = max(0, len(grid.x_boundaries) - 2)
    # Score rewards multiple internal row/column rules and is capped at 1.
    return min(1.0, (row_rules / 5.0) * 0.55 + (col_rules / 4.0) * 0.45)


def classify_with_rules(
    image: np.ndarray,
    *,
    ocr_text: str | None = None,
    max_dimension: int = 1800,
) -> RuleClassification:
    text = ocr_text if ocr_text is not None else ocr_page_text(image, max_dimension=max_dimension)
    normalized = normalize_page_text(text)
    upper = normalized.upper()

    form_rules = [
        (
            "FDA_1572",
            [r"FORM\s+FDA\s+1572", r"STATEMENT\s+OF\s+INVESTIGATOR"],
            "FDA 1572 form number/title",
        ),
        (
            "FDA_3454",
            [
                r"FORM\s+FDA\s+3454",
                r"CERTIFICATION.{0,80}FINANCIAL\s+INTERESTS",
            ],
            "FDA 3454 form number/certification title",
        ),
        (
            "FDA_3455",
            [
                r"FORM\s+FDA\s+3455",
                r"DISCLOSURE.{0,80}FINANCIAL\s+INTERESTS",
            ],
            "FDA 3455 form number/disclosure title",
        ),
    ]
    for label, patterns, reason in form_rules:
        matches = sum(bool(re.search(pattern, upper, flags=re.DOTALL)) for pattern in patterns)
        if matches:
            confidence = 0.99 if matches == len(patterns) else 0.93
            return RuleClassification(label, confidence, reason, normalized, 0.0)

    table_score = ruled_table_score(image)
    protocol_signal = bool(
        re.search(r"\bPROTOCOL\b|CLINICAL\s+STUDY|MODERNATX", upper)
    )
    table_language = bool(
        re.search(
            r"\bTABLE\b|SCHEDULE\s+OF\s+EVENTS|GRADE\s+[0-4]|TREATMENT\s+GROUP|ENDPOINT",
            upper,
        )
    )
    if table_score >= 0.62 or (table_score >= 0.45 and table_language):
        confidence = min(0.97, 0.62 + 0.35 * table_score)
        return RuleClassification(
            "CLINICAL_TABLE",
            confidence,
            "ruled-table geometry with clinical table language",
            normalized,
            table_score,
        )
    if protocol_signal:
        confidence = 0.92 if "CLINICAL STUDY PROTOCOL" in upper else 0.84
        return RuleClassification(
            "CLINICAL_PROTOCOL",
            confidence,
            "clinical protocol header/text",
            normalized,
            table_score,
        )
    return RuleClassification(
        "UNKNOWN",
        0.25,
        "no high-confidence document rule matched",
        normalized,
        table_score,
    )
