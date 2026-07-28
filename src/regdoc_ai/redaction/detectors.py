from __future__ import annotations

import re
from functools import lru_cache
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from regdoc_ai.extraction.template_fields import pdf_rect_to_pixels

from .models import DetectedEntity, RedactionAction
from .policy import RedactionPolicy


PROTOCOL_RE = re.compile(r"\b[A-Za-z]?mRNA[- ]?\d{4}[- ]?P\d{3}\b", re.IGNORECASE)
DATE_RE = re.compile(r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b")
POSTAL_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
NCT_RE = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)
PERSON_RE = re.compile(
    r"\b[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,3}(?:,?\s*(?:MD|DO|PhD))?\b"
)


@dataclass(frozen=True)
class OCRToken:
    text: str
    confidence: float
    bbox_pdf: tuple[float, float, float, float]
    start: int
    end: int


@dataclass(frozen=True)
class FieldOCR:
    text: str
    confidence: float
    tokens: tuple[OCRToken, ...]


def _field_entity_types(field_name: str) -> tuple[str, ...]:
    name = field_name.lower()
    types: list[str] = []
    if any(token in name for token in ("invest_name", "sub_inv_names", "invesname", "appname", "invname")):
        types.append("PERSON")
    if any(token in name for token in ("address", "_city", "_state", "_country", "_zip")):
        types.append("ADDRESS")
    if any(token in name for token in ("loc_name", "lab_name", "irb_name")):
        types.append("SENSITIVE_FACILITY")
    if "sigdate" in name or "sig_date" in name:
        types.append("DATE")
    if "prot_name_code" in name or "nameofstudy" in name:
        types.append("CCI_STUDY_TITLE")
    if "appfirm" in name:
        types.append("CCI_SPONSOR")
    return tuple(types)


def expected_entity_types(field_name: str, value: str) -> tuple[str, ...]:
    types = list(_field_entity_types(field_name))
    if PROTOCOL_RE.search(value):
        types.append("CCI_PROTOCOL_ID")
    return tuple(dict.fromkeys(types))


def ocr_field_tokens(
    page_image: np.ndarray,
    rect_pdf: tuple[float, float, float, float],
    *,
    dpi: int,
    multiline: bool,
) -> FieldOCR:
    x0, y0, x1, y1 = pdf_rect_to_pixels(rect_pdf, dpi=dpi, padding_points=0.0)
    h, w = page_image.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    crop = page_image[y0:y1, x0:x1]
    if crop.size == 0:
        return FieldOCR("", 0.0, ())
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if gray.shape[0] < 100:
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        coordinate_scale = 0.5
    else:
        coordinate_scale = 1.0
    psm = 6 if multiline else 7
    frame = pytesseract.image_to_data(
        gray,
        config=f"--oem 3 --psm {psm}",
        output_type=Output.DATAFRAME,
        pandas_config={"dtype": str},
    )
    if frame is None or frame.empty:
        return FieldOCR("", 0.0, ())
    frame = frame.dropna(subset=["text"]).copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame.loc[frame["text"] != ""]
    numeric_columns = [
        "page_num", "block_num", "par_num", "line_num", "word_num",
        "left", "top", "width", "height", "conf",
    ]
    for column in numeric_columns:
        frame[column] = __import__("pandas").to_numeric(frame[column], errors="coerce")
    frame = frame.loc[frame["conf"] >= 0]
    if frame.empty:
        return FieldOCR("", 0.0, ())

    grouped_lines = []
    grouping = ["page_num", "block_num", "par_num", "line_num"]
    for _, group in frame.groupby(grouping, sort=False):
        ordered = group.sort_values(["left", "word_num"])
        grouped_lines.append((float(group["top"].min()), float(group["left"].min()), ordered))
    grouped_lines.sort(key=lambda item: (item[0], item[1]))

    pieces: list[str] = []
    tokens: list[OCRToken] = []
    cursor = 0
    pt_per_px = 72.0 / dpi
    for line_index, (_, _, line) in enumerate(grouped_lines):
        if line_index > 0:
            pieces.append("\n")
            cursor += 1
        for word_index, row in enumerate(line.itertuples(index=False)):
            token_text = str(row.text)
            if word_index > 0:
                pieces.append(" ")
                cursor += 1
            start = cursor
            pieces.append(token_text)
            cursor += len(token_text)
            left = float(row.left) * coordinate_scale
            top = float(row.top) * coordinate_scale
            width = float(row.width) * coordinate_scale
            height = float(row.height) * coordinate_scale
            bbox = (
                (x0 + left) * pt_per_px,
                (y0 + top) * pt_per_px,
                (x0 + left + width) * pt_per_px,
                (y0 + top + height) * pt_per_px,
            )
            tokens.append(OCRToken(token_text, float(row.conf) / 100.0, bbox, start, cursor))
    text = "".join(pieces)
    confidence = float(frame["conf"].mean()) / 100.0
    return FieldOCR(text, confidence, tuple(tokens))


def _span_bbox(tokens: Iterable[OCRToken], start: int, end: int) -> tuple[float, float, float, float] | None:
    selected = [token for token in tokens if token.end > start and token.start < end]
    if not selected:
        return None
    return (
        min(t.bbox_pdf[0] for t in selected),
        min(t.bbox_pdf[1] for t in selected),
        max(t.bbox_pdf[2] for t in selected),
        max(t.bbox_pdf[3] for t in selected),
    )


@lru_cache(maxsize=1)
def _spacy_person_ruler():
    """Build the generic PERSON EntityRuler once per process."""
    try:
        import spacy

        nlp = spacy.blank("en")
        ruler = nlp.add_pipe("entity_ruler")
        ruler.add_patterns(
            [
                {
                    "label": "PERSON",
                    "pattern": [
                        {"IS_TITLE": True},
                        {"IS_TITLE": True},
                        {"IS_PUNCT": True, "OP": "?"},
                        {"LOWER": {"IN": ["md", "do", "phd"]}, "OP": "?"},
                    ],
                }
            ]
        )
        return nlp
    except Exception:
        return None


def _spacy_person_spans(text: str) -> list[tuple[int, int, str]]:
    """Generic spaCy EntityRuler fallback; no person names are preloaded."""
    nlp = _spacy_person_ruler()
    if nlp is None:
        return []
    return [(ent.start_char, ent.end_char, ent.text) for ent in nlp(text).ents if ent.label_ == "PERSON"]


def detect_regex_only(
    field_name: str,
    field_ocr: FieldOCR,
    rect_pdf: tuple[float, float, float, float],
    page: int,
    policy: RedactionPolicy,
) -> list[DetectedEntity]:
    detections: list[DetectedEntity] = []
    patterns = [
        ("PERSON", PERSON_RE),
        ("DATE", DATE_RE),
        ("ADDRESS", POSTAL_RE),
        ("CCI_PROTOCOL_ID", PROTOCOL_RE),
    ]
    for entity_type, pattern in patterns:
        for match in pattern.finditer(field_ocr.text):
            bbox = _span_bbox(field_ocr.tokens, match.start(), match.end()) or rect_pdf
            confidence = max(0.55, field_ocr.confidence)
            action, review = policy.resolve_action(entity_type, confidence)
            detections.append(
                DetectedEntity(
                    entity_type=entity_type,
                    action=action,
                    page=page,
                    field_name=field_name,
                    detected_text=match.group(0),
                    confidence=confidence,
                    bbox_pdf=bbox,
                    detection_methods=("regex", "ocr_coordinates"),
                    policy_rule=entity_type,
                    needs_review=review,
                )
            )
    return detections


def detect_hybrid_policy(
    field_name: str,
    field_ocr: FieldOCR,
    rect_pdf: tuple[float, float, float, float],
    page: int,
    policy: RedactionPolicy,
) -> list[DetectedEntity]:
    semantic_types = _field_entity_types(field_name)
    if not field_ocr.text.strip():
        # For fixed regulatory forms, a known sensitive field remains sensitive even
        # when OCR fails. Route it to review rather than silently treating it as safe.
        return [
            DetectedEntity(
                entity_type=entity_type,
                action=RedactionAction.REVIEW,
                page=page,
                field_name=field_name,
                detected_text="",
                confidence=0.40,
                bbox_pdf=rect_pdf,
                detection_methods=("field_semantics", "ocr_empty_fail_safe"),
                policy_rule=entity_type,
                needs_review=True,
            )
            for entity_type in semantic_types
        ]
    detections: list[DetectedEntity] = []

    if "PERSON" in semantic_types:
        spans = _spacy_person_spans(field_ocr.text)
        if not spans:
            spans = [(m.start(), m.end(), m.group(0)) for m in PERSON_RE.finditer(field_ocr.text)]
        if spans:
            boxes = [_span_bbox(field_ocr.tokens, start, end) for start, end, _ in spans]
            valid_boxes = [box for box in boxes if box is not None]
            bbox = (
                min(box[0] for box in valid_boxes),
                min(box[1] for box in valid_boxes),
                max(box[2] for box in valid_boxes),
                max(box[3] for box in valid_boxes),
            ) if valid_boxes else rect_pdf
            confidence = min(0.99, 0.70 + 0.30 * field_ocr.confidence)
            action, review = policy.resolve_action("PERSON", confidence)
            detections.append(
                DetectedEntity(
                    "PERSON", action, page, field_name, field_ocr.text, confidence, bbox,
                    ("field_semantics", "spacy_entity_ruler", "ocr_coordinates"),
                    "PERSON", review,
                )
            )

    for entity_type in ("ADDRESS", "SENSITIVE_FACILITY", "CCI_STUDY_TITLE", "CCI_SPONSOR", "DATE"):
        if entity_type not in semantic_types:
            continue
        methods = ["field_semantics", "ocr_coordinates"]
        confidence = min(0.99, 0.72 + 0.28 * field_ocr.confidence)
        detected_text = field_ocr.text
        bbox = rect_pdf
        if entity_type == "DATE":
            match = DATE_RE.search(field_ocr.text)
            if not match:
                continue
            detected_text = match.group(0)
            bbox = _span_bbox(field_ocr.tokens, match.start(), match.end()) or rect_pdf
            methods.append("regex")
        if entity_type == "ADDRESS" and "zip" in field_name.lower():
            match = POSTAL_RE.search(field_ocr.text)
            if match:
                detected_text = match.group(0)
                bbox = _span_bbox(field_ocr.tokens, match.start(), match.end()) or rect_pdf
                methods.append("regex")
        action, review = policy.resolve_action(entity_type, confidence)
        detections.append(
            DetectedEntity(
                entity_type, action, page, field_name, detected_text, confidence, bbox,
                tuple(methods), entity_type, review,
            )
        )

    for match in PROTOCOL_RE.finditer(field_ocr.text):
        bbox = _span_bbox(field_ocr.tokens, match.start(), match.end()) or rect_pdf
        confidence = min(0.99, 0.76 + 0.24 * field_ocr.confidence)
        action, review = policy.resolve_action("CCI_PROTOCOL_ID", confidence)
        detections.append(
            DetectedEntity(
                "CCI_PROTOCOL_ID", action, page, field_name, match.group(0), confidence, bbox,
                ("regex", "ocr_coordinates"), "CCI_PROTOCOL_ID", review,
            )
        )

    unique: dict[tuple[str, str], DetectedEntity] = {}
    for entity in detections:
        key = (entity.entity_type, entity.field_name)
        current = unique.get(key)
        if current is None or entity.confidence > current.confidence:
            unique[key] = entity
    return list(unique.values())
