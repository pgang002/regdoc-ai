from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import fitz

from .models import DetectedEntity, RedactionAction


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _padded_rect(box: tuple[float, float, float, float], padding: float = 0.7) -> fitz.Rect:
    rect = fitz.Rect(*box)
    rect.x0 -= padding
    rect.y0 -= padding
    rect.x1 += padding
    rect.y1 += padding
    return rect


def redact_pdf(
    source_pdf: str | Path,
    output_pdf: str | Path,
    entities: Iterable[DetectedEntity],
    *,
    review_output_pdf: str | Path | None = None,
) -> dict[str, object]:
    source_pdf = Path(source_pdf)
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    entities = list(entities)

    document = fitz.open(source_pdf)
    for entity in entities:
        page = document[entity.page - 1]
        rect = _padded_rect(entity.bbox_pdf)
        if entity.action == RedactionAction.REDACT:
            page.add_redact_annot(rect, fill=(0, 0, 0), text="", cross_out=False)
        elif entity.action == RedactionAction.REVIEW:
            annotation = page.add_rect_annot(rect)
            annotation.set_colors(stroke=(1, 0.65, 0))
            annotation.set_border(width=1.2)
            annotation.set_info(content=f"REVIEW: {entity.entity_type}")
            annotation.update(opacity=0.9)
    for page in document:
        page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_PIXELS,
            graphics=fitz.PDF_REDACT_LINE_ART_NONE,
            text=fitz.PDF_REDACT_TEXT_REMOVE,
        )
    document.save(output_pdf, garbage=4, clean=True, deflate=True)
    document.close()

    if review_output_pdf is not None:
        # The redacted output already retains review annotations. Keep a named copy for UI workflows.
        Path(review_output_pdf).parent.mkdir(parents=True, exist_ok=True)
        Path(review_output_pdf).write_bytes(output_pdf.read_bytes())

    return {
        "source_pdf": str(source_pdf),
        "output_pdf": str(output_pdf),
        "source_sha256": file_sha256(source_pdf),
        "output_sha256": file_sha256(output_pdf),
        "redaction_count": sum(e.action == RedactionAction.REDACT for e in entities),
        "review_count": sum(e.action == RedactionAction.REVIEW for e in entities),
    }


def _normalize_for_verification(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def verify_redaction_regions(
    source_pdf: str | Path,
    redacted_pdf: str | Path,
    entities: Iterable[DetectedEntity],
    *,
    reference_texts: dict[tuple[int, str, str], str] | None = None,
) -> list[dict[str, object]]:
    source_document = fitz.open(source_pdf)
    redacted_document = fitz.open(redacted_pdf)
    rows: list[dict[str, object]] = []
    for entity in entities:
        source_page = source_document[entity.page - 1]
        redacted_page = redacted_document[entity.page - 1]
        rect = _padded_rect(entity.bbox_pdf)
        source_text = source_page.get_textbox(rect).strip()
        redacted_text = redacted_page.get_textbox(rect).strip()
        reference_key = (entity.page, entity.field_name, entity.entity_type)
        verification_text = (reference_texts or {}).get(reference_key, entity.detected_text)
        target = _normalize_for_verification(verification_text)
        source_normalized = _normalize_for_verification(source_text)
        redacted_normalized = _normalize_for_verification(redacted_text)
        target_present_before = bool(target) and target in source_normalized
        target_present_after = bool(target) and target in redacted_normalized
        if entity.action == RedactionAction.REDACT:
            # Surrounding field labels can overlap a widget rectangle. True-redaction
            # verification therefore checks removal of the detected target text rather
            # than requiring the entire rectangle to become text-free.
            passed = target_present_before and not target_present_after
        elif entity.action == RedactionAction.REVIEW:
            passed = bool(redacted_normalized)
        else:
            passed = True
        rows.append(
            {
                "page": entity.page,
                "field_name": entity.field_name,
                "entity_type": entity.entity_type,
                "action": entity.action.value,
                "source_text_in_box_before": source_text,
                "text_in_box_after": redacted_text,
                "verification_reference_source": (
                    "ground_truth_policy_reference" if reference_texts and reference_key in reference_texts
                    else "detected_ocr_text"
                ),
                "target_present_before": target_present_before,
                "target_present_after": target_present_after,
                "verification_passed": passed,
            }
        )
    source_document.close()
    redacted_document.close()
    return rows
