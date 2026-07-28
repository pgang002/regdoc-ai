from __future__ import annotations

import numpy as np
import pytest

from regdoc_ai.ocr.engines import (
    EngineOCRResult,
    TesseractFieldEngine,
    create_engine,
    parse_paddle_recognition_result,
    segment_text_lines,
)


def test_parse_paddle_nested_result() -> None:
    result = parse_paddle_recognition_result(
        {"res": {"rec_text": "ModernaTX, Inc.", "rec_score": 0.97}}
    )
    assert result == EngineOCRResult(text="ModernaTX, Inc.", mean_confidence=97.0)


def test_parse_paddle_plural_result() -> None:
    result = parse_paddle_recognition_result(
        {"rec_texts": ["line one", "line two"], "rec_scores": [0.8, 0.9]}
    )
    assert result.text == "line one\nline two"
    assert result.mean_confidence == pytest.approx(85.0)


def test_parse_paddle_unknown_result_fails() -> None:
    with pytest.raises(ValueError):
        parse_paddle_recognition_result({"unexpected": "shape"})


def test_create_tesseract_engine() -> None:
    engine = create_engine("tesseract", language="eng", psm=6)
    assert isinstance(engine, TesseractFieldEngine)
    assert engine.model_name == "tesseract-eng"


def test_line_segmentation_finds_two_lines() -> None:
    image = np.full((80, 240), 255, dtype=np.uint8)
    for x in range(20, 180, 18):
        image[12:20, x : x + 8] = 0
    for x in range(20, 190, 18):
        image[50:58, x : x + 8] = 0
    lines = segment_text_lines(image)
    assert len(lines) == 2
