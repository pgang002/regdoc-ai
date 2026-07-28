"""OCR engine adapters."""

from regdoc_ai.ocr.engines import (
    EngineOCRResult,
    EngineUnavailableError,
    PaddleOCRFieldEngine,
    TesseractFieldEngine,
    create_engine,
)

__all__ = [
    "EngineOCRResult",
    "EngineUnavailableError",
    "PaddleOCRFieldEngine",
    "TesseractFieldEngine",
    "create_engine",
]
