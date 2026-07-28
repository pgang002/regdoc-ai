from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ProcessingStatus(str, Enum):
    uploaded = "UPLOADED"
    queued = "QUEUED"
    preprocessing = "PREPROCESSING"
    ocr_running = "OCR_RUNNING"
    table_extraction = "TABLE_EXTRACTION"
    redaction = "REDACTION"
    validation = "VALIDATION"
    completed = "COMPLETED"
    failed = "FAILED"
    needs_review = "NEEDS_REVIEW"


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class ExtractedField(BaseModel):
    name: str
    value: str | int | float | bool | None
    page: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: BoundingBox | None = None
    extraction_method: str


class CheckboxResult(BaseModel):
    name: str
    state: str
    page: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: BoundingBox


class TableArtifact(BaseModel):
    page: int = Field(ge=1)
    csv_path: str | None = None
    json_path: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentResult(BaseModel):
    document_id: str
    source_filename: str
    document_type: str | None = None
    status: ProcessingStatus
    fields: list[ExtractedField] = Field(default_factory=list)
    checkboxes: list[CheckboxResult] = Field(default_factory=list)
    tables: list[TableArtifact] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
