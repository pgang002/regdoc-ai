from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from regdoc_ai.schemas.document import (
    CheckboxResult,
    ExtractedField,
    ProcessingStatus,
    TableArtifact,
)


class PageClassification(BaseModel):
    page: int = Field(ge=1)
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    decision_source: str
    rule_reason: str
    table_line_score: float = Field(ge=0.0, le=1.0)


class RedactionCandidate(BaseModel):
    candidate_id: str
    entity_type: str
    action: Literal["redact", "review", "retain", "ignore"]
    page: int = Field(ge=1)
    field_name: str
    masked_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: list[float] = Field(min_length=4, max_length=4)
    detection_methods: list[str] = Field(default_factory=list)
    needs_review: bool = False


class ArtifactLink(BaseModel):
    name: str
    media_type: str
    size_bytes: int = Field(ge=0)
    download_path: str


class ProcessingResponse(BaseModel):
    document_id: str
    source_filename: str
    status: ProcessingStatus
    created_at: str
    completed_at: str | None = None
    processing_seconds: float | None = Field(default=None, ge=0.0)
    page_count: int = Field(ge=0)
    classifications: list[PageClassification] = Field(default_factory=list)
    fields: list[ExtractedField] = Field(default_factory=list)
    checkboxes: list[CheckboxResult] = Field(default_factory=list)
    tables: list[TableArtifact] = Field(default_factory=list)
    redaction_candidates: list[RedactionCandidate] = Field(default_factory=list)
    artifacts: list[ArtifactLink] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RedactionDecision(BaseModel):
    candidate_id: str
    action: Literal["redact", "review", "retain", "ignore"]


class ApplyRedactionsRequest(BaseModel):
    decisions: list[RedactionDecision] = Field(default_factory=list)


class ApplyRedactionsResponse(BaseModel):
    document_id: str
    redacted_count: int
    review_count: int
    retained_count: int
    ignored_count: int
    artifact: ArtifactLink
    audit_artifact: ArtifactLink


class JobResponse(BaseModel):
    job_id: str
    document_id: str
    batch_id: str | None = None
    source_filename: str
    status: str
    progress: int = Field(ge=0, le=100)
    current_stage: str
    attempt_count: int = Field(ge=0)
    max_retries: int = Field(ge=0)
    celery_task_id: str | None = None
    queue_mode: str
    error_type: str | None = None
    error_message: str | None = None
    created_at: str
    queued_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    processing_seconds: float | None = Field(default=None, ge=0.0)
    result_path: str | None = None


class JobEventResponse(BaseModel):
    sequence: int = Field(ge=1)
    status: str
    stage: str
    progress: int = Field(ge=0, le=100)
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class BatchResponse(BaseModel):
    batch_id: str
    job_count: int = Field(ge=1)
    completed_count: int = Field(ge=0)
    progress: float = Field(ge=0.0, le=100.0)
    status_counts: dict[str, int]
    jobs: list[JobResponse]


class BatchSubmissionResponse(BaseModel):
    batch_id: str
    jobs: list[JobResponse]
