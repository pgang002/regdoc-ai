from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from regdoc_ai.schemas.document import ProcessingStatus

from .runtime import WorkerRuntime


class RecoverableProcessingError(RuntimeError):
    """Signals a transient error that a Celery worker may retry."""


ProgressCallback = Callable[[str, int, str, dict | None], None]


def run_processing_job(
    job_id: str, *, runtime: WorkerRuntime | None = None, mark_failure: bool = True
) -> dict[str, object]:
    runtime = runtime or WorkerRuntime()
    repo = runtime.repository
    store = runtime.store
    job = repo.get_job(job_id)
    repo.mark_attempt_started(job_id)
    document_id = str(job["document_id"])
    filename = str(job["source_filename"])
    doc_dir = store.document_dir(document_id)
    source_candidates = sorted(doc_dir.glob("source.*"))
    if not source_candidates:
        error = FileNotFoundError(f"No source upload found for {document_id}")
        repo.mark_failure(job_id, error)
        raise error
    source_path = source_candidates[0]
    data = source_path.read_bytes()
    started = time.perf_counter()

    stage_to_status = {
        "uploaded": "queued",
        "preprocessing": "preprocessing",
        "classifying": "classifying",
        "extracting": "extracting",
        "redacting": "redacting",
        "persisting": "persisting",
        "completed": "completed",
        "needs_review": "needs_review",
    }

    def progress(stage: str, percent: int, message: str, details: dict | None = None) -> None:
        status = stage_to_status.get(stage, "extracting")
        repo.append_event(
            job_id,
            status=status,
            stage=stage,
            progress=percent,
            message=message,
            details=details,
        )

    try:
        result = runtime.pipeline.process(
            filename=filename,
            data=data,
            progress_callback=progress,
            processing_mode="asynchronous_day9",
        )
        repo.persist_result(job_id, result)
        terminal_status = (
            "needs_review" if result.status == ProcessingStatus.needs_review else "completed"
        )
        repo.append_event(
            job_id,
            status=terminal_status,
            stage=terminal_status,
            progress=100,
            message="Processing finished and metadata was persisted.",
            details={
                "document_id": result.document_id,
                "page_count": result.page_count,
                "artifact_count": len(result.artifacts),
                "processing_seconds": round(time.perf_counter() - started, 6),
            },
        )
        return {
            "job_id": job_id,
            "document_id": result.document_id,
            "status": terminal_status,
            "page_count": result.page_count,
            "artifact_count": len(result.artifacts),
        }
    except Exception as exc:
        if mark_failure:
            repo.mark_failure(job_id, exc)
        raise
