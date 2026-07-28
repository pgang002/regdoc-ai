from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select

from regdoc_ai.service.models import ArtifactLink, ProcessingResponse

from .database import Database
from .models import ArtifactRecord, DocumentRecord, JobEventRecord, ProcessingJobRecord


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


class MetadataRepository:
    def __init__(self, database: Database):
        self.database = database

    def upsert_document(
        self,
        *,
        document_id: str,
        source_filename: str,
        source_sha256: str,
        workspace_path: str | Path,
        status: str = "uploaded",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.database.session() as session, session.begin():
            row = session.get(DocumentRecord, document_id)
            if row is None:
                row = DocumentRecord(
                    id=document_id,
                    source_filename=source_filename,
                    source_sha256=source_sha256,
                    workspace_path=str(Path(workspace_path)),
                    status=status,
                    metadata_json=metadata or {},
                )
                session.add(row)
            else:
                row.source_filename = source_filename
                row.source_sha256 = source_sha256
                row.workspace_path = str(Path(workspace_path))
                row.status = status
                row.metadata_json = metadata or row.metadata_json
                row.updated_at = utc_now()

    def create_job(
        self,
        *,
        job_id: str,
        document_id: str,
        batch_id: str | None,
        queue_mode: str,
        max_retries: int,
    ) -> dict[str, Any]:
        now = utc_now()
        with self.database.session() as session, session.begin():
            job = ProcessingJobRecord(
                id=job_id,
                document_id=document_id,
                batch_id=batch_id,
                status="queued",
                progress=0,
                current_stage="uploaded",
                attempt_count=0,
                max_retries=max_retries,
                queue_mode=queue_mode,
                queued_at=now,
            )
            session.add(job)
            session.flush()
            session.add(
                JobEventRecord(
                    job_id=job_id,
                    sequence=1,
                    status="queued",
                    stage="uploaded",
                    progress=0,
                    message="Document accepted and queued for processing.",
                    details_json={"queue_mode": queue_mode},
                )
            )
        return self.get_job(job_id)

    def set_task_id(self, job_id: str, task_id: str | None) -> None:
        with self.database.session() as session, session.begin():
            job = session.get(ProcessingJobRecord, job_id)
            if job is None:
                raise FileNotFoundError(job_id)
            job.celery_task_id = task_id

    def append_event(
        self,
        job_id: str,
        *,
        status: str,
        stage: str,
        progress: int,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        progress = max(0, min(100, int(progress)))
        now = utc_now()
        with self.database.session() as session, session.begin():
            job = session.get(ProcessingJobRecord, job_id)
            if job is None:
                raise FileNotFoundError(job_id)
            next_sequence = int(
                session.scalar(
                    select(func.coalesce(func.max(JobEventRecord.sequence), 0)).where(
                        JobEventRecord.job_id == job_id
                    )
                )
                or 0
            ) + 1
            job.status = status
            job.current_stage = stage
            job.progress = progress
            if status in {"preprocessing", "classifying", "extracting", "redacting", "persisting"}:
                job.started_at = job.started_at or now
            if status in {"completed", "needs_review", "failed"}:
                job.completed_at = now
                if job.started_at:
                    started_at = job.started_at
                    if started_at.tzinfo is None:
                        started_at = started_at.replace(tzinfo=timezone.utc)
                    job.processing_seconds = max(0.0, (now - started_at).total_seconds())
            session.add(
                JobEventRecord(
                    job_id=job_id,
                    sequence=next_sequence,
                    status=status,
                    stage=stage,
                    progress=progress,
                    message=message,
                    details_json=details or {},
                )
            )

    def mark_attempt_started(self, job_id: str) -> None:
        with self.database.session() as session, session.begin():
            job = session.get(ProcessingJobRecord, job_id)
            if job is None:
                raise FileNotFoundError(job_id)
            job.attempt_count += 1
            job.started_at = job.started_at or utc_now()
            job.error_type = None
            job.error_message = None

    def mark_failure(self, job_id: str, exc: Exception) -> None:
        with self.database.session() as session, session.begin():
            job = session.get(ProcessingJobRecord, job_id)
            if job is None:
                raise FileNotFoundError(job_id)
            job.error_type = type(exc).__name__
            job.error_message = str(exc)[:4000]
        self.append_event(
            job_id,
            status="failed",
            stage="failed",
            progress=100,
            message=f"Processing failed: {type(exc).__name__}.",
            details={"error_type": type(exc).__name__, "error_message": str(exc)[:1000]},
        )

    def persist_result(self, job_id: str, result: ProcessingResponse) -> None:
        with self.database.session() as session, session.begin():
            document = session.get(DocumentRecord, result.document_id)
            job = session.get(ProcessingJobRecord, job_id)
            if document is None or job is None:
                raise FileNotFoundError(job_id)
            document.status = str(result.status.value if hasattr(result.status, "value") else result.status)
            document.page_count = result.page_count
            document.result_json = result.model_dump(mode="json")
            document.metadata_json = result.metadata
            document.updated_at = utc_now()
            session.execute(delete(ArtifactRecord).where(ArtifactRecord.job_id == job_id))
            for artifact in result.artifacts:
                session.add(
                    ArtifactRecord(
                        document_id=result.document_id,
                        job_id=job_id,
                        name=artifact.name,
                        media_type=artifact.media_type,
                        size_bytes=artifact.size_bytes,
                        storage_path=artifact.download_path,
                    )
                )

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.database.session() as session:
            job = session.get(ProcessingJobRecord, job_id)
            if job is None:
                raise FileNotFoundError(job_id)
            document = session.get(DocumentRecord, job.document_id)
            return {
                "job_id": job.id,
                "document_id": job.document_id,
                "batch_id": job.batch_id,
                "source_filename": document.source_filename if document else "",
                "status": job.status,
                "progress": job.progress,
                "current_stage": job.current_stage,
                "attempt_count": job.attempt_count,
                "max_retries": job.max_retries,
                "celery_task_id": job.celery_task_id,
                "queue_mode": job.queue_mode,
                "error_type": job.error_type,
                "error_message": job.error_message,
                "created_at": iso(job.created_at),
                "queued_at": iso(job.queued_at),
                "started_at": iso(job.started_at),
                "completed_at": iso(job.completed_at),
                "processing_seconds": job.processing_seconds,
                "result_path": (
                    f"/v1/documents/{job.document_id}" if document and document.result_json else None
                ),
            }

    def get_events(self, job_id: str) -> list[dict[str, Any]]:
        with self.database.session() as session:
            if session.get(ProcessingJobRecord, job_id) is None:
                raise FileNotFoundError(job_id)
            rows = session.scalars(
                select(JobEventRecord)
                .where(JobEventRecord.job_id == job_id)
                .order_by(JobEventRecord.sequence)
            ).all()
            return [
                {
                    "sequence": row.sequence,
                    "status": row.status,
                    "stage": row.stage,
                    "progress": row.progress,
                    "message": row.message,
                    "details": row.details_json,
                    "created_at": iso(row.created_at),
                }
                for row in rows
            ]

    def get_batch(self, batch_id: str) -> dict[str, Any]:
        with self.database.session() as session:
            rows = session.scalars(
                select(ProcessingJobRecord)
                .where(ProcessingJobRecord.batch_id == batch_id)
                .order_by(ProcessingJobRecord.created_at)
            ).all()
            if not rows:
                raise FileNotFoundError(batch_id)
            statuses: dict[str, int] = {}
            for row in rows:
                statuses[row.status] = statuses.get(row.status, 0) + 1
            terminal = {"completed", "needs_review", "failed"}
            done = sum(count for status, count in statuses.items() if status in terminal)
            return {
                "batch_id": batch_id,
                "job_count": len(rows),
                "completed_count": done,
                "progress": round(100.0 * done / len(rows), 2),
                "status_counts": statuses,
                "jobs": [self.get_job(row.id) for row in rows],
            }


    def operational_summary(self) -> dict[str, Any]:
        """Return database-backed operational metrics for dashboards and Prometheus."""
        with self.database.session() as session:
            jobs = session.scalars(select(ProcessingJobRecord)).all()
            artifacts = session.scalars(select(ArtifactRecord)).all()
            documents = int(session.scalar(select(func.count(DocumentRecord.id))) or 0)

        status_counts: dict[str, int] = {}
        for job in jobs:
            status_counts[job.status] = status_counts.get(job.status, 0) + 1

        terminal = [job for job in jobs if job.status in {"completed", "needs_review", "failed"}]
        successful = [job for job in terminal if job.status in {"completed", "needs_review"}]
        processing = sorted(
            float(job.processing_seconds) for job in terminal if job.processing_seconds is not None
        )

        def percentile(values: list[float], fraction: float) -> float:
            if not values:
                return 0.0
            if len(values) == 1:
                return values[0]
            position = (len(values) - 1) * fraction
            lower = int(position)
            upper = min(lower + 1, len(values) - 1)
            weight = position - lower
            return values[lower] * (1.0 - weight) + values[upper] * weight

        total_terminal = len(terminal)
        completed_times = [
            job.completed_at for job in terminal if job.completed_at is not None
        ]
        latest = max(completed_times) if completed_times else None
        return {
            "document_count": documents,
            "job_count": len(jobs),
            "terminal_job_count": total_terminal,
            "status_counts": status_counts,
            "success_rate": round(len(successful) / total_terminal, 6) if total_terminal else 0.0,
            "failure_rate": round(status_counts.get("failed", 0) / total_terminal, 6)
            if total_terminal
            else 0.0,
            "needs_review_rate": round(status_counts.get("needs_review", 0) / total_terminal, 6)
            if total_terminal
            else 0.0,
            "processing_seconds": {
                "mean": round(sum(processing) / len(processing), 6) if processing else 0.0,
                "p50": round(percentile(processing, 0.50), 6),
                "p95": round(percentile(processing, 0.95), 6),
                "max": round(max(processing), 6) if processing else 0.0,
            },
            "artifact_count": len(artifacts),
            "artifact_bytes": int(sum(row.size_bytes for row in artifacts)),
            "latest_terminal_at": iso(latest),
        }

    def list_artifacts(self, job_id: str) -> list[ArtifactLink]:
        with self.database.session() as session:
            rows = session.scalars(
                select(ArtifactRecord).where(ArtifactRecord.job_id == job_id)
            ).all()
            return [
                ArtifactLink(
                    name=row.name,
                    media_type=row.media_type,
                    size_bytes=row.size_bytes,
                    download_path=row.storage_path,
                )
                for row in rows
            ]
