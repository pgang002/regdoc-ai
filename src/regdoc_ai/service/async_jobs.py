from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from regdoc_ai.persistence.repository import MetadataRepository
from regdoc_ai.worker.dispatcher import TaskDispatcher

from .models import BatchSubmissionResponse, JobResponse
from .storage import WorkspaceStore


class AsyncJobService:
    def __init__(
        self,
        *,
        store: WorkspaceStore,
        repository: MetadataRepository,
        dispatcher: TaskDispatcher,
        max_retries: int = 2,
    ):
        self.store = store
        self.repository = repository
        self.dispatcher = dispatcher
        self.max_retries = max_retries

    def submit(self, *, filename: str, data: bytes, batch_id: str | None = None) -> JobResponse:
        suffix = Path(filename).suffix.lower()
        from .pipeline import MAX_UPLOAD_BYTES, SUPPORTED_SUFFIXES

        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {suffix or 'none'}")
        if not data:
            raise ValueError("Uploaded file is empty")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError(f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")

        document_id = self.store.document_id_for(data, filename)
        source_path = self.store.save_upload(document_id, filename, data)
        source_sha256 = hashlib.sha256(data).hexdigest()
        self.repository.upsert_document(
            document_id=document_id,
            source_filename=filename,
            source_sha256=source_sha256,
            workspace_path=self.store.document_dir(document_id),
            status="uploaded",
            metadata={"source_path": str(source_path), "source_size_bytes": len(data)},
        )
        job_id = str(uuid.uuid4())
        self.repository.create_job(
            job_id=job_id,
            document_id=document_id,
            batch_id=batch_id,
            queue_mode=self.dispatcher.mode,
            max_retries=self.max_retries,
        )
        try:
            task_id = self.dispatcher.dispatch(job_id)
            self.repository.set_task_id(job_id, task_id)
        except Exception as exc:
            self.repository.mark_failure(job_id, exc)
            raise
        return JobResponse.model_validate(self.repository.get_job(job_id))

    def submit_batch(self, files: list[tuple[str, bytes]]) -> BatchSubmissionResponse:
        if not files:
            raise ValueError("At least one file is required")
        batch_id = str(uuid.uuid4())
        jobs = [self.submit(filename=name, data=data, batch_id=batch_id) for name, data in files]
        return BatchSubmissionResponse(batch_id=batch_id, jobs=jobs)

    def retry(self, job_id: str) -> JobResponse:
        current = self.repository.get_job(job_id)
        if current["status"] != "failed":
            raise ValueError("Only failed jobs can be retried")
        if int(current["attempt_count"]) > int(current["max_retries"]):
            raise ValueError("Maximum retry count reached")
        self.repository.append_event(
            job_id,
            status="retrying",
            stage="retrying",
            progress=0,
            message="Failed job was manually requeued.",
        )
        task_id = self.dispatcher.dispatch(job_id)
        self.repository.set_task_id(job_id, task_id)
        return JobResponse.model_validate(self.repository.get_job(job_id))
