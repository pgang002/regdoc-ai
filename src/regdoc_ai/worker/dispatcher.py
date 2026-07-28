from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock

from .tasks import run_processing_job


class TaskDispatcher:
    """Dispatch jobs to Celery in production or a local executor for tests."""

    def __init__(self, mode: str | None = None, *, max_workers: int = 2):
        self.mode = (mode or os.getenv("REGDOC_QUEUE_MODE", "celery")).lower()
        if self.mode not in {"celery", "eager", "thread"}:
            raise ValueError(f"Unsupported queue mode: {self.mode}")
        self._executor = ThreadPoolExecutor(max_workers=max_workers) if self.mode == "thread" else None
        self._futures: dict[str, Future] = {}
        self._lock = Lock()

    def dispatch(self, job_id: str) -> str | None:
        if self.mode == "eager":
            run_processing_job(job_id)
            return None
        if self.mode == "thread":
            assert self._executor is not None
            future = self._executor.submit(run_processing_job, job_id)
            with self._lock:
                self._futures[job_id] = future
            return f"local-thread-{job_id}"
        from .celery_app import process_document_task

        if process_document_task is None:
            raise RuntimeError(
                "Celery is not installed. Install requirements-infra.txt or use "
                "REGDOC_QUEUE_MODE=thread for local validation."
            )
        task = process_document_task.apply_async(args=[job_id])
        return str(task.id)

    def future(self, job_id: str) -> Future | None:
        with self._lock:
            return self._futures.get(job_id)

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
