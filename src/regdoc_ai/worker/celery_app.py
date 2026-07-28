from __future__ import annotations

import os

try:
    from celery import Celery
except ImportError:  # Local test environments can run the eager/thread dispatcher.
    Celery = None  # type: ignore[assignment]


BROKER_URL = os.getenv("REGDOC_REDIS_BROKER_URL", "redis://redis:6379/0")
RESULT_BACKEND = os.getenv("REGDOC_REDIS_RESULT_URL", "redis://redis:6379/1")

if Celery is not None:
    celery_app = Celery("regdoc_ai", broker=BROKER_URL, backend=RESULT_BACKEND)
    celery_app.conf.update(
        task_track_started=True,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        broker_connection_retry_on_startup=True,
        result_expires=86400,
        broker_transport_options={"visibility_timeout": 3600},
        task_routes={"regdoc_ai.process_document": {"queue": "documents"}},
    )
else:
    celery_app = None


if celery_app is not None:
    from .runtime import WorkerRuntime
    from .tasks import run_processing_job

    @celery_app.task(
        bind=True,
        name="regdoc_ai.process_document",
        max_retries=2,
        default_retry_delay=10,
    )
    def process_document_task(self, job_id: str):  # type: ignore[no-untyped-def]
        try:
            return run_processing_job(job_id, mark_failure=False)
        except (OSError, TimeoutError, ConnectionError) as exc:
            runtime = WorkerRuntime()
            job = runtime.repository.get_job(job_id)
            if self.request.retries >= self.max_retries:
                runtime.repository.mark_failure(job_id, exc)
                raise
            runtime.repository.append_event(
                job_id,
                status="retrying",
                stage="retrying",
                progress=int(job["progress"]),
                message="Transient worker failure; Celery will retry the task.",
                details={
                    "error_type": type(exc).__name__,
                    "retry_number": self.request.retries + 1,
                },
            )
            # Celery records RETRY as a task state and republishes the same task ID.
            raise self.retry(exc=exc, countdown=min(60, 10 * (self.request.retries + 1)))
        except Exception as exc:
            WorkerRuntime().repository.mark_failure(job_id, exc)
            raise
else:
    process_document_task = None
