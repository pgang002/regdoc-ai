from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path

import pandas as pd
from sqlalchemy import func, select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "results/day9_async"
WORKSPACE = OUTPUT_ROOT / "workspace"
DB_PATH = OUTPUT_ROOT / "regdoc_ai_day9.db"


def wait_for_batch(repository, batch_id: str, timeout: float = 300.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = repository.get_batch(batch_id)
        if state["completed_count"] == state["job_count"]:
            return state
        time.sleep(0.25)
    raise TimeoutError(f"Batch {batch_id} did not finish within {timeout} seconds")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(WORKSPACE, ignore_errors=True)
    DB_PATH.unlink(missing_ok=True)
    os.environ["REGDOC_PROJECT_ROOT"] = str(PROJECT_ROOT)
    os.environ["REGDOC_WORKSPACE"] = str(WORKSPACE)
    os.environ["REGDOC_DATABASE_URL"] = f"sqlite:///{DB_PATH.as_posix()}"
    os.environ["REGDOC_QUEUE_MODE"] = "thread"

    from regdoc_ai.persistence import Database, MetadataRepository
    from regdoc_ai.persistence.models import ArtifactRecord, DocumentRecord, JobEventRecord, ProcessingJobRecord
    from regdoc_ai.service.async_jobs import AsyncJobService
    from regdoc_ai.service.storage import WorkspaceStore
    from regdoc_ai.worker.dispatcher import TaskDispatcher
    from regdoc_ai.worker.runtime import WorkerRuntime
    from regdoc_ai.worker.tasks import run_processing_job

    database = Database.from_env(PROJECT_ROOT)
    database.create_schema()
    repository = MetadataRepository(database)
    store = WorkspaceStore(WORKSPACE)
    dispatcher = TaskDispatcher("thread", max_workers=2)
    service = AsyncJobService(store=store, repository=repository, dispatcher=dispatcher)

    samples = [
        PROJECT_ROOT / "data/processed/populated_forms/flattened/FDA_1572_NCT04796896.pdf",
        PROJECT_ROOT / "data/processed/table_benchmark/images/NCT04796896_p014_t0.png",
    ]
    submitted_at = time.perf_counter()
    batch = service.submit_batch([(path.name, path.read_bytes()) for path in samples])
    state = wait_for_batch(repository, batch.batch_id)
    observed_seconds = time.perf_counter() - submitted_at
    dispatcher.shutdown()

    job_rows = []
    event_rows = []
    for job in state["jobs"]:
        events = repository.get_events(job["job_id"])
        progresses = [event["progress"] for event in events]
        if progresses != sorted(progresses):
            raise AssertionError(f"Non-monotonic progress for {job['job_id']}: {progresses}")
        for event in events:
            event_rows.append({"job_id": job["job_id"], **event})
        result = store.load_result(job["document_id"])
        artifacts = repository.list_artifacts(job["job_id"])
        job_rows.append(
            {
                **job,
                "event_count": len(events),
                "page_count": result.page_count,
                "field_count": len(result.fields),
                "checkbox_count": len(result.checkboxes),
                "table_count": len(result.tables),
                "redaction_candidate_count": len(result.redaction_candidates),
                "artifact_count": len(artifacts),
            }
        )

    # Forced missing-source failure followed by a successful retry using an actual protocol page.
    retry_source = PROJECT_ROOT / "data/processed/document_understanding/base_images/NCT04796896_clinical_protocol_p001.png"
    retry_data = retry_source.read_bytes()
    retry_document_id = store.document_id_for(retry_data, retry_source.name)
    retry_path = store.save_upload(retry_document_id, retry_source.name, retry_data)
    repository.upsert_document(
        document_id=retry_document_id,
        source_filename=retry_source.name,
        source_sha256=__import__("hashlib").sha256(retry_data).hexdigest(),
        workspace_path=store.document_dir(retry_document_id),
    )
    retry_job_id = str(uuid.uuid4())
    repository.create_job(
        job_id=retry_job_id,
        document_id=retry_document_id,
        batch_id=None,
        queue_mode="eager",
        max_retries=2,
    )
    retry_path.unlink()
    runtime = WorkerRuntime(PROJECT_ROOT)
    try:
        run_processing_job(retry_job_id, runtime=runtime)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Missing-source job should have failed")
    retry_path.write_bytes(retry_data)
    repository.append_event(
        retry_job_id,
        status="retrying",
        stage="retrying",
        progress=0,
        message="Integration test restored source and requeued the failed job.",
    )
    retry_result = run_processing_job(retry_job_id, runtime=runtime)
    retry_state = repository.get_job(retry_job_id)
    retry_events = repository.get_events(retry_job_id)
    if retry_state["status"] not in {"completed", "needs_review"}:
        raise AssertionError(retry_state)

    with database.session() as session:
        database_counts = {
            "documents": int(session.scalar(select(func.count()).select_from(DocumentRecord)) or 0),
            "processing_jobs": int(session.scalar(select(func.count()).select_from(ProcessingJobRecord)) or 0),
            "job_events": int(session.scalar(select(func.count()).select_from(JobEventRecord)) or 0),
            "artifacts": int(session.scalar(select(func.count()).select_from(ArtifactRecord)) or 0),
        }

    summary = {
        "batch_id": batch.batch_id,
        "queue_mode": "thread_local_validation",
        "production_queue": "Celery + Redis",
        "production_database": "PostgreSQL",
        "job_count": state["job_count"],
        "status_counts": state["status_counts"],
        "batch_progress": state["progress"],
        "observed_batch_seconds": observed_seconds,
        "documents_per_minute": 60.0 * state["job_count"] / observed_seconds,
        "retry_job_id": retry_job_id,
        "retry_attempt_count": retry_state["attempt_count"],
        "retry_final_status": retry_state["status"],
        "retry_event_count": len(retry_events),
        "retry_result": retry_result,
        "database_counts": database_counts,
        "database_health": database.healthcheck(),
    }
    pd.DataFrame(job_rows).to_csv(OUTPUT_ROOT / "job_summary.csv", index=False)
    pd.DataFrame(event_rows).to_csv(OUTPUT_ROOT / "job_events.csv", index=False)
    (OUTPUT_ROOT / "integration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (OUTPUT_ROOT / "batch_response.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    (OUTPUT_ROOT / "retry_events.json").write_text(
        json.dumps(retry_events, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
