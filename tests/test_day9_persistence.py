from __future__ import annotations

from regdoc_ai.persistence import Database, MetadataRepository


def build_repo(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'metadata.db'}")
    database.create_schema()
    return database, MetadataRepository(database)


def test_job_lifecycle_and_ordered_events(tmp_path):
    database, repository = build_repo(tmp_path)
    repository.upsert_document(
        document_id="doc-1",
        source_filename="sample.pdf",
        source_sha256="a" * 64,
        workspace_path=tmp_path / "doc-1",
    )
    repository.create_job(
        job_id="job-1",
        document_id="doc-1",
        batch_id="batch-1",
        queue_mode="eager",
        max_retries=2,
    )
    repository.mark_attempt_started("job-1")
    repository.append_event(
        "job-1",
        status="preprocessing",
        stage="preprocessing",
        progress=10,
        message="Rendering pages.",
    )
    repository.append_event(
        "job-1",
        status="completed",
        stage="completed",
        progress=100,
        message="Done.",
    )
    job = repository.get_job("job-1")
    events = repository.get_events("job-1")
    batch = repository.get_batch("batch-1")
    assert database.healthcheck()
    assert job["status"] == "completed"
    assert job["attempt_count"] == 1
    assert [row["sequence"] for row in events] == [1, 2, 3]
    assert [row["progress"] for row in events] == [0, 10, 100]
    assert batch["progress"] == 100.0


def test_failed_job_records_error_without_raw_traceback(tmp_path):
    _, repository = build_repo(tmp_path)
    repository.upsert_document(
        document_id="doc-2",
        source_filename="sample.pdf",
        source_sha256="b" * 64,
        workspace_path=tmp_path / "doc-2",
    )
    repository.create_job(
        job_id="job-2",
        document_id="doc-2",
        batch_id=None,
        queue_mode="eager",
        max_retries=2,
    )
    repository.mark_failure("job-2", RuntimeError("temporary OCR failure"))
    job = repository.get_job("job-2")
    assert job["status"] == "failed"
    assert job["error_type"] == "RuntimeError"
    assert job["error_message"] == "temporary OCR failure"
