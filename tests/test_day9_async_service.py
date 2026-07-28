from __future__ import annotations

from regdoc_ai.persistence import Database, MetadataRepository
from regdoc_ai.service.async_jobs import AsyncJobService
from regdoc_ai.service.storage import WorkspaceStore


class FakeDispatcher:
    mode = "celery"

    def __init__(self):
        self.jobs = []

    def dispatch(self, job_id: str) -> str:
        self.jobs.append(job_id)
        return f"task-{job_id}"


def test_async_submission_persists_source_job_and_task_id(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'metadata.db'}")
    database.create_schema()
    repository = MetadataRepository(database)
    store = WorkspaceStore(tmp_path / "workspace")
    dispatcher = FakeDispatcher()
    service = AsyncJobService(store=store, repository=repository, dispatcher=dispatcher)  # type: ignore[arg-type]
    response = service.submit(filename="form.pdf", data=b"%PDF-1.4 test")
    job = repository.get_job(response.job_id)
    assert response.status == "queued"
    assert dispatcher.jobs == [response.job_id]
    assert job["celery_task_id"] == f"task-{response.job_id}"
    assert list(store.document_dir(response.document_id).glob("source.pdf"))


def test_batch_submission_uses_one_batch_id(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'metadata.db'}")
    database.create_schema()
    repository = MetadataRepository(database)
    service = AsyncJobService(
        store=WorkspaceStore(tmp_path / "workspace"),
        repository=repository,
        dispatcher=FakeDispatcher(),  # type: ignore[arg-type]
    )
    batch = service.submit_batch(
        [("first.pdf", b"%PDF first"), ("second.png", b"png bytes")]
    )
    assert len(batch.jobs) == 2
    assert {job.batch_id for job in batch.jobs} == {batch.batch_id}
    assert repository.get_batch(batch.batch_id)["job_count"] == 2
