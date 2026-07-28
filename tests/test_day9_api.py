from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


def test_day9_health_exposes_queue_and_database():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["version"] == "1.0.0"
    assert payload["readiness"]["checks"]["database"]["ready"] is True
    assert payload["queue_mode"] in {"celery", "thread", "eager"}


def test_unknown_job_and_batch_return_404():
    client = TestClient(app)
    assert client.get("/v1/jobs/not-a-job").status_code == 404
    assert client.get("/v1/batches/not-a-batch").status_code == 404
