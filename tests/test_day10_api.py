from fastapi.testclient import TestClient

from api.main import app


def test_liveness_readiness_operations_and_metrics():
    client = TestClient(app)
    live = client.get("/health/live")
    assert live.status_code == 200
    assert live.json() == {"status": "alive", "version": "1.0.0"}

    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["ready"] is True

    operations = client.get("/v1/operations/summary")
    assert operations.status_code == 200
    assert "status_counts" in operations.json()
    assert "processing_seconds" in operations.json()

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "regdoc_http_requests_total" in metrics.text
    assert "regdoc_dependency_ready" in metrics.text


def test_metric_route_does_not_expose_ids():
    client = TestClient(app)
    client.get("/v1/jobs/not-a-job")
    metrics = client.get("/metrics").text
    assert "not-a-job" not in metrics
    assert '/v1/jobs/{id}' in metrics
