from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


def test_health_and_model_info():
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    info = client.get("/model-info")
    assert info.status_code == 200
    assert "pdf" in info.json()["supported_file_types"]


def test_process_rejects_unsupported_extension():
    client = TestClient(app)
    response = client.post(
        "/v1/documents/process",
        files={"file": ("malware.exe", b"not a document", "application/octet-stream")},
    )
    assert response.status_code == 422
    assert "Unsupported file type" in response.json()["detail"]


def test_artifact_path_traversal_is_not_exposed():
    client = TestClient(app)
    response = client.get("/v1/documents/not-found/artifacts/..%2Fprivate_state.json")
    assert response.status_code in {404, 422}
