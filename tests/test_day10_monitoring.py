from pathlib import Path

from regdoc_ai.monitoring.health import readiness_report
from regdoc_ai.monitoring.metrics import MetricsManager
from regdoc_ai.persistence import Database, MetadataRepository


def test_metrics_render_contains_regdoc_names():
    metrics = MetricsManager()
    with metrics.track_request("GET", "/health") as holder:
        holder["status"] = 200
    metrics.update_operational(
        {
            "status_counts": {"completed": 2, "failed": 1},
            "processing_seconds": {"mean": 2.5, "p50": 2.0, "p95": 4.0, "max": 5.0},
            "artifact_count": 7,
            "artifact_bytes": 1024,
        }
    )
    payload, content_type = metrics.render()
    text = payload.decode("utf-8")
    assert "regdoc_http_requests_total" in text
    assert 'route="/health"' in text
    assert 'status="completed"' in text
    assert "text/plain" in content_type


def test_readiness_local_mode(tmp_path: Path):
    database = Database(f"sqlite:///{tmp_path / 'monitoring.db'}")
    database.create_schema()
    report = readiness_report(database=database, workspace=tmp_path / "workspace", queue_mode="thread")
    assert report["ready"] is True
    assert report["checks"]["database"]["ready"] is True
    assert report["checks"]["workspace"]["ready"] is True
    assert report["checks"]["queue"]["ready"] is True


def test_operational_summary_empty_database(tmp_path: Path):
    database = Database(f"sqlite:///{tmp_path / 'summary.db'}")
    database.create_schema()
    summary = MetadataRepository(database).operational_summary()
    assert summary["job_count"] == 0
    assert summary["success_rate"] == 0.0
    assert summary["artifact_count"] == 0
