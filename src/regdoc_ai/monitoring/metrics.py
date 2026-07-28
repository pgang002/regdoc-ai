from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST


class MetricsManager:
    """Application-scoped Prometheus metrics.

    A private registry avoids duplicate collectors when FastAPI applications are
    created repeatedly by the test suite.
    """

    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        self.http_requests = Counter(
            "regdoc_http_requests_total",
            "HTTP requests received by the API.",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.http_latency = Histogram(
            "regdoc_http_request_duration_seconds",
            "HTTP request latency in seconds.",
            ["method", "route"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 15, 30),
            registry=self.registry,
        )
        self.in_flight = Gauge(
            "regdoc_http_requests_in_flight",
            "HTTP requests currently being handled.",
            registry=self.registry,
        )
        self.jobs = Gauge(
            "regdoc_jobs",
            "Persisted processing jobs grouped by status.",
            ["status"],
            registry=self.registry,
        )
        self.job_processing_seconds = Gauge(
            "regdoc_job_processing_seconds",
            "Observed job processing-time statistics.",
            ["statistic"],
            registry=self.registry,
        )
        self.artifacts = Gauge(
            "regdoc_artifacts_total",
            "Persisted document artifacts.",
            registry=self.registry,
        )
        self.artifact_bytes = Gauge(
            "regdoc_artifact_bytes_total",
            "Total bytes represented by persisted artifacts.",
            registry=self.registry,
        )
        self.readiness = Gauge(
            "regdoc_dependency_ready",
            "Dependency readiness (1 ready, 0 unavailable).",
            ["dependency"],
            registry=self.registry,
        )

    @contextmanager
    def track_request(self, method: str, route: str) -> Iterator[dict[str, int]]:
        started = time.perf_counter()
        status_holder = {"status": 500}
        self.in_flight.inc()
        try:
            yield status_holder
        finally:
            elapsed = max(0.0, time.perf_counter() - started)
            status = str(status_holder["status"])
            self.http_requests.labels(method=method, route=route, status=status).inc()
            self.http_latency.labels(method=method, route=route).observe(elapsed)
            self.in_flight.dec()

    def update_operational(self, summary: dict[str, object]) -> None:
        statuses = summary.get("status_counts", {})
        if isinstance(statuses, dict):
            for status in (
                "queued",
                "retrying",
                "preprocessing",
                "classifying",
                "extracting",
                "redacting",
                "persisting",
                "completed",
                "needs_review",
                "failed",
            ):
                self.jobs.labels(status=status).set(float(statuses.get(status, 0)))
        processing = summary.get("processing_seconds", {})
        if isinstance(processing, dict):
            for statistic in ("mean", "p50", "p95", "max"):
                value = processing.get(statistic)
                self.job_processing_seconds.labels(statistic=statistic).set(
                    float(value or 0.0)
                )
        self.artifacts.set(float(summary.get("artifact_count", 0) or 0))
        self.artifact_bytes.set(float(summary.get("artifact_bytes", 0) or 0))

    def update_readiness(self, report: dict[str, object]) -> None:
        checks = report.get("checks", {})
        if not isinstance(checks, dict):
            return
        for name, check in checks.items():
            ready = bool(check.get("ready")) if isinstance(check, dict) else bool(check)
            self.readiness.labels(dependency=str(name)).set(1.0 if ready else 0.0)

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
