from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNTIME = ROOT / "runtime/day10_validation"
RESULTS = ROOT / "results/day10_final"


def main() -> None:
    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)
    RUNTIME.mkdir(parents=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    os.environ["REGDOC_DATABASE_URL"] = f"sqlite:///{(RUNTIME / 'regdoc.db').as_posix()}"
    os.environ["REGDOC_QUEUE_MODE"] = "thread"
    os.environ["REGDOC_WORKSPACE"] = str(RUNTIME / "documents")

    from fastapi.testclient import TestClient
    from api.main import app, dispatcher

    client = TestClient(app)
    live = client.get("/health/live")
    ready = client.get("/health/ready")

    samples = [
        ROOT / "data/processed/populated_forms/flattened/FDA_1572_NCT04796896.pdf",
        ROOT / "data/processed/table_benchmark/images/NCT04796896_p116_t0.png",
    ]
    files = [("files", (p.name, p.read_bytes(), "application/pdf" if p.suffix == ".pdf" else "image/png")) for p in samples]
    started = time.perf_counter()
    submission = client.post("/v1/batches/process", files=files)
    submission.raise_for_status()
    batch_id = submission.json()["batch_id"]
    deadline = time.time() + 120
    batch = None
    while time.time() < deadline:
        response = client.get(f"/v1/batches/{batch_id}")
        response.raise_for_status()
        batch = response.json()
        if batch["completed_count"] == batch["job_count"]:
            break
        time.sleep(0.25)
    if batch is None or batch["completed_count"] != batch["job_count"]:
        raise RuntimeError("Day 10 validation batch did not reach a terminal state")
    elapsed = time.perf_counter() - started

    for _ in range(10):
        client.get("/health/live")
    operations = client.get("/v1/operations/summary")
    metrics = client.get("/metrics")
    model_info = client.get("/model-info")
    if not all(r.status_code == 200 for r in (live, ready, operations, metrics, model_info)):
        raise RuntimeError("Monitoring endpoint validation failed")

    metrics_path = RESULTS / "prometheus_metrics.txt"
    metrics_path.write_text(metrics.text)
    op = operations.json()
    (RESULTS / "operational_summary.json").write_text(json.dumps(op, indent=2) + "\n")
    openapi = client.get("/openapi.json").json()
    (RESULTS / "openapi.json").write_text(json.dumps(openapi, indent=2) + "\n")
    with (RESULTS / "endpoint_catalog.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "path", "summary"])
        writer.writeheader()
        for path, operations in sorted(openapi["paths"].items()):
            for method, specification in sorted(operations.items()):
                writer.writerow({
                    "method": method.upper(),
                    "path": path,
                    "summary": specification.get("summary", ""),
                })

    output = {
        "version": live.json()["version"],
        "liveness_status": live.json()["status"],
        "readiness": ready.json(),
        "batch_id": batch_id,
        "job_count": batch["job_count"],
        "status_counts": batch["status_counts"],
        "observed_batch_seconds": elapsed,
        "operational_summary": op,
        "prometheus_metric_families_present": [
            name
            for name in [
                "regdoc_http_requests_total",
                "regdoc_http_request_duration_seconds",
                "regdoc_jobs",
                "regdoc_job_processing_seconds",
                "regdoc_dependency_ready",
            ]
            if name in metrics.text
        ],
        "openapi_path_count": len(openapi["paths"]),
        "queue_mode": model_info.json()["task_queue"],
    }
    (RESULTS / "monitoring_validation.json").write_text(json.dumps(output, indent=2) + "\n")
    dispatcher.shutdown()
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
