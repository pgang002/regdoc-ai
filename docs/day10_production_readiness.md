# Day 10: production readiness, monitoring, and final evidence package

## Objective

Day 10 closes the portfolio project by adding operational monitoring, deployment checks,
a consolidated evaluation scorecard, CI configuration, and resume-ready evidence. No
previous result was replaced or recomputed with substitute data.

## Monitoring implementation

The FastAPI service is now version `1.0.0` and exposes:

| Endpoint | Purpose |
|---|---|
| `GET /health/live` | Process liveness |
| `GET /health/ready` | Database, workspace, OCR binary, and queue/broker readiness |
| `GET /health` | Backward-compatible aggregate health response |
| `GET /v1/operations/summary` | Database-backed job, status, latency, and artifact summary |
| `GET /metrics` | Prometheus exposition endpoint |

The HTTP middleware records request counts, latency histograms, and in-flight requests.
Dynamic document, job, batch, and artifact identifiers are normalized before becoming
Prometheus labels to prevent high-cardinality metrics and accidental identifier exposure.

Prometheus gauges are also refreshed from persisted SQLAlchemy records for:

- Jobs by status
- Mean, median, 95th-percentile, and maximum processing time
- Artifact count and bytes
- Dependency readiness

## Deployment configuration

The Docker Compose stack includes:

- PostgreSQL 16
- Redis with append-only persistence
- FastAPI service with readiness health check
- Celery worker
- Streamlit review interface
- Prometheus scraper

The UI waits for the API readiness health check. Prometheus scrapes the API every 15
seconds from `/metrics`. A GitHub Actions workflow runs compilation, Ruff, the test suite,
and a Docker image build.

Docker was not available in the execution environment. The Compose and Prometheus YAML
files were parsed and statically validated, but container startup is not claimed.

## Actual monitoring validation

The final validation processed the same actual-data pair used for the asynchronous
milestone:

1. The populated official FDA 1572 associated with public NCT04796896 metadata
2. A real table cropped from the public NCT04796896 Moderna protocol

The local two-worker test mode was used because Redis/Celery/PostgreSQL services were not
available.

| Measure | Result |
|---|---:|
| API version | 1.0.0 |
| Liveness | Passed |
| Readiness | Passed |
| Documents | 2 |
| Terminal jobs | 2 of 2 |
| Final statuses | 1 completed, 1 needs review |
| Observed batch time | 16.78 seconds |
| Persisted artifacts | 12 |
| Persisted artifact bytes | 6,192,869 |
| Operational success rate | 100% |
| Operational failure rate | 0% |
| Mean persisted job time | 12.16 seconds |
| P95 persisted job time | 16.06 seconds |
| OpenAPI paths | 15 |

The metric output contained all required families:

- `regdoc_http_requests_total`
- `regdoc_http_request_duration_seconds`
- `regdoc_jobs`
- `regdoc_job_processing_seconds`
- `regdoc_dependency_ready`

## Consolidated measured scorecard

| Capability | Result | Evaluation scope |
|---|---:|---|
| Degraded-form field exact match | 96.43% | 140 field instances across seven degradations |
| Field exact-match error reduction | 84.85% | Enhanced versus raw OCR |
| Checkbox accuracy | 100% | 77 degraded checkbox instances |
| Clean table exact grid shape | 100% | 20 real Moderna protocol tables |
| Physical-cell F1 | 0.925 | 20 real protocol tables |
| Held-out document classification | 100% | 96 page instances across eight conditions |
| Routed field exact match | 98.21% | 224 field instances |
| Sensitive-entity F1 | 1.00 | 168 expected entities across eight conditions |
| Automatic-redaction coverage | 97.92% | Fields requiring redaction |
| False automatic redactions | 0% | Hybrid policy benchmark |

## Resume evidence

The final resume bullets in `results/day10_final/resume_bullets.md` use only completed,
measured results. The infrastructure bullet distinguishes implemented production
components from the locally executed queue/database fallback. No PaddleOCR, Table
Transformer, PP-StructureV3, pretrained MobileNetV3, Docker-service, Celery-service, or
PostgreSQL-service metric is claimed without execution.

## Verification

- 53 automated tests pass.
- Python source compiles successfully.
- Alembic migration remains part of the repository.
- Deployment configuration passes static validation.
- Actual-data asynchronous monitoring validation passes.
- Consolidated metrics, scorecard figure, limitations, and resume bullets are generated
  reproducibly by `scripts/create_final_scorecard.py`.
