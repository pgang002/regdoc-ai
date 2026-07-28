# Day 9: Persistent asynchronous processing

## Objective

Day 9 moves the Day 8 application from request-bound processing to a persistent job
architecture. PostgreSQL is the production metadata store, Redis is the Celery broker and
result backend, and the existing object-style workspace continues to hold source documents,
previews, tables, audit logs, and redacted PDFs.

The OCR and extraction implementation is unchanged. The synchronous and asynchronous
interfaces call the same `DocumentPipeline`, which now emits progress callbacks at stable
processing boundaries.

## Architecture

```text
Streamlit / API client
        |
        +-- POST /v1/jobs/process or /v1/batches/process
        v
FastAPI submission service
        |-- saves source in object-style workspace
        |-- inserts document + job metadata in PostgreSQL
        v
Redis broker ---> Celery worker(s)
                       |
                       +-- classification / OCR / fields / tables / redaction
                       +-- persistent stage events and retry state
                       v
                PostgreSQL metadata
                filesystem artifacts
```

The Docker Compose configuration starts PostgreSQL, Redis with append-only persistence,
FastAPI, a two-process Celery worker, and Streamlit. The worker uses late acknowledgements,
a one-task prefetch multiplier, worker-loss rejection, connection retry on startup, and a
one-hour Redis visibility timeout.

## Database schema

The SQLAlchemy and Alembic schema contains four tables:

- `documents`: source identity, checksum, workspace location, page count, status, result JSON,
  and document metadata.
- `processing_jobs`: batch linkage, stage, progress, attempts, retry limit, Celery task ID,
  errors, and timing fields.
- `job_events`: immutable ordered progress events with stage-specific details.
- `artifacts`: downloadable artifact metadata linked to both a document and processing job.

A database session is created per repository operation. Sessions are not shared across API
requests or worker tasks.

## API additions

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/v1/jobs/process` | Queue one document and return HTTP 202 |
| POST | `/v1/batches/process` | Queue multiple documents under one batch ID |
| GET | `/v1/jobs/{job_id}` | Return job status, stage, progress, attempts, and errors |
| GET | `/v1/jobs/{job_id}/events` | Return ordered persisted job events |
| POST | `/v1/jobs/{job_id}/retry` | Requeue an eligible failed job |
| GET | `/v1/batches/{batch_id}` | Return aggregate and per-job batch status |

The Day 8 synchronous endpoint remains available for immediate interactive review.

## Progress and failure handling

The processing pipeline emits these stages:

1. uploaded
2. preprocessing
3. classifying
4. extracting
5. redacting
6. persisting
7. completed or needs_review

Page numbers and routed document labels are saved in stage-event details. Errors are stored
as bounded type/message fields; raw Python tracebacks are not returned by the job API.

Celery retries transient `OSError`, `TimeoutError`, and `ConnectionError` failures up to two
times with increasing delays. The API also exposes an explicit retry endpoint for failed jobs.

## Actual-data integration run

The local execution environment did not provide downloadable Celery, Redis client, or
PostgreSQL driver packages, and it did not expose Redis/PostgreSQL services. Therefore, the
production Compose stack was implemented but not falsely described as executed.

The complete persistence and worker service layer was executed with its documented local
validation mode:

- SQLAlchemy with SQLite instead of PostgreSQL
- a two-worker thread dispatcher instead of Celery/Redis
- the same `DocumentPipeline`, repository, job schema, progress callbacks, and artifacts

Actual project inputs were used:

- populated official FDA 1572 for NCT04796896
- a real table image from the public NCT04796896 Moderna protocol
- a real NCT04796896 protocol-cover page for the failure/retry test

### Batch result

| Measure | Result |
|---|---:|
| Documents submitted | 2 |
| Terminal jobs | 2 |
| Batch completion | 100% |
| Observed batch time | 14.43 seconds |
| Local two-worker throughput | 8.32 documents/minute |
| Persisted batch events | 20 |
| Registered batch artifacts | 12 |

The two-page FDA form finished in 14.15 seconds with 28 fields, 4 checkboxes, 29 redaction
candidates, and 5 registered artifacts. Its terminal state was `needs_review`, which is the
correct policy outcome. The real protocol table finished in 4.17 seconds with one reconstructed
table and 7 registered artifacts.

Progress remained monotonic for both batch jobs. The form persisted 11 events and the table
persisted 9 events.

### Retry result

The retry test intentionally removed the source file after job creation. The first attempt
failed with a persisted `FileNotFoundError`. The same actual protocol image was restored and
the job was requeued.

| Measure | Result |
|---|---:|
| Attempts | 2 |
| Final status | completed |
| Persisted events | 11 |
| Page count | 1 |
| Registered artifacts | 4 |

The final SQLite validation database contained 3 documents, 3 jobs, 31 job events, and 16
artifact records. Its health check passed.

## HTTP validation

A real protocol table was also submitted through `POST /v1/jobs/process` using FastAPI's test
client. The endpoint returned HTTP 202, the job progressed to `completed`, nine events were
available, and the resulting document contained one page, one table, and seven artifacts.
The generated OpenAPI schema contains 12 paths.

## Migration and configuration validation

- All YAML configuration files and `docker-compose.yml` parsed successfully.
- The initial Alembic migration upgraded a fresh database successfully.
- The current API OpenAPI schema was generated successfully.
- All 48 automated tests pass.

## Reproduction

### Local validation without external services

```powershell
$env:REGDOC_QUEUE_MODE="thread"
$env:REGDOC_DATABASE_URL="sqlite:///runtime/day9/regdoc_ai.db"
python scripts/init_database.py
python scripts/run_day9_integration.py
python scripts/create_day9_figures.py
pytest
```

### Production-style Docker stack

```powershell
docker compose up --build
```

FastAPI is exposed on port 8000 and Streamlit on port 8501. PostgreSQL and Redis are not
published to the host by default.

## Limitations

- PostgreSQL, Redis, and Celery were not executable in the restricted development runtime;
  no production-service latency or throughput is claimed.
- File storage is an object-storage abstraction on a shared Docker volume, not S3 or Azure
  Blob Storage.
- Celery retries are limited to transient infrastructure-oriented exception classes; future
  error analysis may add narrowly defined OCR/model retry categories.
- The current batch endpoint reads uploaded files before submitting them. Very large enterprise
  submissions should use direct object-storage upload URLs and enqueue only immutable object IDs.
