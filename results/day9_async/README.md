# Day 9 asynchronous-processing results

- `integration_summary.json`: measured batch, throughput, retry, and database-count summary.
- `job_summary.csv`: one row per actual batch job with extraction and artifact counts.
- `job_events.csv`: ordered persisted progress events for the actual batch.
- `batch_response.json`: final aggregate batch response.
- `retry_events.json`: failure, retry, and successful completion history.
- `api_async_verification.json`: HTTP 202 submission and polling verification.
- `openapi.json`: generated Day 9 OpenAPI schema.
- `endpoint_catalog.csv`: compact endpoint inventory.
- `job_processing_time.png`: actual per-job processing duration.
- `job_progress_events.png`: persisted stage progress.
- `architecture.png`: PostgreSQL/Redis/Celery deployment architecture.

The production PostgreSQL/Redis/Celery stack is configured in `docker-compose.yml`. Metrics
in this directory were generated with the documented SQLite/thread local validation mode
because those external services and Python packages were unavailable in the restricted runtime.
