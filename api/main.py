from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from regdoc_ai.monitoring import MetricsManager, readiness_report
from regdoc_ai.persistence import Database, MetadataRepository
from regdoc_ai.service.async_jobs import AsyncJobService
from regdoc_ai.service.models import (
    ApplyRedactionsRequest,
    ApplyRedactionsResponse,
    BatchResponse,
    BatchSubmissionResponse,
    JobEventResponse,
    JobResponse,
    ProcessingResponse,
)
from regdoc_ai.service.pipeline import DocumentPipeline
from regdoc_ai.service.review import apply_redaction_review
from regdoc_ai.service.storage import WorkspaceStore
from regdoc_ai.worker.dispatcher import TaskDispatcher

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(os.getenv("REGDOC_WORKSPACE", PROJECT_ROOT / "runtime/day9/documents"))
store = WorkspaceStore(WORKSPACE_ROOT)
pipeline = DocumentPipeline(PROJECT_ROOT, store)
database = Database.from_env(PROJECT_ROOT)
database.create_schema()
repository = MetadataRepository(database)
dispatcher = TaskDispatcher(os.getenv("REGDOC_QUEUE_MODE", "thread"))
async_service = AsyncJobService(store=store, repository=repository, dispatcher=dispatcher)
metrics = MetricsManager()

app = FastAPI(
    title="RegDocAI API",
    version="1.0.0",
    description=(
        "Production-oriented regulatory Document AI API with synchronous review, "
        "asynchronous processing, persistence, and operational monitoring."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv("REGDOC_CORS_ORIGINS", "http://localhost:8501").split(",")
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

def _metric_route(path: str) -> str:
    """Bound metric-cardinality for document, job, batch, and artifact identifiers."""
    parts = path.strip("/").split("/") if path != "/" else []
    normalized: list[str] = []
    replace_next = False
    for part in parts:
        if replace_next:
            normalized.append("{id}")
            replace_next = False
            continue
        normalized.append(part)
        if part in {"documents", "jobs", "batches"}:
            replace_next = True
        elif part == "artifacts":
            replace_next = True
    return "/" + "/".join(normalized) if normalized else "/"


@app.middleware("http")
async def observe_http(request: Request, call_next):  # type: ignore[no-untyped-def]
    route = _metric_route(request.url.path)
    with metrics.track_request(request.method, route) as holder:
        response = await call_next(request)
        holder["status"] = response.status_code
        return response


@app.get("/health/live")
def liveness() -> dict[str, object]:
    return {"status": "alive", "version": app.version}


@app.get("/health/ready")
def readiness(response: Response) -> dict[str, object]:
    report = readiness_report(database=database, workspace=WORKSPACE_ROOT, queue_mode=dispatcher.mode)
    metrics.update_readiness(report)
    if not report["ready"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return report


@app.get("/v1/operations/summary")
def operations_summary() -> dict[str, object]:
    summary = repository.operational_summary()
    metrics.update_operational(summary)
    return summary


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    summary = repository.operational_summary()
    metrics.update_operational(summary)
    report = readiness_report(database=database, workspace=WORKSPACE_ROOT, queue_mode=dispatcher.mode)
    metrics.update_readiness(report)
    payload, content_type = metrics.render()
    return Response(content=payload, media_type=content_type.split(";", 1)[0])


@app.get("/health")
def health() -> dict[str, object]:
    report = readiness_report(database=database, workspace=WORKSPACE_ROOT, queue_mode=dispatcher.mode)
    metrics.update_readiness(report)
    return {
        "status": "ok" if report["ready"] else "degraded",
        "version": app.version,
        "processing_mode": "production_day10",
        "queue_mode": dispatcher.mode,
        "readiness": report,
    }


@app.get("/model-info")
def model_info() -> dict[str, object]:
    return {
        "classifier": "hybrid OCR rules + HOG/LinearSVC fallback",
        "ocr": "Tesseract",
        "table_extraction": "OpenCV ruled-grid + Tesseract",
        "redaction_policy": {"name": pipeline.policy.name, "version": pipeline.policy.version},
        "metadata_store": "PostgreSQL via SQLAlchemy (SQLite local-test fallback)",
        "monitoring": "Prometheus metrics + liveness/readiness + operational summary",
        "task_queue": "Celery with Redis (thread/eager local-test fallback)",
        "supported_file_types": ["pdf", "png", "jpg", "jpeg", "tif", "tiff"],
        "max_upload_mb": 25,
    }


@app.post("/v1/documents/process", response_model=ProcessingResponse)
async def process_document(file: UploadFile = File(...)) -> ProcessingResponse:
    """Day 8-compatible synchronous endpoint retained for interactive review."""
    filename = file.filename or "upload.bin"
    try:
        data = await file.read()
        return pipeline.process(filename=filename, data=data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Processing failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        await file.close()


@app.post("/v1/jobs/process", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def enqueue_document(file: UploadFile = File(...)) -> JobResponse:
    filename = file.filename or "upload.bin"
    try:
        return async_service.submit(filename=filename, data=await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Queue submission failed: {exc}") from exc
    finally:
        await file.close()


@app.post(
    "/v1/batches/process",
    response_model=BatchSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def enqueue_batch(files: list[UploadFile] = File(...)) -> BatchSubmissionResponse:
    payload: list[tuple[str, bytes]] = []
    try:
        for item in files:
            payload.append((item.filename or "upload.bin", await item.read()))
        return async_service.submit_batch(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Batch submission failed: {exc}") from exc
    finally:
        for item in files:
            await item.close()


@app.get("/v1/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    try:
        return JobResponse.model_validate(repository.get_job(job_id))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")


@app.get("/v1/jobs/{job_id}/events", response_model=list[JobEventResponse])
def get_job_events(job_id: str) -> list[JobEventResponse]:
    try:
        return [JobEventResponse.model_validate(row) for row in repository.get_events(job_id)]
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")


@app.post("/v1/jobs/{job_id}/retry", response_model=JobResponse, status_code=202)
def retry_job(job_id: str) -> JobResponse:
    try:
        return async_service.retry(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Retry submission failed: {exc}") from exc


@app.get("/v1/batches/{batch_id}", response_model=BatchResponse)
def get_batch(batch_id: str) -> BatchResponse:
    try:
        return BatchResponse.model_validate(repository.get_batch(batch_id))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Batch not found")


@app.get("/v1/documents/{document_id}", response_model=ProcessingResponse)
def get_document(document_id: str) -> ProcessingResponse:
    try:
        return store.load_result(document_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail="Document not found")


@app.post(
    "/v1/documents/{document_id}/redactions/apply",
    response_model=ApplyRedactionsResponse,
)
def apply_redactions(document_id: str, request: ApplyRedactionsRequest) -> ApplyRedactionsResponse:
    try:
        return apply_redaction_review(document_id=document_id, request=request, store=store)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail="Document not found or decision invalid")
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Redaction failed: {type(exc).__name__}: {exc}"
        ) from exc


@app.get("/v1/documents/{document_id}/artifacts/{artifact_name}")
def download_artifact(document_id: str, artifact_name: str) -> FileResponse:
    try:
        path = store.resolve_artifact(document_id, artifact_name)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path, filename=path.name)
