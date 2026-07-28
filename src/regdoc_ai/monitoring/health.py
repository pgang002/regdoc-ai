from __future__ import annotations

import os
import shutil
import socket
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from regdoc_ai.persistence import Database


def _workspace_check(workspace: Path) -> dict[str, object]:
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=workspace, prefix="health-", delete=True) as handle:
            handle.write(b"ok")
            handle.flush()
        return {"ready": True, "path": str(workspace)}
    except Exception as exc:
        return {"ready": False, "path": str(workspace), "detail": str(exc)}


def _binary_check(name: str) -> dict[str, object]:
    path = shutil.which(name)
    return {"ready": path is not None, "path": path}


def _tcp_check(url: str | None) -> dict[str, object]:
    if not url:
        return {"ready": False, "detail": "URL not configured"}
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or 6379
    if not host:
        return {"ready": False, "detail": "URL has no host"}
    try:
        with socket.create_connection((host, port), timeout=0.5):
            pass
        return {"ready": True, "host": host, "port": port}
    except OSError as exc:
        return {"ready": False, "host": host, "port": port, "detail": str(exc)}


def readiness_report(
    *, database: Database, workspace: str | Path, queue_mode: str
) -> dict[str, object]:
    checks: dict[str, dict[str, object]] = {
        "database": {"ready": database.healthcheck(), "backend": database.url.split(":", 1)[0]},
        "workspace": _workspace_check(Path(workspace)),
        "tesseract": _binary_check("tesseract"),
    }
    if queue_mode == "celery":
        checks["redis_broker"] = _tcp_check(os.getenv("REGDOC_REDIS_BROKER_URL"))
    else:
        checks["queue"] = {"ready": True, "mode": queue_mode}
    ready = all(bool(item.get("ready")) for item in checks.values())
    return {"status": "ready" if ready else "not_ready", "ready": ready, "checks": checks}
