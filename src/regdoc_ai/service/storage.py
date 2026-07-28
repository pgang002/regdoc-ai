from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from .models import ArtifactLink, ProcessingResponse


class WorkspaceStore:
    """Filesystem-backed Day 8 store with path traversal protection."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def document_id_for(data: bytes, filename: str) -> str:
        digest = hashlib.sha256(data).hexdigest()[:16]
        clean_stem = "".join(ch for ch in Path(filename).stem if ch.isalnum() or ch in "-_")[:32]
        return f"{clean_stem or 'document'}-{digest}"

    def document_dir(self, document_id: str) -> Path:
        if not document_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in document_id):
            raise ValueError("Invalid document ID")
        path = (self.root / document_id).resolve()
        if self.root not in path.parents:
            raise ValueError("Invalid document path")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_upload(self, document_id: str, filename: str, data: bytes) -> Path:
        suffix = Path(filename).suffix.lower()
        path = self.document_dir(document_id) / f"source{suffix}"
        path.write_bytes(data)
        return path

    def result_path(self, document_id: str) -> Path:
        return self.document_dir(document_id) / "result.json"

    def private_state_path(self, document_id: str) -> Path:
        return self.document_dir(document_id) / "private_state.json"

    def save_result(self, result: ProcessingResponse) -> None:
        self.result_path(result.document_id).write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )

    def load_result(self, document_id: str) -> ProcessingResponse:
        path = self.result_path(document_id)
        if not path.exists():
            raise FileNotFoundError(document_id)
        return ProcessingResponse.model_validate_json(path.read_text(encoding="utf-8"))

    def save_private_state(self, document_id: str, payload: dict[str, Any]) -> None:
        self.private_state_path(document_id).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        try:
            os.chmod(self.private_state_path(document_id), 0o600)
        except OSError:
            pass

    def load_private_state(self, document_id: str) -> dict[str, Any]:
        path = self.private_state_path(document_id)
        if not path.exists():
            raise FileNotFoundError(document_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def resolve_artifact(self, document_id: str, artifact_name: str) -> Path:
        doc_dir = self.document_dir(document_id)
        if not artifact_name or Path(artifact_name).name != artifact_name:
            raise ValueError("Invalid artifact name")
        path = (doc_dir / artifact_name).resolve()
        if doc_dir not in path.parents or not path.is_file():
            raise FileNotFoundError(artifact_name)
        if path.name == "private_state.json":
            raise PermissionError("Private processing state is not downloadable")
        return path

    def artifact_link(self, document_id: str, path: str | Path, *, name: str | None = None) -> ArtifactLink:
        file_path = Path(path)
        media_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        return ArtifactLink(
            name=name or file_path.name,
            media_type=media_type,
            size_bytes=file_path.stat().st_size,
            download_path=f"/v1/documents/{document_id}/artifacts/{file_path.name}",
        )
