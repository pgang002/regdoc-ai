from __future__ import annotations

import os
from pathlib import Path

from regdoc_ai.persistence import Database, MetadataRepository
from regdoc_ai.service.pipeline import DocumentPipeline
from regdoc_ai.service.storage import WorkspaceStore


class WorkerRuntime:
    def __init__(self, project_root: str | Path | None = None):
        self.project_root = Path(
            project_root or os.getenv("REGDOC_PROJECT_ROOT", Path(__file__).resolve().parents[3])
        ).resolve()
        workspace = Path(
            os.getenv("REGDOC_WORKSPACE", self.project_root / "runtime/day9/documents")
        )
        self.store = WorkspaceStore(workspace)
        self.database = Database.from_env(self.project_root)
        self.database.create_schema()
        self.repository = MetadataRepository(self.database)
        self.pipeline = DocumentPipeline(self.project_root, self.store)
