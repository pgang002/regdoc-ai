from __future__ import annotations

import pytest

from regdoc_ai.service.storage import WorkspaceStore


def test_workspace_store_rejects_path_traversal(tmp_path):
    store = WorkspaceStore(tmp_path)
    with pytest.raises(ValueError):
        store.document_dir("../escape")
    with pytest.raises(ValueError):
        store.resolve_artifact("safe-id", "../secret.txt")


def test_private_state_is_not_downloadable(tmp_path):
    store = WorkspaceStore(tmp_path)
    store.save_private_state("document-1", {"entities": []})
    with pytest.raises(PermissionError):
        store.resolve_artifact("document-1", "private_state.json")


def test_document_id_is_deterministic():
    first = WorkspaceStore.document_id_for(b"same bytes", "Sample File.pdf")
    second = WorkspaceStore.document_id_for(b"same bytes", "Sample File.pdf")
    assert first == second
    assert first.startswith("SampleFile-")
