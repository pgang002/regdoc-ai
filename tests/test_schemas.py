from regdoc_ai.schemas.document import DocumentResult, ProcessingStatus


def test_document_result_defaults() -> None:
    result = DocumentResult(
        document_id="doc-001",
        source_filename="FDA_1572.pdf",
        status=ProcessingStatus.completed,
    )
    assert result.fields == []
    assert result.status.value == "COMPLETED"
