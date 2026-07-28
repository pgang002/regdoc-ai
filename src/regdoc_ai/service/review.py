from __future__ import annotations

import json
from pathlib import Path

from regdoc_ai.redaction.models import DetectedEntity, RedactionAction
from regdoc_ai.redaction.pdf_redactor import redact_pdf

from .models import ApplyRedactionsRequest, ApplyRedactionsResponse
from .storage import WorkspaceStore


def apply_redaction_review(
    *,
    document_id: str,
    request: ApplyRedactionsRequest,
    store: WorkspaceStore,
) -> ApplyRedactionsResponse:
    state = store.load_private_state(document_id)
    decisions = {item.candidate_id: item.action for item in request.decisions}
    entities: list[DetectedEntity] = []
    counts = {"redact": 0, "review": 0, "retain": 0, "ignore": 0}
    audit_entities = []
    for item in state.get("entities", []):
        candidate_id = str(item["candidate_id"])
        action_value = decisions.get(candidate_id, str(item["action"]))
        action = RedactionAction(action_value)
        counts[action.value] += 1
        entity = DetectedEntity(
            entity_type=str(item["entity_type"]),
            action=action,
            page=int(item["page"]),
            field_name=str(item["field_name"]),
            detected_text="",
            confidence=float(item["confidence"]),
            bbox_pdf=tuple(float(v) for v in item["bbox_pdf"]),
            detection_methods=tuple(str(v) for v in item.get("detection_methods", [])),
            policy_rule=str(item.get("policy_rule", "")),
            needs_review=action == RedactionAction.REVIEW,
        )
        entities.append(entity)
        audit_entities.append(
            {
                "candidate_id": candidate_id,
                "entity_type": entity.entity_type,
                "action": action.value,
                "page": entity.page,
                "field_name": entity.field_name,
                "confidence": entity.confidence,
                "bbox_pdf": list(entity.bbox_pdf),
                "detection_methods": list(entity.detection_methods),
            }
        )
    doc_dir = store.document_dir(document_id)
    output_pdf = doc_dir / "reviewed_redacted.pdf"
    metadata = redact_pdf(Path(state["source_pdf"]), output_pdf, entities)
    audit_path = doc_dir / "reviewed_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "document_id": document_id,
                "redaction_metadata": metadata,
                "decisions": audit_entities,
                "audit_note": "Review artifact stores decisions and coordinates, not raw sensitive text.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ApplyRedactionsResponse(
        document_id=document_id,
        redacted_count=counts["redact"],
        review_count=counts["review"],
        retained_count=counts["retain"],
        ignored_count=counts["ignore"],
        artifact=store.artifact_link(document_id, output_pdf),
        audit_artifact=store.artifact_link(document_id, audit_path),
    )
