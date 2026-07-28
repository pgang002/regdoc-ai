from __future__ import annotations

from regdoc_ai.service.models import RedactionCandidate


def test_redaction_candidate_validates_action_and_box():
    item = RedactionCandidate(
        candidate_id="abc123",
        entity_type="PERSON",
        action="review",
        page=1,
        field_name="investigator",
        masked_text="A***n",
        confidence=0.8,
        bounding_box=[1, 2, 3, 4],
        detection_methods=["field_semantics"],
        needs_review=True,
    )
    assert item.action == "review"
    assert item.bounding_box == [1.0, 2.0, 3.0, 4.0]
