from pathlib import Path

from regdoc_ai.redaction.models import DetectedEntity, RedactionAction
from regdoc_ai.redaction.policy import RedactionPolicy


def test_policy_loads_actions() -> None:
    policy = RedactionPolicy.from_yaml(Path('configs/redaction_policy.yaml'))
    assert policy.entity('PERSON').action == RedactionAction.REDACT
    assert policy.entity('DATE').action == RedactionAction.REVIEW
    assert policy.entity('CCI_PROTOCOL_ID').action == RedactionAction.REDACT


def test_audit_dict_does_not_expose_raw_text() -> None:
    entity = DetectedEntity(
        entity_type='PERSON',
        action=RedactionAction.REDACT,
        page=1,
        field_name='investigator_name',
        detected_text='Maya Chen, MD',
        confidence=0.98,
        bbox_pdf=(10, 20, 100, 40),
        detection_methods=('ner',),
    )
    audit = entity.to_audit_dict()
    assert 'detected_text' not in audit
    assert 'Maya Chen, MD' not in str(audit)
    assert len(audit['text_sha256']) == 64
