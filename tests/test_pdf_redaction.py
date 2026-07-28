from pathlib import Path

import fitz

from regdoc_ai.redaction.models import DetectedEntity, RedactionAction
from regdoc_ai.redaction.pdf_redactor import redact_pdf, verify_redaction_regions


def test_true_pdf_redaction_removes_underlying_text(tmp_path: Path) -> None:
    source = tmp_path / 'source.pdf'
    output = tmp_path / 'redacted.pdf'
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((50, 80), 'Sensitive Person', fontsize=14)
    doc.save(source)
    doc.close()

    entity = DetectedEntity(
        entity_type='PERSON',
        action=RedactionAction.REDACT,
        page=1,
        field_name='name',
        detected_text='Sensitive Person',
        confidence=0.99,
        bbox_pdf=(45, 62, 180, 86),
        detection_methods=('test',),
    )
    redact_pdf(source, output, [entity])
    verification = verify_redaction_regions(source, output, [entity])
    assert verification[0]['verification_passed'] is True
    redacted = fitz.open(output)
    assert 'Sensitive Person' not in redacted[0].get_text()
    redacted.close()
