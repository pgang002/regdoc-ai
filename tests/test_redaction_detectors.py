from regdoc_ai.redaction.detectors import expected_entity_types


def test_expected_person_and_protocol_entities() -> None:
    assert expected_entity_types('db_invest_name', 'Maya Chen, MD') == ('PERSON',)
    types = expected_entity_types(
        'db_prot_name_code',
        'mRNA-1273-P301 - A Phase 3 Vaccine Study',
    )
    assert 'CCI_STUDY_TITLE' in types
    assert 'CCI_PROTOCOL_ID' in types


def test_address_and_sponsor_mapping() -> None:
    assert expected_entity_types('db_inv_zip', '02115') == ('ADDRESS',)
    assert expected_entity_types('topmost.appFirm[0]', 'ModernaTX, Inc.') == ('CCI_SPONSOR',)
