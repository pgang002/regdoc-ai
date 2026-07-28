from regdoc_ai.extraction.protocol_fields import extract_protocol_cover_fields

SAMPLE_TEXT = """
NCT #: NCTO4796896
CLINICAL STUDY PROTOCOL
Protocol Title: A Phase 2/3, Three-Part, Open-Label Study in Healthy Children
Protocol Number: mRNA-1273-P204
Sponsor Name: ModernaTX, Inc.
Legal Registered Address: 200 Technology Square Cambridge, MA 02139
Sponsor Contact and Medical Monitor: PPD
Amendment Number: 9
Date of Amendment 9: 04 Aug 2022
"""


def test_protocol_cover_extraction_and_metadata_resolution() -> None:
    known = {
        "nct_id": "NCT04796896",
        "protocol_number": "mRNA-1273-P204",
        "sponsor_name": "ModernaTX, Inc.",
        "sponsor_address": "200 Technology Square Cambridge, MA 02139",
        "amendment_number": "9",
        "amendment_date": "04 Aug 2022",
        "protocol_title": "A Phase 2/3, Three-Part, Open-Label Study in Healthy Children",
        "phase": "Phase 2/3",
    }
    result = extract_protocol_cover_fields(SAMPLE_TEXT, known_metadata=known)
    assert result.fields == known
    assert result.warnings == []


def test_protocol_cover_normalizes_ocr_confused_nct_zero() -> None:
    result = extract_protocol_cover_fields(SAMPLE_TEXT)
    assert result.fields["nct_id"] == "NCT04796896"
    assert result.fields["phase"] == "Phase 2/3"
