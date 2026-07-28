from regdoc_ai.extraction.field_validation import validate_field_value


def test_zip_preserves_leading_zero() -> None:
    assert validate_field_value("db_inv_zip", "02115") == "02115"


def test_state_code_cleanup() -> None:
    assert validate_field_value("db_irb_state", "MA :") == "MA"


def test_sponsor_entity_resolution() -> None:
    assert (
        validate_field_value(
            "topmostSubform[0].Page1[0].appFirm[0]",
            "ModernaTlX, Inc.",
            known_sponsor="ModernaTX, Inc.",
        )
        == "ModernaTX, Inc."
    )


def test_domain_token_cleanup() -> None:
    assert validate_field_value("db_prot_name_code", "mMRNA-1273 study") == "mRNA-1273 study"
