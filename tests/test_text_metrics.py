from regdoc_ai.evaluation.text_metrics import character_error_rate, normalize_text, word_error_rate


def test_normalize_text() -> None:
    assert normalize_text("Form FDA-1572\nStatement") == "form fda 1572 statement"


def test_identical_text_has_zero_error() -> None:
    assert character_error_rate("abc", "abc") == 0
    assert word_error_rate("one two", "one two") == 0


def test_word_error_rate_detects_substitution() -> None:
    assert word_error_rate("one two three", "one four three") == 1 / 3
