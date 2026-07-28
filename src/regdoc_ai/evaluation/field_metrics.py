from __future__ import annotations

from dataclasses import dataclass

from regdoc_ai.evaluation.text_metrics import character_error_rate, normalize_text


@dataclass(frozen=True)
class FieldScore:
    exact_match: bool
    character_error_rate: float
    character_accuracy: float


def score_field(reference: str, hypothesis: str) -> FieldScore:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    cer = character_error_rate(ref, hyp)
    return FieldScore(
        exact_match=ref == hyp,
        character_error_rate=cer,
        character_accuracy=max(0.0, 1.0 - cer),
    )
