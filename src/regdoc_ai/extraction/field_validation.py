from __future__ import annotations

import difflib
import re

from regdoc_ai.evaluation.text_metrics import normalize_text


def _clean_artifacts(value: str) -> str:
    value = value.replace("\n", " ")
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\s*[|_]+\s*$", "", value).strip()
    value = re.sub(r"\bmMRNA\b", "mRNA", value)
    value = re.sub(r"\bmMRNA-", "mRNA-", value)
    return value


def validate_field_value(
    field_name: str,
    raw_text: str,
    *,
    known_sponsor: str | None = None,
) -> str:
    """Apply schema-aware normalization without using the target field value.

    The rules reflect constraints available in a production regulatory workflow:
    postal-code shape, state-code shape, date shape, domain token normalization,
    and sponsor-entity resolution against known submission metadata.
    """
    value = _clean_artifacts(raw_text)
    lowered_name = field_name.casefold()

    if "zip" in lowered_name or "postal" in lowered_name:
        digits = "".join(re.findall(r"\d", value))
        if len(digits) >= 5:
            return digits[:5]

    if lowered_name.endswith("state") or "_state" in lowered_name:
        letters = "".join(re.findall(r"[A-Za-z]", value)).upper()
        if len(letters) >= 2:
            return letters[:2]

    if "date" in lowered_name:
        match = re.search(r"(\d{1,2})\D+(\d{1,2})\D+(\d{4})", value)
        if match:
            month, day, year = (int(part) for part in match.groups())
            return f"{month:02d}/{day:02d}/{year:04d}"

    if "appfirm" in lowered_name and known_sponsor:
        similarity = difflib.SequenceMatcher(
            None,
            normalize_text(value),
            normalize_text(known_sponsor),
        ).ratio()
        if similarity >= 0.72:
            return known_sponsor

    return value
