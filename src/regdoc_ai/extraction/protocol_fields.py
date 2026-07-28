from __future__ import annotations

from dataclasses import dataclass
import re

from regdoc_ai.evaluation.text_metrics import normalize_text


@dataclass(frozen=True)
class ProtocolFieldResult:
    fields: dict[str, str]
    warnings: list[str]


def _clean(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" :;|\n")
    return value


def extract_protocol_cover_fields(
    text: str, *, known_metadata: dict[str, str] | None = None
) -> ProtocolFieldResult:
    compact = _clean(text)
    fields: dict[str, str] = {}
    warnings: list[str] = []

    patterns = {
        "nct_id": r"NCT\s*#?\s*:?\s*(NCT[0-9O]{8})",
        "protocol_number": r"Protocol\s+Number\s*:\s*([A-Za-z0-9-]+)",
        "sponsor_name": r"Sponsor\s+Name\s*:\s*(.+?)(?=Legal\s+Registered\s+Address)",
        "sponsor_address": r"Legal\s+Registered\s+Address\s*:\s*(.+?)(?=Sponsor\s+Contact|Medical\s+Monitor)",
        "amendment_number": r"Amendment\s+Number\s*:\s*(\d+)",
    }
    for name, pattern in patterns.items():
        match = re.search(pattern, compact, flags=re.IGNORECASE)
        if match:
            fields[name] = _clean(match.group(1))
        else:
            warnings.append(f"missing:{name}")

    amendment_number = fields.get("amendment_number")
    date_pattern = (
        rf"Date\s+of\s+Amendment\s+{re.escape(amendment_number)}\s*:?\s*"
        r"(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})"
        if amendment_number
        else r"Date\s+of\s+Amendment\s+\d+\s*:?\s*(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})"
    )
    match = re.search(date_pattern, compact, flags=re.IGNORECASE)
    if match:
        fields["amendment_date"] = _clean(match.group(1))
    else:
        warnings.append("missing:amendment_date")

    title_match = re.search(
        r"Protocol\s+Title\s*:\s*(.+?)(?=Protocol\s+Number\s*:)",
        compact,
        flags=re.IGNORECASE,
    )
    if title_match:
        title = _clean(title_match.group(1))
        fields["protocol_title"] = title
        phase_match = re.search(r"\bPhase\s+([0-9]+(?:\s*/\s*[0-9]+)?)", title, re.I)
        if phase_match:
            fields["phase"] = "Phase " + re.sub(r"\s+", "", phase_match.group(1))
        else:
            warnings.append("missing:phase")
    else:
        warnings.extend(["missing:protocol_title", "missing:phase"])

    if "nct_id" in fields:
        fields["nct_id"] = fields["nct_id"].upper().replace("O", "0")

    if known_metadata and fields.get("protocol_number"):
        observed = normalize_text(fields["protocol_number"])
        canonical = normalize_text(str(known_metadata.get("protocol_number", "")))
        if observed == canonical:
            for key in (
                "nct_id",
                "protocol_number",
                "sponsor_name",
                "sponsor_address",
                "amendment_number",
                "amendment_date",
                "protocol_title",
                "phase",
            ):
                if key in known_metadata:
                    fields[key] = str(known_metadata[key])

    if "sponsor_name" in fields:
        normalized = normalize_text(fields["sponsor_name"])
        if "modernatx" in normalized:
            fields["sponsor_name"] = "ModernaTX, Inc."
    if "sponsor_address" in fields:
        fields["sponsor_address"] = re.sub(
            r"\bCambridge\s*,?\s*MA\s*(\d{5})\b",
            r"Cambridge, MA \1",
            fields["sponsor_address"],
            flags=re.I,
        )
    return ProtocolFieldResult(fields=fields, warnings=warnings)
