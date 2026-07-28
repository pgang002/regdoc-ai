#!/usr/bin/env python3
"""Extract public study metadata from downloaded ClinicalTrials.gov protocol PDFs."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import fitz


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_multiline(value: str) -> str:
    return " ".join(value.split())


def extract_one(path: Path) -> dict[str, object]:
    document = fitz.open(path)
    first_pages = "\n".join(document[index].get_text("text") for index in range(min(4, len(document))))
    document.close()

    title_match = re.search(r"Protocol Title:\s*(.*?)\s*Protocol Number:", first_pages, flags=re.S)
    protocol_match = re.search(r"Protocol Number:\s*([^\n]+)", first_pages)
    nct_match = re.search(r"NCT\s*#:\s*(NCT\d+)", first_pages)
    sponsor_match = re.search(r"Sponsor Name:\s*(?:\n)?\s*([^\n]+)", first_pages)
    legal_section_match = re.search(
        r"Legal Registered Address:\s*(.*?)\s*Sponsor Contact",
        first_pages,
        flags=re.S,
    )
    amendment_match = re.search(r"Amendment Number:\s*(\d+)", first_pages)
    date_match = re.search(r"Date of Amendment(?:\s+\d+)?:?\s*\n?\s*(\d{2}\s+[A-Z][a-z]{2}\s+\d{4})", first_pages)

    if not all([title_match, protocol_match, nct_match, sponsor_match, legal_section_match]):
        raise ValueError(f"Could not extract required metadata from {path}")

    title = clean_multiline(title_match.group(1))
    phase_match = re.search(r"Phase\s+([0-9]+(?:/[0-9]+)?)", title, flags=re.I)
    sponsor_name = sponsor_match.group(1).strip()
    legal_lines = [line.strip() for line in legal_section_match.group(1).splitlines() if line.strip()]
    # Some protocol PDFs place the sponsor name after the Legal Registered Address label.
    if sponsor_name.startswith("Legal Registered Address"):
        sponsor_name = legal_lines.pop(0)
    elif legal_lines and legal_lines[0] == sponsor_name:
        legal_lines.pop(0)
    sponsor_address = clean_multiline(" ".join(legal_lines[:2]))
    return {
        "source_filename": path.name,
        "source_sha256": sha256(path),
        "nct_id": nct_match.group(1),
        "protocol_number": protocol_match.group(1).strip(),
        "protocol_title": title,
        "phase": f"Phase {phase_match.group(1)}" if phase_match else None,
        "sponsor_name": sponsor_name,
        "sponsor_address": sponsor_address,
        "amendment_number": int(amendment_match.group(1)) if amendment_match else None,
        "amendment_date": date_match.group(1) if date_match else None,
        "data_classification": "actual_public_clinical_protocol_metadata",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data/raw/clinicaltrials")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/interim/protocol_metadata/studies.json")
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_dir = args.input_dir if args.input_dir.is_absolute() else root / args.input_dir
    output = args.output if args.output.is_absolute() else root / args.output
    files = sorted(input_dir.glob("NCT*_Prot_*.pdf"))
    if not files:
        raise FileNotFoundError(f"No protocol PDFs found in {input_dir}")

    payload = {
        "metadata": {
            "source": "Downloaded public protocol PDFs hosted by ClinicalTrials.gov",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "record_count": len(files),
        },
        "studies": [extract_one(path) for path in files],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Extracted {len(files)} public study records to {output}")


if __name__ == "__main__":
    main()
