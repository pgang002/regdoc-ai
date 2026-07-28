#!/usr/bin/env python3
"""Create a checksum manifest for downloaded public ClinicalTrials.gov protocol PDFs."""
from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import fitz
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/data_sources.yaml").read_text(encoding="utf-8"))
    documents = config["sources"]["clinicaltrials"]["public_protocol_documents"]
    rows = []
    retrieved_at = datetime.now(timezone.utc).isoformat()
    for item in documents:
        path = root / item["local_path"]
        if not path.exists():
            raise FileNotFoundError(path)
        document = fitz.open(path)
        rows.append(
            {
                "nct_id": item["nct_id"],
                "protocol_number": item["protocol_number"],
                "source_url": item["url"],
                "local_path": item["local_path"],
                "usage": item["usage"],
                "retrieved_at_utc": retrieved_at,
                "page_count": len(document),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
        document.close()

    output = root / "data/manifests/clinical_protocols_manifest.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} protocol source records to {output}")


if __name__ == "__main__":
    main()
