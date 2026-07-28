#!/usr/bin/env python3
"""Build a reproducible file-level manifest for downloaded source documents."""
from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pypdf import PdfReader


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def pdf_details(path: Path) -> dict[str, Any]:
    reader = PdfReader(str(path), strict=False)
    metadata = reader.metadata or {}
    return {
        "page_count": len(reader.pages),
        "encrypted": bool(reader.is_encrypted),
        "pdf_title": str(metadata.get("/Title", "")),
        "pdf_creator": str(metadata.get("/Creator", "")),
        "pdf_producer": str(metadata.get("/Producer", "")),
        "has_xfa": bool(getattr(reader, "xfa", None)),
        "form_field_count": len(reader.get_fields() or {}),
    }


def build_manifest(project_root: Path, config_path: Path) -> list[dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    retrieved_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []

    for document in config["sources"]["fda_forms"]["documents"]:
        local_path = project_root / document["local_path"]
        if not local_path.exists():
            raise FileNotFoundError(f"Missing source document: {local_path}")

        details = pdf_details(local_path)
        rows.append(
            {
                "source_id": document["id"],
                "title": document["title"],
                "authority": config["sources"]["fda_forms"]["authority"],
                "source_url": document["url"],
                "local_path": str(local_path.relative_to(project_root)),
                "usage": document["usage"],
                "retrieved_at_utc": retrieved_at,
                "size_bytes": local_path.stat().st_size,
                "sha256": sha256_file(local_path),
                **details,
                "notes": document.get("notes", ""),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/data_sources.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/fda_forms_manifest.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    output_path = args.output if args.output.is_absolute() else root / args.output
    rows = build_manifest(root, config_path)
    write_csv(rows, output_path)
    print(f"Wrote {len(rows)} source records to {output_path}")


if __name__ == "__main__":
    main()
