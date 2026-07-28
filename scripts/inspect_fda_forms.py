#!/usr/bin/env python3
"""Inspect actual FDA PDF forms and record form/rendering characteristics."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from pypdf import PdfReader


def inspect_pdf(path: Path) -> dict[str, Any]:
    reader = PdfReader(str(path), strict=False)
    fields = reader.get_fields() or {}
    metadata = reader.metadata or {}
    first_page = reader.pages[0]
    media_box = first_page.mediabox
    creator = str(metadata.get("/Creator", ""))
    producer = str(metadata.get("/Producer", ""))
    is_livecycle = "LiveCycle" in creator or "LiveCycle" in producer
    has_xfa = bool(getattr(reader, "xfa", None))
    standard_renderer_risk = is_livecycle and not fields

    return {
        "filename": path.name,
        "page_count": len(reader.pages),
        "width_points": round(float(media_box.width), 2),
        "height_points": round(abs(float(media_box.height)), 2),
        "encrypted": bool(reader.is_encrypted),
        "form_field_count": len(fields),
        "has_xfa": has_xfa,
        "creator": creator,
        "producer": producer,
        "standard_renderer_risk": standard_renderer_risk,
        "recommended_initial_use": (
            "retain_source_convert_with_adobe_before_ocr"
            if standard_renderer_risk
            else "render_and_use_for_ocr_benchmark"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/fda_forms"))
    parser.add_argument("--output", type=Path, default=Path("results/fda_form_inventory.csv"))
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    input_dir = args.input_dir if args.input_dir.is_absolute() else project_root / args.input_dir
    output_path = args.output if args.output.is_absolute() else project_root / args.output

    rows = [inspect_pdf(path) for path in sorted(input_dir.glob("*.pdf"))]
    if not rows:
        raise FileNotFoundError(f"No PDF files found in {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Inspected {len(rows)} actual FDA forms; wrote {output_path}")


if __name__ == "__main__":
    main()
