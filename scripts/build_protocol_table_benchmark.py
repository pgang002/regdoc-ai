#!/usr/bin/env python3
"""Build exact table annotations from actual public Moderna protocol PDFs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import fitz
import pandas as pd
import pdfplumber
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique(values: list[float], tolerance: float = 0.01) -> list[float]:
    ordered = sorted(values)
    result: list[float] = []
    for value in ordered:
        if not result or abs(value - result[-1]) > tolerance:
            result.append(value)
    return result


def cell_text(page: pdfplumber.page.Page, bbox: tuple[float, float, float, float]) -> str:
    return (page.crop(bbox).extract_text(x_tolerance=2, y_tolerance=2) or "").strip()


def build_table_record(
    *,
    pdf_path: Path,
    pdf_page: pdfplumber.page.Page,
    fitz_page: fitz.Page,
    page_number: int,
    table_index: int,
    category: str,
    nct_id: str,
    protocol_number: str,
    dpi: int,
    padding: float,
    image_path: Path,
) -> dict[str, Any]:
    tables = pdf_page.find_tables()
    if table_index >= len(tables):
        raise IndexError(f"{pdf_path.name} page {page_number} has only {len(tables)} tables")
    table = tables[table_index]
    x0, top, x1, bottom = [float(v) for v in table.bbox]
    page_rect = fitz_page.rect
    clip = fitz.Rect(
        max(page_rect.x0, x0 - padding),
        max(page_rect.y0, top - padding),
        min(page_rect.x1, x1 + padding),
        min(page_rect.y1, bottom + padding),
    )
    scale = dpi / 72.0
    pix = fitz_page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)

    x_boundaries_pdf = unique([v for cell in table.cells for v in (float(cell[0]), float(cell[2]))])
    y_boundaries_pdf = unique([v for cell in table.cells for v in (float(cell[1]), float(cell[3]))])
    x_boundaries_image = [round((value - clip.x0) * scale, 3) for value in x_boundaries_pdf]
    y_boundaries_image = [round((value - clip.y0) * scale, 3) for value in y_boundaries_pdf]

    cells = []
    for index, cell in enumerate(table.cells):
        cx0, cy0, cx1, cy1 = [float(v) for v in cell]
        row_start = min(range(len(y_boundaries_pdf)), key=lambda i: abs(y_boundaries_pdf[i] - cy0))
        row_end = min(range(len(y_boundaries_pdf)), key=lambda i: abs(y_boundaries_pdf[i] - cy1))
        col_start = min(range(len(x_boundaries_pdf)), key=lambda i: abs(x_boundaries_pdf[i] - cx0))
        col_end = min(range(len(x_boundaries_pdf)), key=lambda i: abs(x_boundaries_pdf[i] - cx1))
        cells.append(
            {
                "cell_id": index,
                "bbox_pdf": [cx0, cy0, cx1, cy1],
                "bbox_image": [
                    round((cx0 - clip.x0) * scale, 3),
                    round((cy0 - clip.y0) * scale, 3),
                    round((cx1 - clip.x0) * scale, 3),
                    round((cy1 - clip.y0) * scale, 3),
                ],
                "row_start": row_start,
                "row_end": row_end,
                "column_start": col_start,
                "column_end": col_end,
                "row_span": row_end - row_start,
                "column_span": col_end - col_start,
                "text": cell_text(pdf_page, (cx0, cy0, cx1, cy1)),
            }
        )

    extracted = table.extract(x_tolerance=2, y_tolerance=2)
    normalized_matrix = [["" if value is None else str(value) for value in row] for row in extracted]
    table_id = f"{nct_id}_p{page_number:03d}_t{table_index}"
    return {
        "table_id": table_id,
        "source_type": "actual_public_clinical_protocol_table",
        "source_pdf": str(pdf_path.relative_to(PROJECT_ROOT)),
        "source_pdf_sha256": sha256(pdf_path),
        "nct_id": nct_id,
        "protocol_number": protocol_number,
        "page": page_number,
        "table_index": table_index,
        "category": category,
        "render_dpi": dpi,
        "crop_padding_points": padding,
        "crop_bbox_pdf": list(clip),
        "table_bbox_pdf": [x0, top, x1, bottom],
        "table_bbox_image": [
            round((x0 - clip.x0) * scale, 3),
            round((top - clip.y0) * scale, 3),
            round((x1 - clip.x0) * scale, 3),
            round((bottom - clip.y0) * scale, 3),
        ],
        "image_path": str(image_path.relative_to(PROJECT_ROOT)),
        "image_sha256": sha256(image_path),
        "image_width": image.width,
        "image_height": image.height,
        "x_boundaries_image": x_boundaries_image,
        "y_boundaries_image": y_boundaries_image,
        "logical_rows": len(normalized_matrix),
        "logical_columns": max((len(row) for row in normalized_matrix), default=0),
        "physical_cell_count": len(cells),
        "matrix": normalized_matrix,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/table_benchmark.yaml"))
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = resolve(config["output_data_dir"])
    images_dir = output_root / "images"
    annotations_dir = output_root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for source in config["selection"]:
        pdf_path = resolve(source["source_pdf"])
        fitz_doc = fitz.open(pdf_path)
        with pdfplumber.open(pdf_path) as plumber_doc:
            for item in source["tables"]:
                page_number = int(item["page"])
                table_index = int(item["table_index"])
                table_id = f"{source['nct_id']}_p{page_number:03d}_t{table_index}"
                image_path = images_dir / f"{table_id}.png"
                record = build_table_record(
                    pdf_path=pdf_path,
                    pdf_page=plumber_doc.pages[page_number - 1],
                    fitz_page=fitz_doc[page_number - 1],
                    page_number=page_number,
                    table_index=table_index,
                    category=item["category"],
                    nct_id=source["nct_id"],
                    protocol_number=source["protocol_number"],
                    dpi=int(config["render_dpi"]),
                    padding=float(config["crop_padding_points"]),
                    image_path=image_path,
                )
                annotation_path = annotations_dir / f"{table_id}.json"
                annotation_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
                manifest_rows.append(
                    {
                        key: record[key]
                        for key in (
                            "table_id", "source_type", "source_pdf", "source_pdf_sha256",
                            "nct_id", "protocol_number", "page", "table_index", "category",
                            "image_path", "image_sha256", "image_width", "image_height",
                            "logical_rows", "logical_columns", "physical_cell_count",
                        )
                    }
                    | {"annotation_path": str(annotation_path.relative_to(PROJECT_ROOT))}
                )
        fitz_doc.close()

    manifest = pd.DataFrame(manifest_rows).sort_values(["nct_id", "page", "table_index"])
    manifest.to_csv(output_root / "manifest.csv", index=False)
    metadata = {
        "benchmark_name": config["benchmark_name"],
        "source_type": config["source_type"],
        "table_count": len(manifest),
        "source_pdf_count": manifest["source_pdf"].nunique(),
        "render_dpi": config["render_dpi"],
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "ground_truth_method": "PDF vector rules and embedded text extracted with pdfplumber",
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Built {len(manifest)} actual public clinical protocol table samples in {output_root}")


if __name__ == "__main__":
    main()
