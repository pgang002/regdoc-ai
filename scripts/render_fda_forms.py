#!/usr/bin/env python3
"""Render actual FDA PDFs to images for OCR benchmarking."""
from __future__ import annotations

import argparse
from pathlib import Path

import fitz


def render_pdf(pdf_path: Path, output_dir: Path, dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(pdf_path)
    scale = dpi / 72
    matrix = fitz.Matrix(scale, scale)
    outputs: list[Path] = []
    for page_index, page in enumerate(document):
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        output = output_dir / f"page-{page_index + 1}.png"
        pixmap.save(output)
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/fda_forms"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/pdf_pages"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_dir = args.input_dir if args.input_dir.is_absolute() else root / args.input_dir
    output_root = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir

    total = 0
    for pdf_path in sorted(input_dir.glob("*.pdf")):
        outputs = render_pdf(pdf_path, output_root / pdf_path.stem, args.dpi)
        total += len(outputs)
        print(f"{pdf_path.name}: rendered {len(outputs)} page(s)")
    print(f"Rendered {total} actual FDA source pages")


if __name__ == "__main__":
    main()
