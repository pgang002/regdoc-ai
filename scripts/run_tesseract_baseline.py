#!/usr/bin/env python3
"""Run a reproducible Tesseract baseline on actual FDA form pages.

The PDF text layer is used as a proxy reference for printed form-label text. These
metrics do not yet measure populated-field extraction accuracy; that will use exact
field-level ground truth after form population.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import fitz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from regdoc_ai.evaluation.text_metrics import character_error_rate, word_error_rate
from regdoc_ai.ocr.tesseract_engine import recognize
from regdoc_ai.preprocessing.image import preprocess_image


def pdf_page_text(pdf_path: Path, page_index: int) -> str:
    document = fitz.open(pdf_path)
    return document[page_index].get_text("text")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forms-dir", type=Path, default=Path("data/raw/fda_forms"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/interim/pdf_pages"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/tesseract_baseline"))
    parser.add_argument("--modes", nargs="+", default=["raw", "clahe", "adaptive"])
    parser.add_argument("--psm", type=int, default=6)
    args = parser.parse_args()

    forms_dir = args.forms_dir if args.forms_dir.is_absolute() else PROJECT_ROOT / args.forms_dir
    images_dir = args.images_dir if args.images_dir.is_absolute() else PROJECT_ROOT / args.images_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for pdf_path in sorted(forms_dir.glob("*.pdf")):
        if pdf_path.stem == "FDA_1571":
            continue  # XFA placeholder under standard rendering; documented separately.
        page_dir = images_dir / pdf_path.stem
        for image_path in sorted(page_dir.glob("page-*.png")):
            page_number = int(image_path.stem.split("-")[-1])
            reference = pdf_page_text(pdf_path, page_number - 1)
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Unable to read {image_path}")
            for mode in args.modes:
                processed = preprocess_image(image, mode)
                start = time.perf_counter()
                result = recognize(processed, psm=args.psm)
                latency = time.perf_counter() - start

                text_path = output_dir / f"{pdf_path.stem}_page-{page_number}_{mode}.txt"
                text_path.write_text(result.text, encoding="utf-8")
                boxes_path = output_dir / f"{pdf_path.stem}_page-{page_number}_{mode}_words.csv"
                result.words.to_csv(boxes_path, index=False)

                rows.append(
                    {
                        "document": pdf_path.stem,
                        "page": page_number,
                        "preprocessing": mode,
                        "psm": args.psm,
                        "reference_characters": len(reference),
                        "recognized_characters": len(result.text),
                        "cer_proxy": round(character_error_rate(reference, result.text), 6),
                        "wer_proxy": round(word_error_rate(reference, result.text), 6),
                        "mean_word_confidence": round(result.mean_confidence, 3),
                        "latency_seconds": round(latency, 3),
                    }
                )

    summary = output_dir / "summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Completed {len(rows)} actual-page OCR runs; wrote {summary}")


if __name__ == "__main__":
    main()
