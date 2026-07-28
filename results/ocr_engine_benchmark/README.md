# OCR engine benchmark

This directory contains the controlled field-level OCR comparison introduced in Day 4.
Every engine receives the same enhanced page image, exact PDF field rectangle, selected
field set, and eight scan conditions from `configs/degradation_benchmark.yaml`.

## Executed in the current runtime

- Tesseract 5.5.0 / pytesseract 0.3.13: completed on 160 field instances.
- PaddleOCR: not executed because the optional PaddlePaddle/PaddleOCR binary runtime and
  model assets could not be downloaded in the execution environment.

`benchmark_status.json` is the source of truth. The repository intentionally does not
create placeholder PaddleOCR rows or describe an incomplete run as a comparison.

## Complete the comparison in Colab

Run `notebooks/04_tesseract_vs_paddleocr.ipynb`, or install
`requirements-paddle.txt` and execute:

```bash
python scripts/benchmark_ocr_engines.py --engines tesseract paddleocr --strict
```

The benchmark uses official FDA form templates populated with public Moderna protocol
metadata and controlled test identifiers. Scan corruptions are deterministic synthetic
augmentations and are labeled `augmented_scan` in the manifest.
