# Table Extraction Benchmark Results

This directory contains measured results from 20 table crops taken from two actual
public Moderna clinical study protocol PDFs.

## Completed local engines

- `camelot_lattice`: PDF-native extraction using vector rules.
- `opencv_tesseract`: image-only grid detection, OCR, cell assignment, and export.

## Important limitation

Table Transformer and PaddleOCR PP-StructureV3 were not executed in the restricted
runtime. Their optional runtime status is recorded in `benchmark_status.json`, and no
placeholder model metrics are included. Use
`notebooks/05_table_extraction_deep_models.ipynb` in Colab to run them on the same
images and annotations.

## Main files

- `summary_overall.csv`: clean-table aggregate metrics.
- `summary_by_category.csv`: metrics by clinical table category.
- `table_predictions.csv`: per-table metrics.
- `robustness/summary_by_condition.csv`: clean and degraded image results.
- `robustness/summary_degraded_overall.csv`: aggregate degraded-image results.
- `reconstructions/`: CSV, JSON, and HTML table outputs.
- `overlays/`: ground-truth and predicted rule visualizations.
- `figures/`: portfolio charts and real-data examples.
