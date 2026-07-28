# Day 5 - Real Clinical Protocol Table Extraction

## Objective

Day 5 adds a reproducible table-extraction benchmark using actual public Moderna
clinical study protocol tables. The benchmark tests table structure detection,
cell reconstruction, OCR text recovery, export to CSV/JSON/HTML, and robustness to
scan defects.

## Actual data used

The benchmark uses 20 tables rendered directly from two public protocol PDFs hosted
by ClinicalTrials.gov:

- NCT04470427, protocol mRNA-1273-P301
- NCT04796896, protocol mRNA-1273-P204

The selected tables include document history, amendment summaries, treatment groups,
objectives/endpoints, efficacy assumptions, adverse-reaction grading, schedules of
events, laboratory assessments, and statistical-analysis tables.

No table content is synthetically generated. Ground truth is derived from the source
PDFs' vector rules, cell geometry, and embedded text using `pdfplumber`. The separate
robustness set applies deterministic scan artifacts to these real table crops and is
labeled as augmented data.

## Implemented pipelines

### 1. Camelot lattice

A PDF-native baseline that uses vector line information. It is appropriate for born-
digital submissions and serves as a strong reference when embedded PDF geometry is
available.

### 2. OpenCV + Tesseract

An image-based pipeline that:

1. Detects horizontal and vertical rules with morphology.
2. Clusters line coordinates into row and column boundaries.
3. Reconstructs the logical grid.
4. Removes grid lines before OCR.
5. Upscales the cleaned crop and runs Tesseract.
6. Assigns OCR words to cells by spatial containment.
7. Exports reconstructed tables to CSV, JSON, and HTML.

This path is relevant to scanned regulatory submissions where PDF vector information
is unavailable.

## Clean-table results

| Pipeline | Tables | Exact grid shape | Row boundary F1 | Column boundary F1 | Physical-cell F1 | Mean text CER | Mean latency/table |
|---|---:|---:|---:|---:|---:|---:|---:|
| Camelot lattice | 20 | 95.0% | 1.000 | 0.992 | Not reported | 0.0089 | 0.449 s |
| OpenCV + Tesseract | 20 | 100.0% | 1.000 | 1.000 | 0.925 | 0.1927 | 0.757 s |

Camelot reproduced 19 of 20 tables exactly and achieved a mean text CER of 0.0089.
The image pipeline found every row and column boundary on the clean ruled-table set,
but OCR and merged-cell reconstruction remained the primary error sources.

`physical-cell F1` compares predicted adjacent grid cells with the source PDF's
physical cells. It penalizes over-segmentation of merged cells, which explains why it
is lower than boundary F1.

## Scan-robustness results

The robustness benchmark uses two representative complex tables across clean,
1.5-degree rotation, Gaussian blur, JPEG compression, and a combined degradation.
Across the eight degraded table instances:

| Pipeline | Exact grid shape | Row boundary F1 | Column boundary F1 | Physical-cell F1 | Mean text CER |
|---|---:|---:|---:|---:|---:|
| Raw image pipeline | 50.0% | 0.558 | 0.725 | 0.408 | 0.4987 |
| Enhanced pipeline | 100.0% | 1.000 | 1.000 | 0.795 | 0.2718 |

Deskewing and restoration recovered exact grid shape from 0% to 100% on the rotated
and combined conditions. The enhanced path reduced mean text CER across degraded
conditions by approximately 45.5% relative to the raw image pipeline.

## Deep-model comparison status

The repository includes configuration and a Colab runner for:

- `microsoft/table-transformer-detection`
- `microsoft/table-transformer-structure-recognition-v1.1-all`
- PaddleOCR `PPStructureV3`

These models were not executed in the current runtime because the optional packages
and model files could not be downloaded. Their metrics are deliberately absent from
the result tables; no placeholder values are generated. The Colab notebook uses the
same 20 actual table images and annotations.

## Fine-tuning decision

Fine-tuning Table Transformer is not yet justified because pretrained Table
Transformer metrics have not been measured on this benchmark. After the Colab run,
fine-tuning should be considered only if repeated errors occur or mean row/column
boundary F1 is below 0.85 on the clinical tables.

## Reproduction

```bash
python scripts/build_protocol_table_benchmark.py
python scripts/evaluate_table_extraction.py
python scripts/generate_degraded_tables.py
python scripts/evaluate_table_robustness.py
python scripts/create_table_figures.py
pytest
```

## Key outputs

- `data/processed/table_benchmark/manifest.csv`
- `data/processed/table_benchmark/annotations/`
- `results/table_extraction_benchmark/table_predictions.csv`
- `results/table_extraction_benchmark/summary_overall.csv`
- `results/table_extraction_benchmark/robustness/`
- `results/table_extraction_benchmark/reconstructions/`
- `results/table_extraction_benchmark/figures/`
