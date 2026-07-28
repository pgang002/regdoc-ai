# Populated-form benchmark results

This folder contains exact-ground-truth evaluation outputs for six populated official
FDA form samples derived from Forms 1572, 3454, and 3455.

## Headline clean-render results

- 90 populated text fields per preprocessing configuration
- 94.44% raw OCR exact-match accuracy
- 100.00% exact-match accuracy after schema-aware validation
- 100.00% checkbox accuracy across 22 boxes

## Files

- `field_predictions.csv`: field-level references, raw OCR, validated values, confidence,
  error, and latency.
- `field_summary.csv`: aggregate comparison of raw and adaptive preprocessing.
- `field_summary_by_form.csv`: results split by FDA form type.
- `checkbox_predictions.csv`: checkbox-level expected/predicted states and ink ratios.
- `checkbox_summary.csv`: aggregate checkbox accuracy.
- `annotated/`: visual field/checkbox overlays for review.

## Limitation

These results use clean 300-DPI renderings of template-aligned PDFs. The next benchmark
adds scan degradations and should be used for any final resume metric.
