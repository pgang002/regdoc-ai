# Degradation benchmark results

This directory contains the completed Day 3 robustness evaluation.

## Key degraded-only results

| Pipeline | Validated field exact match | Mean CER | Checkbox accuracy |
|---|---:|---:|---:|
| `baseline_raw` | 76.43% | 0.2833 | 84.42% |
| `enhanced_deskew_restored` | 96.43% | 0.1040 | 100.00% |

The enhanced pipeline increased validated field accuracy by 20.0 percentage points and reduced field exact-match errors by 84.85%.

## Contents

- `field_predictions.csv`: all field predictions for all conditions and pipelines
- `checkbox_predictions.csv`: all checkbox-state predictions
- `page_preprocessing.csv`: skew estimates and page-restoration latency
- `field_summary_by_condition.csv`: condition-level field metrics
- `field_summary_degraded_overall.csv`: aggregate degraded-only field metrics
- `checkbox_summary_by_condition.csv`: condition-level checkbox metrics
- `checkbox_summary_degraded_overall.csv`: aggregate degraded-only checkbox metrics
- `page_summary.csv`: page-level restoration statistics
- `run_metadata.json`: benchmark scope and data-use statement
- `conditions/`: per-condition detailed outputs
- `figures/`: visual examples and summary charts

Source pages are official FDA templates populated with public clinical-protocol metadata and controlled test values. Scan defects are deterministic synthetic augmentations.
