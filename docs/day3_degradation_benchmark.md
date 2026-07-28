# Day 3: Scan-Degradation Robustness Benchmark

## Objective

Evaluate the existing template-aware OCR and checkbox pipeline under realistic scan defects while retaining exact field and checkbox ground truth.

The source pages are not invented blank documents. They are renderings of official FDA Forms 1572, 3454, and 3455 populated in the previous milestone with:

- public Moderna protocol metadata extracted from NCT04470427,
- controlled test identities and addresses,
- controlled checkbox scenarios,
- exact PDF field coordinates and expected values.

The scan defects are deterministic synthetic augmentations. They are labeled `augmented_scan` and are not represented as naturally acquired scans.

## Generated benchmark data

The generator created 64 page images:

- 8 source pages from the six populated official-form samples,
- 8 conditions per page,
- source and output SHA-256 hashes,
- exact augmentation parameters and seeds in the manifest.

Conditions:

1. Clean
2. Rotation by 2 degrees
3. Gaussian blur
4. Gaussian noise
5. Low contrast
6. Directional shadow
7. JPEG compression
8. Combined moderate degradation

The combined condition applies rotation, blur, noise, reduced contrast, shadow, and JPEG compression together.

## Evaluation subset

To keep the benchmark reproducible and fast enough for local development, evaluation uses one representative populated sample of each form type from NCT04470427:

- FDA 1572: 9 stratified fields
- FDA 3454: 6 stratified fields
- FDA 3455: 5 stratified fields
- Total: 20 fields and 11 checkboxes per condition

The field subset includes names, addresses, ZIP codes, dates, sponsor names, investigator lists, and long multiline protocol text. The earlier clean-form milestone still evaluates all 90 populated fields and 22 checkboxes.

Across all eight conditions and two pipelines, the benchmark contains:

- 320 field-prediction records
- 176 checkbox-prediction records
- 64 generated page images

## Compared pipelines

### Baseline

- Original degraded page
- Fixed template coordinates
- Raw Tesseract OCR on each field crop
- Fixed-form checkbox detector
- Existing schema-aware field validation

### Enhanced restoration pipeline

- Hough-line skew estimation
- Corrective affine rotation
- Median denoising
- CLAHE local contrast recovery
- Unsharp masking
- Raw OCR on restored field crops
- Locally normalized checkbox ink detection
- Existing schema-aware field validation

Adaptive thresholding was initially tested as a universal enhancement. It amplified Gaussian noise and caused a complete OCR failure in that condition. The final pipeline therefore restores the full page first and avoids applying the same thresholding rule indiscriminately to every scan.

## Main results

### Degraded conditions only

| Pipeline | Fields | Validated exact match | Mean character accuracy | Mean character error rate | Mean OCR confidence |
|---|---:|---:|---:|---:|---:|
| Raw baseline | 140 | 76.43% | 83.03% | 0.2833 | 83.29 |
| Enhanced restoration | 140 | 96.43% | 96.59% | 0.1040 | 90.60 |

The restoration pipeline produced:

- **+20.0 percentage points** in validated field exact-match accuracy
- **84.85% reduction in field exact-match error rate**
- **63.27% reduction in mean character error rate**
- **+7.31 points in mean OCR confidence**

Field-crop OCR latency decreased from 0.162 seconds to 0.126 seconds on average, but the enhanced path also adds approximately 0.253 seconds of page-level restoration per degraded page. Overall end-to-end latency should therefore be measured after the API and batch-processing stages are added.

### Checkbox robustness

| Pipeline | Degraded checkbox records | Accuracy |
|---|---:|---:|
| Raw baseline | 77 | 84.42% |
| Enhanced restoration | 77 | 100.00% |

The original fixed grayscale threshold failed when an empty checkbox was uniformly darkened by shadow. Local-background normalization corrected this without lowering clean, blurred, noisy, low-contrast, or compressed accuracy.

## Important condition-level findings

| Condition | Raw validated field accuracy | Enhanced validated field accuracy | Raw checkbox accuracy | Enhanced checkbox accuracy |
|---|---:|---:|---:|---:|
| Rotation 2 degrees | 15% | 100% | 45.45% | 100% |
| Combined moderate | 30% | 85% | 45.45% | 100% |
| Directional shadow | 90% | 90% | 100% | 100% |
| Gaussian blur | 100% | 100% | 100% | 100% |
| Gaussian noise | 100% | 100% | 100% | 100% |
| JPEG compression | 100% | 100% | 100% | 100% |
| Low contrast | 100% | 100% | 100% | 100% |

The largest improvements come from geometric correction. Under combined degradation, the three remaining enhanced-pipeline failures are:

- the long multiline FDA 1572 protocol field,
- the FDA 1572 sub-investigator list,
- the FDA 3455 sponsor field.

These failures motivate the next OCR-model benchmark and confidence-based review routing.

## Reproducibility

```powershell
python scripts/generate_degraded_forms.py

python scripts/evaluate_degradation_condition.py clean
python scripts/evaluate_degradation_condition.py rotation_2deg
python scripts/evaluate_degradation_condition.py gaussian_blur
python scripts/evaluate_degradation_condition.py gaussian_noise
python scripts/evaluate_degradation_condition.py low_contrast
python scripts/evaluate_degradation_condition.py directional_shadow
python scripts/evaluate_degradation_condition.py jpeg_compression
python scripts/evaluate_degradation_condition.py combined_moderate

python scripts/summarize_degradation_benchmark.py
python scripts/create_degradation_figures.py
pytest
```

## Files added

- `configs/degradation_benchmark.yaml`
- `src/regdoc_ai/augmentation/degradations.py`
- `src/regdoc_ai/preprocessing/document.py`
- `scripts/generate_degraded_forms.py`
- `scripts/evaluate_degradation_condition.py`
- `scripts/summarize_degradation_benchmark.py`
- `scripts/create_degradation_figures.py`
- `data/processed/degraded_forms/manifest.csv`
- `results/degradation_benchmark/`
- new deterministic degradation, deskew, and shadowed-checkbox tests

## Limitations

- The scan defects are controlled augmentations, not photographs captured from mobile devices or scanners.
- The robustness subset contains 20 representative fields from one public protocol, while the clean-form benchmark contains all 90 fields across both protocols.
- The current OCR engine is Tesseract. PaddleOCR comparison is the next milestone.
- Exact template coordinates assume that the document type and template version are already known.
- Schema validation can repair constrained fields such as ZIP codes, dates, state codes, and known sponsor names, but cannot recover arbitrary long text when OCR loses substantial content.
