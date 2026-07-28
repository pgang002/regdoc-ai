# Day 6: Hybrid document classification and field extraction

## Objective

Day 6 adds document-type routing and structured field extraction to the existing OCR
and table pipeline. The implementation tests whether an incoming page can be routed to
the correct FDA-form, clinical-protocol, or clinical-table workflow before extraction.

## Actual data and split policy

The benchmark uses only documents already acquired and provenance-tracked in the
repository:

- populated official FDA Forms 1572, 3454, and 3455;
- actual public pages from Moderna protocols NCT04470427 and NCT04796896;
- full protocol pages containing real tables identified during Day 5.

The page content is not generated. Deterministic rotation, blur, noise, contrast,
shadow, and JPEG artifacts are applied at runtime and are labeled as augmented scans.
Controlled identities inside populated FDA forms remain clearly marked as test values.

To prevent source leakage, all image-fallback training pages come from NCT04470427.
NCT04796896 is completely held out for evaluation. The FDA templates are the same form
families, but populated values differ by study. The final base dataset contains 24 pages
across five classes:

- FDA_1572
- FDA_3454
- FDA_3455
- CLINICAL_PROTOCOL
- CLINICAL_TABLE

The held-out set contains 12 source pages evaluated under eight scan conditions, or 96
page instances.

## Classification architecture

### Rule layer

The rule layer combines:

- OCR form-number and title patterns;
- clinical-protocol header signals;
- document restoration before OCR;
- ruled-table line geometry and table-language signals.

Deskewing and restoration run before the classification rules. This is important because
rotation can erase long horizontal/vertical line projections even when the page remains
readable to a person.

### Local image fallback

The locally executed fallback uses HOG page-layout features and a class-weighted
LinearSVC. The training set is balanced to 32 augmented instances per class. This model
is deliberately treated as a fallback rather than the main decision path.

MobileNetV3-Small transfer-learning code is included for a model-enabled Colab runtime.
Pretrained weights could not be downloaded locally, so no unavailable or randomly
initialized deep-model result is presented as a completed metric.

### Hybrid decision

High-confidence OCR/layout rules decide first. The image model is called only when no
rule reaches the configured threshold. Low image margins can be routed to human review.

## Classification results

| Pipeline | Page instances | Accuracy | Macro F1 | Mean latency/page |
|---|---:|---:|---:|---:|
| Image-only HOG + LinearSVC | 96 | 83.33% | 0.8933 | 0.0158 s |
| Restored rule-only | 96 | 100.00% | 1.0000 | 0.9077 s |
| Hybrid router | 96 | 100.00% | 1.0000 | 0.9173 s |

The hybrid and rule pipelines classified every held-out page correctly under clean,
rotation, blur, noise, low-contrast, shadow, JPEG, and combined-degradation conditions.
The image-only model's errors were limited to confusing narrative protocol pages with
protocol table pages; all FDA form classes were classified correctly.

These results demonstrate reliability on this controlled held-out benchmark, not on all
possible regulatory documents. More protocols, sponsors, form revisions, and naturally
scanned submissions are needed before interpreting 100% as enterprise-wide accuracy.

## Field-extraction architecture

After routing:

- FDA forms use template coordinates, Tesseract OCR, schema normalization, and known
  submission metadata for sponsor resolution;
- clinical protocol covers use OCR label anchors, regex parsing, and protocol-number-based
  resolution against the public study metadata already stored in the pipeline;
- clinical table pages route to the Day 5 reconstruction workflow.

FDA extraction returns to the original 300-DPI form renderings after classification.
This avoids sacrificing field accuracy merely because the classifier can operate on a
smaller page representation.

## Field-extraction results

The form evaluation uses the same stratified 20-field schema as the Day 3 robustness
study. The protocol cover contributes eight public metadata fields per condition.

| Document type | Field instances | Exact match | Mean CER | Routing accuracy |
|---|---:|---:|---:|---:|
| Clinical protocol | 64 | 100.00% | 0.0000 | 100.00% |
| FDA 1572 | 72 | 97.22% | 0.0480 | 100.00% |
| FDA 3454 | 48 | 100.00% | 0.0000 | 100.00% |
| FDA 3455 | 40 | 95.00% | 0.3208 | 100.00% |
| **Overall** | **224** | **98.21%** | **0.0727** | **100.00%** |

The four remaining exact-match failures were caused by severe OCR corruption in one
long sub-investigator field, a trailing character in one IRB field, and two sponsor-firm
crops under shadow/combined degradation. They remain visible in the prediction file
instead of being silently corrected from ground truth.

## Reproduce

```powershell
python scripts/build_document_understanding_benchmark.py
python scripts/train_document_classifier.py
python scripts/evaluate_document_understanding.py
python scripts/create_document_understanding_figures.py
pytest
```

Optional MobileNetV3-Small execution:

```powershell
python scripts/train_mobilenet_document_classifier.py --epochs 6 --batch-size 16
```

The optional script writes an explicit unavailable status if pretrained weights cannot
be loaded. It does not substitute random-weight results.

## Outputs

- `data/processed/document_understanding/manifest.csv`
- `models/document_hog_svm.joblib`
- `results/document_understanding/classification_predictions.csv`
- `results/document_understanding/classification_summary_overall.csv`
- `results/document_understanding/classification_summary_by_condition.csv`
- `results/document_understanding/field_predictions.csv`
- `results/document_understanding/field_summary_overall.csv`
- `results/document_understanding/field_summary_by_document_type.csv`
- `results/document_understanding/figures/`
- `notebooks/06_document_classification_and_field_extraction.ipynb`
