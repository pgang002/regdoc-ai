# Day 7 - Policy-driven PII/CCI detection and true PDF redaction

## Objective

Day 7 adds a configurable redaction layer after document classification and field extraction. The implementation detects policy-defined PII and simulated confidential commercial information (CCI), maps detections to PDF coordinates, applies true redaction, retains low-confidence items for review, and writes audit-safe logs.

## Data use and interpretation

The benchmark does **not** contain real patient records or genuine confidential submissions.

- The document layouts are official FDA Forms 1572, 3454, and 3455.
- Names, investigator addresses, facility names, and disclosure-related values are controlled test values created for the project.
- Protocol numbers, sponsor names, study titles, and dates come from the public Moderna protocol metadata already used in Days 2-6.
- The CCI policy is a workflow simulation. Labeling public protocol metadata as `CCI_*` tests configurable enterprise redaction behavior and does not claim that the public source is confidential.
- Ground-truth values are used only to score detection and verify that content was removed. They are not supplied to the detector.

## Architecture

```text
Classified FDA form
    -> template field coordinates
    -> OCR text + word boxes
    -> regex + spaCy EntityRuler + field semantics
    -> YAML policy action and confidence threshold
    -> redact / review / retain
    -> true PDF redaction
    -> audit-safe JSON + verification report
```

### Detection components

- **Regex:** protocol identifiers, dates, postal codes, and a general person-name baseline.
- **spaCy EntityRuler:** generic PERSON patterns without preloading benchmark names.
- **Field semantics:** known FDA field roles such as investigator name, address, sponsor, facility, and study title.
- **Fail-safe review:** a known sensitive field with empty or low-confidence OCR is routed to review instead of being treated as safe.
- **Coordinate mapping:** clean-document OCR word boxes are converted from pixels to PDF points; the robustness benchmark reuses Day 6 field rectangles because Day 6 stored field OCR but not word-level boxes.

A statistical spaCy language model was not downloaded in the local runtime. The measured local NER component is the generic EntityRuler. The notebook includes an optional statistical-model extension and keeps its results separate until it is actually run.

## Policy

`configs/redaction_policy.yaml` controls:

- entity action (`redact`, `review`, `retain`, or `ignore`)
- minimum confidence
- global human-review threshold
- protocol-ID regex
- benchmark conditions

The current policy automatically redacts:

- `PERSON`
- `ADDRESS`
- `CCI_PROTOCOL_ID`
- `CCI_SPONSOR`

It sends the following to review:

- `SENSITIVE_FACILITY`
- `DATE`
- `CCI_STUDY_TITLE`
- any sensitive field with insufficient OCR confidence

## Robustness benchmark

The Day 7 benchmark reuses the actual held-out Day 6 OCR predictions for NCT04796896. This avoids repeating 300-DPI OCR while preserving identical document content and scan conditions.

- 3 official FDA form types
- 20 selected sensitive/non-sensitive fields per condition
- 8 scan conditions
- 168 expected entity instances for the hybrid policy
- conditions: clean, rotation, blur, noise, low contrast, shadow, JPEG compression, and combined degradation

### Overall results

| Pipeline | Precision | Recall | F1 | Automatic redaction coverage | False redaction rate | Policy-action accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Regex only | 0.5963 | 0.5714 | 0.5836 | 73.93% | 47.44% | 56.53% |
| Hybrid policy | **1.0000** | **1.0000** | **1.0000** | **97.92%** | **0.00%** | **98.81%** |

The hybrid detector achieved 1.0 F1 for all seven evaluated entity types. Two sponsor-field instances were intentionally not auto-redacted:

1. directional shadow produced empty OCR, so the fail-safe policy routed the field to review;
2. combined degradation produced 0.799 confidence, just below the 0.80 sponsor threshold, so it was also routed to review.

These are counted as successful detections but not as automatic-redaction coverage.

## Permanent redaction outputs

True PDF redaction was applied to all six populated FDA documents with PyMuPDF redaction annotations followed by `apply_redactions`. Review items remain visible and are outlined in orange.

| Measure | Result |
|---|---:|
| Redacted PDFs created | 6 |
| Automatic redaction regions | 71 |
| Review regions | 18 |
| Target-text removal verification | **100%** |
| Review-text retention verification | **100%** |
| Mean redaction write time | **0.0795 s/document** |

Verification compares the source and redacted PDF text inside each applied region. It confirms that the known target text was present before redaction and absent after redaction. Surrounding labels are allowed to remain because form-label text can overlap widget rectangles.

The PDFs were also re-rendered and visually inspected. Black regions are permanent redactions; orange outlines indicate human-review fields.

## Audit behavior

Each document produces a JSON audit record with:

- source and output SHA-256 checksums
- policy name and version
- page and field identifiers
- entity type, confidence, action, and detection method
- bounding box in PDF coordinates
- masked text preview and SHA-256 of the detected text

Raw detected sensitive text is not written to audit logs.

## Main files

- `src/regdoc_ai/redaction/models.py`
- `src/regdoc_ai/redaction/policy.py`
- `src/regdoc_ai/redaction/detectors.py`
- `src/regdoc_ai/redaction/pdf_redactor.py`
- `scripts/evaluate_redaction.py`
- `scripts/create_redaction_figures.py`
- `configs/redaction_policy.yaml`
- `results/redaction_benchmark/`

## Reproduce

```bash
python scripts/evaluate_redaction.py
python scripts/create_redaction_figures.py
pytest -q
```

## Limitations

- The local NER benchmark uses spaCy EntityRuler rather than a downloaded statistical or transformer NER model.
- Robustness metrics use template field coordinates because the stored Day 6 predictions do not contain word-level OCR boxes.
- Permanent redaction was measured on the clean vector PDFs. The detector was tested under scan degradation, but an image-only scanned-PDF corpus should be added before claiming scanned-PDF redaction performance at scale.
- The benchmark includes controlled identities and public protocol metadata, not genuine PII/CCI from regulated submissions.
