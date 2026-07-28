# RegDocAI data card

## Data sources

| Source | Use | Data label |
|---|---|---|
| Official FDA Forms 1571, 1572, 3454, and 3455 | Regulatory layouts and form controls | `source_real` |
| Public Moderna protocols NCT04470427 and NCT04796896 | Protocol metadata, pages, and tables | `source_real` |
| ClinicalTrials.gov public records | Registry and aggregate study metadata | `source_public_aggregate` |
| Controlled form identities and disclosure states | Safe evaluation of PII and checkboxes | `synthetic_pii` |
| Deterministic image degradations | Robustness testing | `augmented_scan` |

## Dataset composition

- Six populated official FDA-form samples
- 90 labeled clean text fields
- 22 labeled clean checkboxes
- 64 deterministic degraded page images
- 20 real protocol tables with PDF-derived structure and text ground truth
- 96 held-out document-classification page instances
- 224 routed field-extraction instances
- 168 expected redaction entities in the robustness benchmark

## Provenance

Source URLs, checksums, retrieval metadata, content labels, and limitations are retained
under `configs/` and `data/manifests/`. Public protocol content is not represented as
private clinical data.

## Privacy and controlled data

No private patient-level records are used. Investigator identities, addresses,
signatures, and disclosure states are controlled test values where real personal data is
unnecessary. CCI labels simulate an enterprise policy; they do not assert that the public
source protocols are confidential.

## Limitations

- Two public Moderna protocols are not representative of every regulatory submission.
- The table set emphasizes ruled tables.
- Generated scan artifacts approximate, but do not replace, naturally scanned document
  collections.
- FDA Form 1571 is XFA-based and requires compatible flattening before standard rendering.
