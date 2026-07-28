# RegDocAI model and system card

## System purpose

RegDocAI is a portfolio-scale system for extracting structured data from regulatory and
clinical documents. It supports document routing, OCR, fixed-form extraction, checkbox
recognition, ruled-table reconstruction, configurable PII/CCI review, permanent PDF
redaction, and human review.

It is not clinically validated, regulatory-qualified, or intended to make clinical
judgments.

## Executed components

| Component | Executed implementation |
|---|---|
| OCR | Tesseract 5.x |
| Image restoration | OpenCV deskewing, denoising, CLAHE, and sharpening |
| Document routing | OCR/layout rules with HOG + LinearSVC fallback |
| Fixed fields | Template coordinates, OCR, regex, and schema validation |
| Checkboxes | Contours and locally normalized ink ratio |
| Ruled tables | OpenCV grid detection plus Tesseract cell OCR |
| PDF-native tables | Camelot lattice baseline |
| Sensitive entities | Field semantics, regex, and spaCy EntityRuler |
| Redaction | PyMuPDF permanent redaction annotations |

## Included but not locally executed

PaddleOCR, PP-StructureV3, Table Transformer, and pretrained MobileNetV3-Small adapters
and notebooks are included. Their packages or model weights could not be downloaded in
the restricted runtime, so their performance is not reported.

## Headline performance

- 96.43% validated field exact match across 140 degraded field instances
- 100% checkbox accuracy across 77 degraded checkbox instances
- 100% exact row/column grid recovery across 20 real protocol tables
- 0.925 physical-cell F1 on the 20-table clean benchmark
- 100% held-out page classification accuracy across 96 page instances
- 98.21% exact match across 224 routed fields
- 1.00 sensitive-entity F1 and 0% false automatic redactions

## Human oversight

Low-confidence or policy-defined entities are routed to review. Reviewers may change an
action to redact, review, retain, or ignore before the final PDF is generated. Audit logs
store masked previews and hashes rather than raw sensitive text.

## Known risks

- Template-based field extraction may not generalize to revised form layouts.
- The table benchmark is dominated by ruled tables.
- OCR performance may deteriorate on handwriting, unusual fonts, severe occlusion, or
  non-English text.
- Controlled PII is used; performance on real-world personal data requires separate
  privacy-approved validation.
- High benchmark accuracy on a small, controlled portfolio dataset is not equivalent to
  enterprise or regulatory qualification.
