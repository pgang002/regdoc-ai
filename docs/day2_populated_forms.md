# Day 2 - Populated official forms and exact field evaluation

## Objective

Create an exact-ground-truth benchmark from actual official FDA form templates while
using public study metadata and avoiding real personal or financial-disclosure data.

## Actual public inputs

- Official FDA Form 1572, Form 3454, and Form 3455 PDFs.
- Public Moderna protocol PDF for NCT04470427 (`mRNA-1273-P301`).
- Public Moderna protocol PDF for NCT04796896 (`mRNA-1273-P204`).

The protocol title, protocol number, NCT identifier, sponsor, amendment number, and
amendment date are extracted directly from the downloaded protocol PDFs. Source hashes
are stored in `data/interim/protocol_metadata/studies.json`.

## Controlled values

Investigator names, addresses, facilities, applicant representatives, checkbox states,
and signatures are controlled test values. They do not represent real investigators,
real sites, or real financial disclosures. Every ground-truth JSON file includes this
privacy notice and identifies the source of each text value.

## Generated benchmark

- 6 populated official-form samples
- 8 rendered pages at 300 DPI
- 90 populated text fields
- 22 checked or unchecked boxes
- Editable and flattened PDFs
- Page images and exact widget coordinates
- Per-document JSON ground truth and a CSV manifest

## Extraction approach

1. Populate AcroForm widgets with PyMuPDF.
2. Flatten the widget appearances so OCR sees stable page content.
3. Render each page at 300 DPI.
4. Crop each known field using the original PDF widget rectangle.
5. Run Tesseract on raw and adaptive-thresholded crops.
6. Apply schema-aware validation for postal codes, state codes, dates, domain tokens,
   and sponsor entity resolution against known submission metadata.
7. Detect checkbox state using inner-region ink density after excluding the printed box
   border.

## Measured clean-document results

| Method | Text fields | Raw OCR exact match | Validated exact match | Mean validated character accuracy |
|---|---:|---:|---:|---:|
| Raw crops | 90 | 94.44% | 100.00% | 100.00% |
| Adaptive thresholding | 90 | 94.44% | 100.00% | 100.00% |

Checkbox detection achieved **100.00% accuracy across 22 boxes**. Checked boxes had a
mean inner dark-pixel ratio of 0.332, while unchecked boxes had a ratio of 0.000 under
the clean rendered condition.

These results are limited to clean, template-aligned renderings. They are not yet a
claim about scanned, photographed, skewed, blurred, or compressed documents.

## Engineering defects found and fixed

### Leading-zero corruption

`pytesseract` returns a Pandas DataFrame. Its automatic type inference converted OCR
text such as `02115` into the numeric-looking value `2115.0`. The OCR adapter now forces
the text column to string type, preserving leading zeros.

### Multi-line reading-order corruption

Sorting OCR words by Tesseract block IDs scrambled long protocol-name fields even
though the recognized words were correct. The OCR adapter now reconstructs lines from
physical `top` and `left` coordinates, preserving visual reading order.

## Verification

Representative flattened samples from all three form types were rendered with both
Poppler and PDFium. The field content and checkbox appearances remained visible and
aligned in both renderers. Renderer-specific anti-aliasing differences are recorded in
`results/render_verification/`.

## Reproduction

```powershell
python scripts/extract_protocol_metadata.py
python scripts/generate_populated_forms.py
python scripts/evaluate_populated_forms.py
pytest
```

## Next milestone

Create controlled scan degradations, then measure how OCR field accuracy and checkbox
accuracy change under rotation, blur, noise, JPEG compression, shadows, and low
contrast. That benchmark will determine the preprocessing and human-review thresholds
used in the application pipeline.
