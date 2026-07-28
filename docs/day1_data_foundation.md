# Day 1 - Data foundation and ingestion findings

## Completed

- Acquired four current official FDA source forms from FDA URLs.
- Recorded file hashes, metadata, source URLs, and intended benchmark use.
- Rendered all PDFs with a standard renderer and visually checked the outputs.
- Identified Form 1571 as a dynamic LiveCycle/XFA compatibility case.
- Established actual ClinicalTrials.gov NCT seeds and a live API v2 fetcher.
- Added deterministic preparation for a PubTables-1M evaluation subset.
- Ran the initial Tesseract comparison on actual FDA pages.

## Initial OCR result

CLAHE improved the small benchmark relative to raw page rendering. This confirms that
preprocessing will remain an experimental variable, but the result is not yet suitable
for a resume claim because the pages are blank templates and the reference is the PDF
text layer rather than exact populated-field ground truth.

## Next implementation task

Create controlled, machine-readable field populations for Forms 1572, 3454, and 3455,
then flatten and render them so field-value accuracy and checkbox-state accuracy can be
measured directly.
