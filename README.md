# RegDocAI

RegDocAI is a portfolio-scale regulatory Document AI project for extracting text,
fields, checkboxes, and tables from scanned pharmaceutical and clinical documents.
It is designed to demonstrate OCR benchmarking, document computer vision, structured
information extraction, validation, redaction, and production-oriented model serving.

## Why this project

The target use case is a regulatory-document workflow similar to those used in drug
development. The repository uses official public FDA form templates, public aggregate
ClinicalTrials.gov records, and a documented PubTables-1M evaluation subset. It does
not use private patient records or represent generated identities as real clinical data.

## Current completed milestones

Ten implementation milestones are complete. Optional deep-model executions remain isolated where model downloads are required, and unavailable infrastructure is never presented as executed:

1. **Data foundation:** official FDA form acquisition, source manifests, PDF inspection,
   rendering checks, and a blank-template Tesseract baseline.
2. **Populated forms:** six official FDA form samples populated with public Moderna
   protocol metadata plus controlled test identities, with 90 exact text-field labels
   and 22 checkbox labels.
3. **Scan robustness:** 64 deterministic degraded page images covering rotation, blur,
   noise, low contrast, directional shadow, JPEG compression, and a combined condition.
4. **OCR-engine comparison harness:** identical field crops and preprocessing for
   Tesseract and PaddleOCR, with explicit per-engine completion status and a Colab runner.
5. **Real table extraction:** 20 actual public Moderna protocol tables with PDF-derived
   structure/text ground truth, Camelot and OpenCV/Tesseract baselines, scan-robustness
   testing, CSV/JSON/HTML reconstruction, and deep-model Colab runners.
6. **Document understanding:** source-separated hybrid classification for FDA forms,
   clinical protocol pages, and clinical table pages, followed by routed template/anchor
   field extraction across eight scan conditions.
7. **PII/CCI redaction:** configurable regex, spaCy EntityRuler, and field-semantic detection;
   fail-safe review routing; OCR/PDF coordinate mapping; true PDF redaction; audit-safe logs;
   and source-versus-output removal verification.
8. **Application layer:** FastAPI upload and artifact endpoints, reusable service orchestration,
   Streamlit human review, policy-decision overrides, structured downloads, and actual API
   integration runs for an FDA form and clinical protocol table.
9. **Persistent asynchronous processing:** SQLAlchemy/Alembic metadata persistence,
   PostgreSQL production configuration, Celery/Redis job dispatch, ordered progress events,
   retries, batch status, Docker Compose, and actual local asynchronous integration validation.
10. **Production readiness:** liveness/readiness endpoints, Prometheus metrics, database-backed
    operational summaries, CI, deployment checks, consolidated evaluation, model/data cards,
    and quantified resume evidence.

**Fifty-three automated tests pass.**

## Final scorecard

| Capability | Measured result | Scope |
|---|---:|---|
| Degraded-form field exact match | **96.43%** | 140 field instances across seven scan degradations |
| Checkbox accuracy | **100%** | 77 degraded checkbox instances |
| Clean table grid recovery | **100%** | 20 real Moderna protocol tables |
| Table physical-cell F1 | **0.925** | 20 real protocol tables |
| Held-out page classification | **100%** | 96 page instances across eight conditions |
| Routed field exact match | **98.21%** | 224 field instances |
| Sensitive-entity detection | **1.00 F1** | 168 expected entities across eight conditions |
| False automatic redactions | **0%** | Hybrid policy benchmark |

The consolidated evidence is in `results/day10_final/`. The final resume bullets use only
executed metrics and explicitly separate the locally measured queue fallback from the
unexecuted PostgreSQL/Redis/Celery container stack.

### Clean populated-form result

| Preprocessing | Text fields | Raw OCR exact match | Validated exact match | Checkbox accuracy |
|---|---:|---:|---:|---:|
| Raw | 90 | 94.44% | 100.00% | 100.00% |
| Adaptive thresholding | 90 | 94.44% | 100.00% | 100.00% |

### Degraded-scan result

The robustness evaluation uses a stratified subset of 20 fields and 11 checkboxes from
one populated sample of each form type. Across seven degraded conditions, each pipeline
produced 140 field predictions and 77 checkbox predictions.

| Pipeline | Validated field exact match | Mean character error rate | Checkbox accuracy |
|---|---:|---:|---:|
| Raw baseline | 76.43% | 0.2833 | 84.42% |
| Enhanced deskew/restoration | 96.43% | 0.1040 | 100.00% |

The enhanced pipeline improved validated field accuracy by **20.0 percentage points**,
reduced field exact-match errors by **84.85%**, and reduced mean character error rate by
**63.27%**. At 2 degrees of rotation, field accuracy recovered from 15% to 100% and
checkbox accuracy from 45.45% to 100%. Under the combined rotation, blur, noise, shadow,
contrast, and compression condition, field accuracy increased from 30% to 85% and
checkbox accuracy from 45.45% to 100%.

The restoration path uses Hough-line deskewing, median denoising, CLAHE, unsharp masking,
and locally normalized checkbox ink detection. Detailed methods and limitations are in
`docs/day3_degradation_benchmark.md`; results are in
`results/degradation_benchmark/`.

The validation layer uses production-available constraints rather than target answers:
postal-code and state formats, date normalization, domain-token cleanup, and sponsor
entity resolution against known submission metadata.

The earlier blank-template OCR comparison remains in
`results/tesseract_baseline/summary.csv`. On four actual FDA pages, CLAHE reduced proxy
CER by 6.9% and proxy WER by 7.4%.


### Day 4 OCR-engine benchmark status

The controlled Day 4 benchmark uses the same 20 selected fields across all eight scan
conditions, for 160 field instances per OCR engine. Tesseract was executed in the current
runtime; PaddleOCR could not be installed because binary-package and model downloads were
blocked, so the repository records it as unavailable rather than creating placeholder
metrics.

| Executed engine | Fields | Raw exact match | Validated exact match | Mean CER | Mean latency/field |
|---|---:|---:|---:|---:|---:|
| Tesseract 5.5.0 | 160 | 88.13% | 96.88% | 0.0910 | 0.1234 s |

The complete, executable comparison path is in
`notebooks/04_tesseract_vs_paddleocr.ipynb`. Install the optional runtime from
`requirements-paddle.txt` and rerun both engines with `--strict`; the command will fail
rather than describe an incomplete run as a completed comparison. Details are in
`docs/day4_ocr_engine_benchmark.md` and `results/ocr_engine_benchmark/`.

## Day 5 table-extraction results

Day 5 uses 20 tables rendered directly from two public Moderna protocols
(NCT04470427 and NCT04796896). Table content is not generated; reference structure
and text come from the source PDFs' vector rules and embedded text.

| Pipeline | Exact grid shape | Row F1 | Column F1 | Physical-cell F1 | Mean text CER |
|---|---:|---:|---:|---:|---:|
| Camelot lattice | 95.0% | 1.000 | 0.992 | Not reported | 0.0089 |
| OpenCV + Tesseract | 100.0% | 1.000 | 1.000 | 0.925 | 0.1927 |

On two representative complex tables across four degraded scan conditions, the
enhanced image pipeline retained 100% exact grid shape and reduced mean text CER from
0.4987 to 0.2718 across degraded images. Details are in
`docs/day5_table_extraction.md` and `results/table_extraction_benchmark/`.

Table Transformer and PP-StructureV3 were not run in the restricted local runtime.
Their metrics are not filled with placeholders. The executable comparison notebook is
`notebooks/05_table_extraction_deep_models.ipynb`.


## Day 6 document-understanding results

Day 6 trains the local image fallback on pages from NCT04470427 and holds out all
NCT04796896 pages. Twelve held-out source pages across five classes are evaluated under
eight scan conditions, producing 96 page instances.

| Pipeline | Accuracy | Macro F1 | Mean latency/page |
|---|---:|---:|---:|
| Image-only HOG + LinearSVC | 83.33% | 0.8933 | 0.0158 s |
| Restored rule-only | 100.00% | 1.0000 | 0.9077 s |
| Hybrid router | 100.00% | 1.0000 | 0.9173 s |

After hybrid routing, the field pipeline evaluated 224 field instances and achieved
**98.21% exact match**, **0.0727 mean CER**, and **100% routing accuracy**. Protocol-cover
fields achieved 100%; FDA 1572, 3454, and 3455 achieved 97.22%, 100%, and 95%,
respectively. Details and limitations are in `docs/day6_document_understanding.md`.

The executable MobileNetV3-Small transfer-learning path is included, but pretrained
weights could not be downloaded locally. No deep-model metrics are reported until that
script is run in a model-enabled environment.

## Day 7 redaction results

Day 7 evaluates policy-driven sensitive-entity detection on the held-out NCT04796896
FDA forms across the same eight scan conditions used in Day 6. The benchmark uses
controlled test identities and public protocol metadata; CCI labels simulate an
enterprise policy and do not imply that the public source documents are confidential.

| Pipeline | Precision | Recall | F1 | Auto-redaction coverage | False-redaction rate |
|---|---:|---:|---:|---:|---:|
| Regex only | 0.5963 | 0.5714 | 0.5836 | 73.93% | 47.44% |
| Hybrid policy | **1.0000** | **1.0000** | **1.0000** | **97.92%** | **0.00%** |

The hybrid pipeline combines field semantics, regex, generic spaCy EntityRuler PERSON
patterns, OCR coordinates, and confidence thresholds. Two heavily degraded sponsor
fields were routed to review instead of being auto-redacted, yielding 98.81% policy-action
accuracy without a false automatic redaction.

All six populated FDA PDFs were processed with true PDF redaction. The system applied
71 permanent redaction regions and 18 review regions. Target-text removal verification
and review-text retention both achieved **100%**. Audit JSON files store hashes and masked
previews rather than raw detected sensitive text. Details are in
`docs/day7_redaction.md` and `results/redaction_benchmark/`.

The local NER run uses a generic spaCy EntityRuler because no statistical English model
was downloaded. The optional extension is in `notebooks/07_pii_cci_redaction.ipynb`; no
unexecuted statistical-model metrics are reported.

## Day 8 application results

Day 8 exposes the existing pipeline through FastAPI and provides a Streamlit review
interface. The actual API integration processed the two-page populated FDA 1572 sample in
**11.08 seconds**, returning 28 fields, 4 checkboxes, and 29 policy candidates. Reviewer
redaction decisions were applied in **0.21 seconds**. A real NCT04796896 protocol table
image was classified and reconstructed to CSV, JSON, and HTML in **1.82 seconds**.

All generated artifacts were downloaded through their HTTP endpoints during the
integration run. The backend, service layer, UI code, OpenAPI schema, endpoint catalog,
and measured outputs are documented in `docs/day8_fastapi_streamlit.md` and
`results/day8_app/`.

Streamlit was not downloadable in the restricted local runtime, so no browser screenshot
or unexecuted UI metric is claimed. The app code compiled successfully and is ready for the
local Python 3.11/Anaconda environment defined by `requirements-app.txt`.


## Day 9 asynchronous-processing results

Day 9 adds persistent job metadata, ordered progress events, batch processing, retry handling,
and production deployment configuration. PostgreSQL is the production metadata store and
Redis is the Celery broker/result backend. The existing workspace remains responsible for
large source and artifact files.

The restricted runtime did not expose PostgreSQL/Redis services and could not install Celery,
the Redis Python client, or the PostgreSQL driver. The production Docker stack is therefore
implemented but not falsely described as executed. The same repository, job schema, pipeline,
and progress callbacks were executed through SQLAlchemy/SQLite and a two-worker thread queue.

An actual two-document batch used the populated official FDA 1572 and a real NCT04796896
protocol table:

| Measure | Result |
|---|---:|
| Jobs completed | 2 of 2 |
| Observed batch time | 14.43 s |
| Local two-worker throughput | 8.32 documents/min |
| Persisted batch events | 20 |
| Registered batch artifacts | 12 |

The FDA form persisted 28 fields, 4 checkboxes, and 29 redaction candidates and correctly
ended in `needs_review`. The real clinical table ended in `completed` with one reconstructed
table. A forced missing-source job failed on attempt 1, was requeued after restoring the same
actual protocol image, and completed on attempt 2. The validation database contained 3
documents, 3 jobs, 31 job events, and 16 artifacts.

A separate HTTP validation returned `202 Accepted` from `/v1/jobs/process`, persisted nine
progress events, and exposed the completed document and seven artifacts. The initial Alembic
migration successfully upgraded a fresh database. Details are in
`docs/day9_async_processing.md` and `results/day9_async/`.

## Day 10 production-readiness results

The API is now version 1.0.0 and exposes liveness, readiness, Prometheus metrics, and a
persisted operational summary. Request metrics normalize dynamic IDs to avoid
high-cardinality labels. Docker Compose adds API health checks and a Prometheus service;
GitHub Actions compiles, lints, tests, and builds the production image.

A final actual-data monitoring validation processed the populated FDA 1572 and a real
NCT04796896 protocol table. Both jobs reached terminal states, readiness passed, 12
artifacts were persisted, operational success was 100%, and all required Prometheus metric
families were emitted. The observed batch time was 16.78 seconds in the local two-worker
validation mode.

Docker was unavailable in the execution runtime. Compose, Prometheus, and CI configuration
were statically validated, but container startup is not claimed. See
`docs/day10_production_readiness.md`, `docs/model_card.md`, `docs/data_card.md`, and
`results/day10_final/`.

## Important source-document finding

The official FDA Form 1571 is a dynamic Adobe LiveCycle/XFA PDF. Standard PDF renderers
show its fallback "Please wait" page instead of the form. RegDocAI keeps the original
source and marks it as requiring an Adobe-compatible flattening/export step before OCR.
The first benchmark therefore uses Forms 1572, 3454, and 3455.

This is deliberately treated as a document-ingestion compatibility issue rather than
silently replacing the official source with an unofficial copy.

## Data provenance

| Source | Use |
|---|---|
| FDA IND forms page | Official regulatory templates and checkbox/form layouts |
| ClinicalTrials.gov API v2 | Public study metadata and posted aggregate results |
| Microsoft PubTables-1M | Table detection and structure-recognition evaluation |

Source URLs, retrieval information, expected use, and limitations are recorded in
`configs/data_sources.yaml`. The ClinicalTrials.gov script requests public registry and
aggregate-results records only.

## Repository layout

```text
regdoc-ai/
├── Dockerfile
├── docker-compose.yml
├── alembic/
├── configs/
│   ├── data_sources.yaml
│   ├── degradation_benchmark.yaml
│   ├── ocr_engine_benchmark.yaml
│   ├── table_benchmark.yaml
│   ├── document_understanding.yaml
│   ├── redaction_policy.yaml
│   ├── prometheus.yml
│   └── app.yaml
├── data/
│   ├── raw/fda_forms/
│   ├── raw/clinicaltrials/
│   ├── raw/pubtables/
│   ├── interim/pdf_pages/
│   ├── interim/protocol_metadata/
│   ├── processed/populated_forms/
│   ├── processed/degraded_forms/
│   ├── processed/table_benchmark/
│   ├── processed/document_understanding/
│   ├── processed/redaction_benchmark/
│   └── manifests/
├── api/
│   └── main.py
├── app/
│   └── streamlit_app.py
├── scripts/
│   ├── download_fda_forms.py
│   ├── build_data_manifest.py
│   ├── inspect_fda_forms.py
│   ├── render_fda_forms.py
│   ├── run_tesseract_baseline.py
│   ├── extract_protocol_metadata.py
│   ├── generate_populated_forms.py
│   ├── evaluate_populated_forms.py
│   ├── generate_degraded_forms.py
│   ├── evaluate_degradation_condition.py
│   ├── summarize_degradation_benchmark.py
│   ├── create_degradation_figures.py
│   ├── benchmark_ocr_engines.py
│   ├── check_paddleocr_runtime.py
│   ├── build_protocol_table_benchmark.py
│   ├── evaluate_table_extraction.py
│   ├── generate_degraded_tables.py
│   ├── evaluate_table_robustness.py
│   ├── create_table_figures.py
│   ├── build_document_understanding_benchmark.py
│   ├── train_document_classifier.py
│   ├── evaluate_document_understanding.py
│   ├── create_document_understanding_figures.py
│   ├── evaluate_redaction.py
│   ├── create_redaction_figures.py
│   ├── train_mobilenet_document_classifier.py
│   ├── build_protocol_manifest.py
│   ├── fetch_clinicaltrials.py
│   ├── build_clinical_tables.py
│   └── prepare_pubtables_subset.py
├── src/regdoc_ai/
│   ├── augmentation/
│   ├── preprocessing/
│   ├── ocr/
│   ├── forms/
│   ├── extraction/
│   ├── checkboxes/
│   ├── evaluation/
│   ├── tables/
│   ├── classification/
│   ├── redaction/
│   ├── persistence/
│   ├── service/
│   ├── worker/
│   ├── monitoring/
│   └── schemas/
├── tests/
├── results/
└── notebooks/
```

## Windows/Anaconda setup

```powershell
conda env create -f environment.yml
conda activate regdoc-ai
pip install -e .
```

Tesseract must be available on `PATH`. The Conda environment installs it from
`conda-forge`.

For the PostgreSQL/Celery/Redis deployment dependencies:

```powershell
pip install -r requirements-infra.txt
```

Run the complete production-style stack with:

```powershell
docker compose up --build
```

The local validation mode remains available with `REGDOC_QUEUE_MODE=thread` and a SQLite
`REGDOC_DATABASE_URL`; it is intended for tests, not as the production architecture.

For the optional PaddleOCR comparison, install its isolated requirements after the core
environment is working:

```powershell
pip install -r requirements-paddle.txt
python scripts/check_paddleocr_runtime.py
python scripts/benchmark_ocr_engines.py --engines tesseract paddleocr --strict
```

The `--strict` flag prevents an unavailable engine from being mistaken for a completed
comparison.

## Reproduce the current milestone

```powershell
python scripts/extract_protocol_metadata.py
python scripts/generate_populated_forms.py
python scripts/evaluate_populated_forms.py
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
python scripts/benchmark_ocr_engines.py --engines tesseract
python scripts/build_protocol_table_benchmark.py
python scripts/evaluate_table_extraction.py
python scripts/generate_degraded_tables.py
python scripts/evaluate_table_robustness.py
python scripts/create_table_figures.py
python scripts/build_document_understanding_benchmark.py
python scripts/train_document_classifier.py
python scripts/evaluate_document_understanding.py
python scripts/create_document_understanding_figures.py
python scripts/evaluate_redaction.py
python scripts/create_redaction_figures.py
python scripts/create_final_scorecard.py
python scripts/validate_deployment.py
python scripts/run_day10_validation.py
pytest
```

The two public protocol PDFs are already included in `data/raw/clinicaltrials/` for the
reproducible benchmark. The live API script remains available for expanding the public
study set.

## Pull actual ClinicalTrials.gov records

The seed file contains actual public NCT identifiers associated with Moderna studies.
The command below fetches each complete current API record:

```powershell
python scripts/fetch_clinicaltrials.py --mode seed
python scripts/build_clinical_tables.py
```

A broader sponsor pull can be run with:

```powershell
python scripts/fetch_clinicaltrials.py `
  --mode sponsor `
  --sponsor "ModernaTX, Inc." `
  --max-studies 50
```

The output is not participant-level data. It contains public registry fields and posted
aggregate results, when available.

## Planned implementation sequence

1. **Complete:** actual source acquisition and blank-template OCR baseline.
2. **Complete:** populate official forms and evaluate exact fields and checkboxes.
3. **Complete:** scan degradations, deskew/restoration, robust checkbox detection, and measured robustness evaluation.
4. **Harness complete; Paddle execution pending:** benchmark Tesseract against PaddleOCR on identical field crops. Tesseract has been measured; run the included Colab notebook to complete PaddleOCR in a runtime that permits binary/model downloads.
5. **Local benchmark complete; optional deep models pending:** compare PP-StructureV3 and Table Transformer on the same 20 actual annotated tables using the included Colab notebook.
6. **Complete:** source-separated document classification and routed field extraction.
7. **Complete:** PII/CCI detection, fail-safe review routing, audit logging, and true PDF redaction.
8. **Complete:** build Streamlit and FastAPI interfaces.
9. **Complete with local execution validation:** add PostgreSQL metadata storage and
   Celery/Redis batch processing; run the Docker stack in an environment with the required
   services to collect production-service metrics.
10. **Complete:** operational monitoring, Prometheus configuration, CI, production-readiness
    validation, consolidated scorecard, model/data cards, and measured resume bullets.

Optional next extension: run PaddleOCR, PP-StructureV3, Table Transformer, and pretrained
MobileNetV3 in a model-enabled environment, then fine-tune only when measured error analysis
justifies it.

## Evaluation plan

- OCR: CER, WER, word confidence, and latency
- Fields: precision, recall, F1, exact match, and validation failure rate
- Checkboxes: state accuracy and ambiguous-case review rate
- Tables: detection AP, cell F1, GriTS/TEDS, and reconstruction exact match
- Redaction: entity precision/recall/F1, missed-redaction rate, and false-redaction rate
- Production: page latency, throughput, failure rate, retry rate, and review routing

## Data-use labels

Every generated document will use one of these labels:

- `source_real`
- `source_public_aggregate`
- `template_real_content_controlled`
- `synthetic_pii`
- `augmented_scan`

This distinction prevents synthetic identifiers or controlled form populations from
being misrepresented as real patient-level clinical records.
