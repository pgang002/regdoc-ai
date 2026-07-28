# Resume-ready project bullets

Use two or three bullets depending on available resume space.

- Built **RegDocAI**, an end-to-end regulatory Document AI pipeline using Python, OpenCV, Tesseract, scikit-learn, FastAPI, and Streamlit to classify FDA forms and clinical-protocol pages, extract fields and checkboxes, reconstruct tables, and export validated JSON/CSV/HTML; achieved **98.2% routed field exact match across 224 fields** and **100% held-out document classification accuracy across 96 degraded page instances**.
- Reconstructed **20 real Moderna clinical-protocol tables** with **100% exact row/column grid recovery** and **0.925 physical-cell F1**, while image restoration raised degraded-form field accuracy from **76.4% to 96.4%** and checkbox accuracy from **84.4% to 100%**.
- Implemented policy-driven PII/CCI detection and permanent PDF redaction with **1.00 entity F1**, **97.9% automatic-redaction coverage**, and **0% false automatic redactions**; designed asynchronous processing with SQLAlchemy/PostgreSQL, Redis/Celery, Docker Compose, retries, persistent progress events, and Prometheus monitoring.

## Accuracy note

The PostgreSQL/Redis/Celery and Docker Compose stack is implemented but was not executed in the restricted build environment. The measured asynchronous throughput of 8.32 documents/minute came from the documented two-worker local validation mode, so that throughput should not be attributed to Celery or PostgreSQL.
