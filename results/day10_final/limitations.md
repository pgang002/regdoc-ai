# Known limitations and honest-use boundaries

- PaddleOCR, PP-StructureV3, Table Transformer, and pretrained MobileNetV3 weights were not executed locally because binary/model downloads were blocked. Their adapters and Colab paths are included, but no metrics are claimed.
- The production PostgreSQL, Redis, Celery, Prometheus, and Docker Compose services were not launched in this runtime because Docker and those external services were unavailable. Local SQLAlchemy/SQLite and thread-queue validation is reported separately.
- FDA Form 1571 is an Adobe LiveCycle/XFA document and requires Adobe-compatible flattening before standard OCR ingestion.
- The table benchmark focuses on ruled tables from two public Moderna protocols; borderless and handwritten tables require additional evaluation.
- PII identities and disclosure states are controlled test data. Public protocol metadata is used for realism, and CCI labels simulate an organizational policy rather than asserting that public data is confidential.
- Results are portfolio benchmarks, not clinical validation or regulatory qualification.
