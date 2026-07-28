Generated inventory, benchmark, visualization, and evaluation outputs are written here.
Only small summary files should be committed to version control.

Day 6 document-classification and routed field-extraction outputs are under
`results/document_understanding/`. The MobileNet status file explicitly distinguishes
an executed model from an unavailable optional runtime.

Day 7 policy-driven sensitive-entity detection and PDF-redaction outputs are under
`results/redaction_benchmark/`.

Day 8 FastAPI/Streamlit application integration outputs are under `results/day8_app/`.
They include the generated OpenAPI schema, endpoint catalog, actual FDA-form and clinical-
table API responses, measured latency, and architecture figures. Local application
workspaces under `runtime/` are intentionally excluded from version control because they
may contain uploaded documents and private processing state.
