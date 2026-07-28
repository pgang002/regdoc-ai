from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - exercised only without optional UI dependency
    raise SystemExit(
        "Streamlit is not installed. Run: pip install -r requirements-app.txt"
    ) from exc

API_URL = os.getenv("REGDOC_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="RegDocAI Review", page_icon="📄", layout="wide")
st.title("RegDocAI regulatory document review")
st.caption("Upload a regulatory PDF or image, inspect extracted content, review PII/CCI decisions, and download structured or redacted outputs.")

with st.sidebar:
    st.subheader("Connection")
    st.code(API_URL)
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        health.raise_for_status()
        st.success("API connected")
    except requests.RequestException as exc:
        st.error(f"API unavailable: {exc}")
    st.info("Day 9 supports PostgreSQL-backed jobs and Celery/Redis background processing.")

processing_mode = st.radio(
    "Processing mode",
    ["Background job", "Immediate review"],
    horizontal=True,
    help="Background jobs return immediately and expose progress events. Immediate review retains the Day 8 synchronous workflow.",
)

uploaded = st.file_uploader(
    "Upload PDF or document image",
    type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"],
    help="Maximum API upload size: 25 MB.",
)

if "regdoc_result" not in st.session_state:
    st.session_state.regdoc_result = None
if "regdoc_job" not in st.session_state:
    st.session_state.regdoc_job = None
if "regdoc_batch_id" not in st.session_state:
    st.session_state.regdoc_batch_id = None

if uploaded is not None and st.button("Process document", type="primary"):
    endpoint = "/v1/jobs/process" if processing_mode == "Background job" else "/v1/documents/process"
    with st.spinner("Submitting document..." if processing_mode == "Background job" else "Running document pipeline..."):
        response = requests.post(
            f"{API_URL}{endpoint}",
            files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
            timeout=300,
        )
    if response.ok:
        if processing_mode == "Background job":
            st.session_state.regdoc_job = response.json()
            st.session_state.regdoc_result = None
            st.success("Document queued")
        else:
            st.session_state.regdoc_result = response.json()
            st.session_state.regdoc_job = None
            st.success("Processing completed")
    else:
        st.error(f"Processing failed: {response.status_code} {response.text}")

job = st.session_state.regdoc_job
if job:
    status_response = requests.get(f"{API_URL}/v1/jobs/{job['job_id']}", timeout=15)
    if status_response.ok:
        job = status_response.json()
        st.session_state.regdoc_job = job
        st.subheader("Background job")
        job_cols = st.columns(4)
        job_cols[0].metric("Status", job["status"])
        job_cols[1].metric("Progress", f"{job['progress']}%")
        job_cols[2].metric("Stage", job["current_stage"])
        job_cols[3].metric("Attempt", f"{job['attempt_count']}/{job['max_retries'] + 1}")
        st.progress(job["progress"] / 100.0)
        events_response = requests.get(f"{API_URL}/v1/jobs/{job['job_id']}/events", timeout=15)
        if events_response.ok:
            st.dataframe(pd.DataFrame(events_response.json()), use_container_width=True, hide_index=True)
        if job["status"] in {"completed", "needs_review"} and job.get("result_path"):
            result_response = requests.get(f"{API_URL}{job['result_path']}", timeout=60)
            if result_response.ok:
                st.session_state.regdoc_result = result_response.json()
        elif job["status"] == "failed":
            st.error(f"{job.get('error_type')}: {job.get('error_message')}")
            if st.button("Retry failed job"):
                retry_response = requests.post(f"{API_URL}/v1/jobs/{job['job_id']}/retry", timeout=30)
                if retry_response.ok:
                    st.session_state.regdoc_job = retry_response.json()
                    st.rerun()
                else:
                    st.error(retry_response.text)
        if st.button("Refresh job status"):
            st.rerun()

with st.expander("Batch processing"):
    batch_files = st.file_uploader(
        "Upload a document batch",
        type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
        key="batch_files",
    )
    if batch_files and st.button("Submit batch"):
        response = requests.post(
            f"{API_URL}/v1/batches/process",
            files=[
                ("files", (item.name, item.getvalue(), item.type or "application/octet-stream"))
                for item in batch_files
            ],
            timeout=120,
        )
        if response.ok:
            payload = response.json()
            st.session_state.regdoc_batch_id = payload["batch_id"]
            st.success(f"Queued {len(payload['jobs'])} documents")
        else:
            st.error(response.text)
    if st.session_state.regdoc_batch_id:
        batch_response = requests.get(
            f"{API_URL}/v1/batches/{st.session_state.regdoc_batch_id}", timeout=15
        )
        if batch_response.ok:
            batch = batch_response.json()
            st.progress(batch["progress"] / 100.0)
            st.write(batch["status_counts"])
            st.dataframe(pd.DataFrame(batch["jobs"]), use_container_width=True, hide_index=True)

result = st.session_state.regdoc_result
if result:
    status = result["status"]
    metric_cols = st.columns(6)
    metric_cols[0].metric("Status", status)
    metric_cols[1].metric("Pages", result["page_count"])
    metric_cols[2].metric("Fields", len(result["fields"]))
    metric_cols[3].metric("Checkboxes", len(result["checkboxes"]))
    metric_cols[4].metric("Tables", len(result["tables"]))
    metric_cols[5].metric("Seconds", f"{result.get('processing_seconds', 0):.2f}")

    tabs = st.tabs(["Document", "Fields", "Tables", "Redaction review", "Downloads", "Warnings"])

    with tabs[0]:
        st.subheader("Page classification")
        st.dataframe(pd.DataFrame(result["classifications"]), use_container_width=True, hide_index=True)
        preview_artifacts = [a for a in result["artifacts"] if a["name"].startswith("preview_page_")]
        for artifact in preview_artifacts:
            image_response = requests.get(f"{API_URL}{artifact['download_path']}", timeout=60)
            if image_response.ok:
                st.image(image_response.content, caption=artifact["name"], use_container_width=True)

    with tabs[1]:
        st.subheader("Extracted fields")
        fields = pd.DataFrame(result["fields"])
        if fields.empty:
            st.info("No structured fields were extracted from this document type.")
        else:
            display = fields.copy()
            if "bounding_box" in display.columns:
                display["bounding_box"] = display["bounding_box"].map(json.dumps)
            st.dataframe(display, use_container_width=True, hide_index=True)
        if result["checkboxes"]:
            st.subheader("Checkboxes")
            checks = pd.DataFrame(result["checkboxes"])
            checks["bounding_box"] = checks["bounding_box"].map(json.dumps)
            st.dataframe(checks, use_container_width=True, hide_index=True)

    with tabs[2]:
        if not result["tables"]:
            st.info("No table page was routed to the table-extraction pipeline.")
        for table in result["tables"]:
            st.subheader(f"Page {table['page']} table")
            csv_name = table.get("csv_path")
            artifact = next((a for a in result["artifacts"] if a["name"] == csv_name), None)
            if artifact:
                csv_response = requests.get(f"{API_URL}{artifact['download_path']}", timeout=60)
                if csv_response.ok:
                    from io import BytesIO

                    frame = pd.read_csv(BytesIO(csv_response.content), header=None)
                    st.dataframe(frame, use_container_width=True, hide_index=True)

    with tabs[3]:
        candidates = result["redaction_candidates"]
        if not candidates:
            st.info("No policy-driven redaction candidates were detected.")
        else:
            review_frame = pd.DataFrame(candidates)
            review_frame["bounding_box"] = review_frame["bounding_box"].map(json.dumps)
            edited = st.data_editor(
                review_frame,
                use_container_width=True,
                hide_index=True,
                disabled=[
                    "candidate_id", "entity_type", "page", "field_name", "masked_text",
                    "confidence", "bounding_box", "detection_methods", "needs_review",
                ],
                column_config={
                    "action": st.column_config.SelectboxColumn(
                        "Action", options=["redact", "review", "retain", "ignore"], required=True
                    )
                },
                key=f"redaction_editor_{result['document_id']}",
            )
            if st.button("Apply reviewed decisions"):
                payload = {
                    "decisions": [
                        {"candidate_id": row["candidate_id"], "action": row["action"]}
                        for row in edited.to_dict(orient="records")
                    ]
                }
                response = requests.post(
                    f"{API_URL}/v1/documents/{result['document_id']}/redactions/apply",
                    json=payload,
                    timeout=120,
                )
                if response.ok:
                    st.session_state.redaction_output = response.json()
                    st.success("Reviewed redaction PDF generated")
                else:
                    st.error(response.text)
            reviewed = st.session_state.get("redaction_output")
            if reviewed and reviewed["document_id"] == result["document_id"]:
                artifact = reviewed["artifact"]
                pdf_response = requests.get(f"{API_URL}{artifact['download_path']}", timeout=60)
                if pdf_response.ok:
                    st.download_button(
                        "Download reviewed redacted PDF",
                        data=pdf_response.content,
                        file_name=artifact["name"],
                        mime="application/pdf",
                    )

    with tabs[4]:
        st.download_button(
            "Download structured result JSON",
            data=json.dumps(result, indent=2),
            file_name=f"{result['document_id']}_result.json",
            mime="application/json",
        )
        for artifact in result["artifacts"]:
            response = requests.get(f"{API_URL}{artifact['download_path']}", timeout=60)
            if response.ok:
                st.download_button(
                    f"Download {artifact['name']}",
                    data=response.content,
                    file_name=artifact["name"],
                    mime=artifact["media_type"],
                    key=f"download_{artifact['name']}",
                )

    with tabs[5]:
        if result["warnings"]:
            for warning in result["warnings"]:
                st.warning(warning)
        else:
            st.success("No processing warnings")
