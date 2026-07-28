#!/usr/bin/env python3
"""Run real Day 8 API workflows on an FDA form PDF and a protocol table image."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from api.main import app
from regdoc_ai.evaluation.text_metrics import character_error_rate, normalize_text


def process_file(client: TestClient, path: Path, media_type: str) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    with path.open("rb") as handle:
        response = client.post(
            "/v1/documents/process",
            files={"file": (path.name, handle, media_type)},
        )
    response.raise_for_status()
    return response.json(), time.perf_counter() - started


def check_artifacts(client: TestClient, result: dict[str, Any]) -> tuple[int, int]:
    checked = 0
    total_bytes = 0
    for artifact in result["artifacts"]:
        response = client.get(artifact["download_path"])
        response.raise_for_status()
        checked += 1
        total_bytes += len(response.content)
    return checked, total_bytes


def main() -> None:
    form_sample = PROJECT_ROOT / "data/processed/populated_forms/flattened/FDA_1572_NCT04796896.pdf"
    table_sample = PROJECT_ROOT / "data/processed/table_benchmark/images/NCT04796896_p116_t0.png"
    for sample in (form_sample, table_sample):
        if not sample.exists():
            raise FileNotFoundError(sample)

    client = TestClient(app)
    form, form_seconds = process_file(client, form_sample, "application/pdf")
    review_decisions = [
        {"candidate_id": item["candidate_id"], "action": item["action"]}
        for item in form["redaction_candidates"]
    ]
    review_started = time.perf_counter()
    review_response = client.post(
        f"/v1/documents/{form['document_id']}/redactions/apply",
        json={"decisions": review_decisions},
    )
    review_response.raise_for_status()
    reviewed = review_response.json()
    review_seconds = time.perf_counter() - review_started
    form_downloads, form_bytes = check_artifacts(client, form)
    for key in ("artifact", "audit_artifact"):
        response = client.get(reviewed[key]["download_path"])
        response.raise_for_status()
        form_downloads += 1
        form_bytes += len(response.content)

    table, table_seconds = process_file(client, table_sample, "image/png")
    table_downloads, table_bytes = check_artifacts(client, table)

    form_gt = json.loads(
        (PROJECT_ROOT / "data/processed/populated_forms/ground_truth/FDA_1572_NCT04796896.json").read_text(encoding="utf-8")
    )
    predicted_fields = {(int(row["page"]), str(row["name"])): str(row.get("value") or "") for row in form["fields"]}
    field_matches = [
        normalize_text(str(item["value"])) == normalize_text(predicted_fields.get((int(item["page"]), str(item["name"])), ""))
        for item in form_gt["fields"]
    ]
    predicted_checks = {
        (int(row["page"]), str(row["name"])): row["state"] == "checked"
        for row in form["checkboxes"]
    }
    checkbox_matches = [
        predicted_checks.get((int(item["page"]), str(item["name"]))) == bool(item["value"])
        for item in form_gt["checkboxes"]
    ]

    table_annotation = json.loads(
        (PROJECT_ROOT / "data/processed/table_benchmark/annotations/NCT04796896_p116_t0.json").read_text(encoding="utf-8")
    )
    table_matrix = json.loads(
        (PROJECT_ROOT / "runtime/day8" / table["document_id"] / "table_page_1.json").read_text(encoding="utf-8")
    )
    table_cers = []
    table_exact = []
    for row_index in range(int(table_annotation["logical_rows"])):
        for col_index in range(int(table_annotation["logical_columns"])):
            reference = table_annotation["matrix"][row_index][col_index]
            prediction = table_matrix[row_index][col_index]
            table_cers.append(character_error_rate(reference, prediction))
            table_exact.append(normalize_text(reference) == normalize_text(prediction))

    form_summary = {
        "workflow": "fda_form_review",
        "sample": str(form_sample.relative_to(PROJECT_ROOT)),
        "document_id": form["document_id"],
        "status": form["status"],
        "page_count": form["page_count"],
        "page_labels": ";".join(row["label"] for row in form["classifications"]),
        "field_count": len(form["fields"]),
        "checkbox_count": len(form["checkboxes"]),
        "field_exact_match": sum(field_matches) / len(field_matches),
        "checkbox_accuracy": sum(checkbox_matches) / len(checkbox_matches),
        "table_count": len(form["tables"]),
        "redaction_candidate_count": len(form["redaction_candidates"]),
        "needs_review_count": sum(
            item["action"] == "review" or item["needs_review"]
            for item in form["redaction_candidates"]
        ),
        "api_reported_processing_seconds": form["processing_seconds"],
        "client_observed_processing_seconds": form_seconds,
        "review_apply_seconds": review_seconds,
        "downloads_checked": form_downloads,
        "download_bytes": form_bytes,
        "reviewed_redacted_count": reviewed["redacted_count"],
        "reviewed_review_count": reviewed["review_count"],
        "reviewed_retained_count": reviewed["retained_count"],
        "reviewed_ignored_count": reviewed["ignored_count"],
        "health_status": client.get("/health").json()["status"],
    }
    table_summary = {
        "workflow": "clinical_table_review",
        "sample": str(table_sample.relative_to(PROJECT_ROOT)),
        "document_id": table["document_id"],
        "status": table["status"],
        "page_count": table["page_count"],
        "page_labels": ";".join(row["label"] for row in table["classifications"]),
        "field_count": len(table["fields"]),
        "checkbox_count": len(table["checkboxes"]),
        "table_count": len(table["tables"]),
        "table_exact_grid_shape": bool(
            len(table_matrix) == int(table_annotation["logical_rows"])
            and max((len(row) for row in table_matrix), default=0) == int(table_annotation["logical_columns"])
        ),
        "table_cell_exact_match": sum(table_exact) / len(table_exact),
        "table_mean_character_error_rate": sum(table_cers) / len(table_cers),
        "redaction_candidate_count": len(table["redaction_candidates"]),
        "needs_review_count": 0,
        "api_reported_processing_seconds": table["processing_seconds"],
        "client_observed_processing_seconds": table_seconds,
        "review_apply_seconds": 0.0,
        "downloads_checked": table_downloads,
        "download_bytes": table_bytes,
        "reviewed_redacted_count": 0,
        "reviewed_review_count": 0,
        "reviewed_retained_count": 0,
        "reviewed_ignored_count": 0,
        "health_status": client.get("/health").json()["status"],
    }
    output = PROJECT_ROOT / "results/day8_app"
    output.mkdir(parents=True, exist_ok=True)
    combined = [form_summary, table_summary]
    (output / "integration_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    pd.DataFrame(combined).to_csv(output / "integration_summary.csv", index=False)
    (output / "sample_form_processing_response.json").write_text(json.dumps(form, indent=2), encoding="utf-8")
    (output / "sample_review_response.json").write_text(json.dumps(reviewed, indent=2), encoding="utf-8")
    (output / "sample_table_processing_response.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    (output / "openapi.json").write_text(json.dumps(app.openapi(), indent=2), encoding="utf-8")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
