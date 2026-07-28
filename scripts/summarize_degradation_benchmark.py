#!/usr/bin/env python3
"""Combine per-condition outputs and generate the degradation benchmark summary."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results/degradation_benchmark"
CONDITION_ROOT = RESULT_ROOT / "conditions"


def read_all(filename: str) -> pd.DataFrame:
    paths = sorted(CONDITION_ROOT.glob(f"*/{filename}"))
    if not paths:
        raise RuntimeError(f"No condition outputs found for {filename}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def main() -> None:
    fields = read_all("field_predictions.csv")
    checkboxes = read_all("checkbox_predictions.csv")
    pages = read_all("page_preprocessing.csv")
    fields.to_csv(RESULT_ROOT / "field_predictions.csv", index=False)
    checkboxes.to_csv(RESULT_ROOT / "checkbox_predictions.csv", index=False)
    pages.to_csv(RESULT_ROOT / "page_preprocessing.csv", index=False)

    field_by_condition = (
        fields.groupby(["condition", "pipeline"], as_index=False)
        .agg(
            field_count=("field_name", "count"),
            raw_exact_match_accuracy=("raw_exact_match", "mean"),
            validated_exact_match_accuracy=("exact_match", "mean"),
            mean_character_accuracy=("character_accuracy", "mean"),
            mean_character_error_rate=("character_error_rate", "mean"),
            mean_ocr_confidence=("mean_ocr_confidence", "mean"),
            mean_field_ocr_latency_seconds=("field_ocr_latency_seconds", "mean"),
        )
        .sort_values(["condition", "pipeline"])
    )
    field_by_condition.to_csv(RESULT_ROOT / "field_summary_by_condition.csv", index=False)

    degraded = fields[fields["condition"] != "clean"]
    field_overall = (
        degraded.groupby("pipeline", as_index=False)
        .agg(
            field_count=("field_name", "count"),
            raw_exact_match_accuracy=("raw_exact_match", "mean"),
            validated_exact_match_accuracy=("exact_match", "mean"),
            mean_character_accuracy=("character_accuracy", "mean"),
            mean_character_error_rate=("character_error_rate", "mean"),
            mean_ocr_confidence=("mean_ocr_confidence", "mean"),
            mean_field_ocr_latency_seconds=("field_ocr_latency_seconds", "mean"),
        )
        .sort_values("validated_exact_match_accuracy", ascending=False)
    )
    field_overall.to_csv(RESULT_ROOT / "field_summary_degraded_overall.csv", index=False)

    checkbox_by_condition = (
        checkboxes.groupby(["condition", "pipeline"], as_index=False)
        .agg(
            checkbox_count=("checkbox_name", "count"),
            accuracy=("correct", "mean"),
            mean_dark_ratio=("inner_dark_ratio", "mean"),
        )
        .sort_values(["condition", "pipeline"])
    )
    checkbox_by_condition.to_csv(
        RESULT_ROOT / "checkbox_summary_by_condition.csv", index=False
    )
    checkbox_overall = (
        checkboxes[checkboxes["condition"] != "clean"]
        .groupby("pipeline", as_index=False)
        .agg(checkbox_count=("checkbox_name", "count"), accuracy=("correct", "mean"))
        .sort_values("accuracy", ascending=False)
    )
    checkbox_overall.to_csv(
        RESULT_ROOT / "checkbox_summary_degraded_overall.csv", index=False
    )

    page_summary = (
        pages.groupby(["condition", "pipeline"], as_index=False)
        .agg(
            page_count=("page", "count"),
            mean_page_preprocessing_latency_seconds=(
                "page_preprocessing_latency_seconds",
                "mean",
            ),
            mean_absolute_rotation_residual=("absolute_rotation_residual", "mean"),
        )
        .sort_values(["condition", "pipeline"])
    )
    page_summary.to_csv(RESULT_ROOT / "page_summary.csv", index=False)

    payload = {
        "conditions": sorted(fields["condition"].unique().tolist()),
        "pipelines": sorted(fields["pipeline"].unique().tolist()),
        "evaluated_field_records": int(len(fields)),
        "evaluated_checkbox_records": int(len(checkboxes)),
        "unique_reference_fields": int(fields["field_name"].nunique()),
        "source_data": (
            "Official FDA form templates populated with public Moderna protocol metadata and "
            "controlled test identifiers; degradations are deterministic synthetic scan artifacts."
        ),
    }
    (RESULT_ROOT / "run_metadata.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("Degraded conditions overall:")
    print(field_overall.to_string(index=False))
    print(checkbox_overall.to_string(index=False))


if __name__ == "__main__":
    main()
