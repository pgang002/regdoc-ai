#!/usr/bin/env python3
"""Evaluate one scan-degradation condition using a stratified field subset."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from regdoc_ai.checkboxes.classical import detect_checkbox
from regdoc_ai.evaluation.field_metrics import score_field
from regdoc_ai.extraction.field_validation import validate_field_value
from regdoc_ai.extraction.template_fields import recognize_field
from regdoc_ai.preprocessing.document import enhance_document_page

PIPELINES = {
    "baseline_raw": {"enhance": False, "crop_preprocessing": "raw"},
    "enhanced_deskew_restored": {"enhance": True, "crop_preprocessing": "raw"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("condition")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/degradation_benchmark.yaml")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/degraded_forms/manifest.csv")
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=Path("data/processed/populated_forms/ground_truth"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/degradation_benchmark/conditions")
    )
    parser.add_argument("--checkbox-threshold", type=float, default=0.08)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    manifest_path = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
    gt_dir = (
        args.ground_truth_dir
        if args.ground_truth_dir.is_absolute()
        else PROJECT_ROOT / args.ground_truth_dir
    )
    output_root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    output_dir = output_root / args.condition
    output_dir.mkdir(parents=True, exist_ok=True)

    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    valid_conditions = {condition["name"] for condition in config["conditions"]}
    if args.condition not in valid_conditions:
        raise ValueError(f"Unknown condition {args.condition!r}; expected {sorted(valid_conditions)}")

    selected_samples = set(config["evaluation_sample_ids"])
    selected_fields: dict[str, set[str]] = {
        form_type: set(names) for form_type, names in config["evaluation_fields"].items()
    }
    manifest = pd.read_csv(manifest_path, dtype={"nct_id": str})
    manifest = manifest[
        (manifest["condition"] == args.condition)
        & (manifest["sample_id"].isin(selected_samples))
    ].copy()

    ground_truth = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in gt_dir.glob("*.json")
    }
    field_rows: list[dict[str, Any]] = []
    checkbox_rows: list[dict[str, Any]] = []
    page_rows: list[dict[str, Any]] = []

    for record in manifest.to_dict(orient="records"):
        gt = ground_truth[record["sample_id"]]
        image_path = PROJECT_ROOT / record["degraded_image"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Unable to read {image_path}")
        dpi = int(gt["rendering"]["dpi"])
        page_number = int(record["page"])

        page_variants: dict[str, tuple[Any, float, float]] = {
            "baseline_raw": (image, 0.0, 0.0)
        }
        started = time.perf_counter()
        enhanced = enhance_document_page(image)
        page_variants["enhanced_deskew_restored"] = (
            enhanced.image,
            time.perf_counter() - started,
            enhanced.estimated_skew_degrees,
        )

        for pipeline_name, (page_image, page_latency, estimated_skew) in page_variants.items():
            pipeline = PIPELINES[pipeline_name]
            page_rows.append(
                {
                    "sample_id": record["sample_id"],
                    "form_type": record["form_type"],
                    "page": page_number,
                    "condition": args.condition,
                    "pipeline": pipeline_name,
                    "true_rotation_degrees": float(record["param_rotation_degrees"]),
                    "estimated_skew_degrees": round(float(estimated_skew), 6),
                    "absolute_rotation_residual": round(
                        abs(float(record["param_rotation_degrees"]) + float(estimated_skew)), 6
                    ),
                    "page_preprocessing_latency_seconds": round(page_latency, 4),
                    "image_path": record["degraded_image"],
                }
            )

            allowed = selected_fields[gt["form_type"]]
            for field in gt["fields"]:
                if int(field["page"]) != page_number or field["name"] not in allowed:
                    continue
                started = time.perf_counter()
                prediction = recognize_field(
                    page_image,
                    tuple(field["rect_pdf"]),
                    dpi=dpi,
                    multiline=bool(field.get("multiline", False)),
                    preprocessing=pipeline["crop_preprocessing"],
                )
                latency = time.perf_counter() - started
                validated = validate_field_value(
                    field["name"],
                    prediction.text,
                    known_sponsor=gt["public_study_source"].get("sponsor_name"),
                )
                raw_score = score_field(str(field["value"]), prediction.text)
                score = score_field(str(field["value"]), validated)
                field_rows.append(
                    {
                        "sample_id": record["sample_id"],
                        "form_type": record["form_type"],
                        "nct_id": gt["public_study_source"]["nct_id"],
                        "page": page_number,
                        "condition": args.condition,
                        "pipeline": pipeline_name,
                        "field_name": field["name"],
                        "value_source": field["value_source"],
                        "reference": field["value"],
                        "raw_prediction": prediction.text,
                        "validated_prediction": validated,
                        "raw_exact_match": raw_score.exact_match,
                        "exact_match": score.exact_match,
                        "character_error_rate": round(score.character_error_rate, 6),
                        "character_accuracy": round(score.character_accuracy, 6),
                        "mean_ocr_confidence": round(prediction.mean_confidence, 3),
                        "field_ocr_latency_seconds": round(latency, 4),
                    }
                )

            for checkbox in gt["checkboxes"]:
                if int(checkbox["page"]) != page_number:
                    continue
                prediction = detect_checkbox(
                    page_image,
                    tuple(checkbox["rect_pdf"]),
                    dpi=dpi,
                    threshold=args.checkbox_threshold,
                )
                expected = bool(checkbox["value"])
                checkbox_rows.append(
                    {
                        "sample_id": record["sample_id"],
                        "form_type": record["form_type"],
                        "nct_id": gt["public_study_source"]["nct_id"],
                        "page": page_number,
                        "condition": args.condition,
                        "pipeline": pipeline_name,
                        "checkbox_name": checkbox["name"],
                        "expected_checked": expected,
                        "predicted_checked": prediction.checked,
                        "correct": prediction.checked == expected,
                        "inner_dark_ratio": round(prediction.inner_dark_ratio, 6),
                        "threshold": prediction.threshold,
                    }
                )

    fields = pd.DataFrame(field_rows).sort_values(["pipeline", "form_type", "field_name"])
    checkboxes = pd.DataFrame(checkbox_rows).sort_values(
        ["pipeline", "form_type", "checkbox_name"]
    )
    pages = pd.DataFrame(page_rows).sort_values(["pipeline", "form_type", "page"])
    fields.to_csv(output_dir / "field_predictions.csv", index=False)
    checkboxes.to_csv(output_dir / "checkbox_predictions.csv", index=False)
    pages.to_csv(output_dir / "page_preprocessing.csv", index=False)

    field_summary = (
        fields.groupby("pipeline", as_index=False)
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
    checkbox_summary = (
        checkboxes.groupby("pipeline", as_index=False)
        .agg(checkbox_count=("checkbox_name", "count"), accuracy=("correct", "mean"))
        .sort_values("accuracy", ascending=False)
    )
    field_summary.to_csv(output_dir / "field_summary.csv", index=False)
    checkbox_summary.to_csv(output_dir / "checkbox_summary.csv", index=False)
    print(f"Condition: {args.condition}")
    print(field_summary.to_string(index=False))
    print(checkbox_summary.to_string(index=False))


if __name__ == "__main__":
    main()
