#!/usr/bin/env python3
"""Evaluate template-aware OCR and checkbox extraction against exact ground truth."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from regdoc_ai.checkboxes.classical import detect_checkbox
from regdoc_ai.evaluation.field_metrics import score_field
from regdoc_ai.extraction.field_validation import validate_field_value
from regdoc_ai.extraction.template_fields import pdf_rect_to_pixels, recognize_field


def load_image(root: Path, page_images: list[str], page_number: int) -> Any:
    path = root / page_images[page_number - 1]
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Unable to read image {path}")
    return image


def annotate_sample(
    gt: dict[str, Any],
    root: Path,
    output_dir: Path,
    field_predictions: dict[tuple[str, str], bool],
    checkbox_predictions: dict[str, bool],
) -> None:
    dpi = int(gt["rendering"]["dpi"])
    for page_number, image_relative in enumerate(gt["rendering"]["page_images"], start=1):
        image = cv2.imread(str(root / image_relative))
        if image is None:
            continue
        for field in gt["fields"]:
            if field["page"] != page_number:
                continue
            box = pdf_rect_to_pixels(tuple(field["rect_pdf"]), dpi=dpi, padding_points=0)
            exact = field_predictions.get((field["name"], "clahe"), False)
            # Use grayscale-friendly line styles rather than relying on color semantics.
            thickness = 3 if exact else 6
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), thickness)
        for checkbox in gt["checkboxes"]:
            if checkbox["page"] != page_number:
                continue
            box = pdf_rect_to_pixels(tuple(checkbox["rect_pdf"]), dpi=dpi, padding_points=0)
            correct = checkbox_predictions.get(checkbox["name"], False)
            thickness = 3 if correct else 6
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), thickness)
        sample_dir = output_dir / gt["sample_id"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sample_dir / f"page-{page_number}.png"), image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=Path("data/processed/populated_forms/ground_truth"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/populated_forms")
    )
    parser.add_argument("--preprocessing", nargs="+", default=["raw", "adaptive"])
    parser.add_argument("--checkbox-threshold", type=float, default=0.08)
    args = parser.parse_args()

    gt_dir = args.ground_truth_dir if args.ground_truth_dir.is_absolute() else PROJECT_ROOT / args.ground_truth_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    field_rows: list[dict[str, Any]] = []
    checkbox_rows: list[dict[str, Any]] = []

    for gt_path in sorted(gt_dir.glob("*.json")):
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        dpi = int(gt["rendering"]["dpi"])
        page_images = gt["rendering"]["page_images"]
        image_cache: dict[int, Any] = {}
        sample_field_predictions: dict[tuple[str, str], bool] = {}
        sample_checkbox_predictions: dict[str, bool] = {}

        for field in gt["fields"]:
            page_number = int(field["page"])
            image = image_cache.setdefault(
                page_number, load_image(PROJECT_ROOT, page_images, page_number)
            )
            for preprocessing in args.preprocessing:
                start = time.perf_counter()
                prediction = recognize_field(
                    image,
                    tuple(field["rect_pdf"]),
                    dpi=dpi,
                    multiline=bool(field.get("multiline", False)),
                    preprocessing=preprocessing,
                )
                latency = time.perf_counter() - start
                validated_text = validate_field_value(
                    field["name"],
                    prediction.text,
                    known_sponsor=gt["public_study_source"].get("sponsor_name"),
                )
                raw_score = score_field(str(field["value"]), prediction.text)
                score = score_field(str(field["value"]), validated_text)
                sample_field_predictions[(field["name"], preprocessing)] = score.exact_match
                field_rows.append(
                    {
                        "sample_id": gt["sample_id"],
                        "form_type": gt["form_type"],
                        "nct_id": gt["public_study_source"]["nct_id"],
                        "field_name": field["name"],
                        "value_source": field["value_source"],
                        "page": page_number,
                        "preprocessing": preprocessing,
                        "reference": field["value"],
                        "raw_prediction": prediction.text,
                        "validated_prediction": validated_text,
                        "raw_exact_match": raw_score.exact_match,
                        "exact_match": score.exact_match,
                        "character_error_rate": round(score.character_error_rate, 6),
                        "character_accuracy": round(score.character_accuracy, 6),
                        "mean_ocr_confidence": round(prediction.mean_confidence, 3),
                        "latency_seconds": round(latency, 4),
                    }
                )

        for checkbox in gt["checkboxes"]:
            page_number = int(checkbox["page"])
            image = image_cache.setdefault(
                page_number, load_image(PROJECT_ROOT, page_images, page_number)
            )
            prediction = detect_checkbox(
                image,
                tuple(checkbox["rect_pdf"]),
                dpi=dpi,
                threshold=args.checkbox_threshold,
            )
            expected = bool(checkbox["value"])
            correct = prediction.checked == expected
            sample_checkbox_predictions[checkbox["name"]] = correct
            checkbox_rows.append(
                {
                    "sample_id": gt["sample_id"],
                    "form_type": gt["form_type"],
                    "nct_id": gt["public_study_source"]["nct_id"],
                    "checkbox_name": checkbox["name"],
                    "page": page_number,
                    "expected_checked": expected,
                    "predicted_checked": prediction.checked,
                    "correct": correct,
                    "inner_dark_ratio": round(prediction.inner_dark_ratio, 6),
                    "threshold": prediction.threshold,
                }
            )

        annotate_sample(
            gt,
            PROJECT_ROOT,
            output_dir / "annotated",
            sample_field_predictions,
            sample_checkbox_predictions,
        )

    fields_df = pd.DataFrame(field_rows)
    checkboxes_df = pd.DataFrame(checkbox_rows)
    fields_df.to_csv(output_dir / "field_predictions.csv", index=False)
    checkboxes_df.to_csv(output_dir / "checkbox_predictions.csv", index=False)

    field_summary = (
        fields_df.groupby(["preprocessing"], as_index=False)
        .agg(
            field_count=("field_name", "count"),
            raw_exact_match_accuracy=("raw_exact_match", "mean"),
            validated_exact_match_accuracy=("exact_match", "mean"),
            mean_character_accuracy=("character_accuracy", "mean"),
            mean_character_error_rate=("character_error_rate", "mean"),
            mean_ocr_confidence=("mean_ocr_confidence", "mean"),
            total_field_latency_seconds=("latency_seconds", "sum"),
        )
        .sort_values("validated_exact_match_accuracy", ascending=False)
    )
    field_summary.to_csv(output_dir / "field_summary.csv", index=False)

    form_summary = (
        fields_df.groupby(["form_type", "preprocessing"], as_index=False)
        .agg(
            field_count=("field_name", "count"),
            raw_exact_match_accuracy=("raw_exact_match", "mean"),
            validated_exact_match_accuracy=("exact_match", "mean"),
            mean_character_accuracy=("character_accuracy", "mean"),
            mean_character_error_rate=("character_error_rate", "mean"),
        )
    )
    form_summary.to_csv(output_dir / "field_summary_by_form.csv", index=False)

    checkbox_summary = pd.DataFrame(
        [
            {
                "checkbox_count": len(checkboxes_df),
                "accuracy": float(checkboxes_df["correct"].mean()),
                "checked_mean_dark_ratio": float(
                    checkboxes_df.loc[checkboxes_df["expected_checked"], "inner_dark_ratio"].mean()
                ),
                "unchecked_mean_dark_ratio": float(
                    checkboxes_df.loc[~checkboxes_df["expected_checked"], "inner_dark_ratio"].mean()
                ),
                "threshold": args.checkbox_threshold,
            }
        ]
    )
    checkbox_summary.to_csv(output_dir / "checkbox_summary.csv", index=False)

    print("Field extraction summary:")
    print(field_summary.to_string(index=False))
    print("\nCheckbox extraction summary:")
    print(checkbox_summary.to_string(index=False))


if __name__ == "__main__":
    main()
