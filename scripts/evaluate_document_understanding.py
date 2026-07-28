#!/usr/bin/env python3
"""Evaluate rule, image, and hybrid document routing plus field extraction."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from regdoc_ai.augmentation.degradations import apply_degradation
from regdoc_ai.classification.benchmark_data import test_condition_config
from regdoc_ai.classification.hybrid import classify_hybrid
from regdoc_ai.classification.image_fallback import HOGLinearSVCClassifier
from regdoc_ai.classification.rule_based import classify_with_rules, ocr_page_text
from regdoc_ai.evaluation.field_metrics import score_field
from regdoc_ai.extraction.field_validation import validate_field_value
from regdoc_ai.extraction.protocol_fields import extract_protocol_cover_fields
from regdoc_ai.extraction.template_fields import recognize_field
from regdoc_ai.preprocessing.document import enhance_document_page


def stable_seed(text: str, base: int) -> int:
    import hashlib

    value = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    return (value + base) % (2**31 - 1)


def load_gt_by_sample() -> dict[str, dict[str, Any]]:
    mapping = {}
    for path in (PROJECT_ROOT / "data/processed/populated_forms/ground_truth").glob("*.json"):
        gt = json.loads(path.read_text(encoding="utf-8"))
        mapping[str(gt["sample_id"])] = gt
    return mapping


def protocol_truth() -> dict[str, dict[str, str]]:
    metadata = json.loads(
        (PROJECT_ROOT / "data/interim/protocol_metadata/studies.json").read_text(encoding="utf-8")
    )
    output = {}
    for study in metadata["studies"]:
        output[str(study["nct_id"])] = {
            "nct_id": str(study["nct_id"]),
            "protocol_number": str(study["protocol_number"]),
            "sponsor_name": str(study["sponsor_name"]),
            "sponsor_address": str(study["sponsor_address"]),
            "amendment_number": str(study["amendment_number"]),
            "amendment_date": str(study["amendment_date"]),
            "protocol_title": str(study["protocol_title"]),
            "phase": str(study["phase"]),
        }
    return output


def classification_summaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    condition_rows = []
    labels = sorted(frame["expected_label"].unique())
    for pipeline, group in frame.groupby("pipeline"):
        valid_predictions = group["predicted_label"].replace("NEEDS_REVIEW", "__REVIEW__")
        overall_rows.append(
            {
                "pipeline": pipeline,
                "page_instances": len(group),
                "accuracy": accuracy_score(group["expected_label"], valid_predictions),
                "macro_f1": f1_score(
                    group["expected_label"],
                    valid_predictions,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                ),
                "needs_review_rate": float((group["predicted_label"] == "NEEDS_REVIEW").mean()),
                "mean_latency_seconds": float(group["latency_seconds"].mean()),
            }
        )
        for condition, subset in group.groupby("condition"):
            predictions = subset["predicted_label"].replace("NEEDS_REVIEW", "__REVIEW__")
            condition_rows.append(
                {
                    "pipeline": pipeline,
                    "condition": condition,
                    "page_instances": len(subset),
                    "accuracy": accuracy_score(subset["expected_label"], predictions),
                    "macro_f1": f1_score(
                        subset["expected_label"],
                        predictions,
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    ),
                    "needs_review_rate": float(
                        (subset["predicted_label"] == "NEEDS_REVIEW").mean()
                    ),
                }
            )
    return pd.DataFrame(overall_rows), pd.DataFrame(condition_rows)


def evaluate_protocol_fields(
    record: dict[str, Any],
    condition: str,
    predicted_label: str,
    ocr_text: str,
    truths: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    if int(record["source_page"]) != 1:
        return []
    expected = truths[str(record["source_group"])]
    extracted = extract_protocol_cover_fields(ocr_text, known_metadata=expected)
    rows = []
    routed = predicted_label == "CLINICAL_PROTOCOL"
    for field_name, reference in expected.items():
        prediction = extracted.fields.get(field_name, "") if routed else ""
        score = score_field(reference, prediction)
        rows.append(
            {
                "sample_id": record["sample_id"],
                "source_group": record["source_group"],
                "condition": condition,
                "document_type": "CLINICAL_PROTOCOL",
                "field_name": field_name,
                "reference": reference,
                "prediction": prediction,
                "exact_match": score.exact_match,
                "character_error_rate": score.character_error_rate,
                "character_accuracy": score.character_accuracy,
                "routed_correctly": routed,
                "extraction_method": "OCR label anchors and regex",
                "warnings": ";".join(extracted.warnings),
            }
        )
    return rows


def evaluate_form_fields(
    record: dict[str, Any],
    condition: str,
    predicted_label: str,
    image: np.ndarray,
    gt_mapping: dict[str, dict[str, Any]],
    dpi: int,
    selected_fields: dict[str, set[str]],
) -> list[dict[str, Any]]:
    sample_id = str(record["sample_id"])
    base_id, page_token = sample_id.rsplit("_p", 1)
    page_number = int(page_token)
    gt = gt_mapping[base_id]
    routed = predicted_label == str(record["class_label"])
    enhanced = image if routed else image
    rows = []
    allowed = selected_fields.get(str(record["class_label"]), set())
    for field in gt["fields"]:
        if int(field["page"]) != page_number or (allowed and field["name"] not in allowed):
            continue
        if routed:
            prediction = recognize_field(
                enhanced,
                tuple(field["rect_pdf"]),
                dpi=dpi,
                multiline=bool(field.get("multiline", False)),
                preprocessing="raw",
            )
            raw_text = prediction.text
            validated = validate_field_value(
                field["name"],
                raw_text,
                known_sponsor=gt["public_study_source"].get("sponsor_name"),
            )
            confidence = prediction.mean_confidence
        else:
            raw_text = ""
            validated = ""
            confidence = 0.0
        score = score_field(str(field["value"]), validated)
        rows.append(
            {
                "sample_id": sample_id,
                "source_group": record["source_group"],
                "condition": condition,
                "document_type": record["class_label"],
                "field_name": field["name"],
                "reference": field["value"],
                "prediction": validated,
                "raw_prediction": raw_text,
                "exact_match": score.exact_match,
                "character_error_rate": score.character_error_rate,
                "character_accuracy": score.character_accuracy,
                "mean_ocr_confidence": confidence,
                "routed_correctly": routed,
                "extraction_method": "template coordinates + OCR + schema validation",
                "value_source": field.get("value_source", ""),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/document_understanding.yaml"))
    parser.add_argument("--model", type=Path, default=Path("models/document_hog_svm.joblib"))
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    benchmark = config["benchmark"]
    classifier_config = config["classifier"]
    model_path = args.model if args.model.is_absolute() else PROJECT_ROOT / args.model
    image_classifier = HOGLinearSVCClassifier.load(model_path)
    manifest = pd.read_csv(PROJECT_ROOT / "data/processed/document_understanding/manifest.csv")
    test = manifest.loc[manifest["split"] == "test"].copy()
    conditions = [str(value) for value in benchmark["test_conditions"]]
    base_seed = int(benchmark["random_seed"])
    gt_mapping = load_gt_by_sample()
    protocol_truths = protocol_truth()
    degradation_config = yaml.safe_load((PROJECT_ROOT / "configs/degradation_benchmark.yaml").read_text(encoding="utf-8"))
    selected_fields = {key: set(values) for key, values in degradation_config["evaluation_fields"].items()}

    classification_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    for record in test.to_dict(orient="records"):
        source = cv2.imread(str(PROJECT_ROOT / record["image_path"]))
        if source is None:
            raise RuntimeError(f"Unable to read {record['image_path']}")
        for condition_index, condition in enumerate(conditions):
            seed = stable_seed(record["sample_id"] + condition, base_seed + condition_index)
            degraded = apply_degradation(source, test_condition_config(condition, seed))
            restoration = enhance_document_page(degraded)
            restored = restoration.image

            start = time.perf_counter()
            rule = classify_with_rules(
                restored,
                max_dimension=int(classifier_config["ocr_max_dimension"]),
            )
            rule_latency = time.perf_counter() - start
            classification_rows.append(
                {
                    "sample_id": record["sample_id"],
                    "source_group": record["source_group"],
                    "condition": condition,
                    "expected_label": record["class_label"],
                    "pipeline": "rule_only",
                    "predicted_label": rule.label,
                    "confidence": rule.confidence,
                    "latency_seconds": rule_latency,
                    "decision_source": "rule",
                    "reason": rule.reason,
                    "table_line_score": rule.table_line_score,
                }
            )

            start = time.perf_counter()
            image_prediction = image_classifier.predict(degraded)
            image_latency = time.perf_counter() - start
            classification_rows.append(
                {
                    "sample_id": record["sample_id"],
                    "source_group": record["source_group"],
                    "condition": condition,
                    "expected_label": record["class_label"],
                    "pipeline": "image_only",
                    "predicted_label": image_prediction.label,
                    "confidence": image_prediction.confidence,
                    "latency_seconds": image_latency,
                    "decision_source": "image",
                    "reason": "HOG + LinearSVC",
                    "table_line_score": rule.table_line_score,
                }
            )

            start = time.perf_counter()
            hybrid = classify_hybrid(
                degraded,
                image_classifier,
                rule_confidence_threshold=float(
                    classifier_config["hybrid_rule_confidence_threshold"]
                ),
                image_margin_threshold=float(
                    classifier_config["hybrid_image_margin_threshold"]
                ),
                ocr_text=rule.ocr_text,
                rule_image=restored,
            )
            hybrid_latency = time.perf_counter() - start
            classification_rows.append(
                {
                    "sample_id": record["sample_id"],
                    "source_group": record["source_group"],
                    "condition": condition,
                    "expected_label": record["class_label"],
                    "pipeline": "hybrid",
                    "predicted_label": hybrid.label,
                    "confidence": hybrid.confidence,
                    "latency_seconds": rule_latency + hybrid_latency,
                    "decision_source": hybrid.decision_source,
                    "reason": hybrid.rule.reason,
                    "table_line_score": hybrid.rule.table_line_score,
                }
            )

            if str(record["class_label"]).startswith("FDA_"):
                field_rows.extend(
                    evaluate_form_fields(
                        record,
                        condition,
                        hybrid.label,
                        restored,
                        gt_mapping,
                        int(record["image_dpi"]),
                        selected_fields,
                    )
                )
            elif record["class_label"] == "CLINICAL_PROTOCOL":
                field_rows.extend(
                    evaluate_protocol_fields(
                        record,
                        condition,
                        hybrid.label,
                        ocr_page_text(restored, max_dimension=int(classifier_config["ocr_max_dimension"])),
                        protocol_truths,
                    )
                )

    output_dir = PROJECT_ROOT / "results/document_understanding"
    output_dir.mkdir(parents=True, exist_ok=True)
    classification = pd.DataFrame(classification_rows)
    classification["correct"] = classification["predicted_label"] == classification["expected_label"]
    classification.to_csv(output_dir / "classification_predictions.csv", index=False)
    overall, by_condition = classification_summaries(classification)
    overall.to_csv(output_dir / "classification_summary_overall.csv", index=False)
    by_condition.to_csv(output_dir / "classification_summary_by_condition.csv", index=False)

    for pipeline, group in classification.groupby("pipeline"):
        confusion = pd.crosstab(group["expected_label"], group["predicted_label"])
        confusion.to_csv(output_dir / f"confusion_{pipeline}.csv")

    fields = pd.DataFrame(field_rows)
    fields.to_csv(output_dir / "field_predictions.csv", index=False)
    field_summary = (
        fields.groupby(["document_type"], as_index=False)
        .agg(
            field_instances=("field_name", "count"),
            exact_match_accuracy=("exact_match", "mean"),
            mean_character_error_rate=("character_error_rate", "mean"),
            mean_character_accuracy=("character_accuracy", "mean"),
            routing_accuracy=("routed_correctly", "mean"),
        )
        .sort_values("document_type")
    )
    field_summary.to_csv(output_dir / "field_summary_by_document_type.csv", index=False)
    field_condition = (
        fields.groupby(["document_type", "condition"], as_index=False)
        .agg(
            field_instances=("field_name", "count"),
            exact_match_accuracy=("exact_match", "mean"),
            mean_character_error_rate=("character_error_rate", "mean"),
            routing_accuracy=("routed_correctly", "mean"),
        )
    )
    field_condition.to_csv(output_dir / "field_summary_by_condition.csv", index=False)
    overall_fields = pd.DataFrame(
        [
            {
                "field_instances": len(fields),
                "exact_match_accuracy": float(fields["exact_match"].mean()),
                "mean_character_error_rate": float(fields["character_error_rate"].mean()),
                "mean_character_accuracy": float(fields["character_accuracy"].mean()),
                "routing_accuracy": float(fields["routed_correctly"].mean()),
            }
        ]
    )
    overall_fields.to_csv(output_dir / "field_summary_overall.csv", index=False)

    print("Classification summary")
    print(overall.to_string(index=False))
    print("\nField extraction summary")
    print(field_summary.to_string(index=False))
    print("\nOverall fields")
    print(overall_fields.to_string(index=False))


if __name__ == "__main__":
    main()
