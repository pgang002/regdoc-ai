#!/usr/bin/env python3
"""Benchmark field OCR engines on the identical FDA form crops and scan conditions.

The script never substitutes one engine for another. If an optional engine is
unavailable, it records the reason in benchmark_status.json and continues with
engines that can be executed.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from regdoc_ai.evaluation.field_metrics import score_field
from regdoc_ai.extraction.field_validation import validate_field_value
from regdoc_ai.extraction.template_fields import recognize_field_with_engine
from regdoc_ai.ocr.engines import EngineUnavailableError, create_engine
from regdoc_ai.preprocessing.document import enhance_document_page


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_configs(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    benchmark = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_path = _resolve(benchmark["source_dataset_config"])
    dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    return benchmark, dataset


def _summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_condition = (
        frame.groupby(["engine", "condition"], as_index=False)
        .agg(
            field_count=("field_name", "count"),
            raw_exact_match_accuracy=("raw_exact_match", "mean"),
            validated_exact_match_accuracy=("exact_match", "mean"),
            mean_character_accuracy=("character_accuracy", "mean"),
            mean_character_error_rate=("character_error_rate", "mean"),
            mean_ocr_confidence=("mean_ocr_confidence", "mean"),
            mean_field_ocr_latency_seconds=("field_ocr_latency_seconds", "mean"),
        )
        .sort_values(["engine", "condition"])
    )
    overall = (
        frame.groupby("engine", as_index=False)
        .agg(
            field_count=("field_name", "count"),
            raw_exact_match_accuracy=("raw_exact_match", "mean"),
            validated_exact_match_accuracy=("exact_match", "mean"),
            mean_character_accuracy=("character_accuracy", "mean"),
            mean_character_error_rate=("character_error_rate", "mean"),
            mean_ocr_confidence=("mean_ocr_confidence", "mean"),
            mean_field_ocr_latency_seconds=("field_ocr_latency_seconds", "mean"),
            total_field_ocr_latency_seconds=("field_ocr_latency_seconds", "sum"),
        )
        .sort_values("validated_exact_match_accuracy", ascending=False)
    )
    return by_condition, overall


def evaluate_engine(
    engine_name: str,
    *,
    engine: Any,
    benchmark: dict[str, Any],
    dataset: dict[str, Any],
    conditions: list[str],
    output_root: Path,
) -> dict[str, Any]:
    manifest = pd.read_csv(_resolve(benchmark["manifest"]), dtype={"nct_id": str})
    selected_samples = set(dataset["evaluation_sample_ids"])
    selected_fields = {
        form_type: set(names) for form_type, names in dataset["evaluation_fields"].items()
    }
    manifest = manifest[
        manifest["condition"].isin(conditions)
        & manifest["sample_id"].isin(selected_samples)
    ].copy()
    manifest = manifest.sort_values(["condition", "sample_id", "page"])

    ground_truth_dir = _resolve(benchmark["ground_truth_dir"])
    ground_truth = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in ground_truth_dir.glob("*.json")
    }

    rows: list[dict[str, Any]] = []
    page_rows: list[dict[str, Any]] = []
    started_engine = time.perf_counter()
    for record in manifest.to_dict(orient="records"):
        gt = ground_truth[record["sample_id"]]
        image_path = _resolve(record["degraded_image"])
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Unable to read benchmark image: {image_path}")

        page_started = time.perf_counter()
        enhanced = enhance_document_page(image)
        preprocessing_latency = time.perf_counter() - page_started
        page_number = int(record["page"])
        page_rows.append(
            {
                "engine": engine_name,
                "sample_id": record["sample_id"],
                "form_type": record["form_type"],
                "condition": record["condition"],
                "page": page_number,
                "source_image": record["source_image"],
                "degraded_image": record["degraded_image"],
                "source_sha256": record["source_sha256"],
                "degraded_sha256": record["degraded_sha256"],
                "estimated_skew_degrees": round(enhanced.estimated_skew_degrees, 6),
                "page_preprocessing_latency_seconds": round(preprocessing_latency, 4),
            }
        )

        allowed = selected_fields[gt["form_type"]]
        for field in gt["fields"]:
            if int(field["page"]) != page_number or field["name"] not in allowed:
                continue
            started = time.perf_counter()
            prediction = recognize_field_with_engine(
                enhanced.image,
                tuple(field["rect_pdf"]),
                dpi=int(gt["rendering"]["dpi"]),
                multiline=bool(field.get("multiline", False)),
                preprocessing=benchmark["field_crop_preprocessing"],
                engine=engine,
            )
            latency = time.perf_counter() - started
            validated = validate_field_value(
                field["name"],
                prediction.text,
                known_sponsor=gt["public_study_source"].get("sponsor_name"),
            )
            raw_score = score_field(str(field["value"]), prediction.text)
            validated_score = score_field(str(field["value"]), validated)
            rows.append(
                {
                    "engine": engine_name,
                    "model_name": engine.model_name,
                    "sample_id": record["sample_id"],
                    "form_type": record["form_type"],
                    "nct_id": gt["public_study_source"]["nct_id"],
                    "condition": record["condition"],
                    "page": page_number,
                    "field_name": field["name"],
                    "multiline": bool(field.get("multiline", False)),
                    "value_source": field["value_source"],
                    "reference": field["value"],
                    "raw_prediction": prediction.text,
                    "validated_prediction": validated,
                    "raw_exact_match": raw_score.exact_match,
                    "exact_match": validated_score.exact_match,
                    "character_error_rate": round(validated_score.character_error_rate, 6),
                    "character_accuracy": round(validated_score.character_accuracy, 6),
                    "mean_ocr_confidence": round(prediction.mean_confidence, 3),
                    "field_ocr_latency_seconds": round(latency, 4),
                }
            )

    engine_dir = output_root / engine_name
    engine_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["condition", "form_type", "page", "field_name"])
    pages = pd.DataFrame(page_rows).sort_values(["condition", "form_type", "page"])
    by_condition, overall = _summarize(frame)
    frame.to_csv(engine_dir / "field_predictions.csv", index=False)
    pages.to_csv(engine_dir / "page_preprocessing.csv", index=False)
    by_condition.to_csv(engine_dir / "summary_by_condition.csv", index=False)
    overall.to_csv(engine_dir / "summary_overall.csv", index=False)
    (engine_dir / "engine_metadata.json").write_text(
        json.dumps(engine.metadata(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "status": "completed",
        "engine_metadata": engine.metadata(),
        "field_records": int(len(frame)),
        "page_records": int(len(pages)),
        "conditions": conditions,
        "wall_clock_seconds": round(time.perf_counter() - started_engine, 4),
        "summary": overall.to_dict(orient="records")[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/ocr_engine_benchmark.yaml")
    )
    parser.add_argument(
        "--engines", nargs="+", default=["tesseract", "paddleocr"],
        choices=["tesseract", "paddleocr"],
    )
    parser.add_argument("--conditions", nargs="+", default=["all"])
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    config_path = _resolve(args.config)
    benchmark, dataset = _load_configs(config_path)
    valid_conditions = [entry["name"] for entry in dataset["conditions"]]
    conditions = valid_conditions if args.conditions == ["all"] else args.conditions
    unknown = sorted(set(conditions) - set(valid_conditions))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}; expected {valid_conditions}")

    output_root = _resolve(benchmark["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "benchmark": benchmark["benchmark_name"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "actual_data_basis": {
            "templates": "official FDA Forms 1572, 3454, and 3455",
            "public_metadata": "public Moderna clinical protocol metadata",
            "identifiers": "controlled test values",
            "scan_conditions": "deterministic synthetic degradation of the populated official forms",
        },
        "requested_engines": args.engines,
        "conditions": conditions,
        "engines": {},
    }

    completed_frames: list[pd.DataFrame] = []
    failures: list[str] = []
    for name in args.engines:
        settings = benchmark["engines"][name]
        try:
            engine = create_engine(name, **settings)
            result = evaluate_engine(
                name,
                engine=engine,
                benchmark=benchmark,
                dataset=dataset,
                conditions=conditions,
                output_root=output_root,
            )
            status["engines"][name] = result
            completed_frames.append(pd.read_csv(output_root / name / "field_predictions.csv"))
            print(f"{name}: completed {result['field_records']} field predictions")
        except EngineUnavailableError as exc:
            message = str(exc)
            status["engines"][name] = {"status": "unavailable", "reason": message}
            failures.append(f"{name}: {message}")
            print(f"{name}: unavailable - {message}", file=sys.stderr)
        except Exception as exc:
            status["engines"][name] = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"{name}: failed - {type(exc).__name__}: {exc}", file=sys.stderr)

    if completed_frames:
        combined = pd.concat(completed_frames, ignore_index=True)
        combined.to_csv(output_root / "field_predictions_completed_engines.csv", index=False)
        by_condition, overall = _summarize(combined)
        by_condition.to_csv(output_root / "summary_by_condition_completed_engines.csv", index=False)
        overall.to_csv(output_root / "summary_overall_completed_engines.csv", index=False)

    status["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    status["comparison_complete"] = all(
        status["engines"].get(name, {}).get("status") == "completed" for name in args.engines
    )
    (output_root / "benchmark_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    if failures and args.strict:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
