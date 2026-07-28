#!/usr/bin/env python3
"""Evaluate the image-based table pipeline across realistic scan artifacts."""
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

from regdoc_ai.evaluation.text_metrics import character_error_rate, normalize_text, word_error_rate
from regdoc_ai.preprocessing.document import enhance_document_page
from regdoc_ai.tables.classical import detect_ruled_table_grid
from regdoc_ai.tables.metrics import boundary_metrics, cell_box_metrics
from regdoc_ai.tables.reconstruction import assign_words_to_grid, ocr_words


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def flatten(matrix: list[list[Any]]) -> str:
    return " ".join(str(v).strip() for row in matrix for v in row if v and str(v).strip())


def evaluate_image(image: Any, annotation: dict[str, Any], settings: dict[str, Any], enhance: bool) -> dict[str, Any]:
    started = time.perf_counter()
    estimated_angle = 0.0
    work = image
    if enhance:
        result = enhance_document_page(image)
        work = result.image
        estimated_angle = result.estimated_skew_degrees
    prediction = detect_ruled_table_grid(
        work,
        horizontal_kernel_fraction=float(settings["horizontal_kernel_fraction"]),
        vertical_kernel_fraction=float(settings["vertical_kernel_fraction"]),
        projection_threshold_fraction=float(settings["projection_threshold_fraction"]),
        merge_tolerance_px=int(settings["boundary_merge_tolerance_px"]),
    )
    ocr_image = work.copy()
    if ocr_image.ndim == 2:
        ocr_image = cv2.cvtColor(ocr_image, cv2.COLOR_GRAY2BGR)
    mask = cv2.bitwise_or(prediction.horizontal_mask, prediction.vertical_mask)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    ocr_image[mask > 0] = 255
    upscale = float(settings.get("ocr_upscale_factor", 2.0))
    ocr_image = cv2.resize(ocr_image, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    ocr_timed_out = False
    try:
        words = ocr_words(
            ocr_image,
            psm=int(settings["tesseract_psm"]),
            timeout=float(settings.get("ocr_timeout_seconds", 8)),
        )
    except RuntimeError:
        words = pd.DataFrame(columns=["text", "left", "top", "width", "height", "conf"])
        ocr_timed_out = True
    if not words.empty:
        for coordinate in ("left", "top", "width", "height"):
            words[coordinate] = pd.to_numeric(words[coordinate], errors="coerce") / upscale
    matrix = assign_words_to_grid(words, prediction.x_boundaries, prediction.y_boundaries)

    gt_x = [int(round(v)) for v in annotation["x_boundaries_image"]]
    gt_y = [int(round(v)) for v in annotation["y_boundaries_image"]]
    gt_cells = [cell["bbox_image"] for cell in annotation["cells"]]
    tolerance = int(settings["boundary_match_tolerance_px"])
    xs = boundary_metrics(prediction.x_boundaries, gt_x, tolerance)
    ys = boundary_metrics(prediction.y_boundaries, gt_y, tolerance)
    cs = cell_box_metrics(prediction.cells, gt_cells, float(settings["cell_iou_threshold"]))
    reference = flatten(annotation["matrix"])
    predicted = flatten(matrix)
    return {
        "shape_exact": int(
            len(prediction.y_boundaries) - 1 == annotation["logical_rows"]
            and len(prediction.x_boundaries) - 1 == annotation["logical_columns"]
        ),
        "row_boundary_f1": ys.f1,
        "column_boundary_f1": xs.f1,
        "physical_cell_f1": cs.f1,
        "table_text_exact": int(normalize_text(reference) == normalize_text(predicted)),
        "table_text_cer": character_error_rate(reference, predicted),
        "table_text_wer": word_error_rate(reference, predicted),
        "estimated_skew_degrees": estimated_angle,
        "predicted_rows": max(0, len(prediction.y_boundaries) - 1),
        "predicted_columns": max(0, len(prediction.x_boundaries) - 1),
        "ocr_timed_out": int(ocr_timed_out),
        "latency_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/table_benchmark.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    manifest = pd.read_csv(resolve(config["robustness"]["output_dir"]) / "manifest.csv")
    output_root = resolve(config["output_results_dir"]) / "robustness"
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in manifest.to_dict(orient="records"):
        annotation = json.loads(resolve(item["annotation_path"]).read_text(encoding="utf-8"))
        image = cv2.imread(str(resolve(item["degraded_image"])))
        if image is None:
            raise RuntimeError(f"Unable to read {item['degraded_image']}")
        for pipeline, enhance in (("raw", False), ("enhanced", True)):
            metrics = evaluate_image(image, annotation, config["classical"], enhance)
            rows.append({
                "pipeline": pipeline,
                "table_id": item["table_id"],
                "nct_id": item["nct_id"],
                "category": item["category"],
                "condition": item["condition"],
                **metrics,
            })
    frame = pd.DataFrame(rows).sort_values(["pipeline", "condition", "table_id"])
    frame.to_csv(output_root / "table_predictions.csv", index=False)
    summary = frame.groupby(["pipeline", "condition"], as_index=False).agg(
        table_count=("table_id", "count"),
        shape_exact_accuracy=("shape_exact", "mean"),
        mean_row_boundary_f1=("row_boundary_f1", "mean"),
        mean_column_boundary_f1=("column_boundary_f1", "mean"),
        mean_physical_cell_f1=("physical_cell_f1", "mean"),
        mean_table_text_cer=("table_text_cer", "mean"),
        mean_table_text_wer=("table_text_wer", "mean"),
        ocr_timeout_rate=("ocr_timed_out", "mean"),
        mean_latency_seconds=("latency_seconds", "mean"),
    )
    summary.to_csv(output_root / "summary_by_condition.csv", index=False)
    degraded = frame[frame["condition"] != "clean"]
    overall = degraded.groupby("pipeline", as_index=False).agg(
        table_count=("table_id", "count"),
        shape_exact_accuracy=("shape_exact", "mean"),
        mean_row_boundary_f1=("row_boundary_f1", "mean"),
        mean_column_boundary_f1=("column_boundary_f1", "mean"),
        mean_physical_cell_f1=("physical_cell_f1", "mean"),
        mean_table_text_cer=("table_text_cer", "mean"),
        mean_table_text_wer=("table_text_wer", "mean"),
        ocr_timeout_rate=("ocr_timed_out", "mean"),
        mean_latency_seconds=("latency_seconds", "mean"),
    )
    overall.to_csv(output_root / "summary_degraded_overall.csv", index=False)
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
