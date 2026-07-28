#!/usr/bin/env python3
"""Evaluate PDF-native and image-based table extraction on actual protocol tables."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import camelot
import cv2
import pandas as pd
import pdfplumber
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from regdoc_ai.evaluation.text_metrics import character_error_rate, normalize_text, word_error_rate
from regdoc_ai.tables.classical import detect_ruled_table_grid
from regdoc_ai.tables.deep_models import ppstructure_status, table_transformer_status
from regdoc_ai.tables.geometry import adjacent_cells, box_iou
from regdoc_ai.tables.metrics import boundary_metrics, cell_box_metrics
from regdoc_ai.tables.reconstruction import assign_words_to_grid, matrix_to_dataframe, ocr_words


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def flatten_matrix(matrix: list[list[Any]]) -> str:
    return " ".join(
        str(value).strip()
        for row in matrix
        for value in row
        if value is not None and str(value).strip()
    )


def _camelot_bbox_top(table: Any, page_height: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = [float(v) for v in table._bbox]
    return x0, page_height - y1, x1, page_height - y0


def _match_camelot_table(tables: Any, gt_bbox: list[float], page_height: float) -> Any:
    scored = [(box_iou(_camelot_bbox_top(table, page_height), gt_bbox), table) for table in tables]
    if not scored:
        raise RuntimeError("Camelot returned no tables for the selected page")
    score, table = max(scored, key=lambda item: item[0])
    if score < 0.5:
        raise RuntimeError(f"No Camelot table matched the reference bbox (best IoU={score:.3f})")
    return table


def _save_matrix(matrix: list[list[str]], output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    frame = matrix_to_dataframe(matrix)
    frame.to_csv(output_base.with_suffix(".csv"), index=False, header=False)
    output_base.with_suffix(".json").write_text(
        json.dumps(matrix, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    output_base.with_suffix(".html").write_text(frame.to_html(index=False, header=False), encoding="utf-8")


def _draw_overlay(
    image: Any,
    gt_x: list[int],
    gt_y: list[int],
    pred_x: list[int],
    pred_y: list[int],
    output_path: Path,
) -> None:
    canvas = image.copy()
    h, w = canvas.shape[:2]
    for x in gt_x:
        cv2.line(canvas, (x, 0), (x, h - 1), (0, 170, 0), 2)
    for y in gt_y:
        cv2.line(canvas, (0, y), (w - 1, y), (0, 170, 0), 2)
    for x in pred_x:
        cv2.line(canvas, (x, 0), (x, h - 1), (0, 0, 220), 1)
    for y in pred_y:
        cv2.line(canvas, (0, y), (w - 1, y), (0, 0, 220), 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def evaluate_classical(record: dict[str, Any], annotation: dict[str, Any], config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    image = cv2.imread(str(resolve(record["image_path"])))
    if image is None:
        raise RuntimeError(f"Unable to load {record['image_path']}")
    settings = config["classical"]
    started = time.perf_counter()
    prediction = detect_ruled_table_grid(
        image,
        horizontal_kernel_fraction=float(settings["horizontal_kernel_fraction"]),
        vertical_kernel_fraction=float(settings["vertical_kernel_fraction"]),
        projection_threshold_fraction=float(settings["projection_threshold_fraction"]),
        merge_tolerance_px=int(settings["boundary_merge_tolerance_px"]),
    )
    ocr_image = image.copy()
    upscale = float(settings.get("ocr_upscale_factor", 1.0))
    if bool(settings.get("remove_grid_lines", True)):
        line_mask = cv2.bitwise_or(prediction.horizontal_mask, prediction.vertical_mask)
        line_mask = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        ocr_image[line_mask > 0] = 255
    if upscale != 1.0:
        ocr_image = cv2.resize(
            ocr_image, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC
        )
    words = ocr_words(ocr_image, psm=int(settings["tesseract_psm"]))
    if upscale != 1.0 and not words.empty:
        for coordinate in ("left", "top", "width", "height"):
            words[coordinate] = pd.to_numeric(words[coordinate], errors="coerce") / upscale
    matrix = assign_words_to_grid(words, prediction.x_boundaries, prediction.y_boundaries)
    latency = time.perf_counter() - started

    gt_x = [int(round(v)) for v in annotation["x_boundaries_image"]]
    gt_y = [int(round(v)) for v in annotation["y_boundaries_image"]]
    gt_cells = [tuple(cell["bbox_image"]) for cell in annotation["cells"]]
    x_score = boundary_metrics(prediction.x_boundaries, gt_x, int(settings["boundary_match_tolerance_px"]))
    y_score = boundary_metrics(prediction.y_boundaries, gt_y, int(settings["boundary_match_tolerance_px"]))
    cell_score = cell_box_metrics(prediction.cells, gt_cells, float(settings["cell_iou_threshold"]))
    reference_text = flatten_matrix(annotation["matrix"])
    predicted_text = flatten_matrix(matrix)

    base = output_root / "reconstructions" / "classical" / annotation["table_id"]
    _save_matrix(matrix, base)
    _draw_overlay(
        image,
        gt_x,
        gt_y,
        prediction.x_boundaries,
        prediction.y_boundaries,
        output_root / "overlays" / "classical" / f"{annotation['table_id']}.png",
    )
    return {
        "engine": "opencv_tesseract",
        "table_id": annotation["table_id"],
        "nct_id": annotation["nct_id"],
        "protocol_number": annotation["protocol_number"],
        "page": annotation["page"],
        "category": annotation["category"],
        "reference_rows": annotation["logical_rows"],
        "reference_columns": annotation["logical_columns"],
        "reference_physical_cells": annotation["physical_cell_count"],
        "predicted_rows": max(0, len(prediction.y_boundaries) - 1),
        "predicted_columns": max(0, len(prediction.x_boundaries) - 1),
        "predicted_cells": len(prediction.cells),
        "shape_exact": int(
            len(prediction.y_boundaries) - 1 == annotation["logical_rows"]
            and len(prediction.x_boundaries) - 1 == annotation["logical_columns"]
        ),
        "row_boundary_precision": x_score.precision * 0 + y_score.precision,
        "row_boundary_recall": y_score.recall,
        "row_boundary_f1": y_score.f1,
        "column_boundary_precision": x_score.precision,
        "column_boundary_recall": x_score.recall,
        "column_boundary_f1": x_score.f1,
        "physical_cell_precision": cell_score.precision,
        "physical_cell_recall": cell_score.recall,
        "physical_cell_f1": cell_score.f1,
        "table_text_exact": int(normalize_text(reference_text) == normalize_text(predicted_text)),
        "table_text_cer": character_error_rate(reference_text, predicted_text),
        "table_text_wer": word_error_rate(reference_text, predicted_text),
        "latency_seconds": latency,
    }


def evaluate_camelot_group(
    pdf_path: Path,
    page_number: int,
    group: list[tuple[dict[str, Any], dict[str, Any]]],
    config: dict[str, Any],
    output_root: Path,
) -> list[dict[str, Any]]:
    settings = config["camelot"]
    started = time.perf_counter()
    tables = camelot.read_pdf(
        str(pdf_path),
        pages=str(page_number),
        flavor=settings["flavor"],
        line_scale=int(settings["line_scale"]),
        suppress_stdout=True,
    )
    group_latency = time.perf_counter() - started
    with pdfplumber.open(pdf_path) as document:
        page_height = float(document.pages[page_number - 1].height)
    output = []
    for record, annotation in group:
        table = _match_camelot_table(tables, annotation["table_bbox_pdf"], page_height)
        matrix = table.df.fillna("").astype(str).values.tolist()
        reference_text = flatten_matrix(annotation["matrix"])
        predicted_text = flatten_matrix(matrix)
        gt_x = [int(round(v)) for v in annotation["x_boundaries_image"]]
        gt_y = [int(round(v)) for v in annotation["y_boundaries_image"]]
        # Convert Camelot's PDF coordinates into the benchmark crop's pixel coordinates.
        clip_x0, clip_top, _, _ = annotation["crop_bbox_pdf"]
        scale = float(annotation["render_dpi"]) / 72.0
        x_pdf = sorted({float(cell.x1) for row in table.cells for cell in row} | {float(cell.x2) for row in table.cells for cell in row})
        y_top_pdf = sorted(
            {page_height - float(cell.y2) for row in table.cells for cell in row}
            | {page_height - float(cell.y1) for row in table.cells for cell in row}
        )
        pred_x = [int(round((value - clip_x0) * scale)) for value in x_pdf]
        pred_y = [int(round((value - clip_top) * scale)) for value in y_top_pdf]
        tolerance = int(config["classical"]["boundary_match_tolerance_px"])
        x_score = boundary_metrics(pred_x, gt_x, tolerance)
        y_score = boundary_metrics(pred_y, gt_y, tolerance)
        base = output_root / "reconstructions" / "camelot_lattice" / annotation["table_id"]
        _save_matrix(matrix, base)
        output.append(
            {
                "engine": "camelot_lattice",
                "table_id": annotation["table_id"],
                "nct_id": annotation["nct_id"],
                "protocol_number": annotation["protocol_number"],
                "page": annotation["page"],
                "category": annotation["category"],
                "reference_rows": annotation["logical_rows"],
                "reference_columns": annotation["logical_columns"],
                "reference_physical_cells": annotation["physical_cell_count"],
                "predicted_rows": int(table.shape[0]),
                "predicted_columns": int(table.shape[1]),
                "predicted_cells": int(table.shape[0] * table.shape[1]),
                "shape_exact": int(table.shape == (annotation["logical_rows"], annotation["logical_columns"])),
                "row_boundary_precision": y_score.precision,
                "row_boundary_recall": y_score.recall,
                "row_boundary_f1": y_score.f1,
                "column_boundary_precision": x_score.precision,
                "column_boundary_recall": x_score.recall,
                "column_boundary_f1": x_score.f1,
                "physical_cell_precision": None,
                "physical_cell_recall": None,
                "physical_cell_f1": None,
                "table_text_exact": int(normalize_text(reference_text) == normalize_text(predicted_text)),
                "table_text_cer": character_error_rate(reference_text, predicted_text),
                "table_text_wer": word_error_rate(reference_text, predicted_text),
                "latency_seconds": group_latency / max(1, len(group)),
                "camelot_reported_accuracy": float(table.parsing_report.get("accuracy", 0.0)),
                "camelot_reported_whitespace": float(table.parsing_report.get("whitespace", 0.0)),
            }
        )
    return output


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = [
        "shape_exact", "row_boundary_precision", "row_boundary_recall", "row_boundary_f1",
        "column_boundary_precision", "column_boundary_recall", "column_boundary_f1",
        "physical_cell_precision", "physical_cell_recall", "physical_cell_f1",
        "table_text_exact", "table_text_cer", "table_text_wer", "latency_seconds",
    ]
    aggregations = {column: (column, "mean") for column in numeric}
    overall = frame.groupby("engine", as_index=False).agg(table_count=("table_id", "count"), **aggregations)
    by_category = frame.groupby(["engine", "category"], as_index=False).agg(table_count=("table_id", "count"), **aggregations)
    return overall, by_category


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/table_benchmark.yaml"))
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_root = resolve(config["output_data_dir"])
    output_root = resolve(config["output_results_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(data_root / "manifest.csv")

    pairs = []
    for record in manifest.to_dict(orient="records"):
        annotation = json.loads(resolve(record["annotation_path"]).read_text(encoding="utf-8"))
        pairs.append((record, annotation))

    rows = [evaluate_classical(record, annotation, config, output_root) for record, annotation in pairs]
    grouped: dict[tuple[str, int], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair in pairs:
        grouped[(pair[0]["source_pdf"], int(pair[0]["page"]))].append(pair)
    for (source_pdf, page), group in grouped.items():
        rows.extend(evaluate_camelot_group(resolve(source_pdf), page, group, config, output_root))

    frame = pd.DataFrame(rows).sort_values(["engine", "nct_id", "page", "table_id"])
    frame.to_csv(output_root / "table_predictions.csv", index=False)
    overall, by_category = summarize(frame)
    overall.to_csv(output_root / "summary_overall.csv", index=False)
    by_category.to_csv(output_root / "summary_by_category.csv", index=False)

    model_statuses = [table_transformer_status(), ppstructure_status()]
    status = {
        "benchmark_name": config["benchmark_name"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "actual_data_basis": {
            "documents": "Two actual public Moderna clinical study protocol PDFs from ClinicalTrials.gov",
            "table_count": int(len(manifest)),
            "ground_truth": "PDF vector rules, cell geometry, and embedded text extracted from the source PDFs",
            "synthetic_content": "None in Day 5 table content; table crops are deterministic renders of actual public pages",
        },
        "completed_engines": ["camelot_lattice", "opencv_tesseract"],
        "optional_model_status": [status.__dict__ for status in model_statuses],
        "comparison_note": (
            "Table Transformer and PP-Structure metrics are intentionally absent when their optional "
            "runtimes/model files are unavailable. The included Colab notebook writes into the same schema."
        ),
    }
    (output_root / "benchmark_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
