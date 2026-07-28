from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_real_table_manifest_and_annotations_are_complete() -> None:
    manifest = pd.read_csv(ROOT / "data/processed/table_benchmark/manifest.csv")
    assert len(manifest) == 20
    assert manifest["nct_id"].nunique() == 2
    assert set(manifest["source_type"]) == {"actual_public_clinical_protocol_table"}
    for row in manifest.to_dict(orient="records"):
        annotation = json.loads((ROOT / row["annotation_path"]).read_text(encoding="utf-8"))
        assert annotation["logical_rows"] >= 3
        assert annotation["logical_columns"] >= 2
        assert annotation["physical_cell_count"] > 0
        assert (ROOT / annotation["image_path"]).exists()


def test_table_results_do_not_invent_optional_model_metrics() -> None:
    status = json.loads(
        (ROOT / "results/table_extraction_benchmark/benchmark_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["completed_engines"] == ["camelot_lattice", "opencv_tesseract"]
    optional = {item["name"]: item for item in status["optional_model_status"]}
    assert optional["table_transformer"]["available"] is False
    assert optional["paddleocr_ppstructure"]["available"] is False
