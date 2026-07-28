#!/usr/bin/env python3
"""Generate deterministic scan artifacts from actual public protocol table crops."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from regdoc_ai.augmentation.degradations import DegradationConfig, apply_degradation


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/table_benchmark.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    source_root = resolve(config["output_data_dir"])
    output_root = resolve(config["robustness"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(source_root / "manifest.csv")
    selected = set(config["robustness"]["sample_table_ids"])
    manifest = manifest[manifest["table_id"].isin(selected)].copy()
    if set(manifest["table_id"]) != selected:
        missing = sorted(selected - set(manifest["table_id"]))
        raise ValueError(f"Missing selected table IDs: {missing}")

    rows = []
    for item in manifest.to_dict(orient="records"):
        source_path = resolve(item["image_path"])
        image = cv2.imread(str(source_path))
        if image is None:
            raise RuntimeError(f"Unable to read {source_path}")
        for condition in config["robustness"]["conditions"]:
            params = dict(condition)
            name = params.pop("name")
            params["name"] = name
            degradation = DegradationConfig(**params)
            output_path = output_root / name / f"{item['table_id']}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), apply_degradation(image, degradation))
            rows.append(
                {
                    "table_id": item["table_id"],
                    "nct_id": item["nct_id"],
                    "category": item["category"],
                    "condition": name,
                    "source_image": item["image_path"],
                    "source_sha256": item["image_sha256"],
                    "degraded_image": str(output_path.relative_to(PROJECT_ROOT)),
                    "degraded_sha256": digest(output_path),
                    "annotation_path": item["annotation_path"],
                    "parameters_json": json.dumps(degradation.to_dict(), sort_keys=True),
                }
            )
    frame = pd.DataFrame(rows).sort_values(["condition", "nct_id", "table_id"])
    frame.to_csv(output_root / "manifest.csv", index=False)
    metadata = {
        "source_type": "deterministic_scan_augmentation_of_actual_public_protocol_tables",
        "sample_table_count": int(frame["table_id"].nunique()),
        "condition_count": int(frame["condition"].nunique()),
        "image_count": int(len(frame)),
        "conditions": config["robustness"]["conditions"],
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Generated {len(frame)} table scan images in {output_root}")


if __name__ == "__main__":
    main()
