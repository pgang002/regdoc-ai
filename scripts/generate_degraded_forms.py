#!/usr/bin/env python3
"""Create deterministic scan degradations from populated official FDA-form pages."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from regdoc_ai.augmentation.degradations import DegradationConfig, apply_degradation


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/degradation_benchmark.yaml")
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=Path("data/processed/populated_forms/ground_truth"),
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/degraded_forms/manifest.csv")
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    gt_dir = (
        args.ground_truth_dir
        if args.ground_truth_dir.is_absolute()
        else PROJECT_ROOT / args.ground_truth_dir
    )
    manifest_path = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    benchmark: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_seed = int(benchmark.get("base_seed", 0))
    rows: list[dict[str, Any]] = []

    for gt_path in sorted(gt_dir.glob("*.json")):
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        for page_number, source_relative in enumerate(gt["rendering"]["page_images"], start=1):
            source_path = PROJECT_ROOT / source_relative
            source_image = cv2.imread(str(source_path))
            if source_image is None:
                raise RuntimeError(f"Unable to read {source_path}")

            for condition_index, raw_condition in enumerate(benchmark["conditions"]):
                seed_material = (
                    f"{gt['sample_id']}:{page_number}:{raw_condition['name']}:{base_seed}"
                )
                stable_offset = int(hashlib.sha256(seed_material.encode()).hexdigest()[:8], 16)
                config = DegradationConfig(
                    seed=(base_seed + stable_offset + condition_index) % (2**32),
                    **raw_condition,
                )
                degraded = apply_degradation(source_image, config)
                relative_output = (
                    Path("data/processed/degraded_forms/images")
                    / gt["sample_id"]
                    / config.name
                    / f"page-{page_number}.png"
                )
                output_path = PROJECT_ROOT / relative_output
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(output_path), degraded):
                    raise RuntimeError(f"Unable to write {output_path}")

                rows.append(
                    {
                        "sample_id": gt["sample_id"],
                        "form_type": gt["form_type"],
                        "nct_id": gt["public_study_source"]["nct_id"],
                        "page": page_number,
                        "condition": config.name,
                        "source_image": source_relative,
                        "source_sha256": sha256_file(source_path),
                        "degraded_image": relative_output.as_posix(),
                        "degraded_sha256": sha256_file(output_path),
                        "source_classification": benchmark["source_classification"],
                        "output_classification": benchmark["output_classification"],
                        **{
                            f"param_{key}": value
                            for key, value in config.to_dict().items()
                            if key != "name"
                        },
                    }
                )

    frame = pd.DataFrame(rows).sort_values(["sample_id", "page", "condition"])
    frame.to_csv(manifest_path, index=False)
    metadata = {
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "image_count": int(len(frame)),
        "source_page_count": int(frame[["sample_id", "page"]].drop_duplicates().shape[0]),
        "condition_count": int(frame["condition"].nunique()),
        "conditions": sorted(frame["condition"].unique().tolist()),
        "privacy_and_data_use": (
            "Pages are renderings of official FDA form templates populated with public clinical-protocol "
            "metadata and controlled test identifiers. Degradations are synthetic scan artifacts."
        ),
    }
    (manifest_path.parent / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(frame.groupby("condition").size().rename("images").to_string())
    print(f"Wrote {len(frame)} images and {manifest_path}")


if __name__ == "__main__":
    main()
