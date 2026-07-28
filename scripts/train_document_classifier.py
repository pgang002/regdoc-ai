#!/usr/bin/env python3
"""Train the local HOG/LinearSVC image fallback on actual document pages."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from regdoc_ai.augmentation.degradations import apply_degradation
from regdoc_ai.classification.benchmark_data import training_augmentation
from regdoc_ai.classification.image_fallback import HOGLinearSVCClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/document_understanding.yaml"))
    parser.add_argument("--model", type=Path, default=Path("models/document_hog_svm.joblib"))
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    benchmark = config["benchmark"]
    classifier_config = config["classifier"]
    manifest = pd.read_csv(PROJECT_ROOT / "data/processed/document_understanding/manifest.csv")
    train = manifest.loc[manifest["split"] == "train"].copy()

    target = int(benchmark["train_samples_per_class"])
    seed = int(benchmark["random_seed"])
    images = []
    labels: list[str] = []
    training_rows = []
    for class_index, (label, group) in enumerate(train.groupby("class_label")):
        records = group.sort_values("sample_id").to_dict(orient="records")
        for index in range(target):
            record = records[index % len(records)]
            source_path = PROJECT_ROOT / record["image_path"]
            image = cv2.imread(str(source_path))
            if image is None:
                raise RuntimeError(f"Unable to read {source_path}")
            augmentation = training_augmentation(index, seed + class_index * 100_000)
            images.append(apply_degradation(image, augmentation))
            labels.append(str(label))
            training_rows.append(
                {
                    "class_label": label,
                    "source_sample_id": record["sample_id"],
                    "source_group": record["source_group"],
                    **augmentation.to_dict(),
                }
            )

    model = HOGLinearSVCClassifier(
        width=int(classifier_config["hog_width"]),
        height=int(classifier_config["hog_height"]),
        random_state=seed,
    ).fit(images, labels)
    model_path = args.model if args.model.is_absolute() else PROJECT_ROOT / args.model
    model.save(model_path)
    results_dir = PROJECT_ROOT / "results/document_understanding"
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(training_rows).to_csv(results_dir / "image_classifier_training_manifest.csv", index=False)
    metadata = {
        "model_type": "HOG features + class-weighted LinearSVC",
        "purpose": "local image fallback when OCR classification rules are uncertain",
        "training_source_group": str(benchmark["train_source_group"]),
        "held_out_source_group": str(benchmark["test_source_group"]),
        "training_samples": len(labels),
        "samples_per_class": target,
        "classes": sorted(set(labels)),
        "pretrained_deep_weights_used": False,
        "note": "MobileNetV3-Small transfer learning is provided separately for a model-enabled environment; no unavailable pretrained metrics are reported.",
    }
    (results_dir / "image_classifier_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
