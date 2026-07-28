#!/usr/bin/env python3
"""Optional MobileNetV3-Small transfer-learning benchmark for Day 6.

The local Day 6 metrics use the reproducible HOG/LinearSVC fallback because this
runtime cannot download pretrained weights. This script is intended for Colab or
another model-enabled environment and uses the identical source-separated manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from regdoc_ai.augmentation.degradations import apply_degradation
from regdoc_ai.classification.benchmark_data import test_condition_config, training_augmentation


class DocumentDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, class_to_index: dict[str, int], *, train: bool, samples_per_class: int, seed: int):
        self.class_to_index = class_to_index
        self.train = train
        self.seed = seed
        records: list[dict[str, object]] = []
        if train:
            for class_index, (label, group) in enumerate(rows.groupby("class_label")):
                base = group.sort_values("sample_id").to_dict(orient="records")
                for index in range(samples_per_class):
                    record = dict(base[index % len(base)])
                    record["augmentation_index"] = index
                    record["augmentation_seed"] = seed + class_index * 100_000
                    records.append(record)
        else:
            conditions = [
                "clean", "rotation_1_5deg", "gaussian_blur", "gaussian_noise",
                "low_contrast", "directional_shadow", "jpeg_compression", "combined_moderate",
            ]
            for record in rows.to_dict(orient="records"):
                for condition in conditions:
                    item = dict(record)
                    item["condition"] = condition
                    records.append(item)
        self.records = records
        weights = MobileNet_V3_Small_Weights.DEFAULT
        base_transform = weights.transforms()
        self.transform = transforms.Compose([base_transform])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = cv2.imread(str(PROJECT_ROOT / str(record["image_path"])))
        if image is None:
            raise RuntimeError(f"Unable to read {record['image_path']}")
        if self.train:
            config = training_augmentation(int(record["augmentation_index"]), int(record["augmentation_seed"]))
        else:
            config = test_condition_config(str(record["condition"]), self.seed + index)
        image = apply_degradation(image, config)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb))
        return tensor, self.class_to_index[str(record["class_label"])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--output", type=Path, default=Path("models/document_mobilenet_v3_small.pt"))
    args = parser.parse_args()

    config = yaml.safe_load((PROJECT_ROOT / "configs/document_understanding.yaml").read_text())
    benchmark = config["benchmark"]
    manifest = pd.read_csv(PROJECT_ROOT / "data/processed/document_understanding/manifest.csv")
    classes = sorted(manifest["class_label"].unique())
    class_to_index = {label: index for index, label in enumerate(classes)}
    seed = int(benchmark["random_seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    status_path = PROJECT_ROOT / "results/document_understanding/mobilenet_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
    except Exception as exc:
        status_path.write_text(json.dumps({
            "status": "unavailable",
            "reason": str(exc),
            "metrics_reported": False,
            "required_weight": "MobileNet_V3_Small_Weights.DEFAULT",
        }, indent=2))
        raise SystemExit(f"Unable to load pretrained MobileNetV3-Small weights: {exc}")

    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_data = DocumentDataset(
        manifest[manifest["split"] == "train"], class_to_index,
        train=True, samples_per_class=int(benchmark["train_samples_per_class"]), seed=seed,
    )
    test_data = DocumentDataset(
        manifest[manifest["split"] == "test"], class_to_index,
        train=False, samples_per_class=0, seed=seed,
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * inputs.size(0)
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                logits = model(inputs.to(device))
                predictions = logits.argmax(dim=1).cpu()
                correct += int((predictions == targets).sum())
                total += int(targets.numel())
        history.append({
            "epoch": epoch,
            "train_loss": running_loss / len(train_data),
            "held_out_accuracy": correct / max(total, 1),
        })
        print(history[-1])

    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": classes, "history": history}, output)
    pd.DataFrame(history).to_csv(PROJECT_ROOT / "results/document_understanding/mobilenet_history.csv", index=False)
    status_path.write_text(json.dumps({
        "status": "complete",
        "metrics_reported": True,
        "device": str(device),
        "classes": classes,
        "final_held_out_accuracy": history[-1]["held_out_accuracy"],
        "model_path": str(output.relative_to(PROJECT_ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
