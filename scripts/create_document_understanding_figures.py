#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "results/document_understanding/figures"
OUTPUT.mkdir(parents=True, exist_ok=True)


def classification_figure() -> None:
    frame = pd.read_csv(PROJECT_ROOT / "results/document_understanding/classification_summary_overall.csv")
    frame = frame.sort_values("accuracy")
    plt.figure(figsize=(7, 4.5))
    plt.bar(frame["pipeline"], frame["accuracy"] * 100)
    plt.ylim(0, 105)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Classification pipeline")
    plt.title("Held-out document classification across eight scan conditions")
    for index, value in enumerate(frame["accuracy"] * 100):
        plt.text(index, value + 1.2, f"{value:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(OUTPUT / "classification_accuracy.png", dpi=180)
    plt.close()


def field_figure() -> None:
    frame = pd.read_csv(PROJECT_ROOT / "results/document_understanding/field_summary_by_document_type.csv")
    frame = frame.sort_values("exact_match_accuracy")
    plt.figure(figsize=(8, 4.8))
    plt.bar(frame["document_type"], frame["exact_match_accuracy"] * 100)
    plt.ylim(0, 105)
    plt.ylabel("Exact-match accuracy (%)")
    plt.xlabel("Document type")
    plt.title("Hybrid routed field extraction on held-out documents")
    plt.xticks(rotation=18, ha="right")
    for index, value in enumerate(frame["exact_match_accuracy"] * 100):
        plt.text(index, value + 1.2, f"{value:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(OUTPUT / "field_accuracy_by_document.png", dpi=180)
    plt.close()


def contact_sheet() -> None:
    manifest = pd.read_csv(PROJECT_ROOT / "data/processed/document_understanding/manifest.csv")
    selected = (
        manifest[manifest["split"] == "test"]
        .groupby("class_label", as_index=False)
        .first()
        .sort_values("class_label")
    )
    thumbs = []
    labels = []
    for record in selected.to_dict(orient="records"):
        image = cv2.imread(str(PROJECT_ROOT / record["image_path"]))
        if image is None:
            continue
        scale = min(320 / image.shape[1], 420 / image.shape[0])
        thumb = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        canvas = 255 * __import__("numpy").ones((440, 340, 3), dtype="uint8")
        y0 = (420 - thumb.shape[0]) // 2
        x0 = (340 - thumb.shape[1]) // 2
        canvas[y0:y0 + thumb.shape[0], x0:x0 + thumb.shape[1]] = thumb
        thumbs.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        labels.append(record["class_label"])
    plt.figure(figsize=(15, 5))
    for index, (image, label) in enumerate(zip(thumbs, labels), start=1):
        ax = plt.subplot(1, len(thumbs), index)
        ax.imshow(image)
        ax.set_title(label.replace("_", " "), fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT / "document_class_examples.png", dpi=170)
    plt.close()


if __name__ == "__main__":
    classification_figure()
    field_figure()
    contact_sheet()
    print(f"Wrote figures to {OUTPUT}")
