#!/usr/bin/env python3
"""Create compact portfolio figures from the degradation benchmark outputs."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results/degradation_benchmark"
FIGURE_ROOT = RESULT_ROOT / "figures"


def create_montage() -> None:
    conditions = [
        "clean",
        "rotation_2deg",
        "gaussian_blur",
        "gaussian_noise",
        "low_contrast",
        "directional_shadow",
        "jpeg_compression",
        "combined_moderate",
    ]
    sample_id = "FDA_1572_NCT04470427"
    tiles = []
    target_width = 420
    for condition in conditions:
        path = (
            PROJECT_ROOT
            / "data/processed/degraded_forms/images"
            / sample_id
            / condition
            / "page-1.png"
        )
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Unable to read {path}")
        scale = target_width / image.shape[1]
        tile = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        label_height = 42
        canvas = cv2.copyMakeBorder(
            tile, label_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        label = condition.replace("_", " ")
        cv2.putText(
            canvas,
            label,
            (12, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        tiles.append(canvas)

    rows = [cv2.hconcat(tiles[i : i + 4]) for i in range(0, len(tiles), 4)]
    montage = cv2.vconcat(rows)
    cv2.imwrite(str(FIGURE_ROOT / "degradation_examples.png"), montage)


def create_field_accuracy_chart() -> None:
    summary = pd.read_csv(RESULT_ROOT / "field_summary_by_condition.csv")
    pivot = summary.pivot(
        index="condition", columns="pipeline", values="validated_exact_match_accuracy"
    )
    order = [
        "clean",
        "rotation_2deg",
        "gaussian_blur",
        "gaussian_noise",
        "low_contrast",
        "directional_shadow",
        "jpeg_compression",
        "combined_moderate",
    ]
    pivot = pivot.reindex(order)
    ax = pivot.plot(kind="bar", figsize=(11, 5.5))
    ax.set_title("Validated Field Exact-Match Accuracy by Scan Condition")
    ax.set_xlabel("Scan condition")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_xticklabels([label.replace("_", " ") for label in order], rotation=32, ha="right")
    ax.legend(title="Pipeline")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURE_ROOT / "field_accuracy_by_condition.png", dpi=160)
    plt.close()


def create_checkbox_accuracy_chart() -> None:
    summary = pd.read_csv(RESULT_ROOT / "checkbox_summary_by_condition.csv")
    pivot = summary.pivot(index="condition", columns="pipeline", values="accuracy")
    order = [
        "clean",
        "rotation_2deg",
        "gaussian_blur",
        "gaussian_noise",
        "low_contrast",
        "directional_shadow",
        "jpeg_compression",
        "combined_moderate",
    ]
    pivot = pivot.reindex(order)
    ax = pivot.plot(kind="bar", figsize=(11, 5.5))
    ax.set_title("Checkbox-State Accuracy by Scan Condition")
    ax.set_xlabel("Scan condition")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_xticklabels([label.replace("_", " ") for label in order], rotation=32, ha="right")
    ax.legend(title="Pipeline")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURE_ROOT / "checkbox_accuracy_by_condition.png", dpi=160)
    plt.close()


def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    create_montage()
    create_field_accuracy_chart()
    create_checkbox_accuracy_chart()
    print(f"Created figures in {FIGURE_ROOT}")


if __name__ == "__main__":
    main()
