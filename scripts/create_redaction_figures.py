#!/usr/bin/env python3
"""Create Day 7 redaction benchmark figures from measured result files."""
from __future__ import annotations

import sys
from pathlib import Path

import fitz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results/redaction_benchmark"
FIGURE_ROOT = RESULT_ROOT / "figures"
FIGURE_ROOT.mkdir(parents=True, exist_ok=True)


def render_first_page(path: Path, dpi: int = 120) -> Image.Image:
    document = fitz.open(path)
    page = document[0]
    pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    document.close()
    return image


def create_detection_f1() -> None:
    frame = pd.read_csv(RESULT_ROOT / "summary_by_condition.csv")
    pivot = frame.pivot(index="condition", columns="pipeline", values="f1")
    order = [
        "clean",
        "rotation_1_5deg",
        "gaussian_blur",
        "gaussian_noise",
        "low_contrast",
        "directional_shadow",
        "jpeg_compression",
        "combined_moderate",
    ]
    pivot = pivot.reindex(order)
    axis = pivot.plot(kind="bar", figsize=(11, 5))
    axis.set_title("Sensitive-entity detection F1 by scan condition")
    axis.set_xlabel("Scan condition")
    axis.set_ylabel("F1")
    axis.set_ylim(0, 1.08)
    axis.tick_params(axis="x", rotation=35)
    axis.legend(title="Pipeline")
    axis.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURE_ROOT / "entity_detection_f1_by_condition.png", dpi=180)
    plt.close()


def create_redaction_coverage() -> None:
    frame = pd.read_csv(RESULT_ROOT / "summary_by_condition.csv")
    pivot = frame.pivot(index="condition", columns="pipeline", values="redaction_coverage")
    order = [
        "clean",
        "rotation_1_5deg",
        "gaussian_blur",
        "gaussian_noise",
        "low_contrast",
        "directional_shadow",
        "jpeg_compression",
        "combined_moderate",
    ]
    pivot = pivot.reindex(order)
    axis = pivot.plot(kind="bar", figsize=(11, 5))
    axis.set_title("Automatic redaction coverage by scan condition")
    axis.set_xlabel("Scan condition")
    axis.set_ylabel("Coverage")
    axis.set_ylim(0, 1.08)
    axis.tick_params(axis="x", rotation=35)
    axis.legend(title="Pipeline")
    axis.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURE_ROOT / "redaction_coverage_by_condition.png", dpi=180)
    plt.close()


def create_before_after() -> None:
    source = PROJECT_ROOT / "data/processed/populated_forms/flattened/FDA_1572_NCT04796896.pdf"
    redacted = RESULT_ROOT / "redacted/FDA_1572_NCT04796896_redacted.pdf"
    before = render_first_page(source)
    after = render_first_page(redacted)
    target_height = min(before.height, after.height)
    before = before.resize((round(before.width * target_height / before.height), target_height))
    after = after.resize((round(after.width * target_height / after.height), target_height))
    gap = 28
    header = 70
    canvas = Image.new("RGB", (before.width + after.width + gap, target_height + header), "white")
    canvas.paste(before, (0, header))
    canvas.paste(after, (before.width + gap, header))
    # Use matplotlib only for headings so no font asset is embedded in the repository.
    plt.figure(figsize=(14, 9))
    plt.imshow(np.asarray(canvas))
    plt.axis("off")
    plt.text(before.width / 2, 25, "Original populated FDA form", ha="center", va="center", fontsize=14)
    plt.text(before.width + gap + after.width / 2, 25, "Permanent redaction + review flags", ha="center", va="center", fontsize=14)
    plt.tight_layout(pad=0)
    plt.savefig(FIGURE_ROOT / "redaction_before_after.png", dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    create_detection_f1()
    create_redaction_coverage()
    create_before_after()
    print(f"Wrote figures to {FIGURE_ROOT}")


if __name__ == "__main__":
    main()
