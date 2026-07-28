#!/usr/bin/env python3
"""Create portfolio figures for the Day 5 table benchmark."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/table_extraction_benchmark"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def engine_structure_chart() -> None:
    frame = pd.read_csv(RESULTS / "summary_overall.csv")
    plot = frame.set_index("engine")[["shape_exact", "row_boundary_f1", "column_boundary_f1"]]
    ax = plot.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Table Structure Accuracy on 20 Public Moderna Protocol Tables")
    ax.set_ylabel("Mean score")
    ax.set_xlabel("Pipeline")
    ax.set_ylim(0, 1.05)
    ax.legend(["Exact grid shape", "Row boundary F1", "Column boundary F1"])
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES / "clean_structure_comparison.png", dpi=180)
    plt.close()


def engine_text_chart() -> None:
    frame = pd.read_csv(RESULTS / "summary_overall.csv")
    ax = frame.plot(x="engine", y="table_text_cer", kind="bar", legend=False, figsize=(8, 5))
    ax.set_title("Table Reconstruction Character Error Rate")
    ax.set_ylabel("Mean CER (lower is better)")
    ax.set_xlabel("Pipeline")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES / "clean_text_cer_comparison.png", dpi=180)
    plt.close()


def robustness_chart(metric: str, title: str, ylabel: str, filename: str) -> None:
    frame = pd.read_csv(RESULTS / "robustness/summary_by_condition.csv")
    order = ["clean", "rotation_1_5deg", "gaussian_blur", "jpeg_compression", "combined_moderate"]
    pivot = frame.pivot(index="condition", columns="pipeline", values=metric).reindex(order)
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Scan condition")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=180)
    plt.close()


def example_montage() -> None:
    examples = [
        ("NCT04470427_p117_t0", "Efficacy sample-size table"),
        ("NCT04470427_p118_t0", "Efficacy power table"),
        ("NCT04796896_p014_t0", "Objectives and endpoints"),
        ("NCT04796896_p116_t0", "Adverse-reaction grading"),
    ]
    tiles = []
    for table_id, label in examples:
        image = Image.open(ROOT / f"data/processed/table_benchmark/images/{table_id}.png").convert("RGB")
        image.thumbnail((700, 420))
        canvas = Image.new("RGB", (720, 470), "white")
        canvas.paste(image, ((720 - image.width) // 2, 38))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), label, fill="black")
        tiles.append(ImageOps.expand(canvas, border=1, fill="black"))
    sheet = Image.new("RGB", (1440, 940), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 2) * 720, (index // 2) * 470))
    sheet.save(FIGURES / "real_protocol_table_examples.png")


def main() -> None:
    engine_structure_chart()
    engine_text_chart()
    robustness_chart(
        "shape_exact_accuracy",
        "Grid-Shape Robustness on Real Protocol Tables",
        "Exact shape accuracy",
        "robustness_shape_accuracy.png",
    )
    robustness_chart(
        "mean_table_text_cer",
        "Table Text Robustness on Real Protocol Tables",
        "Mean CER (lower is better)",
        "robustness_text_cer.png",
    )
    example_montage()
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
