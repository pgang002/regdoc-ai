#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "results/day8_app"


def architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axis("off")
    labels = [
        (0.05, "Streamlit\nreview UI"),
        (0.24, "FastAPI\nupload/API"),
        (0.43, "DocumentPipeline\nclassification + OCR"),
        (0.63, "Fields / tables /\nredaction policy"),
        (0.82, "Workspace artifacts\nJSON, CSV, PDF"),
    ]
    for x, label in labels:
        ax.text(
            x,
            0.55,
            label,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.55", "facecolor": "white", "edgecolor": "black"},
            transform=ax.transAxes,
        )
    for x0, x1 in zip([0.10, 0.29, 0.49, 0.69], [0.19, 0.38, 0.58, 0.77]):
        ax.annotate(
            "",
            xy=(x1, 0.55),
            xytext=(x0, 0.55),
            xycoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "linewidth": 1.5},
        )
    ax.text(0.5, 0.12, "Day 8 synchronous workflow; Day 9 moves the same service layer behind Celery/Redis", ha="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(OUTPUT / "architecture.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def latency_figure() -> None:
    frame = pd.read_csv(OUTPUT / "integration_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(frame["workflow"], frame["api_reported_processing_seconds"])
    ax.set_ylabel("Processing time (seconds)")
    ax.set_title("Day 8 actual API integration latency")
    ax.tick_params(axis="x", rotation=15)
    for index, value in enumerate(frame["api_reported_processing_seconds"]):
        ax.text(index, value, f"{value:.2f}s", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUTPUT / "integration_latency.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def endpoint_catalog() -> None:
    payload = json.loads((OUTPUT / "openapi.json").read_text(encoding="utf-8"))
    rows = []
    for path, methods in payload.get("paths", {}).items():
        for method, spec in methods.items():
            if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                continue
            rows.append(
                {
                    "method": method.upper(),
                    "path": path,
                    "summary": spec.get("summary", ""),
                    "operation_id": spec.get("operationId", ""),
                }
            )
    pd.DataFrame(rows).sort_values(["path", "method"]).to_csv(
        OUTPUT / "endpoint_catalog.csv", index=False
    )


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    architecture_figure()
    latency_figure()
    endpoint_catalog()
