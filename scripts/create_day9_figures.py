from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results/day9_async"


def main() -> None:
    jobs = pd.read_csv(RESULTS / "job_summary.csv")
    events = pd.read_csv(RESULTS / "job_events.csv")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = [Path(name).stem[:24] for name in jobs["source_filename"]]
    ax.bar(labels, jobs["processing_seconds"])
    ax.set_ylabel("Processing time (seconds)")
    ax.set_title("Day 9 asynchronous job processing time")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(RESULTS / "job_processing_time.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for job_id, group in events.groupby("job_id"):
        source = jobs.loc[jobs["job_id"] == job_id, "source_filename"].iloc[0]
        ax.step(group["sequence"], group["progress"], where="post", label=Path(source).stem[:20])
    ax.set_xlabel("Persisted event sequence")
    ax.set_ylabel("Progress (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Persisted job progress across processing stages")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "job_progress_events.png", dpi=180)
    plt.close(fig)


    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis("off")
    nodes = {
        "Client / Streamlit": (0.5, 5.2, 2.0, 0.8),
        "FastAPI": (3.2, 5.2, 1.6, 0.8),
        "PostgreSQL\nmetadata": (7.9, 5.2, 2.0, 0.8),
        "Redis broker": (3.2, 3.2, 1.6, 0.8),
        "Celery workers": (6.0, 3.2, 1.8, 0.8),
        "Document pipeline": (6.0, 1.2, 1.8, 0.8),
        "Object-style\nartifacts": (8.7, 1.2, 1.8, 0.8),
    }
    for label, (x, y, width, height) in nodes.items():
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.08"))
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center")
    arrows = [
        ((2.5, 5.6), (3.2, 5.6)),
        ((4.8, 5.6), (7.9, 5.6)),
        ((4.0, 5.2), (4.0, 4.0)),
        ((4.8, 3.6), (6.0, 3.6)),
        ((6.9, 3.2), (6.9, 2.0)),
        ((7.8, 1.6), (8.7, 1.6)),
        ((7.0, 3.9), (8.7, 5.2)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12))
    ax.set_title("RegDocAI Day 9 production architecture")
    fig.tight_layout()
    fig.savefig(RESULTS / "architecture.png", dpi=180)
    plt.close(fig)

    summary = json.loads((RESULTS / "integration_summary.json").read_text(encoding="utf-8"))
    stages = events.groupby("stage", as_index=False).size().sort_values("size", ascending=False)
    stages.to_csv(RESULTS / "event_stage_counts.csv", index=False)
    (RESULTS / "figure_metadata.json").write_text(
        json.dumps(
            {
                "job_count": summary["job_count"],
                "event_count": int(len(events)),
                "figures": ["job_processing_time.png", "job_progress_events.png", "architecture.png"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
