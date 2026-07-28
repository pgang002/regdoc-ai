from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/day10_final/production_architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")

nodes = {
    "Streamlit review UI": (0.6, 5.2, 2.2, 0.8),
    "FastAPI v1.0": (4.0, 5.2, 2.2, 0.8),
    "Prometheus": (8.8, 5.2, 2.2, 0.8),
    "Redis broker": (0.6, 2.9, 2.2, 0.8),
    "Celery workers": (4.0, 2.9, 2.2, 0.8),
    "Document AI pipeline": (8.8, 2.9, 2.4, 0.8),
    "PostgreSQL metadata": (2.2, 0.7, 2.5, 0.8),
    "Object-style artifacts": (7.2, 0.7, 2.5, 0.8),
}
for label, (x, y, w, h) in nodes.items():
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08"))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)

arrows = [
    ((2.8, 5.6), (4.0, 5.6)),
    ((6.2, 5.6), (8.8, 5.6)),
    ((5.1, 5.2), (1.7, 3.7)),
    ((2.8, 3.3), (4.0, 3.3)),
    ((6.2, 3.3), (8.8, 3.3)),
    ((5.1, 2.9), (3.45, 1.5)),
    ((10.0, 2.9), (8.45, 1.5)),
    ((5.1, 5.2), (3.45, 1.5)),
    ((5.1, 5.2), (8.45, 1.5)),
]
for start, end in arrows:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12))

ax.text(6, 6.65, "RegDocAI production-oriented architecture", ha="center", fontsize=16)
ax.text(
    6,
    0.1,
    "Measured locally with SQLite + thread workers; PostgreSQL/Redis/Celery/Prometheus are the Docker Compose production configuration.",
    ha="center",
    fontsize=9,
)
fig.tight_layout()
fig.savefig(OUT, dpi=180, bbox_inches="tight")
plt.close(fig)
