from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/day10_final"
OUT.mkdir(parents=True, exist_ok=True)


def first(path: str, **filters):
    df = pd.read_csv(ROOT / path)
    for key, value in filters.items():
        df = df[df[key] == value]
    if df.empty:
        raise RuntimeError(f"No matching row in {path}: {filters}")
    return df.iloc[0]


def metric(area: str, name: str, value: float, unit: str, source: str, scope: str) -> dict:
    return {
        "area": area,
        "metric": name,
        "value": float(value),
        "unit": unit,
        "source": source,
        "scope": scope,
    }


def main() -> None:
    degradation = first(
        "results/degradation_benchmark/field_summary_degraded_overall.csv",
        pipeline="enhanced_deskew_restored",
    )
    degradation_raw = first(
        "results/degradation_benchmark/field_summary_degraded_overall.csv",
        pipeline="baseline_raw",
    )
    checkbox = first(
        "results/degradation_benchmark/checkbox_summary_degraded_overall.csv",
        pipeline="enhanced_deskew_restored",
    )
    table = first(
        "results/table_extraction_benchmark/summary_overall.csv",
        engine="opencv_tesseract",
    )
    table_robust = first(
        "results/table_extraction_benchmark/robustness/summary_by_condition.csv",
        pipeline="enhanced",
        condition="combined_moderate",
    )
    classification = first(
        "results/document_understanding/classification_summary_overall.csv",
        pipeline="hybrid",
    )
    field = first("results/document_understanding/field_summary_overall.csv")
    redaction = first(
        "results/redaction_benchmark/summary_overall.csv", pipeline="hybrid_policy"
    )
    day8 = pd.read_csv(ROOT / "results/day8_app/integration_summary.csv")
    form_api = day8[day8.workflow == "fda_form_review"].iloc[0]
    day9 = json.loads((ROOT / "results/day9_async/integration_summary.json").read_text())

    rows = [
        metric(
            "OCR robustness",
            "Validated field exact match",
            degradation.validated_exact_match_accuracy,
            "ratio",
            "Day 3",
            "140 field instances across seven degraded conditions",
        ),
        metric(
            "OCR robustness",
            "Mean character error rate",
            degradation.mean_character_error_rate,
            "ratio",
            "Day 3",
            "Enhanced deskew/restoration pipeline",
        ),
        metric(
            "OCR robustness",
            "Field error reduction vs raw",
            1
            - (1 - degradation.validated_exact_match_accuracy)
            / (1 - degradation_raw.validated_exact_match_accuracy),
            "ratio",
            "Day 3",
            "Validated exact-match error reduction",
        ),
        metric(
            "Checkbox extraction",
            "Accuracy",
            checkbox.accuracy,
            "ratio",
            "Day 3",
            "77 checkbox instances across degraded conditions",
        ),
        metric(
            "Table extraction",
            "Clean exact grid shape",
            table.shape_exact,
            "ratio",
            "Day 5",
            "20 real Moderna protocol tables",
        ),
        metric(
            "Table extraction",
            "Physical-cell F1",
            table.physical_cell_f1,
            "ratio",
            "Day 5",
            "20 real Moderna protocol tables",
        ),
        metric(
            "Table extraction",
            "Combined-degradation exact grid shape",
            table_robust.shape_exact_accuracy,
            "ratio",
            "Day 5",
            "Two representative complex tables",
        ),
        metric(
            "Document understanding",
            "Held-out classification accuracy",
            classification.accuracy,
            "ratio",
            "Day 6",
            "96 page instances; source-separated protocol holdout",
        ),
        metric(
            "Document understanding",
            "Routed field exact match",
            field.exact_match_accuracy,
            "ratio",
            "Day 6",
            "224 field instances",
        ),
        metric(
            "Redaction",
            "Sensitive-entity F1",
            redaction.f1,
            "ratio",
            "Day 7",
            "168 expected entities across eight conditions",
        ),
        metric(
            "Redaction",
            "Automatic redaction coverage",
            redaction.redaction_coverage,
            "ratio",
            "Day 7",
            "Fields requiring redaction",
        ),
        metric(
            "Redaction",
            "False automatic redaction rate",
            redaction.false_redaction_rate,
            "ratio",
            "Day 7",
            "Hybrid policy benchmark",
        ),
        metric(
            "API",
            "FDA form processing latency",
            form_api.api_reported_processing_seconds,
            "seconds/document",
            "Day 8",
            "Two-page populated FDA 1572",
        ),
        metric(
            "Async processing",
            "Local two-worker throughput",
            day9["documents_per_minute"],
            "documents/minute",
            "Day 9",
            "Two-document actual-data validation; not Celery/PostgreSQL performance",
        ),
    ]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "consolidated_metrics.csv", index=False)

    scorecard = {
        "project": "RegDocAI",
        "version": "1.0.0",
        "evidence_policy": "Only locally executed metrics are reported; unavailable deep-model and Docker-service results remain unclaimed.",
        "headline_metrics": {
            "degraded_field_accuracy": round(float(degradation.validated_exact_match_accuracy), 6),
            "checkbox_accuracy": round(float(checkbox.accuracy), 6),
            "clean_table_grid_accuracy": round(float(table.shape_exact), 6),
            "table_physical_cell_f1": round(float(table.physical_cell_f1), 6),
            "held_out_document_accuracy": round(float(classification.accuracy), 6),
            "routed_field_accuracy": round(float(field.exact_match_accuracy), 6),
            "redaction_f1": round(float(redaction.f1), 6),
            "redaction_false_positive_rate": round(float(redaction.false_redaction_rate), 6),
            "local_two_worker_documents_per_minute": round(float(day9["documents_per_minute"]), 6),
        },
        "data": {
            "official_fda_forms": ["1572", "3454", "3455"],
            "real_modern_protocols": ["NCT04470427", "NCT04796896"],
            "real_protocol_tables": 20,
        },
    }
    (OUT / "portfolio_summary.json").write_text(json.dumps(scorecard, indent=2) + "\n")

    bullets = """# Resume-ready project bullets\n\nUse two or three bullets depending on available resume space.\n\n- Built **RegDocAI**, an end-to-end regulatory Document AI pipeline using Python, OpenCV, Tesseract, scikit-learn, FastAPI, and Streamlit to classify FDA forms and clinical-protocol pages, extract fields and checkboxes, reconstruct tables, and export validated JSON/CSV/HTML; achieved **98.2% routed field exact match across 224 fields** and **100% held-out document classification accuracy across 96 degraded page instances**.\n- Reconstructed **20 real Moderna clinical-protocol tables** with **100% exact row/column grid recovery** and **0.925 physical-cell F1**, while image restoration raised degraded-form field accuracy from **76.4% to 96.4%** and checkbox accuracy from **84.4% to 100%**.\n- Implemented policy-driven PII/CCI detection and permanent PDF redaction with **1.00 entity F1**, **97.9% automatic-redaction coverage**, and **0% false automatic redactions**; designed asynchronous processing with SQLAlchemy/PostgreSQL, Redis/Celery, Docker Compose, retries, persistent progress events, and Prometheus monitoring.\n\n## Accuracy note\n\nThe PostgreSQL/Redis/Celery and Docker Compose stack is implemented but was not executed in the restricted build environment. The measured asynchronous throughput of 8.32 documents/minute came from the documented two-worker local validation mode, so that throughput should not be attributed to Celery or PostgreSQL.\n"""
    (OUT / "resume_bullets.md").write_text(bullets)

    limitations = """# Known limitations and honest-use boundaries\n\n- PaddleOCR, PP-StructureV3, Table Transformer, and pretrained MobileNetV3 weights were not executed locally because binary/model downloads were blocked. Their adapters and Colab paths are included, but no metrics are claimed.\n- The production PostgreSQL, Redis, Celery, Prometheus, and Docker Compose services were not launched in this runtime because Docker and those external services were unavailable. Local SQLAlchemy/SQLite and thread-queue validation is reported separately.\n- FDA Form 1571 is an Adobe LiveCycle/XFA document and requires Adobe-compatible flattening before standard OCR ingestion.\n- The table benchmark focuses on ruled tables from two public Moderna protocols; borderless and handwritten tables require additional evaluation.\n- PII identities and disclosure states are controlled test data. Public protocol metadata is used for realism, and CCI labels simulate an organizational policy rather than asserting that public data is confidential.\n- Results are portfolio benchmarks, not clinical validation or regulatory qualification.\n"""
    (OUT / "limitations.md").write_text(limitations)

    chart = pd.DataFrame(
        [
            ("Degraded fields", degradation.validated_exact_match_accuracy * 100),
            ("Checkboxes", checkbox.accuracy * 100),
            ("Table grid", table.shape_exact * 100),
            ("Document routing", classification.accuracy * 100),
            ("Routed fields", field.exact_match_accuracy * 100),
            ("Redaction F1", redaction.f1 * 100),
        ],
        columns=["Capability", "Percent"],
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(chart.Capability, chart.Percent)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Measured performance (%)")
    ax.set_title("RegDocAI final measured scorecard")
    ax.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, chart.Percent):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "final_scorecard.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
