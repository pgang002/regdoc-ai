#!/usr/bin/env python3
"""Build a leakage-controlled document classification benchmark from actual project data."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import fitz
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_pdf_page(pdf_path: Path, page_number: int, dpi: int) -> Any:
    document = fitz.open(pdf_path)
    page = document[page_number - 1]
    pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
    image = cv2.cvtColor(
        __import__("numpy").frombuffer(pixmap.samples, dtype="uint8").reshape(
            pixmap.height, pixmap.width, pixmap.n
        ),
        cv2.COLOR_RGB2BGR,
    )
    document.close()
    return image


def write_image(path: Path, image: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Unable to write {path}")


def add_form_samples(rows: list[dict[str, Any]], root: Path, dpi: int) -> None:
    manifest = pd.read_csv(root / "data/processed/populated_forms/manifest.csv")
    for record in manifest.to_dict(orient="records"):
        gt = json.loads((root / record["ground_truth"]).read_text(encoding="utf-8"))
        source_group = str(record["nct_id"])
        split = "train" if source_group == "NCT04470427" else "test"
        for page_number, source_image_relative in enumerate(gt["rendering"]["page_images"], start=1):
            source_image = root / source_image_relative
            image = cv2.imread(str(source_image))
            if image is None:
                raise RuntimeError(f"Unable to read {source_image}")
            sample_id = f"{record['sample_id']}_p{page_number:03d}"
            output = root / "data/processed/document_understanding/base_images" / f"{sample_id}.png"
            write_image(output, image)
            rows.append(
                {
                    "sample_id": sample_id,
                    "class_label": record["form_type"],
                    "source_group": source_group,
                    "split": split,
                    "source_type": "official_fda_form_with_public_protocol_metadata",
                    "source_path": record["flattened_pdf"],
                    "source_page": page_number,
                    "image_path": str(output.relative_to(root)),
                    "image_sha256": sha256_file(output),
                    "image_dpi": int(gt["rendering"]["dpi"]),
                    "actual_content": True,
                    "privacy": "controlled identities; public protocol metadata",
                }
            )


def table_pages_by_protocol(root: Path, limit: int) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for path in sorted((root / "data/processed/table_benchmark/annotations").glob("*.json")):
        annotation = json.loads(path.read_text(encoding="utf-8"))
        grouped.setdefault(str(annotation["nct_id"]), []).append(int(annotation["page"]))
    return {key: sorted(set(values))[:limit] for key, values in grouped.items()}


def add_protocol_samples(
    rows: list[dict[str, Any]],
    root: Path,
    dpi: int,
    text_pages: list[int],
    table_limit: int,
) -> None:
    metadata = json.loads(
        (root / "data/interim/protocol_metadata/studies.json").read_text(encoding="utf-8")
    )
    table_pages = table_pages_by_protocol(root, table_limit)
    for study in metadata["studies"]:
        nct_id = str(study["nct_id"])
        pdf_path = root / "data/raw/clinicaltrials" / study["source_filename"]
        split = "train" if nct_id == "NCT04470427" else "test"
        selected_table_pages = table_pages.get(nct_id, [])
        for label, pages in (
            ("CLINICAL_PROTOCOL", text_pages),
            ("CLINICAL_TABLE", selected_table_pages),
        ):
            for page_number in pages:
                image = render_pdf_page(pdf_path, page_number, dpi)
                sample_id = f"{nct_id}_{label.lower()}_p{page_number:03d}"
                output = root / "data/processed/document_understanding/base_images" / f"{sample_id}.png"
                write_image(output, image)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "class_label": label,
                        "source_group": nct_id,
                        "split": split,
                        "source_type": "actual_public_clinical_protocol_page",
                        "source_path": str(pdf_path.relative_to(root)),
                        "source_page": page_number,
                        "image_path": str(output.relative_to(root)),
                        "image_sha256": sha256_file(output),
                        "image_dpi": dpi,
                        "actual_content": True,
                        "privacy": "public protocol document",
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/document_understanding.yaml"))
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["benchmark"]
    dpi = int(config["render_dpi"])

    rows: list[dict[str, Any]] = []
    add_form_samples(rows, PROJECT_ROOT, dpi)
    add_protocol_samples(
        rows,
        PROJECT_ROOT,
        dpi,
        [int(value) for value in config["protocol_text_pages"]],
        int(config["table_pages_per_protocol"]),
    )
    frame = pd.DataFrame(rows).sort_values(["split", "class_label", "sample_id"])
    output_dir = PROJECT_ROOT / "data/processed/document_understanding"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "manifest.csv", index=False)
    metadata = {
        "data_classification": "actual public documents; deterministic scan augmentation is applied at runtime",
        "render_dpi": dpi,
        "sample_count": len(frame),
        "train_count": int((frame["split"] == "train").sum()),
        "test_count": int((frame["split"] == "test").sum()),
        "class_counts": frame.groupby(["split", "class_label"]).size().unstack(fill_value=0).to_dict(),
        "split_policy": "NCT04470427 trains the image fallback; NCT04796896 is held out for evaluation",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(frame.groupby(["split", "class_label"]).size().to_string())
    print(f"Wrote {len(frame)} base pages to {output_dir}")


if __name__ == "__main__":
    main()
