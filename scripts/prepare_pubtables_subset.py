#!/usr/bin/env python3
"""Create a deterministic local subset from an already-downloaded PubTables-1M split."""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/pubtables/subset"))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    image_out = output_dir / "images"
    annotation_out = output_dir / "annotations"
    image_out.mkdir(parents=True, exist_ok=True)
    annotation_out.mkdir(parents=True, exist_ok=True)

    candidates = []
    for annotation in sorted(args.annotations_dir.glob("*.xml")):
        image = args.images_dir / f"{annotation.stem}.jpg"
        if image.exists():
            candidates.append((image, annotation))
    if len(candidates) < args.sample_size:
        raise ValueError(f"Only {len(candidates)} matched image/annotation pairs were found")

    rng = random.Random(args.seed)
    selected = rng.sample(candidates, args.sample_size)
    rows = []
    for image, annotation in selected:
        shutil.copy2(image, image_out / image.name)
        shutil.copy2(annotation, annotation_out / annotation.name)
        rows.append(
            {
                "sample_id": image.stem,
                "image_file": image.name,
                "annotation_file": annotation.name,
                "selection_seed": args.seed,
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            }
        )

    with (output_dir / "subset_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created deterministic PubTables-1M subset with {len(rows)} samples")


if __name__ == "__main__":
    main()
