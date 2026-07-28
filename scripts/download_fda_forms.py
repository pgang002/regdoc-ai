#!/usr/bin/env python3
"""Download current official FDA source forms from URLs in the source catalog."""
from __future__ import annotations

import argparse
from pathlib import Path

import requests
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data_sources.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    for item in config["sources"]["fda_forms"]["documents"]:
        destination = root / item["local_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not args.overwrite:
            print(f"SKIP {destination}: already exists")
            continue
        response = requests.get(
            item["url"], timeout=90, headers={"User-Agent": "RegDocAI/0.1 research-project"}
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type.casefold() and not response.content.startswith(b"%PDF"):
            raise RuntimeError(f"Unexpected response for {item['id']}: {content_type}")
        destination.write_bytes(response.content)
        print(f"DOWNLOADED {item['id']} -> {destination}")


if __name__ == "__main__":
    main()
