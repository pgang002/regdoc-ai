#!/usr/bin/env python3
"""Download real public study records from the ClinicalTrials.gov API v2.

The script stores only public registry records and aggregate posted results. It does
not request or create participant-level clinical data.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

API_BASE = "https://clinicaltrials.gov/api/v2/studies"


def request_json(url: str, params: dict[str, Any] | None = None, retries: int = 4) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=60,
                headers={"User-Agent": "RegDocAI/0.1 research-project"},
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt == retries - 1:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"ClinicalTrials.gov request failed: {last_error}")


def load_nct_ids(path: Path) -> list[str]:
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            ids.append(value)
    return ids


def fetch_by_ids(nct_ids: Iterable[str]) -> list[dict[str, Any]]:
    records = []
    for nct_id in nct_ids:
        records.append(request_json(f"{API_BASE}/{nct_id}", params={"format": "json"}))
    return records


def fetch_by_sponsor(sponsor: str, page_size: int, max_studies: int) -> list[dict[str, Any]]:
    studies: list[dict[str, Any]] = []
    page_token: str | None = None

    while len(studies) < max_studies:
        params: dict[str, Any] = {
            "query.spons": sponsor,
            "format": "json",
            "pageSize": min(page_size, max_studies - len(studies)),
            "countTotal": "true",
        }
        if page_token:
            params["pageToken"] = page_token
        payload = request_json(API_BASE, params=params)
        studies.extend(payload.get("studies", []))
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return studies[:max_studies]


def write_payload(records: list[dict[str, Any]], output_path: Path, query: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "source": "ClinicalTrials.gov API v2",
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "record_count": len(records),
            "data_classification": "public_registry_and_aggregate_results",
        },
        "studies": records,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["seed", "sponsor"], default="seed")
    parser.add_argument("--nct-file", type=Path, default=Path("data/raw/clinicaltrials/seed_nct_ids.txt"))
    parser.add_argument("--sponsor", default="ModernaTX, Inc.")
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--max-studies", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("data/raw/clinicaltrials/moderna_studies.json"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output

    if args.mode == "seed":
        nct_file = args.nct_file if args.nct_file.is_absolute() else root / args.nct_file
        nct_ids = load_nct_ids(nct_file)
        records = fetch_by_ids(nct_ids)
        query = {"mode": "seed", "nct_ids": nct_ids}
    else:
        records = fetch_by_sponsor(args.sponsor, args.page_size, args.max_studies)
        query = {"mode": "sponsor", "sponsor": args.sponsor, "max_studies": args.max_studies}

    write_payload(records, output, query)
    print(f"Downloaded {len(records)} actual public study records to {output}")


if __name__ == "__main__":
    main()
