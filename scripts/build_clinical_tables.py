#!/usr/bin/env python3
"""Create table-ready CSV files from real ClinicalTrials.gov aggregate records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def nested_get(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def study_summary(study: dict[str, Any]) -> dict[str, Any]:
    protocol = study.get("protocolSection", {})
    identification = protocol.get("identificationModule", {})
    status = protocol.get("statusModule", {})
    design = protocol.get("designModule", {})
    sponsor = protocol.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {})
    enrollment = design.get("enrollmentInfo", {})
    return {
        "nct_id": identification.get("nctId"),
        "brief_title": identification.get("briefTitle"),
        "official_title": identification.get("officialTitle"),
        "lead_sponsor": sponsor.get("name"),
        "overall_status": status.get("overallStatus"),
        "study_type": design.get("studyType"),
        "phases": "; ".join(design.get("phases", [])),
        "enrollment": enrollment.get("count"),
        "enrollment_type": enrollment.get("type"),
        "conditions": "; ".join(protocol.get("conditionsModule", {}).get("conditions", [])),
    }


def baseline_rows(study: dict[str, Any]) -> list[dict[str, Any]]:
    nct_id = nested_get(study, "protocolSection", "identificationModule", "nctId")
    module = nested_get(study, "resultsSection", "baselineCharacteristicsModule", default={}) or {}
    groups = {g.get("id"): g.get("title") for g in module.get("groups", [])}
    rows: list[dict[str, Any]] = []
    for measure in module.get("measures", []):
        for cls in measure.get("classes", []):
            class_title = cls.get("title")
            for category in cls.get("categories", []):
                category_title = category.get("title")
                for measurement in category.get("measurements", []):
                    rows.append(
                        {
                            "nct_id": nct_id,
                            "measure": measure.get("title"),
                            "class": class_title,
                            "category": category_title,
                            "group": groups.get(measurement.get("groupId"), measurement.get("groupId")),
                            "value": measurement.get("value"),
                            "spread": measurement.get("spread"),
                            "unit": measure.get("unitOfMeasure"),
                        }
                    )
    return rows


def adverse_event_rows(study: dict[str, Any]) -> list[dict[str, Any]]:
    nct_id = nested_get(study, "protocolSection", "identificationModule", "nctId")
    module = nested_get(study, "resultsSection", "adverseEventsModule", default={}) or {}
    groups = {g.get("id"): g.get("title") for g in module.get("eventGroups", [])}
    rows: list[dict[str, Any]] = []
    for seriousness, key in (("serious", "seriousEvents"), ("other", "otherEvents")):
        for event in module.get(key, []):
            for stat in event.get("stats", []):
                rows.append(
                    {
                        "nct_id": nct_id,
                        "seriousness": seriousness,
                        "organ_system": event.get("organSystem"),
                        "adverse_event": event.get("term"),
                        "group": groups.get(stat.get("groupId"), stat.get("groupId")),
                        "affected": stat.get("numAffected"),
                        "at_risk": stat.get("numAtRisk"),
                        "events": stat.get("numEvents"),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/clinicaltrials/moderna_studies.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/clinical_tables"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_path = args.input if args.input.is_absolute() else root / args.input
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    studies = payload["studies"]
    summaries = [study_summary(study) for study in studies]
    baselines = [row for study in studies for row in baseline_rows(study)]
    adverse_events = [row for study in studies for row in adverse_event_rows(study)]

    pd.DataFrame(summaries).to_csv(output_dir / "study_summary.csv", index=False)
    pd.DataFrame(baselines).to_csv(output_dir / "baseline_characteristics.csv", index=False)
    pd.DataFrame(adverse_events).to_csv(output_dir / "adverse_events.csv", index=False)

    print(
        f"Wrote {len(summaries)} study rows, {len(baselines)} baseline rows, "
        f"and {len(adverse_events)} adverse-event rows to {output_dir}"
    )


if __name__ == "__main__":
    main()
