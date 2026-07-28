#!/usr/bin/env python3
"""Populate official FDA forms using public study metadata and controlled test identities.

The regulatory PDF templates and protocol metadata are actual public sources. Personal
identifiers, facilities, investigator names, signatures, and disclosure states are controlled
synthetic test values so that no real individual's PII or financial information is represented.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from regdoc_ai.forms.pdf_widgets import fill_pdf_form, render_pdf

DPI = 300

CONTROLLED_IDENTITIES = [
    {
        "investigator": "Maya Chen, MD",
        "address1": "350 Longwood Avenue",
        "address2": "Suite 410",
        "city": "Boston",
        "state": "MA",
        "country": "United States",
        "zip": "02115",
        "facility": "Cambridge Clinical Research Center",
        "facility_address": "100 Main Street",
        "facility_city": "Cambridge",
        "facility_zip": "02142",
        "lab": "Central Clinical Laboratory",
        "lab_address": "25 Laboratory Way",
        "irb": "Independent Research Ethics Board",
        "irb_address": "10 Ethics Plaza",
        "subinvestigators": "Daniel Brooks, MD; Elena Patel, MD",
        "applicant": "Alex Morgan",
        "applicant_title": "Regulatory Affairs Director",
        "other_investigators": [
            "Daniel Brooks, MD",
            "Elena Patel, MD",
            "Jordan Lee, MD",
            "Priya Shah, MD",
            "Noah Williams, MD",
        ],
    },
    {
        "investigator": "Omar Rahman, MD",
        "address1": "725 Research Boulevard",
        "address2": "Building B",
        "city": "Somerville",
        "state": "MA",
        "country": "United States",
        "zip": "02145",
        "facility": "Pediatric Vaccine Research Unit",
        "facility_address": "88 Health Sciences Road",
        "facility_city": "Boston",
        "facility_zip": "02118",
        "lab": "Northeast Bioanalytical Laboratory",
        "lab_address": "45 Discovery Drive",
        "irb": "Regional Human Research Board",
        "irb_address": "200 Review Avenue",
        "subinvestigators": "Sophia Kim, MD; Lucas Martin, MD",
        "applicant": "Taylor Rivera",
        "applicant_title": "Clinical Compliance Manager",
        "other_investigators": [
            "Sophia Kim, MD",
            "Lucas Martin, MD",
            "Ava Thompson, MD",
            "Ethan Wilson, MD",
            "Leila Hassan, MD",
        ],
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mmddyyyy(value: str | None) -> str:
    if not value:
        return "01/15/2026"
    return datetime.strptime(value, "%d %b %Y").strftime("%m/%d/%Y")


def study_name_for_3455(study: dict[str, Any]) -> tuple[str, str]:
    """Create a concise two-line study name derived from actual protocol metadata."""
    if study["nct_id"] == "NCT04470427":
        return (
            "mRNA-1273-P301: Phase 3 mRNA-1273 Vaccine Study",
            "Adults Aged 18 Years and Older (NCT04470427)",
        )
    if study["nct_id"] == "NCT04796896":
        return (
            "mRNA-1273-P204: Phase 2/3 Pediatric Vaccine Study",
            "Children 6 Months to Less Than 12 Years (NCT04796896)",
        )
    return (
        f'{study["protocol_number"]}: {study.get("phase") or "Clinical"} Study',
        study["nct_id"],
    )


def make_1572(study: dict[str, Any], identity: dict[str, Any], index: int) -> tuple[dict[str, Any], set[str], dict[str, float]]:
    values: dict[str, Any] = {
        "db_invest_name": identity["investigator"],
        "db_inv_address1": identity["address1"],
        "db_inv_address2": identity["address2"],
        "db_inv_city": identity["city"],
        "db_inv_state": identity["state"],
        "db_inv_country": identity["country"],
        "db_inv_zip": identity["zip"],
        "db_cv": index % 2 == 0,
        "db_oth_qual": index % 2 == 1,
        "db_loc_name": identity["facility"],
        "db_loc_address1": identity["facility_address"],
        "db_loc_address2": "",
        "db_loc_city": identity["facility_city"],
        "db_loc_state": identity["state"],
        "db_loc_country": identity["country"],
        "db_loc_zip": identity["facility_zip"],
        "db_lab_name": identity["lab"],
        "db_lab_address1": identity["lab_address"],
        "db_lab_address2": "",
        "db_lab_city": "Boston",
        "db_lab_state": "MA",
        "db_lab_country": "United States",
        "db_lab_zip": "02115",
        "db_irb_name": identity["irb"],
        "db_irb_address1": identity["irb_address"],
        "db_irb_address2": "",
        "db_irb_city": "Boston",
        "db_irb_state": "MA",
        "db_irb_country": "United States",
        "db_irb_zip": "02110",
        "db_sub_inv_names": identity["subinvestigators"],
        "db_prot_name_code": f'{study["protocol_number"]} - {study["protocol_title"]}',
        "db_phase_1": False,
        "db_phase_2_3": True,
        "db_sig_date": mmddyyyy(study.get("amendment_date")),
    }
    multiline = {"db_sub_inv_names", "db_prot_name_code"}
    font_sizes = {"db_sub_inv_names": 8.0, "db_prot_name_code": 6.5}
    return values, multiline, font_sizes


def make_3454(study: dict[str, Any], identity: dict[str, Any], index: int) -> tuple[dict[str, Any], set[str], dict[str, float]]:
    checkbox_scenarios = [
        {"check1": True, "check2": False, "check3": False},
        {"check1": False, "check2": True, "check3": False},
    ]
    checks = checkbox_scenarios[index % len(checkbox_scenarios)]
    prefix = "topmostSubform[0].Page1[0].Subform1[0]"
    names = [identity["investigator"], *identity["other_investigators"]]
    values: dict[str, Any] = {
        f"{prefix}.checkboxList[0].LI1[0].check1[0]": checks["check1"],
        f"{prefix}.checkboxList[0].LI2[0].check2[0]": checks["check2"],
        f"{prefix}.checkboxList[0].LI3[0].check3[0]": checks["check3"],
        f"{prefix}.appName[0]": identity["applicant"],
        f"{prefix}.appTitle[0]": identity["applicant_title"],
        f"{prefix}.appFirm[0]": study["sponsor_name"],
        f"{prefix}.appSigDate[0]": mmddyyyy(study.get("amendment_date")),
    }
    for idx, name in enumerate(names, start=1):
        values[
            f"{prefix}.checkboxList[0].LI1[0].clinInvList[0].LI{idx}[0].invName{idx}[0]"
        ] = name
    return values, set(), {}


def make_3455(study: dict[str, Any], identity: dict[str, Any], index: int) -> tuple[dict[str, Any], set[str], dict[str, float]]:
    scenarios = [
        [True, False, True, False],
        [False, True, False, True],
    ]
    states = scenarios[index % len(scenarios)]
    prefix = "topmostSubform[0].Page1[0]"
    title1, title2 = study_name_for_3455(study)
    values: dict[str, Any] = {
        f"{prefix}.invesname[0]": identity["investigator"],
        f"{prefix}.nameofstudy[0]": title1,
        f"{prefix}.nameofstudy2[0]": title2,
        f"{prefix}.appName[0]": identity["applicant"],
        f"{prefix}.appTitle[0]": identity["applicant_title"],
        f"{prefix}.appFirm[0]": study["sponsor_name"],
        f"{prefix}.appSigDate[0]": mmddyyyy(study.get("amendment_date")),
    }
    for checkbox_index, checked in enumerate(states, start=1):
        values[f"{prefix}.checkboxList[0].LI{checkbox_index}[0].check{checkbox_index}[0]"] = checked
    font_sizes = {
        f"{prefix}.nameofstudy[0]": 7.0,
        f"{prefix}.nameofstudy2[0]": 6.5,
    }
    return values, set(), font_sizes


def build_ground_truth(
    *,
    sample_id: str,
    form_type: str,
    study: dict[str, Any],
    values: dict[str, Any],
    widgets: dict[str, Any],
    multiline: set[str],
    image_paths: list[Path],
    source_form: Path,
    output_pdf: Path,
) -> dict[str, Any]:
    fields = []
    checkboxes = []
    for name, value in values.items():
        widget = widgets[name]
        record = {
            "name": name,
            "page": widget.page,
            "rect_pdf": [round(v, 3) for v in widget.rect_pdf],
            "value": value,
        }
        if widget.field_type == "CheckBox":
            record["controlled_test_scenario"] = True
            checkboxes.append(record)
        elif value != "":
            record["multiline"] = name in multiline
            if str(value) in {
                str(study.get("protocol_number")),
                str(study.get("protocol_title")),
                str(study.get("sponsor_name")),
                mmddyyyy(study.get("amendment_date")),
            }:
                record["value_source"] = "actual_public_protocol_metadata"
            elif (
                str(study.get("protocol_number")) in str(value)
                or str(study.get("nct_id")) in str(value)
            ):
                record["value_source"] = "derived_from_actual_public_protocol_metadata"
            else:
                record["value_source"] = "controlled_synthetic_test_value"
            fields.append(record)

    return {
        "sample_id": sample_id,
        "form_type": form_type,
        "source_form": {
            "filename": source_form.name,
            "sha256": sha256(source_form),
            "classification": "actual_official_fda_form_template",
        },
        "public_study_source": {
            "nct_id": study["nct_id"],
            "protocol_number": study["protocol_number"],
            "protocol_title": study["protocol_title"],
            "sponsor_name": study["sponsor_name"],
            "source_filename": study["source_filename"],
            "source_sha256": study["source_sha256"],
            "classification": study["data_classification"],
        },
        "privacy_notice": (
            "Names, addresses, facilities, applicant representatives, checkbox states, and signatures "
            "are controlled test values. They do not describe real investigators or financial disclosures."
        ),
        "rendering": {"dpi": DPI, "page_images": [str(path.relative_to(PROJECT_ROOT)) for path in image_paths]},
        "flattened_pdf": str(output_pdf.relative_to(PROJECT_ROOT)),
        "fields": fields,
        "checkboxes": checkboxes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/interim/protocol_metadata/studies.json")
    )
    parser.add_argument(
        "--forms-dir", type=Path, default=Path("data/raw/fda_forms")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/populated_forms")
    )
    args = parser.parse_args()

    metadata_path = args.metadata if args.metadata.is_absolute() else PROJECT_ROOT / args.metadata
    forms_dir = args.forms_dir if args.forms_dir.is_absolute() else PROJECT_ROOT / args.forms_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    studies = payload["studies"]

    builders = {
        "FDA_1572": make_1572,
        "FDA_3454": make_3454,
        "FDA_3455": make_3455,
    }
    manifest_rows: list[dict[str, Any]] = []

    for study_index, study in enumerate(studies):
        identity = CONTROLLED_IDENTITIES[study_index % len(CONTROLLED_IDENTITIES)]
        for form_type, builder in builders.items():
            sample_id = f'{form_type}_{study["nct_id"]}'
            source_form = forms_dir / f"{form_type}.pdf"
            values, multiline, font_sizes = builder(study, identity, study_index)
            editable_pdf = output_dir / "editable" / f"{sample_id}.pdf"
            flattened_pdf = output_dir / "flattened" / f"{sample_id}.pdf"
            widgets = fill_pdf_form(
                source_form,
                editable_pdf,
                flattened_pdf,
                values,
                font_sizes=font_sizes,
            )
            image_dir = output_dir / "images" / sample_id
            image_paths = render_pdf(flattened_pdf, image_dir, dpi=DPI)
            ground_truth = build_ground_truth(
                sample_id=sample_id,
                form_type=form_type,
                study=study,
                values=values,
                widgets=widgets,
                multiline=multiline,
                image_paths=image_paths,
                source_form=source_form,
                output_pdf=flattened_pdf,
            )
            gt_path = output_dir / "ground_truth" / f"{sample_id}.json"
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            gt_path.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "form_type": form_type,
                    "nct_id": study["nct_id"],
                    "protocol_number": study["protocol_number"],
                    "source_form": str(source_form.relative_to(PROJECT_ROOT)),
                    "source_protocol": f'data/raw/clinicaltrials/{study["source_filename"]}',
                    "flattened_pdf": str(flattened_pdf.relative_to(PROJECT_ROOT)),
                    "ground_truth": str(gt_path.relative_to(PROJECT_ROOT)),
                    "page_count": len(image_paths),
                    "text_field_count": len(ground_truth["fields"]),
                    "checkbox_count": len(ground_truth["checkboxes"]),
                    "personal_data": "controlled_synthetic_only",
                }
            )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Generated {len(manifest_rows)} populated official-form samples at {output_dir}")


if __name__ == "__main__":
    main()
