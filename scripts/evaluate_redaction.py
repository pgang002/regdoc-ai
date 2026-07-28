#!/usr/bin/env python3
"""Evaluate policy-driven sensitive-entity detection and true PDF redaction."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from regdoc_ai.augmentation.degradations import DegradationConfig, apply_degradation
from regdoc_ai.preprocessing.document import enhance_document_page
from regdoc_ai.redaction.detectors import (
    detect_hybrid_policy,
    detect_regex_only,
    expected_entity_types,
    ocr_field_tokens,
)
from regdoc_ai.redaction.models import DetectedEntity, RedactionAction
from regdoc_ai.redaction.pdf_redactor import redact_pdf, verify_redaction_regions
from regdoc_ai.redaction.policy import RedactionPolicy


def stable_seed(text: str, base: int = 20260727) -> int:
    value = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    return (value + base) % (2**31 - 1)


def condition_config(name: str, degradation_config: dict[str, Any], seed: int) -> DegradationConfig:
    condition = next(item for item in degradation_config["conditions"] if item["name"] == name)
    values = dict(condition)
    values["seed"] = seed
    return DegradationConfig(**values)


def bbox_containment(
    predicted: tuple[float, float, float, float],
    expected: tuple[float, float, float, float],
) -> float:
    px0, py0, px1, py1 = predicted
    ex0, ey0, ex1, ey1 = expected
    ix0, iy0 = max(px0, ex0), max(py0, ey0)
    ix1, iy1 = min(px1, ex1), min(py1, ey1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    predicted_area = max(1e-9, (px1 - px0) * (py1 - py0))
    return intersection / predicted_area


def load_ground_truth(source_group: str) -> list[dict[str, Any]]:
    records = []
    root = PROJECT_ROOT / "data/processed/populated_forms/ground_truth"
    for path in sorted(root.glob(f"*_{source_group}.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        item["_path"] = str(path.relative_to(PROJECT_ROOT))
        records.append(item)
    if not records:
        raise RuntimeError(f"No populated-form ground truth found for {source_group}")
    return records


def page_images_for_gt(gt: dict[str, Any]) -> dict[int, Any]:
    images: dict[int, Any] = {}
    for index, relative in enumerate(gt["rendering"]["page_images"], start=1):
        image = cv2.imread(str(PROJECT_ROOT / relative))
        if image is None:
            raise RuntimeError(f"Unable to read {relative}")
        images[index] = image
    return images


def expected_action(policy: RedactionPolicy, entity_type: str) -> str:
    return policy.entity(entity_type).action.value


def score_predictions(
    *,
    sample_id: str,
    form_type: str,
    condition: str,
    field: dict[str, Any],
    expected_types: tuple[str, ...],
    predicted_entities: list[DetectedEntity],
    pipeline: str,
    latency_seconds: float,
    ocr_text: str,
    ocr_confidence: float,
    policy: RedactionPolicy,
) -> list[dict[str, Any]]:
    predicted_by_type = {entity.entity_type: entity for entity in predicted_entities}
    types = sorted(set(expected_types) | set(predicted_by_type))
    rows: list[dict[str, Any]] = []
    for entity_type in types:
        entity = predicted_by_type.get(entity_type)
        expected = entity_type in expected_types
        predicted = entity is not None
        if expected and predicted:
            outcome = "TP"
        elif expected:
            outcome = "FN"
        else:
            outcome = "FP"
        expected_policy_action = expected_action(policy, entity_type) if expected else ""
        predicted_action = entity.action.value if entity else ""
        rows.append(
            {
                "sample_id": sample_id,
                "form_type": form_type,
                "condition": condition,
                "page": int(field["page"]),
                "field_name": field["name"],
                "value_source": field.get("value_source", ""),
                "pipeline": pipeline,
                "entity_type": entity_type,
                "expected": expected,
                "predicted": predicted,
                "outcome": outcome,
                "expected_action": expected_policy_action,
                "predicted_action": predicted_action,
                "action_correct": bool(expected and predicted and expected_policy_action == predicted_action),
                "confidence": entity.confidence if entity else 0.0,
                "needs_review": entity.needs_review if entity else False,
                "bbox_containment": (
                    bbox_containment(entity.bbox_pdf, tuple(field["rect_pdf"])) if entity else 0.0
                ),
                "detection_methods": ";".join(entity.detection_methods) if entity else "",
                "ocr_text": ocr_text,
                "ocr_confidence": ocr_confidence,
                "latency_seconds": latency_seconds,
            }
        )
    return rows


def metric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pipeline, condition), group in frame.groupby(["pipeline", "condition"], dropna=False):
        tp = int((group["outcome"] == "TP").sum())
        fp = int((group["outcome"] == "FP").sum())
        fn = int((group["outcome"] == "FN").sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        positives = group.loc[group["expected"]]
        expected_redact = positives.loc[positives["expected_action"] == "redact"]
        redaction_coverage = (
            float(
                (
                    (expected_redact["outcome"] == "TP")
                    & (expected_redact["predicted_action"] == "redact")
                ).mean()
            )
            if not expected_redact.empty
            else 1.0
        )
        false_redactions = group.loc[
            (~group["expected"]) & (group["predicted_action"] == "redact")
        ]
        predicted_redactions = group.loc[group["predicted_action"] == "redact"]
        false_redaction_rate = (
            len(false_redactions) / len(predicted_redactions) if len(predicted_redactions) else 0.0
        )
        matched = group.loc[group["outcome"] == "TP"]
        rows.append(
            {
                "pipeline": pipeline,
                "condition": condition,
                "entity_instances": len(group),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "redaction_coverage": redaction_coverage,
                "missed_redaction_rate": 1.0 - redaction_coverage,
                "false_redaction_rate": false_redaction_rate,
                "action_accuracy_on_expected": float(positives["action_correct"].mean()) if not positives.empty else 1.0,
                "mean_bbox_containment": float(matched["bbox_containment"].mean()) if not matched.empty else 0.0,
                "mean_field_latency_seconds": float(group["latency_seconds"].mean()),
            }
        )
    return pd.DataFrame(rows)


def overall_summary(condition_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pipeline, group in condition_summary.groupby("pipeline"):
        weights = group["entity_instances"]
        row = {"pipeline": pipeline, "conditions": len(group), "entity_instances": int(weights.sum())}
        for column in (
            "precision",
            "recall",
            "f1",
            "redaction_coverage",
            "missed_redaction_rate",
            "false_redaction_rate",
            "action_accuracy_on_expected",
            "mean_bbox_containment",
            "mean_field_latency_seconds",
        ):
            row[column] = float((group[column] * weights).sum() / max(weights.sum(), 1))
        rows.append(row)
    return pd.DataFrame(rows)


def clean_entities_for_gt(gt: dict[str, Any], policy: RedactionPolicy, dpi: int) -> list[DetectedEntity]:
    images = page_images_for_gt(gt)
    entities: list[DetectedEntity] = []
    for field in gt["fields"]:
        page = int(field["page"])
        image = images[page]
        field_ocr = ocr_field_tokens(
            image,
            tuple(field["rect_pdf"]),
            dpi=dpi,
            multiline=bool(field.get("multiline", False)),
        )
        entities.extend(
            detect_hybrid_policy(
                field["name"], field_ocr, tuple(field["rect_pdf"]), page, policy
            )
        )
    return entities


def write_audit_log(
    path: Path,
    gt: dict[str, Any],
    policy: RedactionPolicy,
    entities: list[DetectedEntity],
    redaction_metadata: dict[str, object],
) -> None:
    payload = {
        "document_id": gt["sample_id"],
        "form_type": gt["form_type"],
        "source_ground_truth": gt["_path"],
        "source_classification": "template_real_content_controlled",
        "policy": {"name": policy.name, "version": policy.version},
        "redaction_metadata": redaction_metadata,
        "entities": [entity.to_audit_dict() for entity in entities],
        "audit_note": "Raw detected sensitive text is excluded; only hashes and masked previews are logged.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    policy_path = PROJECT_ROOT / "configs/redaction_policy.yaml"
    policy_config = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    policy = RedactionPolicy(policy_config)
    benchmark = policy_config["benchmark"]
    dpi = int(benchmark["render_dpi"])
    conditions = [str(item) for item in benchmark["conditions"]]
    source_group = str(benchmark["source_group"])
    degradation_config = yaml.safe_load(
        (PROJECT_ROOT / "configs/degradation_benchmark.yaml").read_text(encoding="utf-8")
    )
    gt_records = load_ground_truth(source_group)
    result_root = PROJECT_ROOT / benchmark["output_dir"]
    result_root.mkdir(parents=True, exist_ok=True)

    # Reuse the actual Day 6 held-out OCR outputs so Day 7 does not repeat hundreds
    # of expensive 300-DPI Tesseract calls. Clean PDF redaction below still performs
    # fresh OCR with coordinate mapping.
    day6_predictions = pd.read_csv(
        PROJECT_ROOT / "results/document_understanding/field_predictions.csv"
    )
    day6_predictions = day6_predictions.loc[
        (day6_predictions["source_group"] == source_group)
        & day6_predictions["document_type"].astype(str).str.startswith("FDA_")
    ].copy()
    gt_by_sample = {str(item["sample_id"]): item for item in gt_records}
    field_by_sample: dict[tuple[str, str], dict[str, Any]] = {}
    for item in gt_records:
        for field in item["fields"]:
            field_by_sample[(str(item["sample_id"]), str(field["name"]))] = field

    rows: list[dict[str, Any]] = []
    for record in day6_predictions.to_dict(orient="records"):
        sample_with_page = str(record["sample_id"])
        base_sample = sample_with_page.rsplit("_p", 1)[0]
        gt = gt_by_sample[base_sample]
        field = field_by_sample[(base_sample, str(record["field_name"]))]
        raw_value = record.get("raw_prediction")
        if pd.isna(raw_value) or not str(raw_value).strip():
            raw_value = record.get("prediction", "")
        ocr_text = "" if pd.isna(raw_value) else str(raw_value)
        confidence_raw = record.get("mean_ocr_confidence", 0.0)
        confidence = 0.0 if pd.isna(confidence_raw) else float(confidence_raw) / 100.0
        # Day 6 stored field-level OCR text but not word boxes. The field rectangle is
        # therefore the production template coordinate used for the robustness benchmark.
        from regdoc_ai.redaction.detectors import FieldOCR, OCRToken
        token = OCRToken(
            text=ocr_text,
            confidence=confidence,
            bbox_pdf=tuple(field["rect_pdf"]),
            start=0,
            end=len(ocr_text),
        )
        field_ocr = FieldOCR(ocr_text, confidence, (token,) if ocr_text else ())
        expected_types = expected_entity_types(field["name"], str(field["value"]))
        for pipeline, detector in (
            ("regex_only", detect_regex_only),
            ("hybrid_policy", detect_hybrid_policy),
        ):
            detect_start = time.perf_counter()
            entities = detector(
                field["name"],
                field_ocr,
                tuple(field["rect_pdf"]),
                int(field["page"]),
                policy,
            )
            latency = time.perf_counter() - detect_start
            scored = score_predictions(
                sample_id=base_sample,
                form_type=gt["form_type"],
                condition=str(record["condition"]),
                field=field,
                expected_types=expected_types,
                predicted_entities=entities,
                pipeline=pipeline,
                latency_seconds=latency,
                ocr_text=field_ocr.text,
                ocr_confidence=field_ocr.confidence,
                policy=policy,
            )
            for item in scored:
                item["coordinate_source"] = "template_field_rect_from_day6_ocr"
            rows.extend(scored)

    predictions = pd.DataFrame(rows)
    predictions.to_csv(result_root / "entity_predictions.csv", index=False)
    condition_summary = metric_summary(predictions)
    condition_summary.to_csv(result_root / "summary_by_condition.csv", index=False)
    overall = overall_summary(condition_summary)
    overall.to_csv(result_root / "summary_overall.csv", index=False)

    # Create true redacted PDFs for all six populated documents, not only the held-out benchmark source.
    all_gt = []
    for path in sorted((PROJECT_ROOT / benchmark["ground_truth_dir"]).glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        item["_path"] = str(path.relative_to(PROJECT_ROOT))
        all_gt.append(item)
    verification_rows = []
    document_rows = []
    for gt in all_gt:
        entities = clean_entities_for_gt(gt, policy, dpi)
        source_pdf = PROJECT_ROOT / gt["flattened_pdf"]
        redacted_pdf = result_root / "redacted" / f"{gt['sample_id']}_redacted.pdf"
        started = time.perf_counter()
        metadata = redact_pdf(source_pdf, redacted_pdf, entities)
        elapsed = time.perf_counter() - started
        for path_key in ("source_pdf", "output_pdf"):
            candidate = Path(str(metadata[path_key]))
            try:
                metadata[path_key] = str(candidate.relative_to(PROJECT_ROOT))
            except ValueError:
                metadata[path_key] = str(candidate)
        audit_path = result_root / "audit" / f"{gt['sample_id']}.json"
        write_audit_log(audit_path, gt, policy, entities, metadata)
        reference_texts: dict[tuple[int, str, str], str] = {}
        from regdoc_ai.redaction.detectors import PROTOCOL_RE
        for field in gt["fields"]:
            value = str(field["value"])
            for entity_type in expected_entity_types(field["name"], value):
                reference = value
                if entity_type == "CCI_PROTOCOL_ID":
                    match = PROTOCOL_RE.search(value)
                    reference = match.group(0) if match else value
                reference_texts[(int(field["page"]), str(field["name"]), entity_type)] = reference
        document_verification = verify_redaction_regions(
            source_pdf, redacted_pdf, entities, reference_texts=reference_texts
        )
        for item in document_verification:
            item["sample_id"] = gt["sample_id"]
            item["redacted_pdf"] = str(redacted_pdf.relative_to(PROJECT_ROOT))
            verification_rows.append(item)
        redactions = [e for e in entities if e.action == RedactionAction.REDACT]
        reviews = [e for e in entities if e.action == RedactionAction.REVIEW]
        verified_redactions = [
            item for item in document_verification if item["action"] == "redact"
        ]
        verified_reviews = [
            item for item in document_verification if item["action"] == "review"
        ]
        document_rows.append(
            {
                "sample_id": gt["sample_id"],
                "form_type": gt["form_type"],
                "redaction_entities": len(redactions),
                "review_entities": len(reviews),
                "redaction_verification_rate": (
                    sum(bool(item["verification_passed"]) for item in verified_redactions)
                    / max(len(verified_redactions), 1)
                ),
                "review_text_retained_rate": (
                    sum(bool(item["verification_passed"]) for item in verified_reviews)
                    / max(len(verified_reviews), 1)
                ),
                "redaction_latency_seconds": elapsed,
                "source_pdf": str(source_pdf.relative_to(PROJECT_ROOT)),
                "redacted_pdf": str(redacted_pdf.relative_to(PROJECT_ROOT)),
                "audit_log": str(audit_path.relative_to(PROJECT_ROOT)),
            }
        )
    pd.DataFrame(verification_rows).to_csv(result_root / "redaction_verification.csv", index=False)
    pd.DataFrame(document_rows).to_csv(result_root / "redaction_document_summary.csv", index=False)

    status = {
        "policy_name": policy.name,
        "policy_version": policy.version,
        "benchmark_source_group": source_group,
        "conditions": conditions,
        "form_documents_benchmarked": len(gt_records),
        "permanent_redacted_pdfs_created": len(all_gt),
        "statistical_ner_model": "not used locally; spaCy EntityRuler generic PERSON patterns were used",
        "no_placeholder_metrics": True,
    }
    (result_root / "benchmark_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(overall.to_string(index=False))
    print(pd.DataFrame(document_rows).to_string(index=False))


if __name__ == "__main__":
    main()
