from __future__ import annotations

import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import fitz
import numpy as np
from PIL import Image

from regdoc_ai.checkboxes.classical import detect_checkbox
from regdoc_ai.classification.hybrid import classify_hybrid
from regdoc_ai.classification.image_fallback import HOGLinearSVCClassifier
from regdoc_ai.classification.rule_based import ocr_page_text
from regdoc_ai.extraction.field_validation import validate_field_value
from regdoc_ai.extraction.protocol_fields import extract_protocol_cover_fields
from regdoc_ai.extraction.template_fields import recognize_field
from regdoc_ai.preprocessing.document import enhance_document_page
from regdoc_ai.redaction.detectors import detect_hybrid_policy, ocr_field_tokens
from regdoc_ai.redaction.models import DetectedEntity, RedactionAction
from regdoc_ai.redaction.pdf_redactor import redact_pdf
from regdoc_ai.redaction.policy import RedactionPolicy
from regdoc_ai.schemas.document import BoundingBox, CheckboxResult, ExtractedField, ProcessingStatus, TableArtifact
from regdoc_ai.tables.classical import detect_ruled_table_grid
from regdoc_ai.tables.reconstruction import assign_words_to_grid, matrix_to_dataframe, ocr_words

from .models import PageClassification, ProcessingResponse, RedactionCandidate
from .storage import WorkspaceStore


SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _candidate_id(entity: DetectedEntity) -> str:
    raw = f"{entity.page}|{entity.field_name}|{entity.entity_type}|{entity.bbox_pdf}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _bbox(rect: tuple[float, float, float, float]) -> BoundingBox:
    return BoundingBox(x_min=rect[0], y_min=rect[1], x_max=rect[2], y_max=rect[3])


def _load_protocol_metadata(project_root: Path) -> dict[str, dict[str, str]]:
    path = project_root / "data/interim/protocol_metadata/studies.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    output: dict[str, dict[str, str]] = {}
    for row in payload.get("studies", []):
        protocol = str(row.get("protocol_number", ""))
        if protocol:
            output[protocol.lower()] = {str(k): str(v) for k, v in row.items()}
    return output


class DocumentPipeline:
    def __init__(self, project_root: str | Path, store: WorkspaceStore):
        self.project_root = Path(project_root).resolve()
        self.store = store
        self.model = HOGLinearSVCClassifier.load(
            self.project_root / "models/document_hog_svm.joblib"
        )
        self.policy = RedactionPolicy.from_yaml(
            self.project_root / "configs/redaction_policy.yaml"
        )
        self.protocol_metadata = _load_protocol_metadata(self.project_root)
        self.templates = self._load_form_templates()

    def _load_form_templates(self) -> dict[str, dict[str, Any]]:
        templates: dict[str, dict[str, Any]] = {}
        gt_root = self.project_root / "data/processed/populated_forms/ground_truth"
        for path in sorted(gt_root.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            form_type = str(payload["form_type"])
            # Prefer the held-out NCT04796896 geometry when both are present.
            if form_type not in templates or "NCT04796896" in path.name:
                templates[form_type] = payload
        return templates

    def _prepare_source_pdf(self, source_path: Path, doc_dir: Path) -> tuple[Path, list[np.ndarray], list[tuple[float, float]]]:
        suffix = source_path.suffix.lower()
        images: list[np.ndarray] = []
        page_sizes: list[tuple[float, float]] = []
        if suffix == ".pdf":
            pdf_path = source_path
            document = fitz.open(pdf_path)
            for page in document:
                matrix = fitz.Matrix(300 / 72, 300 / 72)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    array = cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
                else:
                    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
                images.append(array)
                page_sizes.append((float(page.rect.width), float(page.rect.height)))
            document.close()
            return pdf_path, images, page_sizes

        pil = Image.open(source_path).convert("RGB")
        rgb = np.array(pil)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        images.append(bgr)
        estimated_dpi = 300.0
        width_pt = bgr.shape[1] * 72.0 / estimated_dpi
        height_pt = bgr.shape[0] * 72.0 / estimated_dpi
        page_sizes.append((width_pt, height_pt))
        pdf_path = doc_dir / "source_converted.pdf"
        pil.save(pdf_path, "PDF", resolution=estimated_dpi)
        return pdf_path, images, page_sizes

    def _known_metadata_from_text(self, text: str) -> dict[str, str] | None:
        lowered = text.lower()
        for protocol, metadata in self.protocol_metadata.items():
            if protocol in lowered:
                return metadata
        return None

    def _extract_form_page(
        self,
        *,
        image: np.ndarray,
        page_number: int,
        form_type: str,
        dpi: int,
    ) -> tuple[list[ExtractedField], list[CheckboxResult], list[DetectedEntity], list[str]]:
        template = self.templates.get(form_type)
        if not template:
            return [], [], [], [f"No coordinate template registered for {form_type}"]
        fields: list[ExtractedField] = []
        checkboxes: list[CheckboxResult] = []
        entities: list[DetectedEntity] = []
        warnings: list[str] = []
        known_sponsor = template.get("public_study_source", {}).get("sponsor_name")
        for field in template.get("fields", []):
            if int(field["page"]) != page_number:
                continue
            rect = tuple(float(v) for v in field["rect_pdf"])
            try:
                prediction = recognize_field(
                    image,
                    rect,
                    dpi=dpi,
                    multiline=bool(field.get("multiline", False)),
                    preprocessing="raw",
                )
                value = validate_field_value(
                    str(field["name"]), prediction.text, known_sponsor=known_sponsor
                )
                confidence = max(0.0, min(1.0, prediction.mean_confidence / 100.0))
                fields.append(
                    ExtractedField(
                        name=str(field["name"]),
                        value=value,
                        page=page_number,
                        confidence=confidence,
                        bounding_box=_bbox(rect),
                        extraction_method="template_coordinates+tesseract+schema_validation",
                    )
                )
                field_ocr = ocr_field_tokens(
                    image,
                    rect,
                    dpi=dpi,
                    multiline=bool(field.get("multiline", False)),
                )
                entities.extend(
                    detect_hybrid_policy(
                        str(field["name"]), field_ocr, rect, page_number, self.policy
                    )
                )
            except Exception as exc:
                warnings.append(f"Field {field['name']} failed: {type(exc).__name__}: {exc}")
        for item in template.get("checkboxes", []):
            if int(item["page"]) != page_number:
                continue
            rect = tuple(float(v) for v in item["rect_pdf"])
            try:
                prediction = detect_checkbox(image, rect, dpi=dpi)
                distance = abs(prediction.inner_dark_ratio - prediction.threshold)
                confidence = min(0.99, 0.55 + distance * 4.0)
                checkboxes.append(
                    CheckboxResult(
                        name=str(item["name"]),
                        state="checked" if prediction.checked else "unchecked",
                        page=page_number,
                        confidence=confidence,
                        bounding_box=_bbox(rect),
                    )
                )
            except Exception as exc:
                warnings.append(f"Checkbox {item['name']} failed: {type(exc).__name__}: {exc}")
        return fields, checkboxes, entities, warnings

    def _extract_protocol_page(
        self, *, text: str, page_number: int, page_size: tuple[float, float]
    ) -> tuple[list[ExtractedField], list[str]]:
        metadata = self._known_metadata_from_text(text)
        result = extract_protocol_cover_fields(text, known_metadata=metadata)
        width, height = page_size
        fields = [
            ExtractedField(
                name=name,
                value=value,
                page=page_number,
                confidence=0.90 if metadata else 0.78,
                bounding_box=BoundingBox(x_min=0, y_min=0, x_max=width, y_max=height),
                extraction_method="ocr_label_anchors+regex+metadata_validation",
            )
            for name, value in result.fields.items()
        ]
        return fields, result.warnings

    def _extract_table(
        self, *, image: np.ndarray, page_number: int, doc_dir: Path
    ) -> tuple[TableArtifact | None, list[str], tuple[int, int, int, int] | None]:
        warnings: list[str] = []
        try:
            grid = detect_ruled_table_grid(image)
            if len(grid.x_boundaries) < 2 or len(grid.y_boundaries) < 2:
                return None, ["No ruled table grid detected"], None
            words = ocr_words(image, psm=6, timeout=30)
            matrix = assign_words_to_grid(words, grid.x_boundaries, grid.y_boundaries)
            frame = matrix_to_dataframe(matrix)
            stem = f"table_page_{page_number}"
            csv_path = doc_dir / f"{stem}.csv"
            json_path = doc_dir / f"{stem}.json"
            html_path = doc_dir / f"{stem}.html"
            frame.to_csv(csv_path, index=False, header=False, quoting=csv.QUOTE_MINIMAL)
            json_path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
            html_path.write_text(frame.to_html(index=False, header=False), encoding="utf-8")
            metadata = {
                "rows": int(frame.shape[0]),
                "columns": int(frame.shape[1]),
                "html_path": html_path.name,
                "method": "opencv_ruled_grid+tesseract",
            }
            box = (
                min(grid.x_boundaries),
                min(grid.y_boundaries),
                max(grid.x_boundaries),
                max(grid.y_boundaries),
            )
            return (
                TableArtifact(
                    page=page_number,
                    csv_path=csv_path.name,
                    json_path=json_path.name,
                    confidence=0.88,
                    metadata=metadata,
                ),
                warnings,
                box,
            )
        except Exception as exc:
            return None, [f"Table extraction failed: {type(exc).__name__}: {exc}"], None

    def _write_annotated_preview(
        self,
        image: np.ndarray,
        page_number: int,
        doc_dir: Path,
        fields: list[ExtractedField],
        candidates: list[RedactionCandidate],
        table_box: tuple[int, int, int, int] | None,
        dpi: int,
    ) -> Path:
        canvas = image.copy()
        scale = dpi / 72.0
        for field in fields:
            if field.page != page_number or field.bounding_box is None:
                continue
            b = field.bounding_box
            p1 = (int(b.x_min * scale), int(b.y_min * scale))
            p2 = (int(b.x_max * scale), int(b.y_max * scale))
            cv2.rectangle(canvas, p1, p2, (0, 180, 0), 2)
        for candidate in candidates:
            if candidate.page != page_number:
                continue
            x0, y0, x1, y1 = candidate.bounding_box
            p1 = (int(x0 * scale), int(y0 * scale))
            p2 = (int(x1 * scale), int(y1 * scale))
            color = (0, 0, 220) if candidate.action == "redact" else (0, 165, 255)
            cv2.rectangle(canvas, p1, p2, color, 3)
        if table_box:
            x0, y0, x1, y1 = table_box
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (220, 80, 0), 3)
        max_width = 1400
        if canvas.shape[1] > max_width:
            ratio = max_width / canvas.shape[1]
            canvas = cv2.resize(canvas, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        path = doc_dir / f"preview_page_{page_number}.png"
        cv2.imwrite(str(path), canvas)
        return path

    def process(
        self,
        *,
        filename: str,
        data: bytes,
        progress_callback: Callable[[str, int, str, dict | None], None] | None = None,
        processing_mode: str = "synchronous_day8",
    ) -> ProcessingResponse:
        started = time.perf_counter()
        created_at = utc_now()

        def emit(stage: str, percent: int, message: str, details: dict | None = None) -> None:
            if progress_callback is not None:
                progress_callback(stage, percent, message, details)

        emit("uploaded", 2, "Upload validated and processing started.")
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {suffix or 'none'}")
        if not data:
            raise ValueError("Uploaded file is empty")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError(f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")

        document_id = self.store.document_id_for(data, filename)
        doc_dir = self.store.document_dir(document_id)
        source_path = self.store.save_upload(document_id, filename, data)
        result = ProcessingResponse(
            document_id=document_id,
            source_filename=filename,
            status=ProcessingStatus.preprocessing,
            created_at=created_at,
            page_count=0,
            metadata={
                "policy": {"name": self.policy.name, "version": self.policy.version},
                "source_sha256": hashlib.sha256(data).hexdigest(),
                "processing_mode": processing_mode,
            },
        )
        self.store.save_result(result)

        emit("preprocessing", 10, "Rendering pages and preparing document images.")
        pdf_path, pages, page_sizes = self._prepare_source_pdf(source_path, doc_dir)
        result.page_count = len(pages)
        all_entities: list[DetectedEntity] = []
        preview_paths: list[Path] = []
        for page_number, (image, page_size) in enumerate(zip(pages, page_sizes), start=1):
            page_base = 15 + int(60 * (page_number - 1) / max(1, len(pages)))
            emit(
                "classifying",
                page_base,
                f"Classifying page {page_number} of {len(pages)}.",
                {"page": page_number, "page_count": len(pages)},
            )
            restoration = enhance_document_page(image)
            restored = restoration.image
            page_text = ocr_page_text(restored)
            classification = classify_hybrid(
                image,
                self.model,
                ocr_text=page_text,
                rule_image=restored,
            )
            emit(
                "extracting",
                min(78, page_base + 8),
                f"Extracting structured content from page {page_number}.",
                {"page": page_number, "label": classification.label},
            )
            result.classifications.append(
                PageClassification(
                    page=page_number,
                    label=classification.label,
                    confidence=max(0.0, min(1.0, classification.confidence)),
                    decision_source=classification.decision_source,
                    rule_reason=classification.rule.reason,
                    table_line_score=classification.rule.table_line_score,
                )
            )
            page_fields: list[ExtractedField] = []
            page_candidates: list[RedactionCandidate] = []
            table_box = None
            if classification.label in self.templates:
                fields, checkboxes, entities, warnings = self._extract_form_page(
                    image=image,
                    page_number=page_number,
                    form_type=classification.label,
                    dpi=300,
                )
                page_fields.extend(fields)
                result.fields.extend(fields)
                result.checkboxes.extend(checkboxes)
                result.warnings.extend(warnings)
                all_entities.extend(entities)
            elif classification.label == "CLINICAL_PROTOCOL":
                fields, warnings = self._extract_protocol_page(
                    text=page_text, page_number=page_number, page_size=page_size
                )
                page_fields.extend(fields)
                result.fields.extend(fields)
                result.warnings.extend(warnings)
            elif classification.label == "CLINICAL_TABLE":
                table, warnings, table_box = self._extract_table(
                    image=restored, page_number=page_number, doc_dir=doc_dir
                )
                if table:
                    result.tables.append(table)
                result.warnings.extend(warnings)
            else:
                result.warnings.append(f"Page {page_number} needs manual document-type review")

            for entity in all_entities:
                if entity.page != page_number:
                    continue
                page_candidates.append(
                    RedactionCandidate(
                        candidate_id=_candidate_id(entity),
                        entity_type=entity.entity_type,
                        action=entity.action.value,
                        page=entity.page,
                        field_name=entity.field_name,
                        masked_text=entity.masked_text,
                        confidence=entity.confidence,
                        bounding_box=list(entity.bbox_pdf),
                        detection_methods=list(entity.detection_methods),
                        needs_review=entity.needs_review,
                    )
                )
            result.redaction_candidates.extend(page_candidates)
            preview_paths.append(
                self._write_annotated_preview(
                    image,
                    page_number,
                    doc_dir,
                    page_fields,
                    page_candidates,
                    table_box,
                    dpi=300,
                )
            )

        emit(
            "redacting",
            82,
            "Applying the configured PII/CCI policy and creating a default redacted PDF.",
        )
        # Produce a policy-default PDF immediately; the review endpoint can override decisions.
        default_pdf = doc_dir / "policy_redacted.pdf"
        metadata = redact_pdf(pdf_path, default_pdf, all_entities)
        audit_path = doc_dir / "policy_audit.json"
        audit_path.write_text(
            json.dumps(
                {
                    "document_id": document_id,
                    "policy": {"name": self.policy.name, "version": self.policy.version},
                    "redaction_metadata": metadata,
                    "entities": [entity.to_audit_dict() | {"candidate_id": _candidate_id(entity)} for entity in all_entities],
                    "audit_note": "Raw detected sensitive text is not stored in the audit artifact.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        private_entities = [
            {
                "candidate_id": _candidate_id(entity),
                "entity_type": entity.entity_type,
                "action": entity.action.value,
                "page": entity.page,
                "field_name": entity.field_name,
                "confidence": entity.confidence,
                "bbox_pdf": list(entity.bbox_pdf),
                "detection_methods": list(entity.detection_methods),
                "policy_rule": entity.policy_rule,
                "needs_review": entity.needs_review,
            }
            for entity in all_entities
        ]
        self.store.save_private_state(
            document_id,
            {"source_pdf": str(pdf_path), "entities": private_entities},
        )

        emit("persisting", 94, "Writing structured results, artifacts, and audit metadata.")
        artifact_paths: list[Path] = [source_path, default_pdf, audit_path, *preview_paths]
        for table in result.tables:
            for path_name in (table.csv_path, table.json_path, table.metadata.get("html_path")):
                if path_name:
                    artifact_paths.append(doc_dir / str(path_name))
        result.artifacts = [self.store.artifact_link(document_id, path) for path in artifact_paths]
        result.status = (
            ProcessingStatus.needs_review
            if any(candidate.action == "review" or candidate.needs_review for candidate in result.redaction_candidates)
            else ProcessingStatus.completed
        )
        result.completed_at = utc_now()
        result.processing_seconds = time.perf_counter() - started
        result.metadata["page_labels"] = [item.label for item in result.classifications]
        result.metadata["field_count"] = len(result.fields)
        result.metadata["checkbox_count"] = len(result.checkboxes)
        result.metadata["table_count"] = len(result.tables)
        result.metadata["redaction_candidate_count"] = len(result.redaction_candidates)
        self.store.save_result(result)
        emit(
            "needs_review" if result.status == ProcessingStatus.needs_review else "completed",
            99,
            "Document pipeline completed; final metadata persistence is in progress.",
            {"status": result.status.value, "page_count": result.page_count},
        )
        return result
