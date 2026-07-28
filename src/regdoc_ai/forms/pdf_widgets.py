from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz


@dataclass(frozen=True)
class WidgetSpec:
    name: str
    page: int
    field_type: str
    rect_pdf: tuple[float, float, float, float]


def list_widgets(pdf_path: Path) -> dict[str, WidgetSpec]:
    """Return terminal PDF widgets keyed by fully qualified field name."""
    document = fitz.open(pdf_path)
    output: dict[str, WidgetSpec] = {}
    for page_index, page in enumerate(document):
        for widget in page.widgets() or []:
            output[widget.field_name] = WidgetSpec(
                name=widget.field_name,
                page=page_index + 1,
                field_type=widget.field_type_string,
                rect_pdf=tuple(float(v) for v in widget.rect),
            )
    document.close()
    return output


def fill_pdf_form(
    source_pdf: Path,
    output_editable: Path,
    output_flattened: Path,
    values: dict[str, Any],
    *,
    font_sizes: dict[str, float] | None = None,
) -> dict[str, WidgetSpec]:
    """Fill an AcroForm with PyMuPDF and save editable and flattened copies.

    Checkbox values must be booleans. Signature and push-button fields are ignored.
    The flattened copy is intended for OCR benchmarking so appearances are stable.
    """
    font_sizes = font_sizes or {}
    document = fitz.open(source_pdf)
    seen: dict[str, WidgetSpec] = {}

    for page_index, page in enumerate(document):
        for widget in list(page.widgets() or []):
            name = widget.field_name
            if name not in values:
                continue
            if widget.field_type_string in {"Signature", "Button"}:
                continue

            seen[name] = WidgetSpec(
                name=name,
                page=page_index + 1,
                field_type=widget.field_type_string,
                rect_pdf=tuple(float(v) for v in widget.rect),
            )
            value = values[name]
            if widget.field_type_string == "CheckBox":
                if not isinstance(value, bool):
                    raise TypeError(f"Checkbox {name} requires a boolean, got {type(value)!r}")
                # PyMuPDF generates the correct appearance when booleans are used.
                widget.field_value = value
            else:
                widget.field_value = "" if value is None else str(value)
                if name in font_sizes:
                    widget.text_fontsize = float(font_sizes[name])
            widget.update()

    missing = sorted(set(values) - set(seen))
    if missing:
        raise KeyError(f"Fields were not found or were unsupported in {source_pdf.name}: {missing}")

    output_editable.parent.mkdir(parents=True, exist_ok=True)
    document.save(output_editable, garbage=4, deflate=True)
    document.close()

    flattened = fitz.open(output_editable)
    flattened.bake(annots=True, widgets=True)
    output_flattened.parent.mkdir(parents=True, exist_ok=True)
    flattened.save(output_flattened, garbage=4, deflate=True)
    flattened.close()
    return seen


def render_pdf(pdf_path: Path, output_dir: Path, *, dpi: int = 300) -> list[Path]:
    """Render every PDF page to a PNG using a deterministic DPI."""
    output_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(pdf_path)
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    outputs: list[Path] = []
    for page_index, page in enumerate(document):
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        output = output_dir / f"page-{page_index + 1}.png"
        pixmap.save(output)
        outputs.append(output)
    document.close()
    return outputs
