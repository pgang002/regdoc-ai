from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRuntimeStatus:
    name: str
    available: bool
    reason: str


def table_transformer_status() -> ModelRuntimeStatus:
    if importlib.util.find_spec("transformers") is None:
        return ModelRuntimeStatus(
            "table_transformer",
            False,
            "The optional transformers package is not installed; model files are not bundled.",
        )
    return ModelRuntimeStatus("table_transformer", True, "Transformers runtime is installed.")


def ppstructure_status() -> ModelRuntimeStatus:
    if importlib.util.find_spec("paddleocr") is None or importlib.util.find_spec("paddle") is None:
        return ModelRuntimeStatus(
            "paddleocr_ppstructure",
            False,
            "PaddleOCR/PaddlePaddle is not installed; model files are not bundled.",
        )
    return ModelRuntimeStatus("paddleocr_ppstructure", True, "PaddleOCR runtime is installed.")
