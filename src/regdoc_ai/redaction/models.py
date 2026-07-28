from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import sha256
from typing import Any


class RedactionAction(str, Enum):
    REDACT = "redact"
    REVIEW = "review"
    RETAIN = "retain"
    IGNORE = "ignore"


@dataclass(frozen=True)
class DetectedEntity:
    entity_type: str
    action: RedactionAction
    page: int
    field_name: str
    detected_text: str
    confidence: float
    bbox_pdf: tuple[float, float, float, float]
    detection_methods: tuple[str, ...] = field(default_factory=tuple)
    policy_rule: str = ""
    needs_review: bool = False

    @property
    def text_sha256(self) -> str:
        return sha256(self.detected_text.encode("utf-8")).hexdigest()

    @property
    def masked_text(self) -> str:
        value = self.detected_text.strip()
        if not value:
            return ""
        if len(value) <= 2:
            return "*" * len(value)
        return value[0] + "*" * min(len(value) - 2, 12) + value[-1]

    def to_audit_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("detected_text", None)
        data["action"] = self.action.value
        data["text_sha256"] = self.text_sha256
        data["masked_text"] = self.masked_text
        data["bbox_pdf"] = list(self.bbox_pdf)
        data["detection_methods"] = list(self.detection_methods)
        return data
