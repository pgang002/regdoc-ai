from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .models import RedactionAction


@dataclass(frozen=True)
class EntityPolicy:
    entity_type: str
    action: RedactionAction
    minimum_confidence: float


class RedactionPolicy:
    """Load configurable PII/CCI actions and confidence thresholds from YAML."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.name = str(config.get("policy_name", "unnamed_policy"))
        self.version = str(config.get("version", "0"))
        self.review_below = float(config.get("human_review", {}).get("confidence_below", 0.80))
        self._entities: dict[str, EntityPolicy] = {}
        for section_name in ("pii", "cci"):
            section = config.get(section_name, {})
            if not isinstance(section, dict):
                continue
            default_action = section.get("default_action", "review")
            for entity_type, item in section.items():
                if entity_type in {"default_action", "keywords", "regex_patterns"}:
                    continue
                if not isinstance(item, dict):
                    continue
                action = RedactionAction(str(item.get("action", default_action)))
                threshold = float(item.get("minimum_confidence", self.review_below))
                self._entities[entity_type] = EntityPolicy(entity_type, action, threshold)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RedactionPolicy:
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(config)

    def entity(self, entity_type: str) -> EntityPolicy:
        if entity_type in self._entities:
            return self._entities[entity_type]
        default = self.config.get("cci", {}).get("default_action", "review")
        return EntityPolicy(entity_type, RedactionAction(str(default)), self.review_below)

    def resolve_action(self, entity_type: str, confidence: float) -> tuple[RedactionAction, bool]:
        rule = self.entity(entity_type)
        below_entity_threshold = confidence < rule.minimum_confidence
        below_review_threshold = confidence < self.review_below
        if below_entity_threshold:
            return RedactionAction.REVIEW, True
        return rule.action, below_review_threshold
