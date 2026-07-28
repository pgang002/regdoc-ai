from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .image_fallback import HOGLinearSVCClassifier, ImageClassification
from .rule_based import RuleClassification, classify_with_rules


@dataclass(frozen=True)
class HybridClassification:
    label: str
    confidence: float
    decision_source: str
    rule: RuleClassification
    image: ImageClassification | None


def classify_hybrid(
    image: np.ndarray,
    image_classifier: HOGLinearSVCClassifier,
    *,
    rule_confidence_threshold: float = 0.80,
    image_margin_threshold: float = 0.05,
    ocr_text: str | None = None,
    rule_image: np.ndarray | None = None,
) -> HybridClassification:
    rule = classify_with_rules(rule_image if rule_image is not None else image, ocr_text=ocr_text)
    if rule.label != "UNKNOWN" and rule.confidence >= rule_confidence_threshold:
        return HybridClassification(rule.label, rule.confidence, "rule", rule, None)
    image_result = image_classifier.predict(image)
    if image_result.margin < image_margin_threshold:
        return HybridClassification(
            "NEEDS_REVIEW",
            image_result.confidence,
            "image_low_margin",
            rule,
            image_result,
        )
    return HybridClassification(
        image_result.label,
        image_result.confidence,
        "image_fallback",
        rule,
        image_result,
    )
