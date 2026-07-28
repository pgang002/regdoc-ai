from pathlib import Path

import cv2
import numpy as np

from regdoc_ai.classification.image_fallback import extract_hog_features
from regdoc_ai.classification.rule_based import classify_with_rules

ROOT = Path(__file__).resolve().parents[1]


def test_rule_classifier_identifies_actual_fda_forms() -> None:
    samples = {
        "FDA_1572": "data/processed/document_understanding/base_images/FDA_1572_NCT04796896_p001.png",
        "FDA_3454": "data/processed/document_understanding/base_images/FDA_3454_NCT04796896_p001.png",
        "FDA_3455": "data/processed/document_understanding/base_images/FDA_3455_NCT04796896_p001.png",
    }
    for expected, relative in samples.items():
        image = cv2.imread(str(ROOT / relative))
        assert image is not None
        prediction = classify_with_rules(image, max_dimension=1000)
        assert prediction.label == expected
        assert prediction.confidence >= 0.9


def test_hog_features_are_deterministic() -> None:
    image = np.full((800, 600, 3), 255, dtype=np.uint8)
    cv2.putText(image, "FORM FDA 1572", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    first = extract_hog_features(image, width=256, height=320)
    second = extract_hog_features(image, width=256, height=320)
    assert first.shape == second.shape
    assert first.size > 1000
    assert np.allclose(first, second)
