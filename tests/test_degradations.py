from __future__ import annotations

import numpy as np

from regdoc_ai.augmentation.degradations import DegradationConfig, apply_degradation


def test_degradation_is_deterministic() -> None:
    image = np.full((80, 120, 3), 240, dtype=np.uint8)
    image[20:60, 40:80] = 20
    config = DegradationConfig(
        name="test",
        gaussian_noise_sigma=12.0,
        shadow_strength=0.3,
        jpeg_quality=50,
        seed=42,
    )
    first = apply_degradation(image, config)
    second = apply_degradation(image, config)
    assert np.array_equal(first, second)


def test_clean_condition_preserves_pixels() -> None:
    image = np.arange(60 * 80 * 3, dtype=np.uint8).reshape(60, 80, 3)
    clean = apply_degradation(image, DegradationConfig(name="clean"))
    assert np.array_equal(image, clean)
