from __future__ import annotations

import cv2
import numpy as np

from regdoc_ai.augmentation.degradations import DegradationConfig, apply_degradation


def test_condition_config(name: str, seed: int) -> DegradationConfig:
    mapping = {
        "clean": DegradationConfig(name="clean", seed=seed),
        "rotation_1_5deg": DegradationConfig(name=name, rotation_degrees=1.5, seed=seed),
        "gaussian_blur": DegradationConfig(name=name, gaussian_blur_kernel=5, seed=seed),
        "gaussian_noise": DegradationConfig(name=name, gaussian_noise_sigma=10.0, seed=seed),
        "low_contrast": DegradationConfig(name=name, contrast_factor=0.60, seed=seed),
        "directional_shadow": DegradationConfig(
            name=name, shadow_strength=0.30, shadow_angle_degrees=28.0, seed=seed
        ),
        "jpeg_compression": DegradationConfig(name=name, jpeg_quality=45, seed=seed),
        "combined_moderate": DegradationConfig(
            name=name,
            rotation_degrees=-1.75,
            gaussian_blur_kernel=3,
            gaussian_noise_sigma=7.0,
            contrast_factor=0.68,
            shadow_strength=0.24,
            shadow_angle_degrees=33.0,
            jpeg_quality=55,
            seed=seed,
        ),
    }
    if name not in mapping:
        raise KeyError(f"Unknown test condition: {name}")
    return mapping[name]


def training_augmentation(index: int, seed: int) -> DegradationConfig:
    rng = np.random.default_rng(seed + index * 1009)
    modes = index % 8
    if modes == 0:
        return DegradationConfig(name="clean", seed=seed + index)
    if modes == 1:
        return DegradationConfig(
            name="rotation", rotation_degrees=float(rng.uniform(-2.0, 2.0)), seed=seed + index
        )
    if modes == 2:
        return DegradationConfig(name="blur", gaussian_blur_kernel=3, seed=seed + index)
    if modes == 3:
        return DegradationConfig(
            name="noise", gaussian_noise_sigma=float(rng.uniform(4.0, 10.0)), seed=seed + index
        )
    if modes == 4:
        return DegradationConfig(
            name="contrast", contrast_factor=float(rng.uniform(0.62, 0.88)), seed=seed + index
        )
    if modes == 5:
        return DegradationConfig(
            name="shadow",
            shadow_strength=float(rng.uniform(0.12, 0.28)),
            shadow_angle_degrees=float(rng.uniform(10.0, 60.0)),
            seed=seed + index,
        )
    if modes == 6:
        return DegradationConfig(
            name="jpeg", jpeg_quality=int(rng.integers(45, 80)), seed=seed + index
        )
    return DegradationConfig(
        name="combined",
        rotation_degrees=float(rng.uniform(-1.5, 1.5)),
        gaussian_blur_kernel=3,
        gaussian_noise_sigma=float(rng.uniform(2.0, 7.0)),
        contrast_factor=float(rng.uniform(0.70, 0.90)),
        shadow_strength=float(rng.uniform(0.08, 0.20)),
        shadow_angle_degrees=float(rng.uniform(15.0, 55.0)),
        jpeg_quality=int(rng.integers(55, 85)),
        seed=seed + index,
    )


def load_and_augment(image_path: str, config: DegradationConfig) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Unable to read image: {image_path}")
    return apply_degradation(image, config)
