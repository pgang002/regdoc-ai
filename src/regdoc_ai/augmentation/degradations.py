from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DegradationConfig:
    """Configuration for deterministic synthetic scan artifacts.

    The source page remains a rendering of an actual official FDA form template.
    These transformations only simulate acquisition defects for robustness testing.
    """

    name: str
    rotation_degrees: float = 0.0
    gaussian_blur_kernel: int = 0
    gaussian_noise_sigma: float = 0.0
    contrast_factor: float = 1.0
    shadow_strength: float = 0.0
    shadow_angle_degrees: float = 25.0
    jpeg_quality: int = 100
    seed: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _rotate_keep_size(image: np.ndarray, degrees: float) -> np.ndarray:
    if abs(degrees) < 1e-9:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), degrees, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _apply_low_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-9:
        return image
    working = image.astype(np.float32)
    faded = 255.0 - factor * (255.0 - working)
    return np.clip(faded, 0, 255).astype(np.uint8)


def _apply_shadow(
    image: np.ndarray,
    strength: float,
    angle_degrees: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if strength <= 0:
        return image
    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    theta = np.deg2rad(angle_degrees)
    projection = xx * np.cos(theta) + yy * np.sin(theta)
    projection = (projection - projection.min()) / max(float(np.ptp(projection)), 1.0)
    center = float(rng.uniform(0.28, 0.72))
    spread = float(rng.uniform(0.17, 0.27))
    band = np.exp(-0.5 * ((projection - center) / spread) ** 2)
    illumination = 1.0 - strength * band
    shaded = image.astype(np.float32) * illumination[..., None]
    return np.clip(shaded, 0, 255).astype(np.uint8)


def _jpeg_round_trip(image: np.ndarray, quality: int) -> np.ndarray:
    if quality >= 100:
        return image
    ok, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decoding failed")
    return decoded


def apply_degradation(image: np.ndarray, config: DegradationConfig) -> np.ndarray:
    """Apply a deterministic sequence of realistic scan defects."""
    if image is None:
        raise ValueError("image cannot be None")
    if not (0 < config.contrast_factor <= 1.5):
        raise ValueError("contrast_factor must be in (0, 1.5]")
    if not (1 <= config.jpeg_quality <= 100):
        raise ValueError("jpeg_quality must be between 1 and 100")
    if config.gaussian_blur_kernel not in {0, 3, 5, 7, 9}:
        raise ValueError("gaussian_blur_kernel must be 0 or a supported odd kernel")

    rng = np.random.default_rng(config.seed)
    output = image.copy()
    output = _apply_low_contrast(output, config.contrast_factor)
    output = _apply_shadow(
        output,
        config.shadow_strength,
        config.shadow_angle_degrees,
        rng,
    )
    if config.gaussian_blur_kernel > 0:
        k = config.gaussian_blur_kernel
        output = cv2.GaussianBlur(output, (k, k), 0)
    if config.gaussian_noise_sigma > 0:
        noise = rng.normal(0.0, config.gaussian_noise_sigma, output.shape).astype(np.float32)
        output = np.clip(output.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    output = _rotate_keep_size(output, config.rotation_degrees)
    output = _jpeg_round_trip(output, config.jpeg_quality)
    return output
