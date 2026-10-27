"""Image preprocessing shared by training, evaluation and the serving API.

Using one code path everywhere removes a classic source of training/serving skew: the API
resizes and normalises an uploaded X-ray exactly the way evaluation images were processed.
Depends only on numpy + OpenCV so the serving image stays torch-free.
"""

from __future__ import annotations

import cv2
import numpy as np

STORE_SIZE = 256  # resolution images are cached at after `prepare`
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def to_uint8_gray(img: np.ndarray) -> np.ndarray:
    """Convert any decoded image (RGB/RGBA/16-bit) to a single-channel uint8 array."""
    if img.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        img = cv2.cvtColor(img, code)
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        lo, hi = float(img.min()), float(img.max())
        img = ((img - lo) / max(hi - lo, 1e-6) * 255.0).astype(np.uint8)
    return img


def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image: expected PNG or JPEG bytes")
    return to_uint8_gray(img)


def pad_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w:
        return img
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left, cv2.BORDER_CONSTANT, value=0)


def resize_for_storage(img: np.ndarray, size: int = STORE_SIZE) -> np.ndarray:
    """Letterbox to square (keeps anatomy proportions) and downsample with area interpolation."""
    img = pad_to_square(to_uint8_gray(img))
    if img.shape[0] == size:
        return img
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def normalize(img: np.ndarray) -> np.ndarray:
    """uint8 (H, W) -> float32 (3, H, W), grayscale replicated, ImageNet statistics."""
    x = img.astype(np.float32) / 255.0
    x = np.repeat(x[None], 3, axis=0)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def to_model_input(img: np.ndarray, img_size: int) -> np.ndarray:
    """Deterministic evaluation/serving transform. Returns float32 (3, img_size, img_size)."""
    stored = img if img.shape == (STORE_SIZE, STORE_SIZE) else resize_for_storage(img)
    if img_size != STORE_SIZE:
        stored = cv2.resize(stored, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return normalize(stored)


STAT_COLUMNS = ["mean_intensity", "std_intensity", "p05_intensity", "p95_intensity", "edge_density"]


def image_stats(img: np.ndarray) -> dict[str, float]:
    """Cheap, label-free descriptors of an image used as features for data-drift monitoring.

    They react to the shifts that actually happen in radiology: a new scanner/vendor,
    different exposure, portable (AP) vs. departmental (PA) acquisitions, burned-in text.
    """
    img = resize_for_storage(img)
    p05, p95 = np.percentile(img, [5, 95])
    edges = cv2.Canny(img, 50, 150)
    return {
        "mean_intensity": float(img.mean()),
        "std_intensity": float(img.std()),
        "p05_intensity": float(p05),
        "p95_intensity": float(p95),
        "edge_density": float((edges > 0).mean()),
    }
