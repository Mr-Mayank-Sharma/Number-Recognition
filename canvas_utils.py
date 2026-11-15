"""Helpers that turn streamlit-drawable-canvas output into a grayscale array."""
from __future__ import annotations

from typing import Optional

import numpy as np


def canvas_to_grayscale(image_data: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Convert the canvas RGBA output (H, W, 4) into a (H, W) uint8 grayscale array.

    Returns None when the canvas has not produced any data yet.
    White strokes -> high intensity, black/transparent background -> 0.
    """
    if image_data is None:
        return None

    arr = np.asarray(image_data)
    if arr.ndim == 2:  # already single channel
        return arr.astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        return None

    rgb = arr[..., :3].astype(np.float32)
    # Standard luminance weights for RGB -> gray.
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    if arr.shape[2] == 4:
        # Composite over black: transparent pixels contribute nothing.
        gray *= arr[..., 3].astype(np.float32) / 255.0

    return np.clip(gray, 0, 255).astype(np.uint8)


def has_ink(gray: Optional[np.ndarray]) -> bool:
    """True if the grayscale array contains at least one drawn (non-zero) pixel."""
    return gray is not None and bool(np.any(gray > 0))
