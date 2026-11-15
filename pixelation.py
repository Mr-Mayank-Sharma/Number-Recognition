"""Pure image-processing functions (no Streamlit code)."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

GRID_SIZE = 32


def _to_gray(arr: np.ndarray) -> np.ndarray:
    """Return a single-channel uint8 image from gray / RGB / RGBA input."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr.astype(np.uint8)
    rgb = arr[..., :3].astype(np.float32)
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if arr.shape[2] == 4:  # composite over black so transparent = background
        gray *= arr[..., 3].astype(np.float32) / 255.0
    return np.clip(gray, 0, 255).astype(np.uint8)


def process_digit(raw_rgba_array: np.ndarray) -> np.ndarray:
    """Raw canvas (RGBA) -> centered 32x32 uint8 grayscale array."""
    # 1. Grayscale: white stroke -> high intensity, background -> 0.
    gray = _to_gray(raw_rgba_array)

    # 2. Bounding box of all drawn (non-zero) pixels; blank canvas -> zeros.
    ys, xs = np.nonzero(gray)
    if ys.size == 0:
        return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    crop = gray[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    # 3. Center: paste the ink crop into the middle of a square canvas whose side
    #    is the longer bbox edge plus 20% margin. Centering uses the ink bbox
    #    (not the raw canvas), so every digit fills the frame consistently.
    h, w = crop.shape
    side = int(round(max(h, w) * 1.2))
    side = max(side, max(h, w) + 2)
    square = np.zeros((side, side), dtype=np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    square[top : top + h, left : left + w] = crop

    # 4. Resize to 32x32 with area averaging -> smooth anti-aliased intensities.
    small = cv2.resize(square, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_AREA)

    # 5. Return 32x32 uint8 (0 = background, 255 = full stroke).
    return small.astype(np.uint8)


def _load_font(size: int) -> ImageFont.ImageFont:
    """Small font; falls back gracefully on older Pillow versions."""
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def render_pixel_grid_with_values(
    pixel_array: np.ndarray, upscale_to: int = 320, show_values: bool = True
) -> Image.Image:
    """Upscale a 32x32 array with crisp blocks, grid lines and per-cell values."""
    pixels = np.asarray(pixel_array, dtype=np.uint8)
    n = pixels.shape[0]

    # Nearest-neighbour upscale keeps every cell a sharp, flat block.
    img = Image.fromarray(pixels, mode="L").resize((upscale_to, upscale_to), Image.NEAREST)
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    cell = upscale_to / n

    # Thin grid lines between cells.
    for i in range(1, n):
        p = round(i * cell)
        draw.line([(p, 0), (p, upscale_to)], fill=(90, 90, 90), width=1)
        draw.line([(0, p), (upscale_to, p)], fill=(90, 90, 90), width=1)

    # Intensity text centered in each cell; contrast color chosen per cell.
    if not show_values:
        return img
    font = _load_font(7)
    for r in range(n):
        for c in range(n):
            v = int(pixels[r, c])
            color = (255, 255, 255) if v < 128 else (0, 0, 0)
            draw.text(
                ((c + 0.5) * cell, (r + 0.5) * cell),
                str(v),
                fill=color,
                font=font,
                anchor="mm",
            )
    return img
