"""
visualizer.py — turn a Grad-CAM map into display images (no matplotlib needed).
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _jet(x: np.ndarray) -> np.ndarray:
    """x in [0,1] (H,W) → uint8 RGB using the classic 'jet' colormap."""
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def resize_heat(heat: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a [0,1] heatmap to (width, height)."""
    h = Image.fromarray((np.clip(heat, 0, 1) * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    return np.asarray(h, dtype=np.float32) / 255.0


def heatmap_image(heat: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
    if size:
        heat = resize_heat(heat, size)
    return _jet(heat)


def overlay(rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a heatmap onto an RGB uint8 image of any size."""
    rgb = np.asarray(rgb)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, -1)
    hm = heatmap_image(heat, (rgb.shape[1], rgb.shape[0])).astype(np.float32)
    return np.clip((1 - alpha) * rgb.astype(np.float32) + alpha * hm, 0, 255).astype(np.uint8)
