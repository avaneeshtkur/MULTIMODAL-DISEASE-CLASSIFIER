"""
image_quality.py — Stage A: deterministic image validation.

Checks (in order, first failure wins):
    extension → file size → decodable → real format → dimensions →
    aspect ratio → blur (global floor) → exposure

All thresholds live in config.QUALITY and were calibrated on the bundled
datasets so that every training image passes (see docs/ARCHITECTURE.md).
Dark backgrounds alone (common in X-rays and fundus photos) never trigger
rejection: exposure checks look at the brightest/darkest 1% of pixels.
"""
from __future__ import annotations

import io
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

import config
from src.gatekeeper import rejection_manager as rm
from src.utils.image_hash import sha256_bytes


@dataclass
class QualityResult:
    ok: bool
    image: Image.Image | None = None
    sha256: str | None = None
    metrics: dict = field(default_factory=dict)
    rejection: rm.Rejection | None = None
    warnings: list[str] = field(default_factory=list)


def _reject(code: str, sha: str | None = None, metrics: dict | None = None, **details) -> QualityResult:
    return QualityResult(False, sha256=sha, metrics=metrics or {},
                         rejection=rm.Rejection(code, rm.STAGE_QUALITY, details))


def _to_8bit(img: Image.Image, warns: list[str]) -> Image.Image:
    """Bring 16-bit / float images to 8-bit by bit-depth scaling (not per-image contrast stretching)."""
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        a = np.asarray(img, dtype=np.float64)
        scale = 65535.0 if a.max() > 255 else 255.0
        warns.append("16-bit image converted to 8-bit; models were trained on 8-bit images.")
        return Image.fromarray(np.clip(a / scale * 255.0, 0, 255).astype(np.uint8), mode="L")
    if img.mode == "F":
        a = np.asarray(img, dtype=np.float64)
        warns.append("Floating-point image converted to 8-bit.")
        return Image.fromarray(np.clip(a if a.max() > 1.0 else a * 255.0, 0, 255).astype(np.uint8), mode="L")
    return img


def grey_array(img: Image.Image, max_side: int = 512) -> np.ndarray:
    g = img.convert("L")
    if max(g.size) > max_side:
        g = g.copy()
        g.thumbnail((max_side, max_side), Image.BILINEAR)
    return np.asarray(g, dtype=np.float32)


def laplacian_variance(g: np.ndarray) -> float:
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def colour_metrics(img: Image.Image) -> dict:
    """Saturation and hue-diversity (2nd eigenvalue of chroma covariance)."""
    rgb = np.asarray(img.convert("RGB").resize((128, 128), Image.BILINEAR), dtype=np.float32)
    mx, mn = rgb.max(2), rgb.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)
    flat = rgb.reshape(-1, 3)
    chroma = np.stack([flat[:, 0] - flat[:, 1], flat[:, 1] - flat[:, 2]], 1)
    ev = np.linalg.eigvalsh(np.cov(chroma.T)) if chroma.std() > 0 else np.zeros(2)
    mean_rgb = flat.mean(0)
    h, w = rgb.shape[:2]
    c = max(4, int(min(h, w) * 0.08))
    grey = rgb.mean(2)
    corners = np.concatenate([grey[:c, :c].ravel(), grey[:c, -c:].ravel(), grey[-c:, :c].ravel(), grey[-c:, -c:].ravel()])
    return {
        "saturation_mean": round(float(sat.mean()), 4),
        "hue_diversity": round(float(np.sqrt(max(ev[0], 0.0))), 3),
        "mean_rgb": [round(float(v), 1) for v in mean_rgb],
        "dark_corner_frac": round(float((corners < 20).mean()), 3),
    }


def compute_metrics(img: Image.Image) -> dict:
    g = grey_array(img)
    m = {
        "width": img.width, "height": img.height,
        "aspect_ratio": round(max(img.size) / max(1, min(img.size)), 3),
        "blur_laplacian_var": round(laplacian_variance(g), 3),
        "brightness_mean": round(float(g.mean()), 2),
        "p1": float(np.percentile(g, 1)), "p99": float(np.percentile(g, 99)),
        "white_frac": round(float((g >= 250).mean()), 4),
    }
    m.update(colour_metrics(img))
    return m


def validate_image(data: bytes, filename: str) -> QualityResult:
    """Validate raw upload bytes. Returns a decoded RGB/L PIL image on success."""
    q = config.QUALITY
    ext = Path(filename or "").suffix.lower()
    if ext not in q["allowed_extensions"]:
        return _reject(rm.IMAGE_UNSUPPORTED_FORMAT, fmt=ext or "unknown")
    sha = sha256_bytes(data)
    if len(data) > q["max_file_mb"] * 1024 * 1024:
        return _reject(rm.IMAGE_TOO_LARGE, sha, limit_mb=q["max_file_mb"])
    if not data:
        return _reject(rm.IMAGE_CORRUPTED, sha)

    warns: list[str] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            probe = Image.open(io.BytesIO(data))
            fmt = probe.format
            probe.verify()                       # structural check
            img = Image.open(io.BytesIO(data))
            img.load()                           # full decode (catches truncation)
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        return _reject(rm.IMAGE_TOO_LARGE, sha, limit_mb=q["max_file_mb"])
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return _reject(rm.IMAGE_CORRUPTED, sha)

    if fmt not in q["allowed_formats"]:
        return _reject(rm.IMAGE_UNSUPPORTED_FORMAT, sha, fmt=fmt or "unknown")
    expected_ext = {"PNG": {".png"}, "JPEG": {".jpg", ".jpeg"}, "MPO": {".jpg", ".jpeg"}, "BMP": {".bmp"},
                    "TIFF": {".tif", ".tiff"}, "WEBP": {".webp"}}[fmt]
    if ext not in expected_ext:
        warns.append(f"File extension {ext} does not match actual format {fmt}; the actual format was used.")
    if getattr(img, "n_frames", 1) > 1:
        warns.append("Multi-frame image: only the first frame was analysed.")
        img.seek(0)

    img = _to_8bit(img, warns)
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")

    w, h = img.size
    if min(w, h) < q["min_side_px"]:
        return _reject(rm.IMAGE_TOO_SMALL, sha, width=w, height=h, min_side=q["min_side_px"])
    if max(w, h) / min(w, h) > q["max_aspect_ratio"]:
        return _reject(rm.IMAGE_EXTREME_ASPECT, sha, ratio=round(max(w, h) / min(w, h), 1),
                       max_ratio=q["max_aspect_ratio"])

    m = compute_metrics(img)
    if m["p99"] < q["underexposed_p99_max"]:
        return _reject(rm.IMAGE_UNDEREXPOSED, sha, m)
    if m["p1"] > q["overexposed_p1_min"] or m["white_frac"] > q["overexposed_white_frac"]:
        return _reject(rm.IMAGE_OVEREXPOSED, sha, m)
    if m["blur_laplacian_var"] < q["blur_min_global"]:
        return _reject(rm.IMAGE_TOO_BLURRY, sha, m, score=m["blur_laplacian_var"], minimum=q["blur_min_global"])
    return QualityResult(True, img, sha, m, None, warns)


def modality_blur_check(metrics: dict, modality: str) -> rm.Rejection | None:
    """Stricter, modality-specific sharpness floor applied after routing."""
    floor = config.QUALITY["blur_min_by_modality"].get(modality)
    if floor is not None and metrics.get("blur_laplacian_var", 0) < floor:
        return rm.Rejection(rm.IMAGE_TOO_BLURRY, rm.STAGE_QUALITY,
                            {"score": metrics["blur_laplacian_var"], "minimum": floor})
    return None
