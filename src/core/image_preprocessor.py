"""
image_preprocessor.py — metadata-driven preprocessing for ONNX inference.

The preprocessing must match training exactly, so it is described in each
model's metadata.json and nothing is guessed. Unknown or missing keys raise
PreprocessingError instead of falling back to defaults.

The default pipeline reproduces torchvision's eval transform used in
src/data.py (PIL backend):
    Resize(256)  →  CenterCrop(224)  →  ToTensor()  →  Normalize(mean, std)

Supported metadata block (all keys explicit):

    "preprocessing": {
        "color_mode": "grayscale_to_rgb" | "rgb",
        "resize": 256,                       # int, or null for no resize
        "resize_mode": "shorter_side" | "exact",
        "center_crop": 224,                  # int, or null
        "input_size": 224,                   # final square side fed to the model
        "interpolation": "bilinear" | "bicubic" | "nearest",
        "scale": "0_1",                      # divide by 255 (ToTensor)
        "normalize": {"mean": [...], "std": [...]} | null,
        "fundus_border_crop": false          # optional: crop black surround first
    }
"""
from __future__ import annotations

import numpy as np
from PIL import Image

REQUIRED_KEYS = {"color_mode", "resize", "center_crop", "input_size", "interpolation", "normalize"}
OPTIONAL_KEYS = {"resize_mode", "scale", "fundus_border_crop", "channel_order", "notes"}
COLOR_MODES = {"grayscale_to_rgb", "rgb"}
INTERPOLATIONS = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC, "nearest": Image.NEAREST}


class PreprocessingError(ValueError):
    pass


def validate_spec(spec: dict) -> dict:
    """Validate a preprocessing spec; return a normalised copy or raise."""
    if not isinstance(spec, dict):
        raise PreprocessingError("preprocessing must be an object")
    missing = REQUIRED_KEYS - spec.keys()
    if missing:
        raise PreprocessingError(f"preprocessing is missing required keys: {sorted(missing)}")
    unknown = set(spec) - REQUIRED_KEYS - OPTIONAL_KEYS
    if unknown:
        raise PreprocessingError(f"unsupported preprocessing keys (refusing to guess): {sorted(unknown)}")

    s = dict(spec)
    if s["color_mode"] not in COLOR_MODES:
        raise PreprocessingError(f"color_mode must be one of {sorted(COLOR_MODES)}, got {s['color_mode']!r}")
    if s["interpolation"] not in INTERPOLATIONS:
        raise PreprocessingError(f"interpolation must be one of {sorted(INTERPOLATIONS)}")
    s.setdefault("resize_mode", "shorter_side")
    if s["resize_mode"] not in ("shorter_side", "exact"):
        raise PreprocessingError("resize_mode must be 'shorter_side' or 'exact'")
    s.setdefault("scale", "0_1")
    if s["scale"] != "0_1":
        raise PreprocessingError("only scale='0_1' (ToTensor) is supported")
    if s.get("channel_order", "NCHW") != "NCHW":
        raise PreprocessingError("only channel_order='NCHW' is supported")
    s.setdefault("fundus_border_crop", False)

    for k in ("resize", "center_crop"):
        if s[k] is not None and (not isinstance(s[k], int) or s[k] <= 0):
            raise PreprocessingError(f"{k} must be a positive int or null")
    if not isinstance(s["input_size"], int) or s["input_size"] <= 0:
        raise PreprocessingError("input_size must be a positive int")

    # The final spatial size must equal input_size, otherwise the model gets the wrong shape.
    if s["center_crop"] is not None:
        final = s["center_crop"]
    elif s["resize"] is not None and s["resize_mode"] == "exact":
        final = s["resize"]
    else:
        raise PreprocessingError("pipeline must end in a fixed square size (center_crop, or exact resize)")
    if final != s["input_size"]:
        raise PreprocessingError(f"pipeline produces {final}px but input_size is {s['input_size']}px")
    if s["resize"] is not None and s["center_crop"] is not None and s["resize"] < s["center_crop"]:
        raise PreprocessingError("resize must be ≥ center_crop")

    norm = s["normalize"]
    if norm is not None:
        if not isinstance(norm, dict) or set(norm) != {"mean", "std"}:
            raise PreprocessingError("normalize must be {'mean': [3 floats], 'std': [3 floats]} or null")
        if len(norm["mean"]) != 3 or len(norm["std"]) != 3 or any(v <= 0 for v in norm["std"]):
            raise PreprocessingError("normalize mean/std must have 3 values and std > 0")
    return s


def to_color_mode(img: Image.Image, color_mode: str) -> Image.Image:
    """Same conversion as src/data.load_image: grayscale → L → 3-channel RGB."""
    if color_mode == "grayscale_to_rgb":
        return img.convert("L").convert("RGB")
    return img.convert("RGB")


def crop_fundus_border(img: Image.Image, tol: int = 10) -> Image.Image:
    """Crop the black surround of a fundus photo to the retina bounding box."""
    g = np.asarray(img.convert("L"))
    mask = g > tol
    if mask.sum() < 0.05 * mask.size:      # nothing meaningful found; leave untouched
        return img
    ys, xs = np.where(mask)
    return img.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


def _resize(img: Image.Image, size: int, mode: str, interp) -> Image.Image:
    w, h = img.size
    if mode == "exact":
        return img.resize((size, size), interp)
    # torchvision Resize(int): shorter side → size, longer side truncated
    if w <= h:
        ow, oh = size, int(size * h / w)
    else:
        oh, ow = size, int(size * w / h)
    if (ow, oh) == (w, h):
        return img
    return img.resize((ow, oh), interp)


def _center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w < size or h < size:
        raise PreprocessingError(f"image {w}x{h} smaller than crop {size}")
    top = int(round((h - size) / 2.0))
    left = int(round((w - size) / 2.0))
    return img.crop((left, top, left + size, top + size))


def prepare_pil(img: Image.Image, spec: dict) -> Image.Image:
    """Geometric + colour steps only (used for display/Grad-CAM alignment)."""
    s = validate_spec(spec)
    img = to_color_mode(img, s["color_mode"])
    if s["fundus_border_crop"]:
        img = crop_fundus_border(img)
    interp = INTERPOLATIONS[s["interpolation"]]
    if s["resize"] is not None:
        img = _resize(img, s["resize"], s["resize_mode"], interp)
    if s["center_crop"] is not None:
        img = _center_crop(img, s["center_crop"])
    return img


def preprocess(img: Image.Image, spec: dict) -> np.ndarray:
    """PIL image → float32 array of shape (1, 3, H, W) ready for ONNX Runtime."""
    s = validate_spec(spec)
    pil = prepare_pil(img, s)
    arr = np.asarray(pil, dtype=np.float32) / 255.0           # H, W, 3
    if s["normalize"] is not None:
        mean = np.asarray(s["normalize"]["mean"], dtype=np.float32)
        std = np.asarray(s["normalize"]["std"], dtype=np.float32)
        arr = (arr - mean) / std
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...], dtype=np.float32)


def default_spec(color_mode: str, img_size: int = 224, mean=None, std=None) -> dict:
    """The spec matching src/data.get_transforms(train=False). Used when *writing* metadata."""
    import config
    return {
        "color_mode": color_mode,
        "resize": img_size + 32,
        "resize_mode": "shorter_side",
        "center_crop": img_size,
        "input_size": img_size,
        "interpolation": "bilinear",
        "scale": "0_1",
        "normalize": {"mean": list(mean or config.MEAN), "std": list(std or config.STD)},
        "fundus_border_crop": False,
    }
