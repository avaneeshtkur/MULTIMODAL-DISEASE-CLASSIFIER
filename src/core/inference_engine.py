"""
inference_engine.py — the common prediction interface for every model.

Every specialist returns the same dict shape:

    {
      "disease": "tb", "class_label": "Tuberculosis" | None,
      "model_score": 0.83, "score_kind": "uncalibrated model score",
      "threshold": 0.41, "uncertainty_band": [0.36, 0.46],
      "prediction_status": "SCREENING_FLAGGED" | "NO_FLAG_DETECTED" | "INCONCLUSIVE",
      "reason_code": None | "MODEL_UNAVAILABLE" | ..., "reason": "...",
      "processing_time_ms": 14.2, "model_version": "...", "model_sha256": "...",
      "simulated": False,
    }

`model_score` is the model's output after the declared activation (and
temperature scaling if the metadata provides it). It is NOT a clinical
probability of disease and is never described as one.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
from PIL import Image

import config
from src.core.image_preprocessor import PreprocessingError, preprocess
from src.core.model_registry import ModelHandle

SCREENING_FLAGGED = "SCREENING_FLAGGED"
NO_FLAG_DETECTED = "NO_FLAG_DETECTED"
INCONCLUSIVE = "INCONCLUSIVE"
STATUSES = (SCREENING_FLAGGED, NO_FLAG_DETECTED, INCONCLUSIVE)

# Reason codes produced at inference time (image-quality codes live in gatekeeper/rejection_manager)
MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
PREPROCESSING_ERROR = "PREPROCESSING_ERROR"
INFERENCE_ERROR = "INFERENCE_ERROR"
MODEL_OUTPUT_INVALID = "MODEL_OUTPUT_INVALID"
WITHIN_UNCERTAINTY_BAND = "WITHIN_UNCERTAINTY_BAND"
OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _temperature(md: dict) -> float | None:
    cal = md.get("calibration")
    return float(cal["temperature"]) if cal else None


class InvalidOutput(ValueError):
    pass


def output_to_score(raw: np.ndarray, md: dict) -> float:
    """Binary model raw output → score in [0, 1] for the positive class."""
    out = np.asarray(raw, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(out)):
        raise InvalidOutput("non-finite model output")
    act, kind = md["output"]["activation"], md["output"]["type"]
    T = _temperature(md) or 1.0
    if kind == "probability":
        if out.size != 1 or not 0.0 <= out[0] <= 1.0:
            raise InvalidOutput(f"probability output outside [0, 1]: {out.tolist()}")
        if T == 1.0:
            return float(out[0])
        p = float(np.clip(out[0], 1e-7, 1 - 1e-7))
        return float(_sigmoid(np.log(p / (1 - p)) / T))
    if act == "sigmoid":
        if out.size != 1:
            raise InvalidOutput(f"expected 1 logit, got {out.size}")
        return float(_sigmoid(out[0] / T))
    if act == "softmax":
        if out.size != 2:
            raise InvalidOutput(f"expected 2 logits, got {out.size}")
        pos = md["class_labels"].index(md["positive_class"])
        return float(_softmax(out / T)[pos])
    raise InvalidOutput(f"unsupported output spec {md['output']}")


def uncertainty_band(md: dict) -> tuple[float, float]:
    """Probability-space margin if metadata gives one; otherwise a logit-space margin."""
    t = float(md["threshold"])
    if md.get("uncertainty_margin") is not None:
        m = float(md["uncertainty_margin"])
        return max(0.0, t - m), min(1.0, t + m)
    m = float(md.get("uncertainty_margin_logit", config.DEFAULT_UNCERTAINTY_MARGIN_LOGIT))
    z = np.log(t / (1 - t))
    return float(_sigmoid(z - m)), float(_sigmoid(z + m))


def decide(score: float, md: dict) -> tuple[str, str | None]:
    """Apply threshold + uncertainty band → (status, reason_code)."""
    lo, hi = uncertainty_band(md)
    if lo < score < hi:
        return INCONCLUSIVE, WITHIN_UNCERTAINTY_BAND
    return (SCREENING_FLAGGED if score >= float(md["threshold"]) else NO_FLAG_DETECTED), None


def negative_label(md: dict) -> str:
    return next(l for l in md["class_labels"] if l != md["positive_class"])


def empty_result(disease: str, reason_code: str, reason: str, handle: ModelHandle | None = None,
                 t0: float | None = None) -> dict:
    md = handle.metadata if handle else {}
    return {
        "disease": disease, "class_label": None, "model_score": None,
        "score_kind": None, "threshold": md.get("threshold"), "uncertainty_band": None,
        "prediction_status": INCONCLUSIVE, "reason_code": reason_code, "reason": reason,
        "processing_time_ms": round((time.perf_counter() - t0) * 1000, 1) if t0 else 0.0,
        "model_version": md.get("model_version") if handle else None,
        "model_sha256": handle.sha256 if handle else None,
        "simulated": False,
    }


def predict_binary(handle: ModelHandle, image: Image.Image, disease: str) -> dict:
    """Run one binary specialist end-to-end. Never raises for model problems."""
    t0 = time.perf_counter()
    if not handle.available:
        return empty_result(disease, MODEL_UNAVAILABLE,
                            "Model not installed or failed validation: " + "; ".join(handle.errors), handle, t0)
    md = handle.metadata
    try:
        x = preprocess(image, md["preprocessing"])
    except PreprocessingError as e:
        return empty_result(disease, PREPROCESSING_ERROR, f"Preprocessing failed: {e}", handle, t0)
    try:
        raw = handle.run(x)
    except Exception as e:  # noqa: BLE001 — runtime failures must not crash the app
        return empty_result(disease, INFERENCE_ERROR, f"Inference failed: {type(e).__name__}", handle, t0)
    try:
        score = output_to_score(raw, md)
    except InvalidOutput as e:
        return empty_result(disease, MODEL_OUTPUT_INVALID, f"Model output rejected: {e}", handle, t0)

    status, code = decide(score, md)
    label = None
    if status == SCREENING_FLAGGED:
        label = md["positive_class"]
    elif status == NO_FLAG_DETECTED:
        label = negative_label(md)
    lo, hi = uncertainty_band(md)
    return {
        "disease": disease, "class_label": label, "model_score": round(score, 4),
        "raw_output": np.asarray(raw).reshape(-1).tolist(),
        "raw_positive_score": output_to_score(raw, {k: v for k, v in md.items() if k != 'calibration'}),
        "precise_model_score": score,
        "score_kind": "temperature-scaled model score" if _temperature(md) else "uncalibrated model score",
        "threshold": float(md["threshold"]), "uncertainty_band": [round(lo, 4), round(hi, 4)],
        "prediction_status": status, "reason_code": code,
        "reason": "Score is within the uncertainty band around the threshold." if code else "",
        "processing_time_ms": round((time.perf_counter() - t0) * 1000, 1),
        "model_version": md["model_version"], "model_sha256": handle.sha256, "simulated": False,
    }


def predict_multiclass(handle: ModelHandle, image: Image.Image) -> dict:
    """Gatekeeper: returns {"probs": {label: p}, "processing_time_ms": .., "error": None|str}."""
    t0 = time.perf_counter()
    md = handle.metadata
    try:
        raw = handle.run(preprocess(image, md["preprocessing"]))
        out = np.asarray(raw, dtype=np.float64).reshape(-1)
        if out.size != len(md["class_labels"]) or not np.all(np.isfinite(out)):
            raise InvalidOutput("gatekeeper output has wrong size or non-finite values")
        if md["output"]["type"] == "probability":
            probs = out
            if np.any(probs < 0) or abs(probs.sum() - 1) > 1e-3:
                raise InvalidOutput("gatekeeper probabilities do not sum to 1")
        else:
            probs = _softmax(out / (_temperature(md) or 1.0))
        return {"probs": dict(zip(md["class_labels"], map(float, probs))),
                "processing_time_ms": round((time.perf_counter() - t0) * 1000, 1), "error": None}
    except (PreprocessingError, InvalidOutput) as e:
        return {"probs": {}, "processing_time_ms": 0.0, "error": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"probs": {}, "processing_time_ms": 0.0, "error": f"inference failed: {type(e).__name__}"}


# ── Simulation (UI rehearsal only; never a model prediction) ─────────────────
SIM_THRESHOLD = 0.5


def simulate_binary(disease: str, image_sha256: str, positive_label: str, negative_label_: str) -> dict:
    """Deterministic pseudo-score from the file hash. Labelled SIMULATED everywhere."""
    t0 = time.perf_counter()
    score = int(hashlib.sha256(("sim:" + image_sha256).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    md = {"threshold": SIM_THRESHOLD, "uncertainty_margin": config.DEFAULT_UNCERTAINTY_MARGIN}
    status, code = decide(score, md)
    label = {SCREENING_FLAGGED: positive_label, NO_FLAG_DETECTED: negative_label_}.get(status)
    lo, hi = uncertainty_band(md)
    return {
        "disease": disease, "class_label": label, "model_score": round(score, 4),
        "score_kind": "SIMULATED (not a model output)", "threshold": SIM_THRESHOLD,
        "uncertainty_band": [lo, hi], "prediction_status": status, "reason_code": code,
        "reason": "SIMULATED OUTPUT for interface rehearsal — no trained model was run.",
        "processing_time_ms": round((time.perf_counter() - t0) * 1000, 1),
        "model_version": "SIMULATION", "model_sha256": None, "simulated": True,
    }
