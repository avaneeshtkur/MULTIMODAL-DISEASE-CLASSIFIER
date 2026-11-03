"""
classifier.py — Stage B: image modality classification and routing.

With a trained gatekeeper model (artifacts/models/gatekeeper):
    route only if top score ≥ GATEKEEPER_MIN_CONFIDENCE and the margin over the
    runner-up ≥ GATEKEEPER_MIN_MARGIN; otherwise UNCERTAIN_MODALITY.
    A top class that is not a supported modality → UNSUPPORTED_MODALITY.

Without the trained model:
    automatic routing is DISABLED. Simple colour statistics give a "hint"
    which is shown to the user but never used to route, because hand-written
    rules are not a validated classifier. Manual module selection is still
    allowed; conservative consistency checks catch obvious mismatches
    (e.g. a multi-coloured photo submitted as a chest X-ray).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config
from src.core.artifact_validator import canonical_modality
from src.core.inference_engine import predict_multiclass
from src.core.model_registry import ModelRegistry
from src.gatekeeper import rejection_manager as rm

SOURCE_MODEL = "trained_gatekeeper"
SOURCE_MANUAL = "manual_selection"
SOURCE_NONE = "unavailable"

# Conservative consistency limits (calibrated on bundled data; see docs/ARCHITECTURE.md)
XRAY_MAX_HUE_DIVERSITY = 25.0      # chest X-rays (incl. tinted film photos) ≤ 15.8 in data
COLOUR_MIN_SATURATION = 0.01       # fundus ≥ 0.024, histopathology ≥ 0.11 in data


@dataclass
class ModalityDecision:
    modality: str | None              # chest_xray | fundus | histopathology | None
    source: str
    confidence: float | None = None
    probs: dict = field(default_factory=dict)
    hint: str | None = None
    rejection: rm.Rejection | None = None
    note: str | None = None
    processing_time_ms: float = 0.0

    @property
    def disease_key(self) -> str | None:
        return config.MODALITIES.get(self.modality) if self.modality else None


def heuristic_hint(metrics: dict) -> str:
    """Descriptive only. Never used for routing."""
    sat, hue = metrics.get("saturation_mean", 0), metrics.get("hue_diversity", 0)
    r, g, b = metrics.get("mean_rgb", [0, 0, 0])
    if sat < COLOUR_MIN_SATURATION or hue < 1.0:
        return "greyscale/monochrome image (consistent with an X-ray)"
    if b > r > g:
        return "pink/purple stain colours (consistent with H&E histopathology)"
    if r >= g >= b and metrics.get("dark_corner_frac", 0) > 0.5:
        return "reddish image with dark surround (consistent with a fundus photograph)"
    if hue > XRAY_MAX_HUE_DIVERSITY:
        return "multi-coloured image (not consistent with supported medical modalities)"
    return "no clear pattern"


def _consistency_rejection(modality: str, metrics: dict) -> rm.Rejection | None:
    if modality == "chest_xray" and metrics.get("hue_diversity", 0) > XRAY_MAX_HUE_DIVERSITY:
        return rm.Rejection(rm.MODALITY_MISMATCH, rm.STAGE_MODALITY,
                            {"expected": "chest_xray", "observed": "multi-coloured image"})
    if modality in ("fundus", "histopathology") and metrics.get("saturation_mean", 1) < COLOUR_MIN_SATURATION:
        return rm.Rejection(rm.MODALITY_MISMATCH, rm.STAGE_MODALITY,
                            {"expected": modality, "observed": "greyscale image"})
    return None


class Gatekeeper:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    @property
    def model_available(self) -> bool:
        return self.registry.is_available("gatekeeper")

    def _model_decision(self, image) -> ModalityDecision:
        res = predict_multiclass(self.registry.get("gatekeeper"), image)
        if res["error"]:
            return ModalityDecision(None, SOURCE_MODEL, rejection=rm.Rejection(
                rm.UNCERTAIN_MODALITY, rm.STAGE_MODALITY, {"reason": res["error"]}))
        ranked = sorted(res["probs"].items(), key=lambda kv: kv[1], reverse=True)
        (top_label, top_p), second_p = ranked[0], (ranked[1][1] if len(ranked) > 1 else 0.0)
        modality = canonical_modality(top_label)
        d = ModalityDecision(modality, SOURCE_MODEL, round(top_p, 4), res["probs"],
                             processing_time_ms=res["processing_time_ms"])
        # Per-model routing limits from metadata (e.g. stricter for a gatekeeper without an "other" class)
        routing = self.registry.get("gatekeeper").metadata.get("routing") or {}
        min_conf = max(config.GATEKEEPER_MIN_CONFIDENCE, routing.get("min_confidence", 0.0))
        min_margin = max(config.GATEKEEPER_MIN_MARGIN, routing.get("min_margin", 0.0))
        if top_p < min_conf or top_p - second_p < min_margin:
            d.modality = None
            d.rejection = rm.Rejection(rm.UNCERTAIN_MODALITY, rm.STAGE_MODALITY,
                                       {"reason": f"top score {top_p:.2f}, margin {top_p - second_p:.2f}"})
        elif modality is None:
            d.rejection = rm.Rejection(rm.UNSUPPORTED_MODALITY, rm.STAGE_MODALITY, {"label": top_label})
        return d

    def classify(self, image, metrics: dict, manual_modality: str | None = None) -> ModalityDecision:
        """manual_modality=None means automatic routing."""
        hint = heuristic_hint(metrics)
        if self.model_available:
            d = self._model_decision(image)
            d.hint = hint
            if not d.rejection:
                # A softmax gatekeeper always picks *some* class; deterministic checks catch obvious
                # non-medical inputs it routes confidently (e.g. a colourful photo as "chest X-ray").
                rej = _consistency_rejection(d.modality, metrics)
                if rej:
                    d.rejection = rm.Rejection(rm.UNSUPPORTED_MODALITY, rm.STAGE_MODALITY, {
                        "label": f"{d.modality} by model, but {rej.details['observed']} fails consistency checks"})
                    d.modality = None
            if manual_modality is None or d.rejection:
                return d
            if d.modality != manual_modality:
                d.rejection = rm.Rejection(rm.MODALITY_MISMATCH, rm.STAGE_MODALITY,
                                           {"expected": manual_modality, "observed": d.modality})
                d.modality = None
            else:
                d.source = SOURCE_MANUAL + "+" + SOURCE_MODEL
            return d

        # No trained gatekeeper
        if manual_modality is None:
            return ModalityDecision(None, SOURCE_NONE, hint=hint, rejection=rm.Rejection(
                rm.UNCERTAIN_MODALITY, rm.STAGE_MODALITY, {"reason": "gatekeeper model not installed"}),
                note="Automatic modality detection is unavailable. Select the screening module manually.")
        rej = _consistency_rejection(manual_modality, metrics)
        return ModalityDecision(None if rej else manual_modality, SOURCE_MANUAL, hint=hint, rejection=rej,
                                note="Modality selected manually; not verified by a trained classifier.")
