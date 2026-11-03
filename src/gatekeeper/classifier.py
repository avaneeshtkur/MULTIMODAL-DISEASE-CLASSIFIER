"""Selected-modality suitability checks used before disease inference.

The legacy trained gatekeeper predicts modality; it was not trained on
informative/uninformative labels. It is intentionally not loaded or invoked by
the screening pipeline. Disease selection is explicit and this module applies
only deterministic compatibility checks before handing the image to a specialist.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config
from src.gatekeeper import rejection_manager as rm

SOURCE_SELECTED = "selected_specialist"
SOURCE_MANUAL = "manual_selection"  # compatibility name for older records
SOURCE_MODEL = "trained_gatekeeper"  # legacy identifier; never emitted by this pipeline
SOURCE_NONE = "unavailable"

# Conservative consistency limits (see docs/ARCHITECTURE.md). These are input
# compatibility heuristics, not a trained modality or clinical-gradability model.
XRAY_MAX_HUE_DIVERSITY = 25.0
COLOUR_MIN_SATURATION = 0.01


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
    """Descriptive only; never routes an image or changes a disease score."""
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


class InputSuitabilityGate:
    """Technical and coarse modality checks; does not diagnose or predict disease."""

    model_available = False

    def assess_selected(self, disease_key: str, metrics: dict) -> ModalityDecision:
        modality = next((m for m, disease in config.MODALITIES.items() if disease == disease_key), None)
        if modality is None:
            return ModalityDecision(None, SOURCE_SELECTED, rejection=rm.Rejection(
                rm.UNSUPPORTED_MODALITY, rm.STAGE_MODALITY, {"label": disease_key}))
        rejection = _consistency_rejection(modality, metrics)
        return ModalityDecision(modality, SOURCE_SELECTED, hint=heuristic_hint(metrics),
                                rejection=rejection,
                                note="Selected specialist directly; deterministic suitability checks only.")

    def classify(self, image, metrics: dict, manual_modality: str | None = None) -> ModalityDecision:
        """Compatibility API: require a chosen modality; never run learned routing."""
        if manual_modality is None:
            return ModalityDecision(None, SOURCE_NONE, hint=heuristic_hint(metrics),
                                    rejection=rm.Rejection(rm.DISEASE_SELECTION_REQUIRED,
                                                           rm.STAGE_MODALITY))
        disease = config.MODALITIES.get(manual_modality)
        return self.assess_selected(disease, metrics) if disease else ModalityDecision(
            None, SOURCE_SELECTED, rejection=rm.Rejection(rm.UNSUPPORTED_MODALITY,
                                                            rm.STAGE_MODALITY,
                                                            {"label": manual_modality}))


# Older imports can still construct this class, but its learned classifier path
# has been removed from screening.
Gatekeeper = InputSuitabilityGate
