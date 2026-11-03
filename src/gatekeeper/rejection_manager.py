"""
rejection_manager.py — structured, translatable rejection events.

Every rejection carries a reason code, a human-readable explanation, a
suggested corrective action and a UTC timestamp. Text lives in
src/clinical/translations.py so English and Hindi stay in sync.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

IMAGE_CORRUPTED = "IMAGE_CORRUPTED"
IMAGE_UNSUPPORTED_FORMAT = "IMAGE_UNSUPPORTED_FORMAT"
IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
IMAGE_TOO_SMALL = "IMAGE_TOO_SMALL"
IMAGE_EXTREME_ASPECT = "IMAGE_EXTREME_ASPECT"
IMAGE_TOO_BLURRY = "IMAGE_TOO_BLURRY"
IMAGE_OVEREXPOSED = "IMAGE_OVEREXPOSED"
IMAGE_UNDEREXPOSED = "IMAGE_UNDEREXPOSED"
DUPLICATE_IMAGE = "DUPLICATE_IMAGE"
UNSUPPORTED_MODALITY = "UNSUPPORTED_MODALITY"
UNCERTAIN_MODALITY = "UNCERTAIN_MODALITY"
MODALITY_MISMATCH = "MODALITY_MISMATCH"
MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
DISEASE_SELECTION_REQUIRED = "DISEASE_SELECTION_REQUIRED"

ALL_CODES = [
    IMAGE_CORRUPTED, IMAGE_UNSUPPORTED_FORMAT, IMAGE_TOO_LARGE, IMAGE_TOO_SMALL, IMAGE_EXTREME_ASPECT,
    IMAGE_TOO_BLURRY, IMAGE_OVEREXPOSED, IMAGE_UNDEREXPOSED, DUPLICATE_IMAGE, UNSUPPORTED_MODALITY,
    UNCERTAIN_MODALITY, MODALITY_MISMATCH, MODEL_UNAVAILABLE, DISEASE_SELECTION_REQUIRED,
]

# Which stage produced the rejection (used for analytics grouping)
STAGE_QUALITY = "quality"
STAGE_DUPLICATE = "duplicate"
STAGE_MODALITY = "modality"
STAGE_MODEL = "model"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Rejection:
    code: str
    stage: str
    details: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=_now)

    def _localised_details(self, lang: str) -> dict:
        """Translate detail values that are disease/modality keys (e.g. 'tb', 'fundus')."""
        from src.clinical.translations import has, t
        out = {}
        for k, v in self.details.items():
            if isinstance(v, str):
                for prefix in ("disease.", "modality."):
                    if has(prefix + v, "en"):
                        v = t(prefix + v, lang)
                        break
            out[k] = v
        return out

    def explanation(self, lang: str = "en") -> str:
        from src.clinical.translations import t
        return t(f"reject.{self.code}.why", lang, **self._localised_details(lang))

    def action(self, lang: str = "en") -> str:
        from src.clinical.translations import t
        return t(f"reject.{self.code}.fix", lang, **self._localised_details(lang))

    def to_dict(self, lang: str = "en") -> dict:
        return {"code": self.code, "stage": self.stage, "explanation": self.explanation(lang),
                "suggested_action": self.action(lang), "timestamp": self.timestamp,
                "details": self.details}
