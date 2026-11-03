"""
base.py — shared behaviour for the three specialist screening modules.

Each module owns: model availability, preprocessing (via metadata),
inference, threshold/uncertainty, input-domain checks and explanation.
A failure in one module never affects the others.
"""
from __future__ import annotations

from PIL import Image

import config
from src.core import inference_engine as ie
from src.core.model_registry import ModelHandle, ModelRegistry


class SpecialistModule:
    key: str = ""
    modality: str = ""
    default_labels: tuple[str, str] = ("Negative", "Positive")   # (negative, positive) for simulation only

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    @property
    def handle(self) -> ModelHandle:
        return self.registry.get(self.key)

    @property
    def available(self) -> bool:
        return self.handle.available

    @property
    def name(self) -> str:
        return config.DISEASES[self.key]["name"]

    def input_domain_issue(self, image: Image.Image, metrics: dict) -> str | None:
        """Override: return a reason string if the image is outside what the model was trained on."""
        return None

    def predict(self, image: Image.Image, metrics: dict | None = None, image_sha256: str | None = None) -> dict:
        if not self.available:
            if config.SIMULATION_MODE and image_sha256:
                return ie.simulate_binary(self.key, image_sha256, self.default_labels[1], self.default_labels[0])
            return ie.empty_result(self.key, ie.MODEL_UNAVAILABLE, self.setup_message(), self.handle)
        issue = self.input_domain_issue(image, metrics or {})
        if issue:
            return ie.empty_result(self.key, ie.OUT_OF_DISTRIBUTION, issue, self.handle)
        return ie.predict_binary(self.handle, image, self.key)

    def explain(self, image: Image.Image, result: dict) -> dict:
        from src.explainability.gradcam import explain
        if result.get("simulated"):
            return {"ok": False, "message": "No explanation: result is SIMULATED (no model was run)."}
        if result.get("model_score") is None:
            return {"ok": False, "message": "No explanation: no model score was produced."}
        return explain(self.handle, image, onnx_score=result["model_score"])

    def setup_message(self) -> str:
        h = self.handle
        why = "; ".join(h.errors) or "not installed"
        return (f"{self.name} model unavailable ({why}). Install trained artifacts with: "
                f"python scripts/import_artifacts.py --archive trained_outputs.zip")
