"""
Shared fixtures.

Test models are TINY SYNTHETIC ONNX graphs (logit = a·mean(pixels) + b) built
with onnx.helper. They exercise the real ONNX Runtime path but have no medical
meaning; nothing in this test suite measures diagnostic performance.
Test images are synthetic (generated with numpy/PIL) unless a test explicitly
samples the bundled datasets to check that real images pass quality checks.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.core.image_preprocessor import default_spec  # noqa: E402

LABELS = {"tb": ["Normal", "Tuberculosis"], "dr": ["No_DR", "DR"], "oral": ["Normal", "OSCC"]}
POSITIVE = {"tb": "Tuberculosis", "dr": "DR", "oral": "OSCC"}
GK_LABELS = ["chest_xray", "fundus", "histopathology", "other"]


# ── synthetic ONNX models ────────────────────────────────────────
def make_onnx(path: Path, n_out: int = 1, bias=0.0, weight=0.0, size: int = 224) -> Path:
    import onnx
    from onnx import TensorProto, helper
    bias = np.atleast_1d(np.asarray(bias, dtype=np.float32))
    if bias.size == 1 and n_out > 1:
        bias = np.repeat(bias, n_out)
    nodes = [
        helper.make_node("ReduceMean", ["image", "axes"], ["m"], keepdims=0),
        helper.make_node("Unsqueeze", ["m", "one"], ["m2"]),
        helper.make_node("MatMul", ["m2", "W"], ["mw"]),
        helper.make_node("Add", ["mw", "B"], ["logit"]),
    ]
    inits = [
        helper.make_tensor("axes", TensorProto.INT64, [3], [1, 2, 3]),
        helper.make_tensor("one", TensorProto.INT64, [1], [1]),
        helper.make_tensor("W", TensorProto.FLOAT, [1, n_out], [float(weight)] * n_out),
        helper.make_tensor("B", TensorProto.FLOAT, [n_out], bias.tolist()),
    ]
    graph = helper.make_graph(
        nodes, "tiny", [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["N", 3, size, size])],
        [helper.make_tensor_value_info("logit", TensorProto.FLOAT, ["N", n_out])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9)
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


def binary_metadata(key: str, threshold: float = 0.5, **over) -> dict:
    md = {
        "schema_version": "1.0", "model_key": key, "model_version": f"{key}-test", "architecture": "tiny_test",
        "task": "binary", "class_labels": LABELS[key], "positive_class": POSITIVE[key],
        "output": {"type": "logit", "activation": "sigmoid"}, "threshold": threshold, "uncertainty_margin": 0.05,
        "preprocessing": default_spec("grayscale_to_rgb" if key == "tb" else "rgb"),
    }
    md.update(over)
    return md


def gatekeeper_metadata(**over) -> dict:
    md = {"schema_version": "1.0", "model_key": "gatekeeper", "model_version": "gk-test", "architecture": "tiny_test",
          "task": "multiclass", "class_labels": GK_LABELS, "output": {"type": "logit", "activation": "softmax"},
          "preprocessing": default_spec("rgb")}
    md.update(over)
    return md


def install_model(models_dir: Path, key: str, logit=0.0, metadata: dict | None = None, n_out: int | None = None):
    folder = models_dir / key
    if key == "gatekeeper":
        make_onnx(folder / "model.onnx", n_out=n_out or 4, bias=logit)
        md = metadata or gatekeeper_metadata()
    else:
        make_onnx(folder / "model.onnx", n_out=n_out or 1, bias=logit)
        md = metadata or binary_metadata(key)
    (folder / "metadata.json").write_text(json.dumps(md))
    return folder


def logit_for(p: float) -> float:
    return float(np.log(p / (1 - p)))


# ── synthetic images ─────────────────────────────────────────────
def png_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    b = io.BytesIO()
    img.save(b, fmt)
    return b.getvalue()


def xray_like(size=(320, 320), seed=0) -> Image.Image:
    """Greyscale image with texture (passes blur/exposure checks)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size[1], 0:size[0]]
    base = 90 + 60 * np.sin(x / 9.0) * np.cos(y / 13.0) + rng.normal(0, 18, (size[1], size[0]))
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "L")


def histo_like(size=(256, 256), seed=1) -> Image.Image:
    """Pink/purple textured colour image (H&E-like colours)."""
    rng = np.random.default_rng(seed)
    r = 170 + rng.normal(0, 25, size[::-1])
    g = 110 + rng.normal(0, 25, size[::-1])
    b = 200 + rng.normal(0, 20, size[::-1])
    return Image.fromarray(np.clip(np.dstack([r, g, b]), 0, 255).astype(np.uint8), "RGB")


def colourful(size=(320, 240), seed=2) -> Image.Image:
    """Non-medical image: large blocks of random colours plus fine texture (like a photo of objects)."""
    rng = np.random.default_rng(seed)
    blocks = Image.fromarray(rng.integers(0, 255, (6, 8, 3), dtype=np.uint8), "RGB").resize(size, Image.NEAREST)
    noise = rng.normal(0, 12, (size[1], size[0], 3))
    return Image.fromarray(np.clip(np.asarray(blocks, dtype=np.float64) + noise, 0, 255).astype(np.uint8), "RGB")


# ── fixtures ─────────────────────────────────────────────────────
@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    """Isolated artifacts dir; returns models dir."""
    adir = tmp_path / "artifacts"
    monkeypatch.setattr(config, "ARTIFACTS_DIR", adir)
    monkeypatch.setattr(config, "MODELS_DIR", adir / "models")
    monkeypatch.setattr(config, "ARTIFACT_RESULTS_DIR", adir / "results")
    monkeypatch.setattr(config, "MANIFESTS_DIR", adir / "manifests")
    monkeypatch.setattr(config, "INSTALLED_MANIFEST", adir / "manifests" / "installed_manifest.json")
    monkeypatch.setattr(config, "SIMULATION_MODE", False)
    (adir / "models").mkdir(parents=True)
    return adir / "models"


@pytest.fixture
def make_ctx(tmp_path, artifacts):
    from src.core.pipeline import build_context

    def _make():
        return build_context(db_path=tmp_path / "test.db", models_dir=artifacts)
    return _make
