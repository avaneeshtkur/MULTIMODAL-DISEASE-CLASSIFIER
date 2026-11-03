"""Model registry, artifact validation, preprocessing, thresholds, uncertainty, invalid outputs.

Uses tiny synthetic ONNX models (no medical meaning)."""
import json

import numpy as np
import pytest
from PIL import Image

from src.core import inference_engine as ie
from src.core.artifact_validator import validate_model_dir
from src.core.image_preprocessor import PreprocessingError, default_spec, preprocess, validate_spec
from src.core.model_registry import ModelRegistry
from src.diseases import build_modules
from tests.conftest import binary_metadata, histo_like, install_model, logit_for, make_onnx, xray_like


# ── loaders / registry ───────────────────────────────────────────
def test_missing_models_reported_not_crashing(artifacts):
    reg = ModelRegistry(models_dir=artifacts)
    assert all(r["status"] == "missing" for r in reg.status_rows())
    res = build_modules(reg)["tb"].predict(xray_like())
    assert res["prediction_status"] == ie.INCONCLUSIVE and res["reason_code"] == ie.MODEL_UNAVAILABLE
    assert "import_artifacts.py" in res["reason"]


@pytest.mark.parametrize("key", ["tb", "dr", "oral"])
def test_each_disease_loader(artifacts, key):
    install_model(artifacts, key, logit=logit_for(0.9))
    reg = ModelRegistry(models_dir=artifacts)
    assert reg.is_available(key)
    img = xray_like() if key == "tb" else histo_like()
    res = build_modules(reg)[key].predict(img)
    assert res["prediction_status"] == ie.SCREENING_FLAGGED
    assert res["class_label"] == binary_metadata(key)["positive_class"]
    assert res["processing_time_ms"] > 0 and res["model_version"] == f"{key}-test"


def test_failed_model_does_not_block_others(artifacts):
    install_model(artifacts, "tb", logit=0.0)
    install_model(artifacts, "dr", metadata=binary_metadata("dr", threshold=2.0))   # invalid threshold
    reg = ModelRegistry(models_dir=artifacts)
    assert reg.is_available("tb") and reg.get("dr").status == "invalid"
    assert build_modules(reg)["dr"].predict(histo_like())["reason_code"] == ie.MODEL_UNAVAILABLE


@pytest.mark.parametrize("mutate,needle", [
    (lambda m: m.pop("preprocessing"), "preprocessing"),
    (lambda m: m["preprocessing"].update(color_mode="hsv"), "color_mode"),
    (lambda m: m["preprocessing"].update(magic_filter=True), "unsupported preprocessing keys"),
    (lambda m: m.update(threshold=1.5), "threshold"),
    (lambda m: m.update(class_labels=["Tuberculosis", "Normal"]), "index 1"),
    (lambda m: m.update(positive_class="Normal"), "not a recognised positive"),
    (lambda m: m.update(task="multiclass"), "only binary"),
    (lambda m: m.update(schema_version="9.9"), "schema_version"),
    (lambda m: m.update(model_key="dr"), "does not match"),
])
def test_invalid_metadata_rejected(artifacts, mutate, needle):
    md = binary_metadata("tb")
    mutate(md)
    folder = install_model(artifacts, "tb", metadata=md)
    rep = validate_model_dir(folder, "tb")
    assert not rep.ok and any(needle in e for e in rep.errors), rep.errors


def test_input_size_mismatch_rejected(artifacts):
    folder = artifacts / "tb"
    make_onnx(folder / "model.onnx", size=300)
    (folder / "metadata.json").write_text(json.dumps(binary_metadata("tb")))
    rep = validate_model_dir(folder, "tb")
    assert not rep.ok and any("incompatible with preprocessing" in e for e in rep.errors)


def test_output_width_mismatch_rejected(artifacts):
    folder = install_model(artifacts, "tb", n_out=3)
    rep = validate_model_dir(folder, "tb")
    assert not rep.ok and any("outputs 3" in e for e in rep.errors)


def test_hash_mismatch_rejected(artifacts):
    folder = install_model(artifacts, "tb")
    md = binary_metadata("tb", sha256={"model.onnx": "0" * 64})
    (folder / "metadata.json").write_text(json.dumps(md))
    rep = validate_model_dir(folder, "tb")
    assert not rep.ok and any("SHA-256" in e for e in rep.errors)


# ── preprocessing ────────────────────────────────────────────────
def test_preprocess_shape_and_normalisation():
    x = preprocess(xray_like((500, 400)), default_spec("grayscale_to_rgb"))
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    assert np.allclose(x[0, 0] * 0.229 + 0.485, x[0, 1] * 0.224 + 0.456, atol=1e-5)   # grey → identical RGB


def test_preprocess_matches_torchvision_eval_transform():
    pytest.importorskip("torchvision")
    from src.data import get_transforms
    for img, mode in ((xray_like((517, 389)), "grayscale_to_rgb"), (histo_like((301, 333)), "rgb")):
        src = img.convert("L").convert("RGB") if mode == "grayscale_to_rgb" else img.convert("RGB")
        ref = get_transforms(train=False)(src).numpy()
        assert np.abs(ref - preprocess(img, default_spec(mode))[0]).max() < 1e-6


def test_preprocessing_spec_errors():
    bad = default_spec("rgb")
    bad["center_crop"] = 200
    with pytest.raises(PreprocessingError, match="input_size"):
        validate_spec(bad)


# ── thresholds, uncertainty, invalid outputs ─────────────────────
MD = binary_metadata("tb", threshold=0.4)


@pytest.mark.parametrize("score,status", [(0.9, ie.SCREENING_FLAGGED), (0.41, ie.INCONCLUSIVE),
                                          (0.37, ie.INCONCLUSIVE), (0.2, ie.NO_FLAG_DETECTED),
                                          (0.45, ie.SCREENING_FLAGGED)])
def test_threshold_and_uncertainty_band(score, status):
    assert ie.decide(score, MD)[0] == status


def test_softmax_and_probability_outputs():
    md = binary_metadata("tb", output={"type": "logit", "activation": "softmax"})
    assert ie.output_to_score(np.array([[0.0, 2.0]]), md) == pytest.approx(1 / (1 + np.exp(-2)))
    mdp = binary_metadata("tb", output={"type": "probability", "activation": "none"})
    assert ie.output_to_score(np.array([[0.7]]), mdp) == pytest.approx(0.7)


def test_temperature_calibration_applied():
    md = binary_metadata("tb", calibration={"method": "temperature", "temperature": 2.0})
    assert ie.output_to_score(np.array([[2.0]]), md) == pytest.approx(1 / (1 + np.exp(-1)))


@pytest.mark.parametrize("raw,md", [
    (np.array([[np.nan]]), MD),
    (np.array([[1.0, 2.0]]), MD),
    (np.array([[1.4]]), binary_metadata("tb", output={"type": "probability", "activation": "none"})),
])
def test_invalid_outputs_raise(raw, md):
    with pytest.raises(ie.InvalidOutput):
        ie.output_to_score(raw, md)


def test_invalid_model_output_gives_inconclusive(artifacts, monkeypatch):
    install_model(artifacts, "tb")
    reg = ModelRegistry(models_dir=artifacts)
    monkeypatch.setattr(reg.get("tb"), "run", lambda x: np.array([[np.inf]]))   # mocked runtime failure mode
    res = build_modules(reg)["tb"].predict(xray_like())
    assert res["prediction_status"] == ie.INCONCLUSIVE and res["reason_code"] == ie.MODEL_OUTPUT_INVALID


def test_inference_exception_gives_inconclusive(artifacts, monkeypatch):
    install_model(artifacts, "tb")
    reg = ModelRegistry(models_dir=artifacts)

    def boom(x):
        raise RuntimeError("ORT failure")
    monkeypatch.setattr(reg.get("tb"), "run", boom)
    assert build_modules(reg)["tb"].predict(xray_like())["reason_code"] == ie.INFERENCE_ERROR


def test_dr_out_of_distribution_raw_fundus(artifacts):
    install_model(artifacts, "dr", logit=logit_for(0.9),
                  metadata=binary_metadata("dr", input_domain="ben_graham_filtered"))
    reg = ModelRegistry(models_dir=artifacts)
    res = build_modules(reg)["dr"].predict(Image.new("RGB", (300, 300)), {"dark_corner_frac": 0.9})
    assert res["prediction_status"] == ie.INCONCLUSIVE and res["reason_code"] == ie.OUT_OF_DISTRIBUTION


def test_session_reused(artifacts):
    install_model(artifacts, "tb")
    h = ModelRegistry(models_dir=artifacts).get("tb")
    assert h.session() is h.session()


def test_simulation_is_labelled(artifacts, monkeypatch):
    import config
    monkeypatch.setattr(config, "SIMULATION_MODE", True)
    res = build_modules(ModelRegistry(models_dir=artifacts))["oral"].predict(histo_like(), {}, image_sha256="ab" * 32)
    assert res["simulated"] and res["model_version"] == "SIMULATION" and "SIMULATED" in res["score_kind"]
