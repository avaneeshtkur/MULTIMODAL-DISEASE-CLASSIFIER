"""Grad-CAM success and failure handling. Uses a RANDOMLY INITIALISED EfficientNet-B0
(no medical meaning) exported to ONNX, to test the checkpoint/ONNX consistency logic."""
import json

import numpy as np
import pytest

from src.core.image_preprocessor import default_spec
from src.core.model_registry import ModelRegistry
from src.diseases import build_modules
from src.explainability.gradcam import explain
from tests.conftest import binary_metadata, install_model, xray_like


def test_no_checkpoint_message(artifacts):
    install_model(artifacts, "tb", metadata=binary_metadata("tb", architecture="efficientnet_b0"))
    res = explain(ModelRegistry(models_dir=artifacts).get("tb"), xray_like(), 0.5)
    assert not res["ok"] and "checkpoint" in res["message"]


def test_unsupported_architecture_message(artifacts):
    install_model(artifacts, "tb")    # architecture tiny_test
    (artifacts / "tb" / "best_model.pth").write_bytes(b"x")
    res = explain(ModelRegistry(models_dir=artifacts).get("tb"), xray_like(), 0.5)
    assert not res["ok"] and "architecture" in res["message"]


def test_corrupt_checkpoint_never_raises(artifacts):
    install_model(artifacts, "tb", metadata=binary_metadata("tb", architecture="efficientnet_b0"))
    (artifacts / "tb" / "best_model.pth").write_bytes(b"not a checkpoint")
    res = explain(ModelRegistry(models_dir=artifacts).get("tb"), xray_like(), 0.5)
    assert not res["ok"] and "could not be generated" in res["message"]


def test_explanation_failure_does_not_break_prediction(artifacts):
    install_model(artifacts, "tb", metadata=binary_metadata("tb", architecture="efficientnet_b0"))
    (artifacts / "tb" / "best_model.pth").write_bytes(b"garbage")
    mod = build_modules(ModelRegistry(models_dir=artifacts))["tb"]
    res = mod.predict(xray_like())
    assert res["model_score"] is not None
    assert mod.explain(xray_like(), res)["ok"] is False


@pytest.fixture(scope="module")
def effnet_files(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from src.model import build_model
    d = tmp_path_factory.mktemp("effnet")
    torch.manual_seed(0)
    m = build_model(pretrained=False).eval()
    torch.save(m.state_dict(), d / "a.pth")
    torch.onnx.export(m, torch.randn(1, 3, 224, 224), str(d / "a.onnx"), input_names=["image"],
                      output_names=["logit"], dynamic_axes={"image": {0: "b"}, "logit": {0: "b"}},
                      opset_version=17, dynamo=False)
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    sd["classifier.1.bias"] += 3.0                                        # checkpoint ≠ exported ONNX
    torch.save(sd, d / "b.pth")
    return d


def _install_effnet(artifacts, effnet_files, ckpt):
    folder = artifacts / "tb"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.onnx").write_bytes((effnet_files / "a.onnx").read_bytes())
    (folder / "best_model.pth").write_bytes((effnet_files / ckpt).read_bytes())
    md = binary_metadata("tb", architecture="efficientnet_b0", preprocessing=default_spec("grayscale_to_rgb"))
    (folder / "metadata.json").write_text(json.dumps(md))
    return build_modules(ModelRegistry(models_dir=artifacts))["tb"]


def test_gradcam_on_matching_checkpoint(artifacts, effnet_files):
    mod = _install_effnet(artifacts, effnet_files, "a.pth")
    img = xray_like()
    res = mod.predict(img)
    ex = mod.explain(img, res)
    assert ex["ok"], ex["message"]
    assert ex["overlay"].shape == (224, 224, 3) and ex["heatmap"].dtype == np.uint8
    assert abs(ex["torch_score"] - res["model_score"]) < 1e-3


def test_gradcam_withheld_when_checkpoint_does_not_match_onnx(artifacts, effnet_files):
    mod = _install_effnet(artifacts, effnet_files, "b.pth")
    img = xray_like()
    res = mod.predict(img)
    ex = mod.explain(img, res)
    assert not ex["ok"] and "withheld" in ex["message"]
