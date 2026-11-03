"""
gradcam.py — Grad-CAM for the EfficientNet-B0 models (PyTorch checkpoint required).

Grad-CAM highlights image regions that most increased the model's output for
the positive class. It shows where the model looked, NOT whether the model is
right, and must not be read as a lesion segmentation.

Safety checks before an explanation is shown:
  * torch + torchvision installed, checkpoint present and hash-valid
  * metadata architecture == "efficientnet_b0" (the layer we hook is known)
  * the checkpoint reproduces the ONNX score within GRADCAM_MAX_SCORE_DIFF —
    otherwise the heatmap would explain a different model, so it is withheld.

There is no ONNX-only fallback: gradient-based maps need the PyTorch graph.
"""
from __future__ import annotations

import threading

import numpy as np

import config
from src.utils.logging_config import get_logger

log = get_logger("gradcam")

SUPPORTED_ARCHITECTURES = {"efficientnet_b0"}
_model_cache: dict[tuple[str, str], object] = {}
_cache_lock = threading.Lock()


class GradCAM:
    """
    Usage (also used by src/evaluate.py):
        cam = GradCAM(model)
        score, heat = cam(x)          # x: (1,3,H,W) tensor; heat: (h,w) in [0,1]
    For multi-output models pass target_index; for 2-logit softmax models it is the positive index.
    """

    def __init__(self, model, layer=None, target_index: int = 0, activation: str = "sigmoid"):
        import torch  # noqa: F401
        from src.model import gradcam_layer
        self.model = model.eval()
        self.layer = layer if layer is not None else gradcam_layer(model)
        self.target_index = target_index
        self.activation = activation

    def __call__(self, x):
        import torch
        store = {}

        def fwd_hook(_m, _i, out):
            store["act"] = out
            out.register_hook(lambda g: store.__setitem__("grad", g))

        handle = self.layer.register_forward_hook(fwd_hook)
        try:
            x = x.clone().requires_grad_(False)
            with torch.enable_grad():
                out = self.model(x)
                target = out[0, self.target_index]
                self.model.zero_grad(set_to_none=True)
                target.backward()
        finally:
            handle.remove()

        act, grad = store["act"].detach()[0], store["grad"].detach()[0]   # C,h,w
        weights = grad.mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * act).sum(0)).cpu().numpy()
        cam = cam / cam.max() if cam.max() > 0 else cam
        logits = out.detach()[0].cpu().numpy()
        if self.activation == "softmax":
            e = np.exp(logits - logits.max())
            score = float((e / e.sum())[self.target_index])
        else:
            score = float(1 / (1 + np.exp(-logits[self.target_index])))
        return score, cam.astype(np.float32)


def overlay(rgb, heat, alpha: float = 0.4):
    from src.explainability.visualizer import overlay as _overlay
    return _overlay(rgb, heat, alpha)


def _load_state_dict(path, torch):
    obj = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    if not isinstance(obj, dict):
        raise ValueError("checkpoint is not a state_dict")
    return {k.removeprefix("module."): v for k, v in obj.items()}


def _torch_model(handle):
    from src.utils.image_hash import sha256_file
    # Key on both files: a replaced checkpoint must never reuse cached weights.
    key = (handle.key, handle.sha256 or "", sha256_file(handle.checkpoint_path))
    with _cache_lock:
        if key in _model_cache:
            return _model_cache[key]
        import torch
        from src.model import build_model
        md = handle.metadata
        n_out = len(md["class_labels"]) if (md.get("task") == "multiclass"
                                           or md["output"]["activation"] == "softmax") else 1
        model = build_model(pretrained=False, num_outputs=n_out)
        model.load_state_dict(_load_state_dict(handle.checkpoint_path, torch))
        model.eval()
        _model_cache[key] = model
        return model


def explain(handle, image, onnx_score: float | None = None) -> dict:
    """
    Returns {"ok": True, "heatmap": HxWx3 uint8, "overlay": HxWx3 uint8, "base": HxWx3 uint8,
             "torch_score": float, "message": str}  or  {"ok": False, "message": str}.
    Never raises.
    """
    try:
        md = handle.metadata
        if not handle.available:
            return {"ok": False, "message": "Explanation unavailable: model not installed."}
        if md.get("architecture") not in SUPPORTED_ARCHITECTURES:
            return {"ok": False, "message": f"Explanation unavailable: Grad-CAM not validated for "
                                            f"architecture {md.get('architecture')!r}."}
        if not handle.checkpoint_path:
            return {"ok": False, "message": "Explanation unavailable: no PyTorch checkpoint "
                                            "(best_model.pth) installed; ONNX-only Grad-CAM is not supported."}
        try:
            import torch
        except ImportError:
            return {"ok": False, "message": "Explanation unavailable: PyTorch is not installed."}

        from src.core.image_preprocessor import preprocess, prepare_pil
        from src.core.inference_engine import output_to_score
        from src.explainability.visualizer import heatmap_image, overlay as _overlay

        model = _torch_model(handle)
        x = torch.from_numpy(preprocess(image, md["preprocessing"]))
        softmax = md["output"]["activation"] == "softmax"
        target = md["class_labels"].index(md["positive_class"]) if softmax else 0
        _, heat = GradCAM(model, target_index=target, activation="softmax" if softmax else "sigmoid")(x)

        with torch.no_grad():
            torch_score = output_to_score(model(x).numpy(), md)
        if onnx_score is not None and abs(torch_score - onnx_score) > config.GRADCAM_MAX_SCORE_DIFF:
            log.warning("Grad-CAM withheld for %s: torch %.4f vs onnx %.4f", handle.key, torch_score, onnx_score)
            return {"ok": False, "message": "Explanation withheld: the PyTorch checkpoint does not reproduce "
                                            "the ONNX model's score, so a heatmap would describe a different model."}
        base = np.asarray(prepare_pil(image, md["preprocessing"]).convert("RGB"))
        size = (base.shape[1], base.shape[0])
        return {"ok": True, "heatmap": heatmap_image(heat, size), "overlay": _overlay(base, heat),
                "base": base, "torch_score": round(torch_score, 4),
                "message": "Grad-CAM: warmer colours mark regions that most increased the model's score. "
                           "It shows where the model looked, not whether the result is correct."}
    except Exception as e:  # noqa: BLE001 — explanation failure must never break screening
        log.exception("Grad-CAM failed for %s", getattr(handle, "key", "?"))
        return {"ok": False, "message": f"Explanation could not be generated ({type(e).__name__})."}
