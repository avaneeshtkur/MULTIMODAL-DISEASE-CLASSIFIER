"""
model.py — EfficientNet-B0 binary classifier, reused for all 3 diseases.

Why EfficientNet-B0?
  - Pretrained on ImageNet (1.2M natural images) → already knows edges, textures, shapes.
  - Small enough (~16 MB as ONNX) to run on a clinic laptop CPU; measure latency
    with scripts/benchmark_cpu.py rather than assuming a number.

The same architecture is used for TB, DR, and Oral — only the saved weights differ.
"""
import torch
import torch.nn as nn
from torchvision import models

import config


def build_model(pretrained: bool = True, num_outputs: int = 1) -> nn.Module:
    """
    Create a fresh EfficientNet-B0 with a binary classification head.

    pretrained=False skips the ImageNet download — use it when the weights will be
    overwritten by a trained checkpoint (inference, Grad-CAM, offline use).
    num_outputs > 1 is used by the multi-class gatekeeper.

    The head replaces ImageNet's 1000-class output with:
      Dropout(0.3) → Linear(1280 → 1)

    Dropout randomly zeroes 30% of neurons during training, which forces the
    network to not rely on any single feature — crucial with only 150 images
    per class to prevent memorising the training set.

    Output is a single raw logit (not a probability). We apply sigmoid later
    to get the 0-1 probability.
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = models.efficientnet_b0(weights=weights)
    backbone.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1280, num_outputs),
    )
    return backbone


def gradcam_layer(model: nn.Module):
    """
    Return the layer we hook into for Grad-CAM heatmaps.

    model.features[-1] is the last convolutional block — it has the richest
    spatial information about WHAT the model is looking at and WHERE.
    """
    return model.features[-1]


def load_trained(device: str = "cpu", disease: str = "tb") -> nn.Module:
    """Load a trained checkpoint for a specific disease."""
    dcfg = config.task_config(disease)
    ckpt = dcfg["checkpoint"]
    if not ckpt.exists():
        raise FileNotFoundError(f"No trained model for {disease} at {ckpt}. Train it first.")
    n_out = 1 if len(dcfg["classes"]) == 2 else len(dcfg["classes"])
    model = build_model(pretrained=False, num_outputs=n_out)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval().to(device)
    return model
