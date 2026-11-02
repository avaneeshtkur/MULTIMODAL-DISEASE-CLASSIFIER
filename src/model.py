"""
model.py — EfficientNet-B0 binary classifier, reused for all 3 diseases.

Why EfficientNet-B0?
  - Pretrained on ImageNet (1.2M natural images) → already knows edges, textures, shapes.
  - Small enough to run on a clinic laptop CPU in ~12ms per image.
  - Accurate enough to compete with radiologists on many tasks in published papers.

The same architecture is used for TB, DR, and Oral — only the saved weights differ.
"""
import torch
import torch.nn as nn
from torchvision import models

import config


def build_model() -> nn.Module:
    """
    Create a fresh EfficientNet-B0 with a binary classification head.

    The head replaces ImageNet's 1000-class output with:
      Dropout(0.3) → Linear(1280 → 1)

    Dropout randomly zeroes 30% of neurons during training, which forces the
    network to not rely on any single feature — crucial with only 150 images
    per class to prevent memorising the training set.

    Output is a single raw logit (not a probability). We apply sigmoid later
    to get the 0-1 probability.
    """
    backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    backbone.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1280, 1),
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
    dcfg = config.DISEASES[disease]
    ckpt = dcfg["checkpoint"]
    if not ckpt.exists():
        raise FileNotFoundError(f"No trained model for {disease} at {ckpt}. Train it first.")
    model = build_model()
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval().to(device)
    return model
