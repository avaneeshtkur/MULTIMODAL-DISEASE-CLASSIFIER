"""
gradcam.py — compatibility shim. src/evaluate.py imports GradCAM and overlay
from here; the implementation lives in src/explainability/gradcam.py.
"""
from src.explainability.gradcam import GradCAM, overlay  # noqa: F401
