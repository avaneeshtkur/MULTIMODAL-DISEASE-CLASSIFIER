"""
tb.py — Tuberculosis screening on chest X-rays.

Preprocessing comes from metadata; the training pipeline (src/data.py)
converts X-rays grayscale → L → 3-channel RGB, so metadata should declare
color_mode = "grayscale_to_rgb" (the validator warns otherwise).
"""
from src.diseases.base import SpecialistModule


class TBModule(SpecialistModule):
    key = "tb"
    modality = "chest_xray"
    default_labels = ("Normal", "Tuberculosis")
