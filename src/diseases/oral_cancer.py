"""
oral_cancer.py — Oral squamous cell carcinoma (OSCC) screening on H&E
histopathology images. Colour is preserved (RGB); output is Normal vs OSCC.
"""
from src.diseases.base import SpecialistModule


class OralCancerModule(SpecialistModule):
    key = "oral"
    modality = "histopathology"
    default_labels = ("Normal", "OSCC")
