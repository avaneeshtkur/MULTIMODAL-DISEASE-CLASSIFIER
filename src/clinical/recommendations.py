"""
recommendations.py — next-step text for an image-model screening status.

Pure lookup: (disease, status) → ordered list of translated recommendation
strings. Never prescribes medicines or treatment.
"""
from __future__ import annotations

from src.clinical.translations import has, t


def recommendation_keys(disease: str, status: str) -> list[str]:
    prefix = f"rec.{disease}.{status}." if status in ("SCREENING_FLAGGED", "NO_FLAG_DETECTED") \
        else f"rec.any.{status}."
    keys = []
    i = 1
    while has(f"{prefix}{i}", "en"):
        keys.append(f"{prefix}{i}")
        i += 1
    return keys + ["rec.common.no_treatment"]


def recommendations(disease: str, status: str, lang: str = "en") -> list[str]:
    return [t(k, lang) for k in recommendation_keys(disease, status)]
