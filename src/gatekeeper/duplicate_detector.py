"""
duplicate_detector.py — exact duplicate detection by SHA-256.

A hash match means the *file bytes* are identical to a previously screened
file. It is NOT evidence that two images come from the same patient, and
visually similar images are not detected. Duplicates are never deleted.
"""
from __future__ import annotations

from dataclasses import dataclass

import config
from src.gatekeeper import rejection_manager as rm


@dataclass
class DuplicateInfo:
    is_duplicate: bool
    same_patient: bool = False
    previous_result_id: int | None = None
    previous_at: str | None = None


def check_duplicate(image_sha256: str, patient_id: int | None, screening_service) -> DuplicateInfo:
    prev = screening_service.find_by_hash(image_sha256)
    if not prev:
        return DuplicateInfo(False)
    same = [p for p in prev if patient_id is not None and p["patient_id"] == patient_id]
    ref = (same or prev)[0]
    return DuplicateInfo(True, bool(same), ref["id"], ref["created_at"])


def duplicate_rejection(info: DuplicateInfo, allow_rescreen: bool) -> rm.Rejection | None:
    """Return a rejection if policy says so; None means proceed (possibly with a warning)."""
    if not info.is_duplicate or allow_rescreen or config.DUPLICATE_POLICY == "warn":
        return None
    # Do not reveal another patient's record ID; only say that a match exists.
    details = {"when": info.previous_at or "",
               "where": f"#{info.previous_result_id}" if info.same_patient else "another record"}
    return rm.Rejection(rm.DUPLICATE_IMAGE, rm.STAGE_DUPLICATE, details)
