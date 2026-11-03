"""
validators.py — small input validators shared by the UI and services.
"""
import re

AGE_RANGES = ["0-17", "18-29", "30-44", "45-59", "60-74", "75+"]
SEX_CHOICES = ["Not recorded", "Female", "Male", "Other"]

_REF_RE = re.compile(r"^[A-Za-z0-9._\-/]{1,40}$")


class ValidationError(ValueError):
    pass


def clean_reference(ref: str | None) -> str | None:
    """Optional clinic reference number: short, no spaces/PII-looking free text."""
    if ref is None:
        return None
    ref = ref.strip()
    if not ref:
        return None
    if not _REF_RE.match(ref):
        raise ValidationError("Reference may contain letters, digits and . _ - / only (max 40 chars). "
                              "Do not enter names or phone numbers.")
    return ref


def clean_age(age) -> int | None:
    if age in (None, ""):
        return None
    try:
        a = int(float(age))
    except (TypeError, ValueError):
        raise ValidationError("Age must be a number.")
    if not 0 <= a <= 120:
        raise ValidationError("Age must be between 0 and 120.")
    return a


def clean_age_range(r: str | None) -> str | None:
    if not r:
        return None
    if r not in AGE_RANGES:
        raise ValidationError(f"Age range must be one of {AGE_RANGES}.")
    return r


def age_to_range(age: int | None) -> str | None:
    if age is None:
        return None
    for r in AGE_RANGES[:-1]:
        lo, hi = (int(x) for x in r.split("-"))
        if lo <= age <= hi:
            return r
    return "75+"


def clean_sex(sex: str | None) -> str | None:
    if not sex or sex == "Not recorded":
        return None
    if sex not in SEX_CHOICES:
        raise ValidationError(f"Sex must be one of {SEX_CHOICES}.")
    return sex
