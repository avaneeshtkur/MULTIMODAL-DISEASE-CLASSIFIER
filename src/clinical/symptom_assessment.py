"""
symptom_assessment.py — optional, disease-specific questionnaire.

Answers are evaluated with transparent, rule-ID'd referral rules. They are
stored separately from the image-model output and are never combined with
the model score into a "disease probability".

Rules are derived from public screening guidance (WHO TB symptom screen /
India NTEP presumptive-TB criteria; routine annual diabetic eye examination;
2-week persistent oral lesion red flag). They have NOT yet been reviewed by
a clinician for this deployment — see docs/CLINICAL_LIMITATIONS.md.
"""
from __future__ import annotations

from dataclasses import dataclass

ROUTINE = "ROUTINE"
PRIORITY = "PRIORITY"

# Question definitions: (id, kind, options) — labels come from translations "q.<disease>.<id>"
QUESTIONS = {
    "tb": [("cough_2w", "bool", None), ("fever", "bool", None), ("night_sweats", "bool", None),
           ("weight_loss", "bool", None), ("tb_contact", "bool", None)],
    "dr": [("diabetes_type", "choice", ["type1", "type2", "other"]), ("diabetes_years", "number", None),
           ("eye_complications", "bool", None), ("last_eye_exam", "choice", ["lt1y", "gt1y", "never"])],
    "oral": [("tobacco", "bool", None), ("gutka_paan", "bool", None), ("ulcer_2w", "bool", None),
             ("dysphagia", "bool", None)],
}


@dataclass
class RuleHit:
    rule_id: str
    priority: str

    @property
    def text_key(self) -> str:
        return f"rule.{self.rule_id}"


def clean_answers(disease: str, answers: dict | None) -> dict:
    """Keep only known question IDs with valid values; drop blanks."""
    out = {}
    for qid, kind, options in QUESTIONS.get(disease, []):
        v = (answers or {}).get(qid)
        if v in (None, "", []):
            continue
        if kind == "bool":
            out[qid] = bool(v)
        elif kind == "choice" and v in options:
            out[qid] = v
        elif kind == "number":
            try:
                n = float(v)
            except (TypeError, ValueError):
                continue
            if 0 <= n <= 100:
                out[qid] = n
    return out


def evaluate_rules(disease: str, answers: dict) -> list[RuleHit]:
    a = clean_answers(disease, answers)
    hits: list[RuleHit] = []
    if disease == "tb":
        if a.get("cough_2w"):
            hits.append(RuleHit("TB-Q1", PRIORITY))
        if sum(bool(a.get(k)) for k in ("fever", "night_sweats", "weight_loss")) >= 2:
            hits.append(RuleHit("TB-Q2", PRIORITY))
        if a.get("tb_contact"):
            hits.append(RuleHit("TB-Q3", ROUTINE))
    elif disease == "dr":
        if a.get("last_eye_exam") in ("gt1y", "never"):
            hits.append(RuleHit("DR-Q1", ROUTINE))
        if a.get("eye_complications"):
            hits.append(RuleHit("DR-Q2", PRIORITY))
        if a.get("diabetes_years", 0) >= 10:
            hits.append(RuleHit("DR-Q3", ROUTINE))
    elif disease == "oral":
        if a.get("ulcer_2w"):
            hits.append(RuleHit("ORAL-Q1", PRIORITY))
        if a.get("dysphagia"):
            hits.append(RuleHit("ORAL-Q2", PRIORITY))
        if a.get("tobacco") or a.get("gutka_paan"):
            hits.append(RuleHit("ORAL-Q3", ROUTINE))
    return hits
