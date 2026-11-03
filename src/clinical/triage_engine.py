"""
triage_engine.py — combines image-model status and questionnaire rules into a
referral suggestion WITHOUT changing either.

Output keeps the two sources visibly separate:
    image:          status + recommendations from the image model
    questionnaire:  rule hits (with rule IDs) from the symptom answers
    referral:       the higher priority of the two, with every reason attributed

There is no combined disease probability.
"""
from __future__ import annotations

from src.clinical.recommendations import recommendations
from src.clinical.symptom_assessment import PRIORITY, ROUTINE, clean_answers, evaluate_rules
from src.clinical.translations import t

IMAGE_PRIORITY = {"SCREENING_FLAGGED": PRIORITY, "NO_FLAG_DETECTED": ROUTINE,
                  "INCONCLUSIVE": ROUTINE, "REJECTED": ROUTINE}


def triage(disease: str, image_status: str, answers: dict | None, lang: str = "en") -> dict:
    answers = clean_answers(disease, answers)
    hits = evaluate_rules(disease, answers)
    image_pri = IMAGE_PRIORITY.get(image_status, ROUTINE)
    q_pri = PRIORITY if any(h.priority == PRIORITY for h in hits) else ROUTINE
    overall = PRIORITY if PRIORITY in (image_pri, q_pri) else ROUTINE

    reasons = [f"{t('source.image', lang)}: {t('status.' + image_status, lang)}"]
    reasons += [f"{t('source.questionnaire', lang)} [{h.rule_id}]: {t(h.text_key, lang)}" for h in hits]
    return {
        "image": {"status": image_status, "priority": image_pri,
                  "recommendations": recommendations(disease, image_status, lang)},
        "questionnaire": {"answers": answers, "priority": q_pri if answers else None,
                          "rules": [{"rule_id": h.rule_id, "priority": h.priority, "text": t(h.text_key, lang)}
                                    for h in hits]},
        "referral": {"priority": overall, "label": t(f"priority.{overall}", lang), "reasons": reasons},
    }
