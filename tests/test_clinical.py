"""Clinical decision support and translations."""
import pytest

from src.clinical.recommendations import recommendations
from src.clinical.symptom_assessment import PRIORITY, ROUTINE, clean_answers, evaluate_rules
from src.clinical.translations import EN, HI, missing_keys, t
from src.clinical.triage_engine import triage
from src.gatekeeper import rejection_manager as rm


def test_tb_flagged_recommends_naat_and_referral():
    recs = " ".join(recommendations("tb", "SCREENING_FLAGGED"))
    assert "NAAT" in recs and "CBNAAT" in recs and "clinician" in recs


def test_dr_flagged_recommends_eye_specialist_dilated_oct():
    recs = " ".join(recommendations("dr", "SCREENING_FLAGGED"))
    assert "ophthalmologist" in recs and "dilated" in recs and "OCT" in recs


def test_oral_flagged_recommends_pathologist_and_specialist():
    recs = " ".join(recommendations("oral", "SCREENING_FLAGGED"))
    assert "pathologist" in recs and "specialist" in recs and "biopsy" in recs.lower()


@pytest.mark.parametrize("disease", ["tb", "dr", "oral"])
def test_no_flag_never_claims_healthy(disease):
    recs = " ".join(recommendations(disease, "NO_FLAG_DETECTED")).lower()
    assert "does not rule out" in recs
    for word in ("healthy", "no disease", "negative for"):
        assert word not in recs


@pytest.mark.parametrize("status", ["SCREENING_FLAGGED", "NO_FLAG_DETECTED", "INCONCLUSIVE"])
def test_no_medicine_or_treatment_prescribed(status):
    for disease in ("tb", "dr", "oral"):
        recs = " ".join(recommendations(disease, status)).lower()
        assert "does not recommend medicines" in recs
        for drug in ("isoniazid", "rifampicin", "anti-vegf", "laser", "chemotherapy", "prescribe"):
            assert drug not in recs


def test_questionnaire_rules_and_separation():
    hits = evaluate_rules("tb", {"cough_2w": True, "fever": True, "night_sweats": True})
    assert {h.rule_id for h in hits} == {"TB-Q1", "TB-Q2"}
    out = triage("tb", "NO_FLAG_DETECTED", {"cough_2w": True}, "en")
    assert out["image"]["status"] == "NO_FLAG_DETECTED"          # model output untouched
    assert out["image"]["priority"] == ROUTINE
    assert out["questionnaire"]["priority"] == PRIORITY
    assert out["referral"]["priority"] == PRIORITY
    assert "probability" not in str(out).lower()


def test_questionnaire_cleaning_drops_unknown_and_invalid():
    assert clean_answers("dr", {"diabetes_years": "abc", "last_eye_exam": "yesterday", "hack": 1}) == {}
    assert clean_answers("dr", {"diabetes_years": 12, "last_eye_exam": "never"}) == \
        {"diabetes_years": 12.0, "last_eye_exam": "never"}


def test_dr_and_oral_rules():
    assert {h.rule_id for h in evaluate_rules("dr", {"diabetes_years": 15, "last_eye_exam": "gt1y",
                                                     "eye_complications": True})} == {"DR-Q1", "DR-Q2", "DR-Q3"}
    assert {h.rule_id for h in evaluate_rules("oral", {"gutka_paan": True, "dysphagia": True})} == {"ORAL-Q2", "ORAL-Q3"}


def test_hindi_dictionary_complete_and_placeholders_match():
    assert missing_keys("hi") == []
    import string
    fields = lambda s: {f for _, f, _, _ in string.Formatter().parse(s) if f}  # noqa: E731
    for k in EN:
        assert fields(EN[k]) == fields(HI[k]), k


def test_hindi_outputs():
    assert t("status.SCREENING_FLAGGED", "hi") != t("status.SCREENING_FLAGGED", "en")
    assert "डायबिटिक" in t("disease.dr", "hi")
    assert "चिकित्सा निदान नहीं" in t("disclaimer", "hi")
    r = rm.Rejection(rm.IMAGE_TOO_SMALL, rm.STAGE_QUALITY, {"width": 90, "height": 90, "min_side": 128})
    assert "90×90" in r.explanation("hi") and "पिक्सेल" in r.explanation("hi")
    m = rm.Rejection(rm.MODEL_UNAVAILABLE, rm.STAGE_MODEL, {"module": "dr"})
    assert t("disease.dr", "hi") in m.explanation("hi")


def test_disclaimer_text_exact():
    assert t("disclaimer") == ("This is an experimental AI screening aid, not a medical diagnosis. Results must be "
                               "reviewed by a qualified healthcare professional and confirmed through appropriate "
                               "clinical testing.")
