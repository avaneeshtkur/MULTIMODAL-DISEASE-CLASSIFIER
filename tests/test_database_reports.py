"""SQLite patient management, history, analytics and PDF reports."""
import pytest

from src.database.analytics_service import AnalyticsService
from src.database.db import Database
from src.database.patient_service import PatientService
from src.database.screening_service import ScreeningService
from src.gatekeeper import rejection_manager as rm
from src.reports.pdf_generator import generate_report
from src.utils.validators import ValidationError


@pytest.fixture
def svc(tmp_path):
    db = Database(tmp_path / "t.db")
    return db, PatientService(db), ScreeningService(db), AnalyticsService(db)


RESULT = {"disease": "tb", "prediction_status": "SCREENING_FLAGGED", "model_score": 0.8, "threshold": 0.4,
          "model_version": "v", "processing_time_ms": 12.0, "simulated": False}


def test_register_and_search(svc):
    _, p, _, _ = svc
    pid = p.register("PHC-001", age=33, sex="Female")
    assert p.get(pid)["age_range"] == "30-44"
    assert [r["id"] for r in p.search("PHC")] == [pid]
    assert [r["id"] for r in p.search(f"#{pid}")] == [pid]
    with pytest.raises(ValidationError):
        p.register("PHC-001")                     # duplicate reference
    with pytest.raises(ValidationError):
        p.register("Ram Kumar 9876543210")         # names/phones not allowed in reference
    with pytest.raises(ValidationError):
        p.register(age=200)


def test_sql_injection_is_inert(svc):
    _, p, _, _ = svc
    p.register("A-1")
    assert p.search("' OR 1=1 --") == []


def test_history_and_details(svc):
    _, p, s, _ = svc
    pid = p.register("H-1")
    sid = s.create_session(pid, "tb", "en", "tb", {"tb": {"cough_2w": True}})
    rid = s.save_result(sid, pid, RESULT, "a" * 64, {"blur": 1}, "manual_selection", None, "rec", "PRIORITY", False)
    h = p.history(pid)
    assert len(h) == 1 and h[0]["screening_status"] == "SCREENING_FLAGGED" and len(h[0]["image_hash"]) == 12
    d = s.get_result(rid)
    assert d["questionnaire"] == {"tb": {"cough_2w": True}} and d["reference"] == "H-1"
    assert s.find_by_hash("a" * 64)[0]["id"] == rid
    csv = p.export_history_csv(pid)
    assert "SCREENING_FLAGGED" in csv and "a" * 64 not in csv


def test_delete_only_demo_with_confirmation(svc):
    db, p, s, _ = svc
    real = p.register("REAL-1")
    demo = p.register("DEMO-1", is_demo=True)
    s.save_result(s.create_session(demo, "tb"), demo, RESULT, "b" * 64, {}, None, None, "", "ROUTINE", False, True)
    with pytest.raises(ValidationError):
        p.delete_demo_patient(real, "REAL-1")
    with pytest.raises(ValidationError):
        p.delete_demo_patient(demo, "wrong")
    p.delete_demo_patient(demo, "DEMO-1")
    assert p.get(demo) is None and s.find_by_hash("b" * 64) == []
    with db.connect() as c:
        assert c.execute("SELECT COUNT(*) FROM audit_logs WHERE event_type='patient_deleted'").fetchone()[0] == 1


def test_analytics_from_stored_records(svc):
    _, p, s, a = svc
    pid = p.register("X")
    demo = p.register("D", is_demo=True)
    sid = s.create_session(pid, "auto")
    s.save_result(sid, pid, RESULT, "c" * 64, {}, None, None, "", "PRIORITY", False)
    s.save_result(sid, pid, {**RESULT, "disease": "dr", "prediction_status": "INCONCLUSIVE"}, "d" * 64, {}, None,
                  None, "", "ROUTINE", False)
    s.save_result(sid, pid, {**RESULT, "simulated": True}, "e" * 64, {}, None, None, "", "ROUTINE", False)
    for code in (rm.IMAGE_TOO_BLURRY, rm.IMAGE_TOO_BLURRY, rm.UNSUPPORTED_MODALITY, rm.IMAGE_CORRUPTED):
        s.save_rejection(sid, pid, rm.Rejection(code, rm.STAGE_QUALITY), "auto", None)
    s.save_rejection(None, demo, rm.Rejection(rm.IMAGE_CORRUPTED, rm.STAGE_QUALITY), "auto", None, is_demo=True)

    summ = a.summary(include_demo=False)
    assert summ["results"] == 2 and summ["simulated_results"] == 1      # simulated excluded by default
    assert summ["count_tb"] == 1 and summ["count_dr"] == 1 and summ["INCONCLUSIVE"] == 1
    br = {r["code"]: r["pct"] for r in a.rejection_breakdown(include_demo=False)}
    assert br == {"IMAGE_TOO_BLURRY": 50.0, "UNSUPPORTED_MODALITY": 25.0, "IMAGE_CORRUPTED": 25.0}
    assert a.summary(include_demo=True)["rejections"] == 5
    assert a.daily_trend()
    # aggregates never include identifiers
    assert "X" not in str(a.status_by_disease()) and "c" * 12 not in str(a.daily_trend())


REPORT = {"result_id": 3, "patient_id": 1, "patient_ref": "PHC-9", "disease": "tb", "status": "SCREENING_FLAGGED",
          "score": 0.81, "threshold": 0.4, "uncertainty_band": [0.35, 0.45], "score_kind": "uncalibrated model score",
          "model_version": "tb-v1", "routing": "Chest X-ray", "recommendations": ["Refer."],
          "questionnaire_rules": [{"rule_id": "TB-Q1", "text": "Cough"}], "referral_label": "Priority referral",
          "processing_time_ms": 20.0}


@pytest.mark.parametrize("lang", ["en", "hi"])
def test_pdf_generation(tmp_path, lang):
    import numpy as np
    img = (np.random.default_rng(0).random((224, 224, 3)) * 255).astype(np.uint8)
    path = generate_report(REPORT, lang, tmp_path, gradcam={"ok": True, "base": img, "overlay": img})
    data = path.read_bytes()
    assert data.startswith(b"%PDF") and len(data) > 10_000


def test_pdf_simulated_and_without_gradcam(tmp_path):
    path = generate_report({**REPORT, "simulated": True, "status": "INCONCLUSIVE", "score": None}, "en", tmp_path,
                           gradcam={"message": "not retained"})
    assert path.exists()
