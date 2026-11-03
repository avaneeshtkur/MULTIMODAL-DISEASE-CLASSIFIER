"""
screening_service.py — persist sessions, results and rejection events.
"""
from __future__ import annotations

import json

from src.database.db import Database, utcnow


class ScreeningService:
    def __init__(self, db: Database):
        self.db = db

    def create_session(self, patient_id: int | None, requested_module: str, language: str = "en",
                       questionnaire_disease: str | None = None, questionnaire: dict | None = None,
                       is_demo: bool = False) -> int:
        with self.db.connect() as c:
            cur = c.execute(
                "INSERT INTO screening_sessions(patient_id, requested_module, language, questionnaire_disease, "
                "questionnaire_json, is_demo, created_at) VALUES (?,?,?,?,?,?,?)",
                (patient_id, requested_module, language, questionnaire_disease,
                 json.dumps(questionnaire) if questionnaire else None, int(is_demo), utcnow()))
            return cur.lastrowid

    def save_result(self, session_id: int, patient_id: int | None, result: dict, image_hash: str,
                    quality_metrics: dict, routing_source: str | None, modality_confidence: float | None,
                    recommendation: str, referral_priority: str, gradcam_available: bool,
                    is_demo: bool = False) -> int:
        with self.db.connect() as c:
            cur = c.execute(
                "INSERT INTO screening_results(session_id, patient_id, disease, image_hash, model_version, "
                "model_sha256, model_score, threshold, screening_status, reason_code, quality_status, "
                "quality_metrics_json, routing_source, modality_confidence, recommendation, referral_priority, "
                "processing_time_ms, gradcam_available, simulated, is_demo, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (session_id, patient_id, result["disease"], image_hash, result.get("model_version"),
                 result.get("model_sha256"), result.get("model_score"), result.get("threshold"),
                 result["prediction_status"], result.get("reason_code"), "PASSED",
                 json.dumps(quality_metrics), routing_source, modality_confidence, recommendation,
                 referral_priority, result.get("processing_time_ms"), int(gradcam_available),
                 int(bool(result.get("simulated"))), int(is_demo), utcnow()))
            rid = cur.lastrowid
            provenance = {k: result.get(k) for k in ('provider', 'provider_status', 'provider_request_id',
                'target_definition', 'positive_class', 'preprocessing_version', 'decision_policy_version',
                'raw_output', 'raw_positive_score', 'precise_model_score', 'score_kind', 'uncertainty_band')}
            c.execute('INSERT INTO result_provenance(result_id,payload) VALUES (?,?)',
                      (rid, json.dumps(provenance, allow_nan=False)))
            self.db.audit("screening_completed", "screening_result", rid,
                          {"disease": result["disease"], "status": result["prediction_status"],
                           "simulated": bool(result.get("simulated"))}, conn=c)
        return rid

    def save_rejection(self, session_id: int | None, patient_id: int | None, rejection, requested_module: str,
                       image_hash: str | None, is_demo: bool = False) -> int:
        with self.db.connect() as c:
            cur = c.execute(
                "INSERT INTO rejection_events(session_id, patient_id, code, stage, requested_module, image_hash, "
                "details_json, is_demo, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (session_id, patient_id, rejection.code, rejection.stage, requested_module, image_hash,
                 json.dumps(rejection.details, default=str), int(is_demo), utcnow()))
            rid = cur.lastrowid
            self.db.audit("image_rejected", "rejection_event", rid, {"code": rejection.code}, conn=c)
        return rid

    def find_by_hash(self, image_hash: str) -> list[dict]:
        with self.db.connect() as c:
            rows = c.execute("SELECT id, patient_id, created_at FROM screening_results WHERE image_hash = ? "
                             "ORDER BY id DESC", (image_hash,)).fetchall()
        return [dict(r) for r in rows]

    def get_result(self, result_id: int) -> dict | None:
        with self.db.connect() as c:
            row = c.execute(
                "SELECT r.*, p.reference, p.age, p.age_range, p.sex, p.is_demo AS patient_is_demo, "
                "s.questionnaire_json, s.questionnaire_disease, s.language, s.requested_module "
                "FROM screening_results r LEFT JOIN patients p ON p.id = r.patient_id "
                "JOIN screening_sessions s ON s.id = r.session_id WHERE r.id = ?", (result_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["questionnaire"] = json.loads(d.pop("questionnaire_json") or "null") or {}
        d["quality_metrics"] = json.loads(d.pop("quality_metrics_json") or "null") or {}
        with self.db.connect() as c:
            provenance = c.execute('SELECT payload FROM result_provenance WHERE result_id=?', (result_id,)).fetchone()
        d['provenance'] = json.loads(provenance['payload']) if provenance else {}
        return d
