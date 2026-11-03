"""
patient_service.py — register, search, history, export and (demo-only) delete.
"""
from __future__ import annotations

import csv
import io

from src.database.db import Database, utcnow
from src.utils import validators as v


class PatientService:
    def __init__(self, db: Database):
        self.db = db

    def register(self, reference: str | None = None, age=None, age_range: str | None = None,
                 sex: str | None = None, is_demo: bool = False) -> int:
        ref = v.clean_reference(reference)
        age_i = v.clean_age(age)
        rng = v.clean_age_range(age_range) or v.age_to_range(age_i)
        sx = v.clean_sex(sex)
        with self.db.connect() as c:
            if ref and c.execute("SELECT 1 FROM patients WHERE reference = ?", (ref,)).fetchone():
                raise v.ValidationError(f"Reference {ref!r} already exists. Search for it instead.")
            cur = c.execute("INSERT INTO patients(reference, age, age_range, sex, is_demo, created_at) "
                            "VALUES (?,?,?,?,?,?)", (ref, age_i, rng, sx, int(is_demo), utcnow()))
            pid = cur.lastrowid
            self.db.audit("patient_registered", "patient", pid, {"is_demo": bool(is_demo)}, conn=c)
        return pid

    def get(self, patient_id: int) -> dict | None:
        with self.db.connect() as c:
            row = c.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
        return dict(row) if row else None

    def search(self, query: str = "", include_demo: bool = True, limit: int = 200) -> list[dict]:
        q = (query or "").strip()
        sql = ("SELECT p.*, (SELECT COUNT(*) FROM screening_results r WHERE r.patient_id = p.id) AS screenings, "
               "(SELECT MAX(created_at) FROM screening_results r WHERE r.patient_id = p.id) AS last_screening "
               "FROM patients p WHERE 1=1")
        args: list = []
        if q:
            if q.lstrip("#").isdigit():
                sql += " AND (p.id = ? OR p.reference LIKE ?)"
                args += [int(q.lstrip("#")), f"%{q}%"]
            else:
                sql += " AND p.reference LIKE ?"
                args.append(f"%{q}%")
        if not include_demo:
            sql += " AND p.is_demo = 0"
        sql += " ORDER BY p.id DESC LIMIT ?"
        args.append(limit)
        with self.db.connect() as c:
            return [dict(r) for r in c.execute(sql, args).fetchall()]

    def history(self, patient_id: int) -> list[dict]:
        with self.db.connect() as c:
            rows = c.execute(
                "SELECT r.id, r.created_at, r.disease, r.screening_status, r.model_score, r.threshold, "
                "r.model_version, r.quality_status, r.reason_code, r.referral_priority, r.simulated, "
                "substr(r.image_hash, 1, 12) AS image_hash "
                "FROM screening_results r WHERE r.patient_id = ? ORDER BY r.id DESC", (patient_id,)).fetchall()
        return [dict(r) for r in rows]

    def rejections(self, patient_id: int) -> list[dict]:
        with self.db.connect() as c:
            rows = c.execute("SELECT id, created_at, code, stage, requested_module FROM rejection_events "
                             "WHERE patient_id = ? ORDER BY id DESC", (patient_id,)).fetchall()
        return [dict(r) for r in rows]

    def delete_demo_patient(self, patient_id: int, confirmation: str) -> None:
        """Only demo patients can be deleted; confirmation must equal the patient ID or reference."""
        p = self.get(patient_id)
        if not p:
            raise v.ValidationError("Patient not found.")
        if not p["is_demo"]:
            raise v.ValidationError("Only demo patient records can be deleted from the app.")
        if (confirmation or "").strip() not in {str(p["id"]), f"#{p['id']}", p["reference"] or "\0"}:
            raise v.ValidationError("Confirmation does not match. Type the patient ID or reference to confirm.")
        with self.db.connect() as c:
            n = c.execute("SELECT COUNT(*) FROM screening_results WHERE patient_id = ?", (patient_id,)).fetchone()[0]
            c.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
            self.db.audit("patient_deleted", "patient", patient_id, {"is_demo": True, "results_deleted": n}, conn=c)

    def export_history_csv(self, patient_id: int | None = None, include_demo: bool = True) -> str:
        sql = ("SELECT r.id AS result_id, r.patient_id, p.reference, r.created_at, r.disease, r.screening_status, "
               "r.model_score, r.threshold, r.model_version, r.quality_status, r.reason_code, r.referral_priority, "
               "r.recommendation, r.simulated, p.is_demo FROM screening_results r "
               "LEFT JOIN patients p ON p.id = r.patient_id WHERE 1=1")
        args: list = []
        if patient_id is not None:
            sql += " AND r.patient_id = ?"
            args.append(patient_id)
        if not include_demo:
            sql += " AND r.is_demo = 0"
        sql += " ORDER BY r.id"
        with self.db.connect() as c:
            rows = c.execute(sql, args).fetchall()
            self.db.audit("history_exported", "patient" if patient_id else "all", patient_id,
                          {"rows": len(rows)}, conn=c)
        buf = io.StringIO()
        w = csv.writer(buf)
        if rows:
            w.writerow(rows[0].keys())
            w.writerows([tuple(r) for r in rows])
        return buf.getvalue()
