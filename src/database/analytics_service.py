"""
analytics_service.py — aggregate statistics computed from stored records.

Only counts and dates are returned — never references, ages or hashes —
so dashboards cannot expose identifiable patient information.
"""
from __future__ import annotations

from src.database.db import Database


class AnalyticsService:
    def __init__(self, db: Database):
        self.db = db

    @staticmethod
    def _demo_clause(include_demo: bool, alias: str = "") -> str:
        return "" if include_demo else f" AND {alias}is_demo = 0"

    def summary(self, include_demo: bool = True, include_simulated: bool = False) -> dict:
        d = self._demo_clause(include_demo)
        sim = "" if include_simulated else " AND simulated = 0"
        with self.db.connect() as c:
            one = lambda sql: c.execute(sql).fetchone()[0]  # noqa: E731 — SQL fragments are constants
            out = {
                "sessions": one(f"SELECT COUNT(*) FROM screening_sessions WHERE 1=1{d}"),
                "results": one(f"SELECT COUNT(*) FROM screening_results WHERE 1=1{d}{sim}"),
                "rejections": one(f"SELECT COUNT(*) FROM rejection_events WHERE 1=1{d}"),
                "patients": one(f"SELECT COUNT(*) FROM patients WHERE 1=1{d}"),
                "simulated_results": one(f"SELECT COUNT(*) FROM screening_results WHERE simulated = 1{d}"),
            }
            for k in ("tb", "dr", "oral"):
                out[f"count_{k}"] = c.execute(
                    f"SELECT COUNT(*) FROM screening_results WHERE disease = ?{d}{sim}", (k,)).fetchone()[0]
            for s in ("SCREENING_FLAGGED", "NO_FLAG_DETECTED", "INCONCLUSIVE"):
                out[s] = c.execute(
                    f"SELECT COUNT(*) FROM screening_results WHERE screening_status = ?{d}{sim}", (s,)).fetchone()[0]
        return out

    def status_by_disease(self, include_demo: bool = True, include_simulated: bool = False) -> list[dict]:
        d = self._demo_clause(include_demo)
        sim = "" if include_simulated else " AND simulated = 0"
        with self.db.connect() as c:
            rows = c.execute(f"SELECT disease, screening_status AS status, COUNT(*) AS n FROM screening_results "
                             f"WHERE 1=1{d}{sim} GROUP BY disease, screening_status").fetchall()
        return [dict(r) for r in rows]

    def rejection_breakdown(self, include_demo: bool = True) -> list[dict]:
        d = self._demo_clause(include_demo)
        with self.db.connect() as c:
            rows = c.execute(f"SELECT code, COUNT(*) AS n FROM rejection_events WHERE 1=1{d} "
                             f"GROUP BY code ORDER BY n DESC").fetchall()
        total = sum(r["n"] for r in rows)
        return [{"code": r["code"], "n": r["n"], "pct": round(100 * r["n"] / total, 1) if total else 0.0}
                for r in rows]

    def daily_trend(self, include_demo: bool = True, include_simulated: bool = False, days: int = 60) -> list[dict]:
        d = self._demo_clause(include_demo)
        sim = "" if include_simulated else " AND simulated = 0"
        with self.db.connect() as c:
            res = c.execute(
                f"SELECT substr(created_at, 1, 10) AS day, disease AS series, COUNT(*) AS n FROM screening_results "
                f"WHERE created_at >= date('now', ?){d}{sim} GROUP BY day, disease", (f"-{int(days)} days",)).fetchall()
            rej = c.execute(
                f"SELECT substr(created_at, 1, 10) AS day, 'rejected' AS series, COUNT(*) AS n FROM rejection_events "
                f"WHERE created_at >= date('now', ?){d} GROUP BY day", (f"-{int(days)} days",)).fetchall()
        return sorted([dict(r) for r in res] + [dict(r) for r in rej], key=lambda r: (r["day"], r["series"]))
