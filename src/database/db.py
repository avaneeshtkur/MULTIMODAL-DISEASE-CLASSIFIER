"""
db.py — SQLite connection handling.

One short-lived connection per operation (safe with Gradio's worker threads),
foreign keys enforced, WAL journal for concurrent reads. All queries in the
services use parameter binding — never string formatting of values.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import config
from src.database.models import SCHEMA, SCHEMA_VERSION


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class Database:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or config.DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init(self) -> None:
        with self.connect() as c:
            exists = c.execute("SELECT name FROM sqlite_master WHERE name='meta'").fetchone()
            version = c.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone() if exists else None
            if version and int(version['value']) > SCHEMA_VERSION:
                raise RuntimeError('Database schema is newer than this application. Refusing to downgrade.')
            c.execute("PRAGMA journal_mode = WAL")
            c.executescript(SCHEMA)
            c.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))

    def audit(self, event_type: str, entity_type: str | None = None, entity_id: int | None = None,
              details: dict | None = None, conn=None) -> None:
        sql = "INSERT INTO audit_logs(event_type, entity_type, entity_id, details_json, created_at) VALUES (?,?,?,?,?)"
        args = (event_type, entity_type, entity_id, json.dumps(details or {}), utcnow())
        if conn is not None:
            conn.execute(sql, args)
        else:
            with self.connect() as c:
                c.execute(sql, args)

    def sync_model_registry(self, rows: list[dict]) -> None:
        with self.connect() as c:
            for r in rows:
                c.execute("INSERT OR REPLACE INTO model_registry(model_key, version, sha256, status, errors, checked_at) "
                          "VALUES (?,?,?,?,?,?)",
                          (r["key"], r["version"], r["sha256"], r["status"], r["errors"], utcnow()))
