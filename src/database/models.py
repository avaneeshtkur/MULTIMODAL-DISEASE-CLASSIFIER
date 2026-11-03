"""
models.py — SQLite schema.

The patients table keeps a reference, age and sex. Demo scans and names, and
named PDF snapshots, are stored by their respective services. Ordinary uploads
are not retained by default. Jobs keep status/results but not uploaded bytes.

Questionnaire answers live on screening_sessions, separate from the
image-model output in screening_results.
"""
SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    reference     TEXT UNIQUE,
    age           INTEGER,
    age_range     TEXT,
    sex           TEXT,
    is_demo       INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS screening_sessions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id            INTEGER REFERENCES patients(id) ON DELETE CASCADE,
    requested_module      TEXT NOT NULL,            -- 'auto' | 'tb' | 'dr' | 'oral'
    language              TEXT NOT NULL DEFAULT 'en',
    questionnaire_disease TEXT,
    questionnaire_json    TEXT,                     -- answers only; never mixed with model output
    is_demo               INTEGER NOT NULL DEFAULT 0,
    created_at            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS screening_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          INTEGER NOT NULL REFERENCES screening_sessions(id) ON DELETE CASCADE,
    patient_id          INTEGER REFERENCES patients(id) ON DELETE CASCADE,
    disease             TEXT NOT NULL,
    image_hash          TEXT NOT NULL,
    model_version       TEXT,
    model_sha256        TEXT,
    model_score         REAL,
    threshold           REAL,
    screening_status    TEXT NOT NULL,
    reason_code         TEXT,
    quality_status      TEXT NOT NULL,
    quality_metrics_json TEXT,
    routing_source      TEXT,
    modality_confidence REAL,
    recommendation      TEXT,
    referral_priority   TEXT,
    processing_time_ms  REAL,
    gradcam_available   INTEGER NOT NULL DEFAULT 0,
    simulated           INTEGER NOT NULL DEFAULT 0,
    is_demo             INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_results_patient ON screening_results(patient_id);
CREATE INDEX IF NOT EXISTS idx_results_hash ON screening_results(image_hash);
CREATE INDEX IF NOT EXISTS idx_results_created ON screening_results(created_at);

CREATE TABLE IF NOT EXISTS rejection_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       INTEGER REFERENCES screening_sessions(id) ON DELETE CASCADE,
    patient_id       INTEGER REFERENCES patients(id) ON DELETE CASCADE,
    code             TEXT NOT NULL,
    stage            TEXT NOT NULL,
    requested_module TEXT,
    image_hash       TEXT,
    details_json     TEXT,
    is_demo          INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rejections_created ON rejection_events(created_at);

CREATE TABLE IF NOT EXISTS model_registry (
    model_key   TEXT PRIMARY KEY,
    version     TEXT,
    sha256      TEXT,
    status      TEXT NOT NULL,
    errors      TEXT,
    checked_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type   TEXT NOT NULL,
    entity_type  TEXT,
    entity_id    INTEGER,
    details_json TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS result_provenance (
    result_id INTEGER PRIMARY KEY REFERENCES screening_results(id) ON DELETE CASCADE,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS screening_jobs (
    id TEXT PRIMARY KEY,
    patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    idempotency_key TEXT UNIQUE NOT NULL,
    request_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    response_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""
