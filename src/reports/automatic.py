"""Immutable report snapshots and automatically generated PDFs for web screenings."""
import json
import tempfile
from uuid import uuid4
from pathlib import Path

from src.core.pipeline import report_from_outcome, routing_text
from src.database.db import utcnow
from src.reports.pdf_generator import generate_report


def quality_summary(outcome):
    failed = bool(outcome.rejection and outcome.rejection.stage == 'quality')
    return {'status': 'FAILED' if failed else 'CHECK_WARNINGS' if outcome.quality_warnings else 'PASSED_BASIC_CHECKS' if outcome.quality_metrics else 'NOT_ASSESSED',
            'metrics': outcome.quality_metrics, 'warnings': outcome.quality_warnings,
            'note': 'Technical image checks only; passing does not establish diagnostic quality. Image quality is separate from disease score.'}


class AutomaticReports:
    def __init__(self, db):
        self.db = db
        with db.connect() as c:
            c.execute('''CREATE TABLE IF NOT EXISTS generated_reports (
                token TEXT PRIMARY KEY, session_id INTEGER UNIQUE NOT NULL
                REFERENCES screening_sessions(id) ON DELETE CASCADE,
                snapshot TEXT NOT NULL, pdf BLOB)''')

    def create(self, outcome, name=None):
        if outcome.stage == 'screened':
            report = report_from_outcome(outcome, 'en')
        else:
            report = {'patient_id': outcome.patient['id'], 'patient_ref': outcome.patient.get('reference'),
                      'status': 'REJECTED', 'disease': None, 'score': None, 'threshold': None,
                      'reason': outcome.rejection.explanation('en'),
                      'recommendations': [outcome.rejection.action('en')],
                      'routing': routing_text(outcome.modality), 'questionnaire_rules': None}
        quality = quality_summary(outcome)
        report.update(patient_name=name or 'Not provided', created_at=utcnow(),
                      patient_age=outcome.patient.get('age'), patient_sex=outcome.patient.get('sex'),
                      quality=quality['status'].replace('_', ' '), quality_metrics=quality['metrics'],
                      quality_warnings=quality['warnings'],
                      model_sha256=(outcome.result or {}).get('model_sha256'),
                      image_sha256=outcome.image_sha256)
        token = uuid4().hex
        with self.db.connect() as c:
            c.execute('INSERT INTO generated_reports(token,session_id,snapshot) VALUES (?,?,?)',
                      (token, outcome.session_id, json.dumps(report)))
        error = None
        try:
            self.render(token, outcome.explanation)
        except Exception:
            # Screening is already saved. Keep its successful response and permit a PDF retry.
            error = 'PDF generation failed. Use Download screening report to retry; the screening is saved.'
        return token, error

    def render(self, token, explanation=None):
        with self.db.connect() as c:
            row = c.execute('SELECT snapshot,pdf FROM generated_reports WHERE token=?', (token,)).fetchone()
        if row is None:
            return None
        if row['pdf'] is not None:
            return bytes(row['pdf'])
        with tempfile.TemporaryDirectory(prefix='mediscan_pdf_') as td:
            path = generate_report(json.loads(row['snapshot']), out_dir=Path(td), gradcam=explanation)
            data = path.read_bytes()
        with self.db.connect() as c:
            c.execute('UPDATE generated_reports SET pdf=? WHERE token=?', (data, token))
        return data
