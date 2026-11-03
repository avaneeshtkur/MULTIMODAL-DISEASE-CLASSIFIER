"""Persistent demo cases. Reference labels never influence inference."""
from uuid import uuid4
from src.database.db import utcnow
from src.utils.validators import clean_age, clean_sex, ValidationError


class DemoService:
    def __init__(self, db):
        self.db = db
        with db.connect() as c:
            c.execute('''CREATE TABLE IF NOT EXISTS demo_cases (
                patient_id INTEGER PRIMARY KEY REFERENCES patients(id) ON DELETE CASCADE,
                name TEXT NOT NULL, details TEXT NOT NULL, module TEXT NOT NULL,
                expected_label TEXT NOT NULL, filename TEXT NOT NULL, image BLOB NOT NULL,
                source TEXT NOT NULL, seed_key TEXT UNIQUE)''')

    def create(self, name, details, module, expected_label, filename, image,
               age=None, sex=None, source='User supplied; label not independently verified', seed_key=None):
        name, details = name.strip(), details.strip()
        if not name or len(name) > 120 or len(details) > 2000:
            raise ValidationError('Enter a name (up to 120 characters) and details (up to 2000 characters).')
        if module not in {'tb', 'dr', 'oral'} or expected_label not in {'positive', 'negative', 'non-informative', 'unknown'}:
            raise ValidationError('Choose a supported disease and reference label.')
        age, sex = clean_age(age), clean_sex(sex)
        with self.db.connect() as c:
            if seed_key:
                row = c.execute('SELECT patient_id FROM demo_cases WHERE seed_key=?', (seed_key,)).fetchone()
                if row:
                    return row[0]
            pid = c.execute('INSERT INTO patients(reference,age,sex,is_demo,created_at) VALUES (?,?,?,1,?)',
                            ('DEMO-' + uuid4().hex[:12], age, sex, utcnow())).lastrowid
            c.execute('INSERT INTO demo_cases VALUES (?,?,?,?,?,?,?,?,?)',
                      (pid, name, details, module, expected_label, filename, image, source, seed_key))
        return pid

    def list(self):
        with self.db.connect() as c:
            return [dict(r) for r in c.execute('SELECT d.patient_id,d.name,d.details,d.module,d.expected_label,d.filename,d.source,p.age,p.sex FROM demo_cases d JOIN patients p ON p.id=d.patient_id ORDER BY d.patient_id')]

    def scan(self, pid):
        with self.db.connect() as c:
            row = c.execute('SELECT image FROM demo_cases WHERE patient_id=?', (pid,)).fetchone()
        return bytes(row[0]) if row else None
