"""Single-process bounded worker with durable status and idempotent submissions.

Uploads stay in memory. Interrupted jobs fail explicitly on restart and are never
silently replayed against a remote provider. Run one app process per database.
"""
import fcntl
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from src.database.db import utcnow


class QueueFull(Exception):
    pass


class JobConflict(Exception):
    pass


class ScreeningJobs:
    def __init__(self, db, worker, capacity=3):
        self.db, self.worker = db, worker
        self.capacity = threading.BoundedSemaphore(capacity)
        self.lock = threading.Lock()
        self.executor = None
        self.process_lock = None

    def start(self):
        self.process_lock = self.db.path.with_suffix('.jobs.lock').open('a')
        try:
            fcntl.flock(self.process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.process_lock.close()
            raise RuntimeError('Another screening worker owns this database. Use one app worker.') from None
        with self.db.connect() as c:
            c.execute("UPDATE screening_jobs SET status='failed', error=?, updated_at=? WHERE status IN ('queued','running')",
                      ('Processing interrupted by restart. Check patient history before submitting again.', utcnow()))
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='screening')

    def close(self):
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        if self.process_lock:
            self.process_lock.close()

    def get(self, job_id):
        with self.db.connect() as c:
            row = c.execute('SELECT * FROM screening_jobs WHERE id=?', (job_id,)).fetchone()
        if not row:
            return None
        return {'id': row['id'], 'status': row['status'], 'created_at': row['created_at'],
                'updated_at': row['updated_at'], 'error': row['error'],
                'result': json.loads(row['response_json']) if row['response_json'] else None}

    def submit(self, key, patient_id, data, filename, options):
        if not self.executor:
            raise QueueFull('Screening worker is not started.')
        fingerprint = hashlib.sha256(data + json.dumps([patient_id, filename, options], sort_keys=True).encode()).hexdigest()
        with self.lock:
            with self.db.connect() as c:
                row = c.execute('SELECT id,request_hash FROM screening_jobs WHERE idempotency_key=?', (key,)).fetchone()
                if row:
                    if row['request_hash'] != fingerprint:
                        raise JobConflict('Submission key already belongs to different input.')
                    return self.get(row['id'])
                if not self.capacity.acquire(blocking=False):
                    raise QueueFull('Screening queue is full. Please retry shortly.')
                job_id = uuid4().hex
                try:
                    c.execute('INSERT INTO screening_jobs(id,patient_id,idempotency_key,request_hash,status,created_at,updated_at) '
                              'VALUES (?,?,?,?,?,?,?)', (job_id, patient_id, key, fingerprint, 'queued', utcnow(), utcnow()))
                except Exception:
                    self.capacity.release()
                    raise
            self.executor.submit(self._run, job_id, data, filename, patient_id, options)
        return self.get(job_id)

    def cancel(self, job_id):
        with self.db.connect() as c:
            changed = c.execute("UPDATE screening_jobs SET status='cancelled',updated_at=? WHERE id=? AND status='queued'",
                                (utcnow(), job_id)).rowcount
        if not changed:
            raise JobConflict('Only queued jobs can be cancelled; running requests may already be processing remotely.')
        return self.get(job_id)

    def _run(self, job_id, data, filename, patient_id, options):
        try:
            with self.db.connect() as c:
                claimed = c.execute("UPDATE screening_jobs SET status='running',updated_at=? WHERE id=? AND status='queued'",
                                    (utcnow(), job_id)).rowcount
            if not claimed:
                return
            response = self.worker(data, filename, patient_id, **options)
            state = response.get('provider_status') or ('unsupported' if response['stage'] == 'rejected' else 'completed')
            if response['stage'] == 'rejected':
                state = 'ungradable' if response['quality']['status'] == 'FAILED' else 'failed' if response.get('rejection_code') == 'MODEL_UNAVAILABLE' else 'unsupported'
            with self.db.connect() as c:
                c.execute('UPDATE screening_jobs SET status=?,response_json=?,updated_at=? WHERE id=?',
                          (state, json.dumps(response, allow_nan=False), utcnow(), job_id))
        except Exception:
            with self.db.connect() as c:
                c.execute("UPDATE screening_jobs SET status='failed',error=?,updated_at=? WHERE id=?",
                          ('Processing failed. No disease decision is available; check patient history before retrying.', utcnow(), job_id))
        finally:
            self.capacity.release()
