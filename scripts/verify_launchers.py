"""Real HTTP/restart checks on disposable databases, leaving port 7860 alone."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=7861)
    parser.add_argument('--out', type=Path, default=Path('docs/launcher-audit.json'))
    args = parser.parse_args()
    if args.port == 7860:
        parser.error('Choose a test port, not the operational port 7860')
    root = Path(__file__).resolve().parents[1]
    import socket
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    records = []
    for entry in ['app.py', 'run_pretrained.py']:
        with tempfile.TemporaryDirectory(prefix='mediscan-launch-') as temp:
            env = dict(os.environ, MEDISCAN_DB=str(Path(temp)/'audit.db'),
                       MEDISCAN_LOG_DIR=str(Path(temp)/'logs'), MEDISCAN_PORT=str(args.port),
                       MEDISCAN_ORT_THREADS='2', MEDISCAN_RETAIN_IMAGES='0')
            env.pop('MEDISCAN_PROVIDER_CONFIG', None)
            env.pop('MEDISCAN_SIMULATION', None)
            process = None
            log = (Path(temp)/'server.log').open('w+')
            def start():
                nonlocal process
                begin = time.monotonic()
                process = subprocess.Popen([sys.executable, entry], cwd=root, env=env,
                                           stdout=log, stderr=log)
                while time.monotonic() - begin < 45:
                    if process.poll() is not None:
                        log.seek(0)
                        raise RuntimeError(log.read()[-6000:])
                    try:
                        reply = client.get('/api/status')
                        if reply.status_code == 200:
                            return round(time.monotonic()-begin, 3), reply.json()
                    except httpx.TransportError:
                        pass
                    time.sleep(.1)
                raise TimeoutError('Server startup exceeded 45 seconds')
            def stop():
                if process and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(5)
            try:
                with httpx.Client(base_url=f'http://127.0.0.1:{args.port}', timeout=30, trust_env=False) as client:
                    startup, state = start()
                    assert client.get('/').status_code == 200
                    scans = {'tb': root/'data/samples/sample_chest_xray.png',
                             'dr': root/('output/pretrained/dr/raw-fixtures/mild-0.jpg' if entry == 'run_pretrained.py' else 'data/samples/sample_fundus.png'),
                             'oral': root/'data/samples/sample_histopathology.jpg'}
                    checks = []
                    for disease, scan in scans.items():
                        create = client.post('/api/patients/demo', data={'name': 'Disposable audit '+disease, 'module': disease},
                                             files={'image': (scan.name, scan.read_bytes())})
                        create.raise_for_status()
                        pid = create.json()['id']
                        fields = {'patient_id': pid, 'module': disease, 'explain': 'false'}
                        files = {'image': (scan.name, scan.read_bytes())}
                        headers = {'Idempotency-Key': uuid.uuid4().hex}
                        begin = time.monotonic()
                        submit = client.post('/api/jobs', data=fields, files=files, headers=headers)
                        assert submit.status_code == 202
                        job = submit.json()
                        assert client.post('/api/jobs', data=fields, files=files, headers=headers).json()['id'] == job['id']
                        deadline = time.monotonic()+45
                        while job['status'] in ['queued', 'running'] and time.monotonic()<deadline:
                            time.sleep(.1)
                            job = client.get('/api/jobs/'+job['id']).json()
                        assert job['status'] == 'completed', job
                        body = job['result']
                        assert body['metrics']['provider'] == ('local_research' if entry == 'run_pretrained.py' else 'local_onnx')
                        pdf_start = time.monotonic()
                        pdf = client.get(body['report_url'])
                        assert pdf.content.startswith(b'%PDF')
                        checks.append({'disease': disease, 'score': body['metrics']['model_score'],
                                       'version': body['metrics']['model_version'],
                                       'screen_and_pdf_seconds': round(time.monotonic()-begin, 3),
                                       'cached_pdf_download_seconds': round(time.monotonic()-pdf_start, 3),
                                       'pdf_bytes': len(pdf.content)})
                    bad = client.post('/api/screen', data=fields, files={'image': ('bad.png', b'invalid')}).json()
                    assert bad['stage'] == 'rejected' and bad['metrics']['model_score'] is None
                    job_id, report_url = job['id'], body['report_url']
                    stop()
                    restart, _ = start()
                    assert client.get('/api/jobs/'+job_id).json()['result']['result_id'] == body['result_id']
                    assert client.get(report_url).content == pdf.content
                    import sqlite3
                    with sqlite3.connect(env['MEDISCAN_DB']) as db:
                        assert db.execute('SELECT COUNT(*) FROM screening_results').fetchone()[0] == 3
                        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
                    records.append({'entry': entry, 'startup_seconds': startup, 'restart_seconds': restart,
                                    'research_evaluation': state['research_evaluation'], 'cases': checks,
                                    'invalid_upload': 'rejected without score', 'restart_persistence': 'passed'})
                    print(entry, 'HTTP, inference, idempotency, PDFs and restart passed', flush=True)
            finally:
                stop()
                log.close()
    args.out.write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()
