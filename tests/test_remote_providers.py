"""Contract and integration tests use synthetic weights, never clinical claims."""
import hashlib
import json
import threading
import time

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import make_onnx, png_bytes, xray_like
from services.research.model import ResearchModel
from services.research.app import create_app
from src.core.image_preprocessor import default_spec
from src.core.jobs import ScreeningJobs, JobConflict, QueueFull
from src.core.pipeline import build_context, run_screening
from src.providers.remote_research import RemoteResearchProvider, validate_response
from src.ui.web import create_web_app


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / 'bundle'
    root.mkdir()
    make_onnx(root / 'model.onnx', n_out=2, bias=[4, -4])
    fixtures = []
    for i in range(2):
        im = xray_like()
        if i:
            im = im.transpose(0)
        data = png_bytes(im)
        (root / f'fixture{i}.png').write_bytes(data)
        fixtures.append(dict(file=f'fixture{i}.png', sha256=hashlib.sha256(data).hexdigest(), raw_output=[4, -4]))
    spec = dict(schema_version='1.0', disease='tb', runtime='onnx', model_file='model.onnx',
                files={'model.onnx': hashlib.sha256((root / 'model.onnx').read_bytes()).hexdigest()},
                model_version='synthetic-test-v1', source='synthetic fixture', license='test-only',
                reference_revision='test-v1', target_definition='test TB output', positive_class='Tuberculosis',
                class_labels=['Normal', 'Tuberculosis'], positive_indices=[1], output_activation='softmax',
                preprocessing=default_spec('grayscale_to_rgb'),
                decision_policy={'threshold': 0.5, 'uncertainty_margin': 0.05, 'validation_source': 'synthetic test'},
                reference_fixtures=fixtures)
    (root / 'manifest.json').write_text(json.dumps(spec))
    return root


def config_for(model):
    return dict(kind='remote_research', url='https://research.invalid', token_env='TEST_REMOTE_TOKEN', **model.identity)


def test_direct_local_bundle_without_network_or_tokens(tmp_path, bundle, monkeypatch):
    def forbid_network(*args, **kwargs):
        raise AssertionError('Local inference must not use HTTP')
    monkeypatch.setattr(httpx.Client, 'send', forbid_network)
    p = tmp_path / 'local.json'
    p.write_text(json.dumps({'tb': {'kind': 'local_research', 'bundle': 'bundle'}}))
    ctx = build_context(tmp_path / 'local.db', tmp_path / 'empty', p)
    assert 'tb' not in ctx.registry.keys
    pid = ctx.patients.register('DIRECT-LOCAL')
    out = run_screening(ctx, png_bytes(xray_like()), 'scan.png', pid, module='tb', explain=False)
    assert out.result['provider'] == 'local_research'
    assert out.result['provider_status'] == 'completed'
    assert out.result['model_score'] < .001
    assert ctx.screenings.get_result(out.result_id)['provenance']['raw_output'] == [4, -4]
    assert ctx.modules['tb'].status_row()['status'] == 'configured'


def test_direct_local_rejects_mismatched_or_changed_bundle(bundle):
    from src.providers.local_research import LocalResearchProvider
    wrong = LocalResearchProvider('dr', {'bundle': str(bundle)})
    assert wrong.predict(xray_like())['model_score'] is None
    p = bundle / 'manifest.json'
    spec = json.loads(p.read_text())
    spec['files']['model.onnx'] = '0' * 64
    p.write_text(json.dumps(spec))
    result = LocalResearchProvider('tb', {'bundle': str(bundle)}).predict(xray_like())
    assert result['provider_status'] == 'failed'
    assert result['prediction_status'] == 'INCONCLUSIVE'
    assert result['model_score'] is None


def test_service_auth_quality_and_positive_mapping(bundle):
    app = create_app(bundle, 'a' * 32)
    with TestClient(app) as client:
        assert client.get('/health').status_code == 401
        auth = {'Authorization': 'Bearer ' + 'a' * 32}
        assert client.get('/health', headers=auth).json()['status'] == 'ready'
        data = {'disease': 'tb', 'request_id': 'a' * 32}
        scan = png_bytes(xray_like())
        response = client.post('/v1/predict', headers=auth, data=data, files={'image': ('x.png', scan)})
        body = response.json()
        assert body['model_score'] < 0.001  # normal's >99% confidence must NOT become TB score
        assert body['prediction_status'] == 'NO_FLAG_DETECTED'
        assert body['raw_output'] == [4, -4]
        assert validate_response(body, 'tb', app.state.model.identity, data['request_id'], hashlib.sha256(scan).hexdigest())
        bad = client.post('/v1/predict', headers=auth, data=data, files={'image': ('x.png', b'bad')}).json()
        assert bad['provider_status'] == 'ungradable' and bad['model_score'] is None
        assert client.post('/v1/predict', headers=auth, data={**data, 'disease': 'dr'}, files={'image': ('x.png', scan)}).status_code == 422


@pytest.mark.parametrize('change', ['hash', 'logits', 'missing_fixtures'])
def test_bundle_fails_closed(bundle, change):
    p = bundle / 'manifest.json'
    s = json.loads(p.read_text())
    if change == 'hash':
        s['files']['model.onnx'] = '0' * 64
    elif change == 'logits':
        s['reference_fixtures'][0]['raw_output'] = [-4, 4]
    else:
        s['reference_fixtures'] = []
    p.write_text(json.dumps(s))
    with pytest.raises(ValueError):
        ResearchModel(bundle)


def test_five_grade_target_uses_declared_positive_classes(bundle):
    p = bundle / 'manifest.json'
    s = json.loads(p.read_text())
    make_onnx(bundle / 'model.onnx', n_out=5, bias=[-3, -2, 1, 2, 3])
    s.update(disease='dr', class_labels=['0','1','2','3','4'], positive_indices=[2,3,4], positive_class='referable_DR')
    s['files']['model.onnx'] = hashlib.sha256((bundle / 'model.onnx').read_bytes()).hexdigest()
    for fixture in s['reference_fixtures']:
        fixture['raw_output'] = [-3, -2, 1, 2, 3]
    p.write_text(json.dumps(s))
    model = ResearchModel(bundle)
    r = model.predict(xray_like())
    logits = np.array([-3, -2, 1, 2, 3], dtype=float)
    expected = np.exp(logits)[2:].sum()/np.exp(logits).sum()
    assert r['model_score'] == pytest.approx(expected)


def test_schema_upgrade_preserves_existing_records(make_ctx):
    from src.database.db import Database
    ctx = make_ctx()
    pid = ctx.patients.register('KEEP-THIS')
    with ctx.db.connect() as c:
        c.execute("UPDATE meta SET value='1' WHERE key='schema_version'")
        c.execute('DROP TABLE result_provenance')
        c.execute('DROP TABLE screening_jobs')
    upgraded = Database(ctx.db.path)
    with upgraded.connect() as c:
        assert c.execute('SELECT reference FROM patients WHERE id=?', (pid,)).fetchone()['reference'] == 'KEEP-THIS'
        assert c.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()['value'] == '2'
        c.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
    with pytest.raises(RuntimeError, match='newer'):
        Database(ctx.db.path)


@pytest.mark.parametrize('bad', ['version', 'request', 'nan', 'decision', 'large', 'timeout', 'redirect'])
def test_remote_failure_never_becomes_negative(bundle, monkeypatch, bad):
    model = ResearchModel(bundle)
    monkeypatch.setenv('TEST_REMOTE_TOKEN', 'a' * 32)
    def handler(request):
        if bad == 'timeout':
            raise httpx.ReadTimeout('secret endpoint', request=request)
        if bad == 'redirect':
            return httpx.Response(307, headers={'Location': 'https://untrusted.invalid'})
        if bad == 'large':
            return httpx.Response(200, content=b' ' * 65537)
        # Extract request id without assuming the random client value.
        request_id = request.content.split(b'name="request_id"\r\n\r\n')[1].split(b'\r\n')[0].decode()
        scan = request.content.split(b'Content-Type: image/png\r\n\r\n')[1].rsplit(b'\r\n--', 1)[0]
        body = dict(schema_version='1.0', disease='tb', request_id=request_id,
                    input_sha256=hashlib.sha256(scan).hexdigest(), processing_time_ms=1, **model.predict(xray_like()))
        if bad == 'version': body['model_version'] = 'changed'
        if bad == 'request': body['request_id'] = 'another-scan'
        if bad == 'nan': body['model_score'] = float('nan')
        if bad == 'decision': body['prediction_status'] = 'SCREENING_FLAGGED'
        return httpx.Response(200, content=json.dumps(body).encode())
    provider = RemoteResearchProvider('tb', config_for(model), httpx.MockTransport(handler))
    r = provider.predict(xray_like())
    assert r['provider_status'] == 'failed' and r['model_score'] is None
    assert r['prediction_status'] == 'INCONCLUSIVE'
    assert 'secret' not in r['reason']


def test_remote_only_context_skips_local_specialist_loading(tmp_path, bundle, monkeypatch):
    model = ResearchModel(bundle)
    monkeypatch.setenv('TEST_REMOTE_TOKEN', 'a' * 32)
    conf = {key: {**config_for(model), 'target_definition': key} for key in ('tb', 'dr', 'oral')}
    p = tmp_path / 'providers.json'
    p.write_text(json.dumps(conf))
    ctx = build_context(tmp_path / 'app.db', tmp_path / 'empty', p)
    assert ctx.registry.keys == ['gatekeeper']
    assert all(m.kind == 'remote_research' for m in ctx.modules.values())


def test_end_to_end_remote_score_persisted_in_report(tmp_path, bundle, monkeypatch):
    model = ResearchModel(bundle)
    cfg = config_for(model)
    monkeypatch.setenv('TEST_REMOTE_TOKEN', 'a' * 32)
    p = tmp_path / 'providers.json'
    p.write_text(json.dumps({'tb': cfg}))
    ctx = build_context(tmp_path / 'app.db', tmp_path / 'empty', p)
    provider = ctx.modules['tb']
    remote = TestClient(create_app(bundle, 'a' * 32))
    # Real multipart → service decoder → ONNX → contract validator, in process.
    def forward(request):
        reply = remote.post('/v1/predict', content=request.content, headers=dict(request.headers))
        return httpx.Response(reply.status_code, content=reply.content)
    provider.transport = httpx.MockTransport(forward)
    pid = ctx.patients.register('REMOTE-TEST')
    with TestClient(create_web_app(ctx)) as client:
        r = client.post('/api/screen', data={'patient_id': pid, 'module': 'tb', 'explain': 'true'},
                        files={'image': ('x.png', png_bytes(xray_like()))}).json()
        assert r['status'] == 'NO_FLAG_DETECTED'
        assert r['metrics']['model_score'] == model.predict(xray_like())['model_score']
        assert r['metrics']['provider'] == 'remote_research' and not r['images']
        record = ctx.screenings.get_result(r['result_id'])
        assert record['provenance']['raw_output'] == [4, -4]
        assert record['provenance']['provider_request_id'] == r['metrics']['provider_request_id']
        assert client.get(r['report_url']).content.startswith(b'%PDF')


def test_job_idempotency_capacity_cancel_and_recovery(make_ctx):
    ctx = make_ctx()
    pid = ctx.patients.register('JOBS')
    entered, release = threading.Event(), threading.Event()
    calls = []
    def worker(*args, **kwargs):
        calls.append(1)
        entered.set()
        release.wait(5)
        return {'stage': 'screened', 'provider_status': 'completed'}
    jobs = ScreeningJobs(ctx.db, worker, capacity=2)
    jobs.start()
    try:
        first = jobs.submit('a'*32, pid, b'x', 'x.png', {})
        assert entered.wait(2)
        assert jobs.submit('a'*32, pid, b'x', 'x.png', {})['id'] == first['id']
        with pytest.raises(JobConflict): jobs.submit('a'*32, pid, b'y', 'x.png', {})
        second = jobs.submit('b'*32, pid, b'x', 'x.png', {})
        with pytest.raises(QueueFull): jobs.submit('c'*32, pid, b'x', 'x.png', {})
        assert jobs.cancel(second['id'])['status'] == 'cancelled'
        with pytest.raises(JobConflict): jobs.cancel(first['id'])
    finally:
        release.set()
        jobs.close()
    assert len(calls) == 1 and jobs.get(first['id'])['status'] == 'completed'
    with ctx.db.connect() as c:
        c.execute("UPDATE screening_jobs SET status='running',response_json=NULL WHERE id=?", (first['id'],))
    recovered = ScreeningJobs(ctx.db, worker)
    recovered.start()
    try:
        assert recovered.get(first['id'])['status'] == 'failed'
        assert 'restart' in recovered.get(first['id'])['error']
    finally:
        recovered.close()


def test_async_endpoint_survives_app_restart(make_ctx, artifacts):
    from conftest import install_model
    install_model(artifacts, 'tb', logit=2)
    ctx = make_ctx()
    pid = ctx.patients.register('ASYNC')
    with TestClient(create_web_app(ctx)) as client:
        data = {'patient_id': pid, 'module': 'tb', 'explain': 'false'}
        files = {'image': ('x.png', png_bytes(xray_like()))}
        headers = {'Idempotency-Key': 'async-integration-test'}
        response = client.post('/api/jobs', headers=headers, data=data, files=files)
        assert response.status_code == 202
        job = response.json()
        assert client.post('/api/jobs', headers=headers, data=data, files=files).json()['id'] == job['id']
        deadline = time.monotonic() + 10
        while job['status'] in ('queued', 'running') and time.monotonic() < deadline:
            time.sleep(0.02)
            job = client.get('/api/jobs/' + job['id']).json()
        assert job['status'] == 'completed'
        assert len(ctx.patients.history(pid)) == 1
        report_url = job['result']['report_url']
        assert client.get(report_url).content.startswith(b'%PDF')
    with TestClient(create_web_app(ctx)) as client:
        recovered = client.get('/api/jobs/' + job['id']).json()
        assert recovered == job
        assert client.get(report_url).content.startswith(b'%PDF')
