"""HTTP integration checks; synthetic models test behaviour, not medical accuracy."""
from fastapi.testclient import TestClient

from conftest import install_model, png_bytes, xray_like
from src.ui.web import create_web_app


def test_page_and_assets(make_ctx):
    with TestClient(create_web_app(make_ctx())) as client:
        page = client.get('/')
        assert page.status_code == 200
        assert 'Mediscan AI' in page.text and '{%' not in page.text
        for asset in ['css/style.css', 'js/app.js', 'images/architecture.png', 'images/architecture-edges.png']:
            assert client.get('/static/' + asset).status_code == 200
        assert client.get('/api/status').headers['cache-control'] == 'no-store'
        assert client.get('/api/samples/unknown').status_code == 404


def test_browser_cross_origin_writes_refused(make_ctx):
    ctx = make_ctx()
    with TestClient(create_web_app(ctx)) as client:
        assert client.post('/api/patients/demo', headers={'origin': 'https://elsewhere.example'}).status_code == 403
        assert not ctx.patients.search('')
        assert client.post('/api/patients/demo', headers={'origin': 'http://testserver'}).status_code == 200
        assert client.get('/api/patients').json()[0]['is_demo'] is True


def test_screening_validation_and_duplicate_contract(make_ctx, artifacts):
    install_model(artifacts, 'tb', logit=2)
    ctx = make_ctx()
    with TestClient(create_web_app(ctx)) as client:
        pid = client.post('/api/patients/demo').json()['id']
        data = {'patient_id': pid, 'module': 'tb', 'explain': 'false'}
        files = {'image': ('xray.png', png_bytes(xray_like()), 'image/png')}
        result = client.post('/api/screen', data=data, files=files)
        assert result.status_code == 200
        body = result.json()
        assert body['stage'] == 'screened' and body['result_id']
        assert body['metrics']['disease'] == 'tb' and not body['images']
        assert len(ctx.patients.history(pid)) == 1
        repeated = client.post('/api/screen', data=data, files=files).json()
        assert repeated['stage'] == 'rejected' and repeated['result_id'] is None
        assert not repeated['images'] and repeated['metrics']['model_score'] is None
        invalid = client.post('/api/screen', data={**data, 'module': 'canny'}, files=files)
        assert invalid.status_code == 422
        missing = client.post('/api/screen', data={**data, 'patient_id': 99999}, files=files)
        assert missing.status_code == 422
        corrupt = client.post('/api/screen', data=data, files={'image': ('broken.png', b'not an image')}).json()
        assert corrupt['stage'] == 'rejected'
