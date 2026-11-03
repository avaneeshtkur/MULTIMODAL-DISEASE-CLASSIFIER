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
        assert client.post('/api/patients/demo', headers={'origin': 'http://testserver'}, data={'name': 'Demo TB', 'module': 'tb'}, files={'image': ('scan.png', png_bytes(xray_like()), 'image/png')}).status_code == 200
        assert client.get('/api/patients').json()[0]['is_demo'] is True


def test_screening_validation_and_duplicate_contract(make_ctx, artifacts):
    install_model(artifacts, 'tb', logit=2)
    ctx = make_ctx()
    with TestClient(create_web_app(ctx)) as client:
        pid = client.post('/api/patients/demo', data={'name': 'Demo TB', 'module': 'tb'}, files={'image': ('scan.png', png_bytes(xray_like()), 'image/png')}).json()['id']
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


def test_demo_persistence_and_validation(make_ctx):
    ctx = make_ctx()
    scan = png_bytes(xray_like())
    with TestClient(create_web_app(ctx)) as client:
        assert client.post('/api/patients/demo').status_code == 422
        data = {'name': 'Demo TB positive', 'details': 'Type 2 diabetes (fictional)',
                'module': 'tb', 'expected_label': 'positive', 'age': '52', 'sex': 'Female'}
        response = client.post('/api/patients/demo', data=data, files={'image': ('scan.png', scan)})
        assert response.status_code == 200
        pid = response.json()['id']
        assert client.get(f'/api/demos/{pid}/scan').content == scan
        assert client.get('/api/demos').json()[0]['expected_label'] == 'positive'
        assert ctx.patients.get(pid)['age'] == 52
        assert not ctx.patients.history(pid)  # Saving a reference label never fabricates a prediction.
        invalid = client.post('/api/patients/demo', data={**data, 'module': 'ct'}, files={'image': ('scan.png', scan)})
        assert invalid.status_code == 422
        assert len(client.get('/api/demos').json()) == 1
    with TestClient(create_web_app(ctx)) as client:
        assert client.get('/api/demos').json()[0]['name'] == data['name']
        assert client.get(f'/api/demos/{pid}/scan').content == scan


def test_reference_label_does_not_override_specialist(make_ctx, artifacts):
    install_model(artifacts, 'tb', logit=-4)
    ctx = make_ctx()
    scan = png_bytes(xray_like())
    with TestClient(create_web_app(ctx)) as client:
        pid = client.post('/api/patients/demo', data={'name': 'Positive reference', 'module': 'tb', 'expected_label': 'positive'},
                          files={'image': ('scan.png', scan)}).json()['id']
        body = client.post('/api/screen', data={'patient_id': pid, 'module': 'tb', 'explain': 'false'},
                           files={'image': ('scan.png', scan)}).json()
        assert body['status'] == 'NO_FLAG_DETECTED'
        assert body['metrics']['model_score'] < 0.1
        assert body['metrics']['model_sha256']
        assert '%' in body['html']


def test_automatic_report_name_persistence_and_quality(make_ctx, artifacts):
    import io
    from pypdf import PdfReader
    from PIL import Image
    install_model(artifacts, 'tb', logit=2)
    ctx = make_ctx()
    scan = png_bytes(xray_like())
    with TestClient(create_web_app(ctx)) as client:
        pid = client.post('/api/patients/demo', data={'name': 'Demo Named Patient', 'module': 'tb'},
                          files={'image': ('scan.png', scan)}).json()['id']
        body = client.post('/api/screen', data={'patient_id': pid, 'module': 'tb', 'explain': 'false', 'patient_name': 'Wrong name'},
                           files={'image': ('scan.png', scan)}).json()
        assert body['report_error'] is None
        assert body['quality']['status'] == 'PASSED_BASIC_CHECKS'
        pdf = client.get(body['report_url'])
        assert pdf.headers['content-type'] == 'application/pdf'
        assert pdf.content.startswith(b'%PDF')
        text = '\n'.join(p.extract_text() for p in PdfReader(io.BytesIO(pdf.content)).pages)
        assert 'Demo Named Patient' in text and 'Wrong name' not in text
        assert '88.08%' in text and 'Image quality measurements' in text
        assert client.get('/api/reports/' + '0'*32).status_code == 404
        dark = client.post('/api/screen', data={'patient_id': pid, 'module': 'tb', 'explain': 'false'},
                           files={'image': ('dark.png', png_bytes(Image.new('RGB', (320,320), (1,1,1))))}).json()
        assert dark['stage'] == 'rejected' and dark['metrics']['model_score'] is None
        assert dark['quality']['status'] == 'FAILED'
        assert client.get(dark['report_url']).content.startswith(b'%PDF')
    with TestClient(create_web_app(ctx)) as client:
        assert client.get(body['report_url']).content == pdf.content


def test_pdf_failure_keeps_screening_and_allows_retry(make_ctx, artifacts, monkeypatch):
    import src.reports.automatic as automatic
    install_model(artifacts, 'tb', logit=2)
    ctx = make_ctx()
    pid = ctx.patients.register('REPORT-TEST')
    render = automatic.generate_report
    def fail(*args, **kwargs):
        raise RuntimeError('Renderer unavailable')
    monkeypatch.setattr(automatic, 'generate_report', fail)
    with TestClient(create_web_app(ctx)) as client:
        body = client.post('/api/screen', data={'patient_id': pid, 'module': 'tb', 'explain': 'false'},
                           files={'image': ('scan.png', png_bytes(xray_like()))}).json()
        assert body['stage'] == 'screened' and body['result_id'] and body['report_error']
        assert client.get(body['report_url']).status_code == 503
        monkeypatch.setattr(automatic, 'generate_report', render)
        assert client.get(body['report_url']).content.startswith(b'%PDF')
