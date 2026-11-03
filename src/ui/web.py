"""MediScan medical imaging studio backed by the existing screening pipeline."""
from __future__ import annotations

import base64
import io
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.concurrency import run_in_threadpool

import config
from src.core.pipeline import routing_text, run_screening
from src.ui.components import clinical_html, result_html

WEB_ROOT = Path(__file__).resolve().parents[2] / 'web'
SAMPLES = {'tb': 'sample_chest_xray.png', 'dr': 'sample_fundus.png',
           'oral': 'sample_histopathology.jpg'}
MAX_UPLOAD = 10 * 1024 * 1024


def image_url(value):
    if value is None:
        return None
    image = value if isinstance(value, Image.Image) else Image.fromarray(value)
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def create_web_app(ctx, legacy_workspace=False) -> FastAPI:
    from src.database.demo_service import DemoService
    from src.utils.validators import ValidationError
    demos = DemoService(ctx.db)
    from src.reports.automatic import AutomaticReports, quality_summary
    reports = AutomaticReports(ctx.db)
    from src.core.jobs import ScreeningJobs, QueueFull, JobConflict
    jobs = ScreeningJobs(ctx.db, lambda *a, **kw: perform_screen(*a, **kw))

    @asynccontextmanager
    async def lifespan(app):
        jobs.start()
        yield
        jobs.close()

    app = FastAPI(title='Mediscan AI', docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.jobs = jobs

    @app.middleware('http')
    async def local_requests(request: Request, call_next):
        # Reject cross-site browser mutations to the local clinical database.
        if request.method not in {'GET', 'HEAD', 'OPTIONS'}:
            origin = request.headers.get('origin')
            if origin and urlsplit(origin).netloc != request.headers.get('host'):
                return JSONResponse({'detail': 'Cross-origin request refused.'}, status_code=403)
        response = await call_next(request)
        if request.url.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/')
    def index():
        return FileResponse(WEB_ROOT / 'index.html')

    @app.get('/api/status')
    def status():
        rows = [p.status_row() for p in ctx.modules.values()]
        return {'models': [{k: r.get(k) for k in ('key', 'status', 'provider', 'version')} for r in rows],
                'processing': 'remote' if any(p.kind == 'remote_research' for p in ctx.modules.values()) else 'local',
                'research_evaluation': any(p.kind == 'local_research' for p in ctx.modules.values()),
                'simulation': config.SIMULATION_MODE, 'legacy_workspace': legacy_workspace}

    @app.get('/api/patients')
    def patients():
        return [{'id': p['id'], 'reference': p['reference'], 'is_demo': bool(p['is_demo'])}
                for p in ctx.patients.search('', limit=500)]

    @app.get('/api/demos')
    def demo_cases():
        return demos.list()

    @app.get('/api/demos/{patient_id}/scan')
    def demo_scan(patient_id: int):
        data = demos.scan(patient_id)
        if data is None:
            raise HTTPException(404, 'Demo scan unavailable.')
        return Response(data, media_type='application/octet-stream')

    @app.post('/api/patients/demo')
    async def demo_patient(name: str = Form(...), details: str = Form(''),
                           module: str = Form(...), expected_label: str = Form('unknown'),
                           age: str = Form(''), sex: str = Form(''), image: UploadFile = File(...)):
        data = await image.read(MAX_UPLOAD + 1)
        await image.close()
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, 'Select an image smaller than 10 MB.')
        try:
            # Decode only: low-quality images must remain available for rejection demos.
            with Image.open(io.BytesIO(data)) as im:
                if im.width * im.height > config.QUALITY['max_pixels']:
                    raise ValueError('Image exceeds the decoded pixel limit.')
                if im.format not in {'PNG', 'JPEG', 'BMP'}:
                    raise ValueError('Use PNG, JPEG, or BMP.')
                im.verify()
            pid = demos.create(name, details, module, expected_label,
                               Path(image.filename or 'scan.png').name, data, age or None, sex or None)
        except (ValueError, ValidationError, OSError, Image.DecompressionBombError) as e:
            raise HTTPException(422, str(e)) from e
        return {'id': pid}

    @app.get('/api/samples/{module}')
    def sample(module: str):
        if module not in SAMPLES or not (config.SAMPLES_DIR / SAMPLES[module]).exists():
            raise HTTPException(404, 'Sample unavailable.')
        return FileResponse(config.SAMPLES_DIR / SAMPLES[module])

    @app.get('/api/reports/{token}')
    def download_report(token: str):
        if len(token) != 32 or any(c not in '0123456789abcdef' for c in token):
            raise HTTPException(404, 'Report not found.')
        try:
            data = reports.render(token)
        except Exception as e:
            raise HTTPException(503, 'PDF generation failed. Please retry.') from e
        if data is None:
            raise HTTPException(404, 'Report not found.')
        return Response(data, media_type='application/pdf',
                        headers={'Content-Disposition': f'attachment; filename="mediscan-screening-{token[:8]}.pdf"'})

    async def read_input(image, patient_id, module):
        if module not in {'auto', 'tb', 'dr', 'oral'}:
            raise HTTPException(422, 'Unsupported screening module.')
        if not ctx.patients.get(patient_id):
            raise HTTPException(422, 'Select an existing patient record.')
        data = await image.read(MAX_UPLOAD + 1)
        await image.close()
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, 'Select an image smaller than 10 MB.')
        return data

    def perform_screen(data, filename, patient_id, module='auto', explain=True, allow_rescreen=False, patient_name=''):
        outcome = run_screening(ctx, data, filename, patient_id, module, None, 'en', allow_rescreen, explain)
        saved_demo = next((d for d in demos.list() if d['patient_id'] == patient_id), None)
        name = saved_demo['name'] if saved_demo else patient_name.strip()
        report_token, report_error = reports.create(outcome, name)
        ex = outcome.explanation or {}
        images = {k: image_url(ex.get(k)) for k in ('base', 'heatmap', 'overlay')} if ex.get('ok') else {}
        r = outcome.result or {}
        return {'stage': outcome.stage, 'status': outcome.status, 'provider_status': r.get('provider_status'),
                'rejection_code': outcome.rejection.code if outcome.rejection else None,
                'html': result_html(outcome, 'en', routing_text(outcome.modality)),
                'clinical': clinical_html(outcome, 'en'), 'images': images,
                'explanation_message': ex.get('message') or ('Grad-CAM not requested.' if not explain else 'No explanation available.'),
                'metrics': {k: r.get(k) for k in ('model_score', 'threshold', 'processing_time_ms',
                                                'model_version', 'model_sha256', 'score_kind', 'uncertainty_band', 'disease', 'simulated',
                                                'provider', 'provider_request_id', 'target_definition', 'positive_class',
                                                'preprocessing_version', 'decision_policy_version', 'raw_positive_score')},
                'result_id': outcome.result_id, 'quality': quality_summary(outcome),
                'report_url': f'/api/reports/{report_token}', 'report_error': report_error}

    @app.post('/api/screen')
    async def screen(image: UploadFile = File(...), patient_id: int = Form(...),
                     module: str = Form('auto'), explain: bool = Form(True),
                     allow_rescreen: bool = Form(False), patient_name: str = Form('', max_length=120)):
        # Compatibility route; share the same bounded queue as background requests.
        import asyncio
        from uuid import uuid4
        data = await read_input(image, patient_id, module)
        try:
            job = jobs.submit(uuid4().hex, patient_id, data, image.filename or 'upload.png',
                              dict(module=module, explain=explain, allow_rescreen=allow_rescreen, patient_name=patient_name))
        except QueueFull as e:
            raise HTTPException(503, str(e)) from None
        while job['status'] in ('queued', 'running'):
            await asyncio.sleep(0.05)
            job = jobs.get(job['id'])
        if not job.get('result'):
            raise HTTPException(503, job['error'] or 'Screening unavailable.')
        return job['result']

    @app.post('/api/jobs', status_code=202)
    async def submit_job(image: UploadFile = File(...), patient_id: int = Form(...),
                         module: str = Form('auto'), explain: bool = Form(False),
                         allow_rescreen: bool = Form(False), patient_name: str = Form('', max_length=120),
                         idempotency_key: str = Header(..., min_length=16, max_length=128)):
        data = await read_input(image, patient_id, module)
        try:
            return jobs.submit(idempotency_key, patient_id, data, image.filename or 'upload.png',
                               dict(module=module, explain=explain, allow_rescreen=allow_rescreen, patient_name=patient_name))
        except QueueFull as e:
            raise HTTPException(503, str(e)) from None
        except JobConflict as e:
            raise HTTPException(409, str(e)) from None

    @app.get('/api/jobs/{job_id}')
    def job_status(job_id: str):
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, 'Screening job not found.')
        return job

    @app.post('/api/jobs/{job_id}/cancel')
    def cancel_job(job_id: str):
        job_status(job_id)
        try:
            return jobs.cancel(job_id)
        except JobConflict as e:
            raise HTTPException(409, str(e)) from None

    app.mount('/static', StaticFiles(directory=WEB_ROOT / 'static'), name='static')
    return app
