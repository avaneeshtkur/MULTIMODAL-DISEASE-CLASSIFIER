"""MediScan medical imaging studio backed by the existing screening pipeline."""
from __future__ import annotations

import base64
import io
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
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


def create_web_app(ctx) -> FastAPI:
    app = FastAPI(title='Mediscan AI', docs_url=None, redoc_url=None)

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
        return {'models': [{'key': r['key'], 'status': r['status']} for r in ctx.registry.status_rows()],
                'simulation': config.SIMULATION_MODE}

    @app.get('/api/patients')
    def patients():
        return [{'id': p['id'], 'reference': p['reference'], 'is_demo': bool(p['is_demo'])}
                for p in ctx.patients.search('', limit=500)]

    @app.post('/api/patients/demo')
    def demo_patient():
        return {'id': ctx.patients.register(is_demo=True)}

    @app.get('/api/samples/{module}')
    def sample(module: str):
        if module not in SAMPLES or not (config.SAMPLES_DIR / SAMPLES[module]).exists():
            raise HTTPException(404, 'Sample unavailable.')
        return FileResponse(config.SAMPLES_DIR / SAMPLES[module])

    @app.post('/api/screen')
    async def screen(image: UploadFile = File(...), patient_id: int = Form(...),
                     module: str = Form('auto'), explain: bool = Form(True),
                     allow_rescreen: bool = Form(False)):
        if module not in {'auto', 'tb', 'dr', 'oral'}:
            raise HTTPException(422, 'Unsupported screening module.')
        if not ctx.patients.get(patient_id):
            raise HTTPException(422, 'Select an existing patient record.')
        data = await image.read(MAX_UPLOAD + 1)
        await image.close()
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, 'Select an image smaller than 10 MB.')
        outcome = await run_in_threadpool(run_screening, ctx, data, image.filename or 'upload.png',
                                         patient_id, module, None, 'en', allow_rescreen, explain)
        ex = outcome.explanation or {}
        images = {k: image_url(ex.get(k)) for k in ('base', 'heatmap', 'overlay')} if ex.get('ok') else {}
        r = outcome.result or {}
        return {'stage': outcome.stage, 'status': outcome.status,
                'html': result_html(outcome, 'en', routing_text(outcome.modality)),
                'clinical': clinical_html(outcome, 'en'), 'images': images,
                'explanation_message': ex.get('message') or ('Grad-CAM not requested.' if not explain else 'No explanation available.'),
                'metrics': {k: r.get(k) for k in ('model_score', 'threshold', 'processing_time_ms',
                                                'model_version', 'disease', 'simulated')},
                'result_id': outcome.result_id}

    app.mount('/static', StaticFiles(directory=WEB_ROOT / 'static'), name='static')
    return app
