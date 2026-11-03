"""Private research API: bounded uploads, one model, authenticated requests."""
import hashlib
import hmac
import os
import threading
import time

from fastapi import FastAPI, Depends, File, Form, Header, HTTPException, UploadFile, Request
from fastapi.responses import JSONResponse

from services.research.model import ResearchModel
from src.gatekeeper.image_quality import validate_image, modality_blur_check
from src.gatekeeper.classifier import _consistency_rejection

MODALITIES = {'tb': 'chest_xray', 'dr': 'fundus', 'oral': 'histopathology'}


def create_app(bundle=None, token=None):
    token = token or os.environ.get('MEDISCAN_SERVICE_TOKEN')
    if not token or len(token) < 32:
        raise ValueError('Set a private MEDISCAN_SERVICE_TOKEN of at least 32 characters.')
    model = ResearchModel(bundle or os.environ['MEDISCAN_MODEL_BUNDLE'])
    app = FastAPI(docs_url=None, redoc_url=None)
    app.state.model = model
    slot = threading.BoundedSemaphore(1)

    @app.middleware('http')
    async def bound_request(request: Request, call_next):
        # Authenticate and bound the multipart body before FastAPI parses/spools it.
        if not hmac.compare_digest(request.headers.get('authorization', ''), 'Bearer ' + token):
            return JSONResponse({'detail': 'Authentication required.'}, status_code=401)
        if request.method == 'POST':
            try:
                length = int(request.headers.get('content-length', '-1'))
            except ValueError:
                length = -1
            if length < 0 or length > 26 * 1024 * 1024:
                return JSONResponse({'detail': 'A bounded Content-Length is required.'}, status_code=413)
        return await call_next(request)

    def authenticate(authorization: str = Header('')):
        if not hmac.compare_digest(authorization, 'Bearer ' + token):
            raise HTTPException(401, 'Authentication required.')

    @app.get('/health', dependencies=[Depends(authenticate)])
    def health():
        return {'status': 'ready', 'disease': model.spec['disease'], **model.identity}

    @app.post('/v1/predict', dependencies=[Depends(authenticate)])
    def predict(image: UploadFile = File(...), disease: str = Form(...),
                request_id: str = Form(..., min_length=16, max_length=128)):
        if disease != model.spec['disease']:
            raise HTTPException(422, 'This service does not support the requested disease.')
        if not slot.acquire(blocking=False):
            raise HTTPException(429, 'Model is busy. Retry later.')
        started = time.perf_counter()
        try:
            data = image.file.read(25 * 1024 * 1024 + 1)
            if len(data) > 25 * 1024 * 1024:
                raise HTTPException(413, 'Upload exceeds limit.')
            base = dict(schema_version='1.0', request_id=request_id, disease=disease,
                        input_sha256=hashlib.sha256(data).hexdigest(), **model.identity)
            quality = validate_image(data, 'scan.png')
            modality = MODALITIES[disease]
            invalid = not quality.ok or modality_blur_check(quality.metrics, modality)
            mismatch = _consistency_rejection(modality, quality.metrics) if quality.ok else None
            domain_mismatch = (model.spec.get('input_domain') == 'ben_graham_filtered' and
                               quality.metrics.get('dark_corner_frac', 0) > 0.5)
            if invalid or mismatch or domain_mismatch:
                return {**base, 'provider_status': 'unsupported' if mismatch or domain_mismatch else 'ungradable',
                        'model_score': None, 'prediction_status': 'INCONCLUSIVE',
                        'reason_code': 'OUT_OF_DISTRIBUTION', 'reason': 'Image does not meet this specialist input contract.',
                        'processing_time_ms': (time.perf_counter() - started) * 1000}
            return {**base, **model.predict(quality.image), 'processing_time_ms': (time.perf_counter() - started) * 1000}
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(503, 'Model inference failed; no disease result is available.') from None
        finally:
            image.file.close()
            slot.release()
    return app
