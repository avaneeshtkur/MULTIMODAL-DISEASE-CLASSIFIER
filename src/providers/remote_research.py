"""Bounded private HTTP transport and strict, pinned specialist-result validation."""
import hashlib
import io
import json
import math
import os
import time
from uuid import uuid4

import httpx

from src.core import inference_engine as ie

IDENTITY_FIELDS = ('model_version', 'model_sha256', 'target_definition', 'positive_class',
                   'preprocessing_version', 'decision_policy_version')


def number(value, low=0, high=1):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError('Invalid numeric model output')
    return value


def validate_response(body, key, expected, request_id, image_hash):
    if not isinstance(body, dict) or body.get('schema_version') != '1.0':
        raise ValueError('Unknown response schema')
    if body.get('request_id') != request_id or body.get('input_sha256') != image_hash or body.get('disease') != key:
        raise ValueError('Response does not belong to this scan')
    for field in IDENTITY_FIELDS:
        if body.get(field) != expected[field]:
            raise ValueError(f'Provider identity changed: {field}')
    state = body.get('provider_status')
    if state not in ('completed', 'ungradable', 'unsupported'):
        raise ValueError('Invalid provider status')
    if state != 'completed':
        if body.get('model_score') is not None or body.get('prediction_status') != ie.INCONCLUSIVE:
            raise ValueError('Unusable input must not have a disease score or negative decision')
    else:
        score = number(body.get('model_score'))
        number(body.get('raw_positive_score'))
        threshold = number(body.get('threshold'))
        band = body.get('uncertainty_band')
        if not isinstance(band, list) or len(band) != 2 or not 0 <= number(band[0]) <= threshold <= number(band[1]) <= 1:
            raise ValueError('Invalid decision band')
        decision = ie.INCONCLUSIVE if band[0] < score < band[1] else ie.SCREENING_FLAGGED if score >= threshold else ie.NO_FLAG_DETECTED
        if body.get('prediction_status') != decision:
            raise ValueError('Provider decision contradicts its declared policy')
        if body.get('score_kind') not in ('uncalibrated model score', 'temperature-scaled model score'):
            raise ValueError('Unsupported score semantics')
        if body['score_kind'] == 'uncalibrated model score' and abs(score - body['raw_positive_score']) > 1e-8:
            raise ValueError('Uncalibrated positive-class score was modified')
        raw = body.get('raw_output')
        if not isinstance(raw, list) or not 1 <= len(raw) <= 100:
            raise ValueError('Raw model output missing')
        for value in raw:
            number(value, -1e9, 1e9)
    number(body.get('processing_time_ms', 0), 0, 1e9)
    for field in ('reason', 'class_label', 'reason_code'):
        if body.get(field) is not None and (not isinstance(body[field], str) or len(body[field]) > 1000):
            raise ValueError('Invalid response text')
    return {k: body.get(k) for k in (*IDENTITY_FIELDS, 'provider_status', 'model_score', 'raw_positive_score',
            'raw_output', 'threshold', 'uncertainty_band', 'prediction_status', 'score_kind', 'class_label',
            'reason_code', 'reason', 'processing_time_ms')}


class RemoteResearchProvider:
    kind = 'remote_research'

    def __init__(self, key, settings, transport=None):
        self.key, self.settings, self.transport = key, settings, transport

    @property
    def available(self):
        return bool(os.environ.get(self.settings['token_env']))

    def setup_message(self):
        return 'Configure the remote research service and its server-side API token.'

    def status_row(self):
        return dict(key=self.key, status='configured' if self.available else 'missing', provider=self.kind,
                    version=self.settings['model_version'], sha256=self.settings['model_sha256'],
                    architecture='remote specialist', classes=self.settings['positive_class'], threshold=None,
                    gradcam=False, errors='' if self.available else self.setup_message())

    def predict(self, image, metrics=None, image_sha256=None):
        started = time.perf_counter()
        request_id = uuid4().hex
        result = ie.empty_result(self.key, ie.INFERENCE_ERROR, 'Remote analysis unavailable; no disease decision was produced.')
        result.update({k: self.settings[k] for k in IDENTITY_FIELDS})
        result.update(provider=self.kind, provider_status='failed', provider_request_id=request_id)
        if not self.available:
            return result
        # Send decoded pixels as a fresh PNG: no patient name, filename, EXIF or report.
        buf = io.BytesIO()
        from PIL import Image
        Image.frombytes(image.mode, image.size, image.tobytes()).save(buf, format='PNG')
        data = buf.getvalue()
        if len(data) > 25 * 1024 * 1024:
            result['reason'] = 'Decoded image exceeds the remote upload limit.'
            return result
        input_hash = hashlib.sha256(data).hexdigest()
        try:
            with httpx.Client(timeout=self.settings.get('timeout_seconds', 60), follow_redirects=False,
                              trust_env=False, transport=self.transport) as client:
                with client.stream('POST', self.settings['url'].rstrip('/') + '/v1/predict',
                                   headers={'Authorization': 'Bearer ' + os.environ[self.settings['token_env']]},
                                   data={'disease': self.key, 'request_id': request_id},
                                   files={'image': ('scan.png', data, 'image/png')}) as response:
                    response.raise_for_status()
                    chunks, size = [], 0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > 65536:
                            raise ValueError('Provider response exceeds limit')
                        chunks.append(chunk)
            body = json.loads(b''.join(chunks))
            result.update(validate_response(body, self.key, self.settings, request_id, input_hash))
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            # Do not expose tokens, endpoint URLs, or untrusted error bodies in logs/reports.
            result['reason'] = 'Remote service failed or returned an incompatible result. No disease decision was produced.'
        result['processing_time_ms'] = round((time.perf_counter() - started) * 1000, 1)
        return result

    def explain(self, image, result):
        return {'ok': False, 'message': 'This remote provider does not supply a validated explanation.'}
