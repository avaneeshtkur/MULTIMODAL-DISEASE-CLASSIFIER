"""Direct research inference in the current server; no network or credentials.

Serialize research inference and release model objects after each request. This
trades reload latency for lower retained weight memory on small laptops. Runtime
allocators may retain memory; this is not a hard process-memory limit.
"""
import gc
import json
import logging
import threading
import time
from pathlib import Path

from src.core import inference_engine as ie

_inference_lock = threading.Lock()
log = logging.getLogger(__name__)


class LocalResearchProvider:
    kind = 'local_research'

    def __init__(self, key, settings):
        self.key = key
        self.folder = Path(settings['bundle'])

    @property
    def available(self):
        return (self.folder / 'manifest.json').is_file()

    def setup_message(self):
        return 'Supply a verified local research bundle containing manifest.json and its model files.'

    def status_row(self):
        try:
            spec = json.loads((self.folder / 'manifest.json').read_text())
        except (OSError, ValueError):
            spec = {}
        return dict(key=self.key, status='configured' if self.available else 'missing', provider=self.kind,
                    version=spec.get('model_version', 'Not verified'), sha256='', architecture='local research specialist',
                    classes=', '.join(spec.get('class_labels', [])),
                    threshold=spec.get('decision_policy', {}).get('threshold'), gradcam=False,
                    errors='' if self.available else self.setup_message())

    def predict(self, image, metrics=None, image_sha256=None):
        started = time.perf_counter()
        result = ie.empty_result(self.key, ie.INFERENCE_ERROR,
                                'Local model could not be verified or executed. No disease decision was produced.')
        result.update(provider=self.kind, provider_status='failed')
        model = None
        with _inference_lock:
            try:
                # Inspect task before loading potentially large weights.
                manifest = json.loads((self.folder / 'manifest.json').read_text())
                if manifest.get('disease') != self.key:
                    raise ValueError('Bundle disease does not match selected specialist')
                if (manifest.get('input_domain') == 'ben_graham_filtered' and
                        (metrics or {}).get('dark_corner_frac', 0) > 0.5):
                    result.update(provider_status='unsupported', reason_code=ie.OUT_OF_DISTRIBUTION,
                                  reason='This bundle requires Ben Graham filtered fundus images.')
                else:
                    from services.research.model import ResearchModel
                    model = ResearchModel(self.folder)
                    prediction = model.predict(image)
                    result.update(prediction)
            except Exception:
                # Runtime/weight errors never become negative disease predictions.
                log.exception('Local research specialist %s failed verification or inference', self.key)
            finally:
                if model is not None and model.spec['runtime'] == 'tbnet_tf':
                    model.model.close()
                model = None
                gc.collect()
        result['processing_time_ms'] = round((time.perf_counter() - started) * 1000, 1)
        return result

    def explain(self, image, result):
        return {'ok': False, 'message': 'This research bundle does not provide a verified attention explanation.'}
