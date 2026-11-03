"""Adapter preserving the existing specialist and its exact decision policy."""
import hashlib
import json


class LocalONNXProvider:
    kind = 'local_onnx'

    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        return getattr(self.module, name)

    def predict(self, *args, **kwargs):
        result = self.module.predict(*args, **kwargs)
        md = self.handle.metadata
        policy = {k: md.get(k) for k in ('threshold', 'uncertainty_margin', 'uncertainty_margin_logit', 'calibration')}
        result.update(provider=self.kind, provider_status='completed' if result.get('model_score') is not None else
                      'unsupported' if result.get('reason_code') == 'OUT_OF_DISTRIBUTION' else 'failed',
                      positive_class=md.get('positive_class'),
                      target_definition=md.get('target_definition') or md.get('positive_class'),
                      preprocessing_version=hashlib.sha256(json.dumps(md.get('preprocessing', {}), sort_keys=True).encode()).hexdigest(),
                      decision_policy_version=hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest())
        return result

    def status_row(self):
        return {**self.handle.as_row(), 'provider': self.kind}
