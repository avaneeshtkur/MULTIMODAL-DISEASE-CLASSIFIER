"""Read server-side configuration; credentials are referenced by environment name."""
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

KEYS = ('tb', 'dr', 'oral')


def load_provider_config(path=None):
    path = path or os.environ.get('MEDISCAN_PROVIDER_CONFIG')
    settings = json.loads(Path(path).read_text()) if path else {}
    if not isinstance(settings, dict) or set(settings) - set(KEYS):
        raise ValueError('Provider configuration must map tb, dr and oral to provider settings.')
    out = {}
    for key in KEYS:
        cfg = dict(settings.get(key, {'kind': 'local_onnx'}))
        if cfg.get('kind') not in ('local_onnx', 'local_research', 'remote_research'):
            raise ValueError(f'Unsupported provider for {key}')
        if cfg['kind'] == 'local_research':
            if not isinstance(cfg.get('bundle'), str) or not cfg['bundle'].strip():
                raise ValueError(f'{key}: local_research requires a bundle directory.')
            bundle = Path(cfg['bundle']).expanduser()
            cfg['bundle'] = str((Path(path).resolve().parent / bundle).resolve())
        if cfg['kind'] == 'remote_research':
            for field in ('url', 'token_env', 'model_version', 'model_sha256',
                          'target_definition', 'positive_class', 'preprocessing_version', 'decision_policy_version'):
                if not isinstance(cfg.get(field), str) or not cfg[field].strip():
                    raise ValueError(f'{key}: missing {field}')
            u = urlsplit(cfg['url'])
            local = u.hostname in ('127.0.0.1', 'localhost', '::1')
            if not u.hostname or u.username or u.password or u.query or u.fragment or (
                    u.scheme != 'https' and not (u.scheme == 'http' and local)):
                raise ValueError(f'{key}: use HTTPS (HTTP allowed only for loopback tests).')
            if not re.fullmatch('[a-f0-9]{64}', cfg['model_sha256']):
                raise ValueError(f'{key}: model_sha256 must pin the verified checkpoint hash.')
            if not re.fullmatch('[A-Z][A-Z0-9_]+', cfg['token_env']):
                raise ValueError(f'{key}: token_env must name a server-side environment variable.')
            timeout = cfg.get('timeout_seconds', 60)
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 1 <= timeout <= 180:
                raise ValueError(f'{key}: timeout_seconds must be between 1 and 180.')
        out[key] = cfg
    return out
