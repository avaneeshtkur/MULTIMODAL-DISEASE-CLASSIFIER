"""Explicit specialist selection. Remote failure never selects a local fallback."""
from .configuration import load_provider_config


def build_providers(registry, settings):
    from src.diseases import build_modules
    from .local_onnx import LocalONNXProvider
    from .remote_research import RemoteResearchProvider
    from .local_research import LocalResearchProvider
    local = build_modules(registry)
    return {key: RemoteResearchProvider(key, cfg) if cfg['kind'] == 'remote_research'
            else LocalResearchProvider(key, cfg) if cfg['kind'] == 'local_research'
            else LocalONNXProvider(local[key]) for key, cfg in settings.items()}
