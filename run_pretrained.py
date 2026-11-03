"""Run all three verified pretrained bundles on this computer only.

Usage: .venv/bin/python run_pretrained.py
No hosting account, API key, Docker, or remote inference is used.
"""
import json
import os
from pathlib import Path


def configure_environment():
    root = Path(__file__).resolve().parent
    configuration = root / 'deploy' / 'providers.pretrained.local.json'
    settings = json.loads(configuration.read_text())
    for disease, provider in settings.items():
        bundle = (configuration.parent / provider['bundle']).resolve()
        if not (bundle / 'manifest.json').is_file():
            raise SystemExit(f'{disease}: verified bundle missing at {bundle}; refusing partial or fallback activation.')
    os.environ['MEDISCAN_PROVIDER_CONFIG'] = str(configuration)
    os.environ.setdefault('MEDISCAN_ORT_THREADS', '2')
    research_db = Path(os.environ.get('MEDISCAN_DB', root / 'output/research/mediscan.db')).resolve()
    if research_db == (root / 'data/mediscan.db').resolve():
        raise SystemExit('Research mode cannot use data/mediscan.db. Set MEDISCAN_DB to a separate research database.')
    os.environ['MEDISCAN_DB'] = str(research_db)
    os.environ.setdefault('MEDISCAN_LOG_DIR', str(root / 'output/research/logs'))
    return research_db


def main():
    research_db = configure_environment()
    print(f'Research evaluation only. Isolated database: {research_db}', flush=True)
    from server import create_app
    import uvicorn
    uvicorn.run(create_app(), host='127.0.0.1', port=int(os.environ.get('MEDISCAN_PORT', '7860')), workers=1)


if __name__ == '__main__':
    main()
