"""Exercise installed research weights through the real local app and PDFs.

Uses an isolated temporary database, no patient uploads or remote inference.
This verifies integration, not medical accuracy. The separate pilot reports
record model failures and must be considered before any replacement decision.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='deploy/providers.pretrained.local.json')
    p.add_argument('--out', type=Path, default=Path('docs/pretrained-integration-audit.json'))
    args = p.parse_args()
    from fastapi.testclient import TestClient
    from src.core.pipeline import build_context
    from src.ui.web import create_web_app
    files = {
        'tb': Path('data/tb/Tuberculosis/Tuberculosis-116.png'),
        'dr': Path('output/pretrained/dr/raw-fixtures/mild-0.jpg'),
        'oral': Path('data/oral/OSCC/aug_109_9290.jpg'),
    }
    results = []
    with tempfile.TemporaryDirectory() as temporary:
        ctx = build_context(Path(temporary) / 'integration.db', provider_config=args.config)
        pid = ctx.patients.register('PRETRAINED-INTEGRATION-DEMO', is_demo=True)
        with TestClient(create_web_app(ctx)) as client:
            for disease, path in files.items():
                start = time.perf_counter()
                response = client.post('/api/screen', data={'patient_id': pid, 'module': disease,
                                                           'explain': 'false'},
                                       files={'image': (path.name, path.read_bytes())})
                response.raise_for_status()
                body = response.json()
                if body.get('metrics', {}).get('provider') != 'local_research':
                    raise AssertionError(f'{disease} did not reach the research specialist: {body}')
                result = ctx.screenings.get_result(body['result_id'])
                provenance = result['provenance']
                if provenance.get('provider_status') != 'completed':
                    raise AssertionError(f'{disease} inference failed: {provenance}')
                report = client.get(body['report_url'])
                assert report.status_code == 200 and report.content.startswith(b'%PDF')
                results.append({'disease': disease, 'status': body['status'],
                                'score': body['metrics']['model_score'],
                                'provider': provenance['provider'],
                                'model_version': result['model_version'],
                                'raw_output': provenance['raw_output'],
                                'pdf_bytes': len(report.content),
                                'elapsed_seconds': round(time.perf_counter() - start, 3)})
                print(disease, 'local model, persistence and PDF passed', flush=True)
    args.out.write_text(json.dumps({'purpose': 'Integration only; not clinical validation',
                                   'cases': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
