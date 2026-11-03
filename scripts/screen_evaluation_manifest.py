"""Run labelled research images through the actual pipeline in a temporary DB.

Input JSON: cases with file, patient_id, label (0/1), sha256. Supply a declared
source name and endpoint. Images are never copied into the operational database.
The resulting CSV matches evaluate_external.py; failures retain empty scores.
This does not establish dataset independence or clinical label validity.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.pipeline import build_context, run_screening


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--source', required=True)
    p.add_argument('--disease', choices=['tb', 'dr', 'oral'], required=True)
    p.add_argument('--config')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    cases = json.loads(args.manifest.read_text())['cases']
    patients, hashes = set(), set()
    for case in cases:
        if case['label'] not in (0, 1) or not case['patient_id']:
            raise ValueError('Explicit binary label and patient identifier required')
        if case['patient_id'] in patients or case['sha256'] in hashes:
            raise ValueError('Duplicate patient/image; aggregate before evaluation')
        patients.add(case['patient_id']); hashes.add(case['sha256'])
        if hashlib.sha256(Path(case['file']).read_bytes()).hexdigest() != case['sha256']:
            raise ValueError('Evaluation image hash mismatch')
    with tempfile.TemporaryDirectory() as temp:
        ctx = build_context(Path(temp)/'evaluation.db', provider_config=args.config)
        provider = ctx.modules[args.disease]
        version = provider.status_row()['version']
        with args.out.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['patient_id','source','split','image_sha256',
                                                       'label','status','score','model_version'])
            writer.writeheader()
            for case in cases:
                pid = ctx.patients.register(case['patient_id'], is_demo=True)
                path = Path(case['file'])
                outcome = run_screening(ctx, path.read_bytes(), path.name, pid,
                                        module=args.disease, explain=False)
                result = outcome.result or {}
                if outcome.rejection:
                    status = 'ungradable' if outcome.rejection.stage == 'quality' else 'unsupported'
                    if outcome.rejection.code == 'MODEL_UNAVAILABLE': status = 'failed'
                else:
                    status = result['prediction_status']
                    if result.get('provider_status') in ('failed', 'unsupported', 'ungradable'):
                        status = result['provider_status']
                writer.writerow(dict(patient_id=case['patient_id'], source=args.source, split='test',
                                     image_sha256=case['sha256'], label=case['label'], status=status,
                                     score=result.get('model_score', ''), model_version=version))
    print('Saved measured research CSV:', args.out)


if __name__ == '__main__':
    main()
