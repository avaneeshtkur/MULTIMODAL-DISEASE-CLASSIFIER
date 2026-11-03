"""Package reviewed model files with independent reference logits; refuse bad parity.

Run locally with the selected model runtime installed.
Settings contain model provenance, preprocessing, class mapping and decision policy.
Reference JSON is a list of {file, raw_output}, produced by the original implementation.
This command verifies reproduction; it does not measure clinical accuracy.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.research.model import ResearchModel
from src.utils.image_hash import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--extra-file', type=Path, action='append', default=[])
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.settings.read_text())
    refs = json.loads(args.references.read_text())
    files = [args.model, *args.extra_file]
    if len({p.name for p in files}) != len(files) or any(p.name in ('manifest.json', 'client-identity.json') for p in files):
        parser.error('Model files must have distinct nonreserved names.')
    args.out.mkdir(parents=True, exist_ok=False)
    spec.update(schema_version='1.0', model_file=args.model.name,
                files={p.name: sha256_file(p) for p in files}, reference_fixtures=[])
    for path in files:
        shutil.copyfile(path, args.out / path.name)
    fixture_dir = args.out / 'fixtures'
    fixture_dir.mkdir()
    for i, fixture in enumerate(refs):
        path = (args.references.parent / fixture['file']).resolve()
        name = f'fixtures/{i}{path.suffix}'
        shutil.copyfile(path, args.out / name)
        spec['reference_fixtures'].append(dict(file=name, sha256=sha256_file(path), raw_output=fixture['raw_output']))
    (args.out / 'manifest.json').write_text(json.dumps(spec, indent=2) + '\n')
    model = ResearchModel(args.out)
    identity = {spec['disease']: dict(kind='local_research', bundle=str(args.out.resolve()))}
    (args.out / 'client-identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    print('Reference inference reproduced. Bundle:', args.out)
    print('Clinical accuracy remains unmeasured. Set MEDISCAN_PROVIDER_CONFIG to the generated client-identity.json for local inference.')


if __name__ == '__main__':
    main()
