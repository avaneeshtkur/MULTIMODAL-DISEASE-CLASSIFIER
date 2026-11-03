"""Measure each installed bundle in a separate CPU process, sequentially."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=Path)
    p.add_argument('--out', type=Path, default=Path('docs/research-performance.json'))
    args = p.parse_args()
    if args.worker:
        from services.research.model import ResearchModel
        from PIL import Image
        start = time.perf_counter()
        model = ResearchModel(args.worker)
        load = time.perf_counter()-start
        fixture = model.path(model.spec['reference_fixtures'][0]['file'])
        with Image.open(fixture) as im:
            image = im.convert('RGB')
        start = time.perf_counter()
        result = model.predict(image)
        inference = time.perf_counter()-start
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != 'darwin': rss *= 1024
        print(json.dumps({'model_version': model.spec['model_version'],
                          'load_hash_and_reference_seconds': load,
                          'one_inference_seconds': inference, 'peak_rss_bytes': rss,
                          'provider_status': result['provider_status']}))
        return
    rows = []
    for key in ['tb', 'dr', 'oral', 'tb-vit']:
        result = subprocess.run([sys.executable, __file__, '--worker', f'output/research-bundles/{key}'],
                                check=True, text=True, capture_output=True, timeout=120)
        rows.append(json.loads(result.stdout))
        print(key, 'measured', flush=True)
    args.out.write_text(json.dumps({'method': 'Separate sequential processes, CPU, one inference per model; not a latency distribution or memory bound', 'models': rows},indent=2)+'\n')


if __name__ == '__main__':
    main()
