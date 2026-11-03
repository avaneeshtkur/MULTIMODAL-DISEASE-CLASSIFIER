"""Audit installed models; propose validation-only thresholds, never install them.
Local split independence from external model training is unverified.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.core.model_registry import ModelRegistry
from src.core.inference_engine import predict_binary, decide


def summary(rows, md):
    counts = {'tp': 0, 'fn': 0, 'tn': 0, 'fp': 0, 'inconclusive': 0}
    for r in rows:
        status, _ = decide(r['score'], md)
        flag = status == 'SCREENING_FLAGGED'
        counts['inconclusive'] += status == 'INCONCLUSIVE'
        counts['tp' if flag and r['label'] else 'fp' if flag else 'fn' if r['label'] else 'tn'] += 1
    counts['flag_sensitivity'] = counts['tp'] / max(1, counts['tp'] + counts['fn'])
    counts['nonflag_specificity'] = counts['tn'] / max(1, counts['tn'] + counts['fp'])
    counts['note'] = 'Inconclusive counts as not flagged in these sensitivity/specificity calculations; it is not a negative diagnosis.'
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('docs/threshold-evaluation.json'))
    args = parser.parse_args()
    registry = ModelRegistry()
    report = {'limitation': 'Local split independence from external training is unverified. Candidate cutoffs are research only, not installed. No internet example is used for tuning.', 'models': {}}
    for key, d in config.DISEASES.items():
        h = registry.get(key)
        rows = []
        if d['splits_csv'].exists():
            with d['splits_csv'].open() as f:
                split_rows = list(csv.DictReader(f))
            split_source = 'Existing local splits.csv; independence from supplied weights unverified'
        else:
            split_rows = []
            for label, cls in enumerate(d['classes']):
                paths = sorted(p for p in (d['data_dir'] / cls).iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'})
                shuffled = pd.Series(paths).sample(frac=1, random_state=config.SEED).tolist()
                nt, nv = max(1, int(len(paths)*config.TEST_FRAC)), max(1, int(len(paths)*config.VAL_FRAC))
                split_rows.extend({'path':str(p), 'label':label, 'class_name':cls, 'split':'test' if i < nt else 'val'} for i,p in enumerate(shuffled[:nt+nv]))
            split_source = 'Reconstructed local split; original external training membership unknown. Exploratory only.'
        for r in split_rows:
            if r['split'] not in {'val', 'test'}:
                continue
            name = r['path'].replace('\\', '/').rsplit('/', 1)[-1]
            p = d['data_dir'] / r['class_name'] / name
            with Image.open(p) as image:
                result = predict_binary(h, image, key)
            if result['model_score'] is None:
                raise RuntimeError(result['reason'])
            rows.append({'file': name, 'split': r['split'], 'label': int(r['label']), 'score': result['model_score']})
        val = [r for r in rows if r['split'] == 'val']
        test = [r for r in rows if r['split'] == 'test']
        if not val or not test or {r['label'] for r in val} != {0,1}:
            raise ValueError('Validation and test splits with both classes are required.')
        md = h.metadata
        # Retain the uncertainty band; search for the largest cutoff flagging all validation positives.
        candidates = sorted(set(np.geomspace(0.000001, 0.999999, 2000).tolist() + [md['threshold']]), reverse=True)
        threshold = next(t for t in candidates if summary(val, {**md, 'threshold': t})['flag_sensitivity'] == 1.0)
        research_md = {**md, 'threshold': threshold}
        report['models'][key] = {'split_source': split_source, 'model_version': h.version, 'sha256': h.sha256,
            'current_threshold': md['threshold'], 'candidate_threshold': threshold,
            'lower_cutoff_test_comparison': {str(t): summary(test, {**md, 'threshold': t}) for t in [md['threshold']/2, md['threshold']/4]}, 'current_validation': summary(val, md), 'candidate_validation': summary(val, research_md),
            'current_test': summary(test, md), 'candidate_test': summary(test, research_md), 'scores': rows}
        print(key, json.dumps({k:v for k,v in report['models'][key].items() if k != 'scores'}), flush=True)
    args.output.write_text(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
