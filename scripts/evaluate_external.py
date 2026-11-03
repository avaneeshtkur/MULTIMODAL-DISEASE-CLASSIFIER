"""Evaluate one locked specialist/endpoint from a patient-level external CSV.

Columns: patient_id, source, split, image_sha256, label, status, score,
model_version. One row per patient per split: aggregate eyes/patches using a
predeclared rule upstream. Include all rejected/failed patients with empty score.
Development rows are used only to detect leakage; metrics use split=test.
"""
import argparse
import csv
import json
import math
from pathlib import Path


def interval(success, count):
    if not count:
        return None
    z = 1.96
    p = success / count
    center = (p + z*z/(2*count)) / (1 + z*z/count)
    half = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / (1 + z*z/count)
    return [max(0, center-half), min(1, center+half)]


def validate(rows):
    patients, images = {}, {}
    for row in rows:
        for key in ('patient_id', 'source', 'split', 'image_sha256', 'model_version'):
            if not row.get(key):
                raise ValueError(f'Missing {key}; independence cannot be assessed')
        if row['split'] not in ('train', 'val', 'calibration', 'test') or row['label'] not in ('0', '1'):
            raise ValueError('Invalid split or reference label')
        if row['patient_id'] in patients:
            raise ValueError('Repeated patient: aggregate once and prevent cross-split leakage')
        if row['image_sha256'] in images:
            raise ValueError('Duplicate image in evaluation manifest')
        patients[row['patient_id']] = row['split']
        images[row['image_sha256']] = row['split']
        status = row['status']
        if status not in ('SCREENING_FLAGGED', 'NO_FLAG_DETECTED', 'INCONCLUSIVE', 'ungradable', 'unsupported', 'failed'):
            raise ValueError('Unknown prediction status')
        score = row.get('score', '')
        if score and (not math.isfinite(float(score)) or not 0 <= float(score) <= 1):
            raise ValueError('Invalid score')
        if status in ('ungradable', 'unsupported', 'failed') and score:
            raise ValueError('Unassessed patients cannot have a disease score')
        if status in ('SCREENING_FLAGGED', 'NO_FLAG_DETECTED') and not score:
            raise ValueError('A scored decision requires its model score')
    test = [r for r in rows if r['split'] == 'test']
    if not test or len({r['model_version'] for r in test}) != 1:
        raise ValueError('One fixed model version and nonempty external test cohort required')
    return test


def metrics(rows):
    n = len(rows)
    positive = [r for r in rows if r['label'] == '1']
    negative = [r for r in rows if r['label'] == '0']
    tp = sum(r['status'] == 'SCREENING_FLAGGED' for r in positive)
    fp = sum(r['status'] == 'SCREENING_FLAGGED' for r in negative)
    tn = sum(r['status'] == 'NO_FLAG_DETECTED' for r in negative)
    fn = sum(r['status'] == 'NO_FLAG_DETECTED' for r in positive)
    result = dict(patients=n, positives=len(positive), negatives=len(negative),
                  flagged_positive=tp, flagged_negative=fp, unflagged_positive=fn, unflagged_negative=tn,
                  undecided=n-tp-fp-tn-fn, decision_coverage=(tp+fp+tn+fn)/n if n else None,
                  positive_flag_rate_all_patients=tp/len(positive) if positive else None,
                  positive_flag_rate_95ci=interval(tp, len(positive)),
                  sensitivity_among_decided=tp/(tp+fn) if tp+fn else None,
                  specificity_among_decided=tn/(tn+fp) if tn+fp else None,
                  sensitivity_among_decided_95ci=interval(tp, tp+fn),
                  specificity_among_decided_95ci=interval(tn, tn+fp))
    scored = [r for r in rows if r.get('score')]
    result['score_coverage'] = len(scored)/n if n else None
    if len({r['label'] for r in scored}) == 2:
        from sklearn.metrics import roc_auc_score, average_precision_score
        labels, scores = [int(r['label']) for r in scored], [float(r['score']) for r in scored]
        result.update(auroc_among_scored=float(roc_auc_score(labels, scores)),
                      average_precision_among_scored=float(average_precision_score(labels, scores)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with args.csv.open() as f:
        rows = list(csv.DictReader(f))
    test = validate(rows)
    development_sources = {r['source'] for r in rows if r['split'] != 'test'}
    result = dict(model_version=test[0]['model_version'], overall=metrics(test),
                  by_source={s: metrics([r for r in test if r['source'] == s]) for s in sorted({r['source'] for r in test})},
                  source_overlap_with_development=sorted(development_sources & {r['source'] for r in test}),
                  note='Pretraining overlap and clinical reference labels require independent provenance review. Rejections are never negatives.')
    with args.out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print('Saved measured evaluation:', args.out)


if __name__ == '__main__':
    main()
