import pytest
from scripts.evaluate_external import validate, metrics


def row(patient, label, status, score='', split='test'):
    return dict(patient_id=patient, source='external', split=split, image_sha256=patient,
                label=label, status=status, score=score, model_version='v1')


def test_rejected_positives_remain_in_denominator():
    rows = [row('a', '1', 'SCREENING_FLAGGED', '0.9'), row('b', '1', 'ungradable'),
            row('c', '0', 'NO_FLAG_DETECTED', '0.1'), row('d', '0', 'failed')]
    report = metrics(validate(rows))
    assert report['positive_flag_rate_all_patients'] == 0.5
    assert report['sensitivity_among_decided'] == 1
    assert report['decision_coverage'] == 0.5
    assert report['undecided'] == 2


def test_patient_leakage_rejected():
    with pytest.raises(ValueError, match='Repeated patient'):
        validate([row('a', '1', 'SCREENING_FLAGGED', '0.9'), row('a', '1', 'SCREENING_FLAGGED', '0.9', 'train')])
