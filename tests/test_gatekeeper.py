"""Gatekeeper routing, unsupported/uncertain modality, manual selection, duplicates.

Gatekeeper models here are tiny synthetic ONNX graphs with fixed logits."""
import numpy as np
import pytest

from src.core.model_registry import ModelRegistry
from src.gatekeeper import rejection_manager as rm
from src.gatekeeper.classifier import SOURCE_MODEL, Gatekeeper
from src.gatekeeper.duplicate_detector import DuplicateInfo, duplicate_rejection
from src.gatekeeper.image_quality import compute_metrics
from tests.conftest import colourful, histo_like, install_model, xray_like


def gk_with(artifacts, logits):
    install_model(artifacts, "gatekeeper", logit=np.asarray(logits, dtype=np.float32))
    return Gatekeeper(ModelRegistry(models_dir=artifacts))


@pytest.mark.parametrize("logits,modality,disease", [
    ([6, 0, 0, 0], "chest_xray", "tb"), ([0, 6, 0, 0], "fundus", "dr"), ([0, 0, 6, 0], "histopathology", "oral")])
def test_routes_confident_predictions(artifacts, logits, modality, disease):
    img = xray_like() if modality == "chest_xray" else histo_like()
    d = gk_with(artifacts, logits).classify(img, compute_metrics(img))
    assert d.rejection is None and d.modality == modality and d.disease_key == disease and d.source == SOURCE_MODEL


def test_unsupported_class(artifacts):
    img = colourful()
    d = gk_with(artifacts, [0, 0, 0, 6]).classify(img, compute_metrics(img))
    assert d.rejection.code == rm.UNSUPPORTED_MODALITY and d.disease_key is None


def test_low_confidence_is_uncertain(artifacts):
    img = xray_like()
    d = gk_with(artifacts, [1.0, 0.9, 0, 0]).classify(img, compute_metrics(img))
    assert d.rejection.code == rm.UNCERTAIN_MODALITY and d.modality is None


def test_manual_selection_checked_against_model(artifacts):
    img = xray_like()
    d = gk_with(artifacts, [6, 0, 0, 0]).classify(img, compute_metrics(img), manual_modality="fundus")
    assert d.rejection.code == rm.MODALITY_MISMATCH
    d = gk_with(artifacts, [6, 0, 0, 0]).classify(img, compute_metrics(img), manual_modality="chest_xray")
    assert d.rejection is None and d.source == "manual_selection+trained_gatekeeper"


def test_without_model_auto_routing_is_disabled(artifacts):
    img = xray_like()
    d = Gatekeeper(ModelRegistry(models_dir=artifacts)).classify(img, compute_metrics(img))
    assert d.rejection.code == rm.UNCERTAIN_MODALITY and d.modality is None and d.hint


def test_without_model_manual_consistency_checks(artifacts):
    gk = Gatekeeper(ModelRegistry(models_dir=artifacts))
    c = colourful()
    assert gk.classify(c, compute_metrics(c), "chest_xray").rejection.code == rm.MODALITY_MISMATCH
    x = xray_like()
    assert gk.classify(x, compute_metrics(x), "histopathology").rejection.code == rm.MODALITY_MISMATCH
    h = histo_like()
    d = gk.classify(h, compute_metrics(h), "histopathology")
    assert d.rejection is None and "not verified" in d.note


def test_duplicate_policy():
    info = DuplicateInfo(True, same_patient=False, previous_result_id=7, previous_at="2026-01-01")
    rej = duplicate_rejection(info, allow_rescreen=False)
    assert rej.code == rm.DUPLICATE_IMAGE and "#7" not in rej.explanation()   # other patient's ID not revealed
    assert "same patient" in rej.explanation()
    assert duplicate_rejection(info, allow_rescreen=True) is None
    assert duplicate_rejection(DuplicateInfo(False), allow_rescreen=False) is None


def test_confident_softmax_on_non_medical_image_is_still_rejected(artifacts):
    """3-class gatekeepers (no 'other') always pick a class; consistency checks must catch it."""
    from tests.conftest import gatekeeper_metadata
    md = gatekeeper_metadata(class_labels=["chest_xray", "fundus", "histopathology"])
    install_model(artifacts, "gatekeeper", logit=np.array([6, 0, 0], dtype=np.float32), metadata=md, n_out=3)
    img = colourful()
    d = Gatekeeper(ModelRegistry(models_dir=artifacts)).classify(img, compute_metrics(img))
    assert d.rejection.code == rm.UNSUPPORTED_MODALITY and d.modality is None


def test_metadata_routing_limits_are_applied(artifacts):
    from tests.conftest import gatekeeper_metadata
    install_model(artifacts, "gatekeeper", logit=np.array([2.5, 0, 0, 0], dtype=np.float32),
                  metadata=gatekeeper_metadata(routing={"min_confidence": 0.95}))
    img = xray_like()
    d = Gatekeeper(ModelRegistry(models_dir=artifacts)).classify(img, compute_metrics(img))
    assert d.confidence < 0.95 and d.rejection.code == rm.UNCERTAIN_MODALITY
