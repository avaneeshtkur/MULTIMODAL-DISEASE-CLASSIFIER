"""Routing confidence and quality gates must never rewrite disease scores."""
from dataclasses import replace
from src.core.pipeline import run_screening
from src.core.inference_engine import predict_binary
from src.gatekeeper.classifier import ModalityDecision
from conftest import install_model, xray_like, png_bytes


def test_routing_confidence_does_not_scale_specialist(make_ctx, artifacts, monkeypatch):
    install_model(artifacts, 'tb', logit=-1.66)
    ctx = make_ctx()
    pid = ctx.patients.register('SEPARATION')
    image = xray_like()
    raw_result = predict_binary(ctx.registry.get('tb'), image, 'tb')
    decision = ModalityDecision('chest_xray', 'trained_gatekeeper', confidence=0.8)
    for confidence in [0.8, 0.9999]:
        monkeypatch.setattr(ctx.gatekeeper, 'classify', lambda *a, **k: replace(decision, confidence=confidence))
        out = run_screening(ctx, png_bytes(image), 'x.png', pid, allow_rescreen=True, explain=False)
        assert out.result['model_score'] == raw_result['model_score']
        assert out.result['model_sha256'] == ctx.registry.get('tb').sha256
