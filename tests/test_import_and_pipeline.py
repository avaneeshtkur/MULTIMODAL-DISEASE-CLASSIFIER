"""Artifact import (trained_outputs.zip) and the end-to-end screening workflow.

End-to-end tests use tiny synthetic ONNX models with FIXED outputs (they are
effectively mocked specialists). The statuses asserted below test software
routing and persistence only — they are not medical model performance."""
import json
import stat
import zipfile

import numpy as np
import pytest

import config
from src.core.artifact_importer import import_archive
from src.core.model_registry import ModelRegistry
from src.core.pipeline import report_from_outcome, report_from_record, run_screening
from src.reports.pdf_generator import generate_report
from src.utils.image_hash import sha256_bytes
from tests.conftest import (binary_metadata, colourful, histo_like, install_model, logit_for, make_onnx,
                            png_bytes, xray_like)


# ── import ───────────────────────────────────────────────────────
def build_zip(path, tmp_path, keys=("tb",), manifest=True, tamper=None, extra=None):
    src = tmp_path / "src_models"
    files = {}
    with zipfile.ZipFile(path, "w") as z:
        for key in keys:
            install_model(src, key, logit=logit_for(0.8))
            for f in sorted((src / key).iterdir()):
                data = f.read_bytes()
                rel = f"models/{key}/{f.name}"
                files[rel] = sha256_bytes(data)
                z.writestr(f"trained_outputs/{rel}", data + (b"X" if tamper == rel else b""))
        if manifest:
            z.writestr("trained_outputs/manifests/manifest.json", json.dumps({"files": files}))
        for name, data in (extra or {}).items():
            z.writestr(name, data)
    return path


def test_import_valid_archive(artifacts, tmp_path):
    z = build_zip(tmp_path / "trained_outputs.zip", tmp_path, keys=("tb", "oral"))
    dry = import_archive(z, dry_run=True)
    assert dry.ok and not (artifacts / "tb" / "model.onnx").exists()
    res = import_archive(z)
    assert res.ok and sorted(res.installed) == ["oral", "tb"]
    reg = ModelRegistry(models_dir=artifacts)
    assert reg.is_available("tb") and reg.is_available("oral") and not reg.is_available("dr")
    assert set(json.loads(config.INSTALLED_MANIFEST.read_text())["models"]) == {"tb", "oral"}


def test_checksum_mismatch_keeps_previous_model(artifacts, tmp_path):
    install_model(artifacts, "tb", metadata=binary_metadata("tb", model_version="previous"))
    z = build_zip(tmp_path / "t.zip", tmp_path, tamper="models/tb/model.onnx")
    res = import_archive(z)
    assert "tb" not in res.installed and res.checksum_errors["tb"]
    assert json.loads((artifacts / "tb" / "metadata.json").read_text())["model_version"] == "previous"


def test_invalid_metadata_not_installed(artifacts, tmp_path):
    src = tmp_path / "bad"
    install_model(src, "tb", metadata={**binary_metadata("tb"), "preprocessing": None})
    files = {f"models/tb/{f.name}": sha256_bytes(f.read_bytes()) for f in (src / "tb").iterdir()}
    with zipfile.ZipFile(tmp_path / "b.zip", "w") as zf:
        for f in (src / "tb").iterdir():
            zf.writestr(f"trained_outputs/models/tb/{f.name}", f.read_bytes())
        zf.writestr("trained_outputs/manifests/manifest.json", json.dumps({"files": files}))
    res = import_archive(tmp_path / "b.zip")
    assert not res.installed and not res.reports["tb"].ok


@pytest.mark.parametrize("evil", ["trained_outputs/../../evil.txt", "/etc/evil", "trained_outputs\\..\\evil"])
def test_path_traversal_rejected(artifacts, tmp_path, evil):
    z = build_zip(tmp_path / "e.zip", tmp_path, extra={evil: b"x"})
    res = import_archive(z)
    assert not res.ok and res.global_errors and not (tmp_path / "evil.txt").exists()


def test_symlink_member_rejected(artifacts, tmp_path):
    z = build_zip(tmp_path / "s.zip", tmp_path)
    with zipfile.ZipFile(z, "a") as zf:
        info = zipfile.ZipInfo("trained_outputs/models/tb/link")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "/etc/passwd")
    assert any("symlink" in e for e in import_archive(z).global_errors)


def test_missing_manifest_requires_flag(artifacts, tmp_path):
    z = build_zip(tmp_path / "m.zip", tmp_path, manifest=False)
    assert not import_archive(z).ok
    res = import_archive(z, allow_missing_manifest=True)
    assert res.ok and res.installed == ["tb"] and res.warnings


def test_missing_archive(artifacts, tmp_path):
    assert not import_archive(tmp_path / "nope.zip").ok


# ── end-to-end ───────────────────────────────────────────────────
def test_end_to_end_auto_routing(make_ctx, artifacts, tmp_path):
    install_model(artifacts, "gatekeeper", logit=np.array([6, 0, 0, 0], dtype=np.float32))
    install_model(artifacts, "tb", logit=logit_for(0.85))
    ctx = make_ctx()
    pid = ctx.patients.register("E2E-1", age=40, is_demo=True)
    data = png_bytes(xray_like())
    o = run_screening(ctx, data, "cxr.png", pid, "auto", {"tb": {"cough_2w": True}}, "en")
    assert o.stage == "screened" and o.result["prediction_status"] == "SCREENING_FLAGGED"
    assert o.modality.source == "trained_gatekeeper" and o.clinical["referral"]["priority"] == "PRIORITY"
    assert ctx.patients.history(pid)[0]["id"] == o.result_id

    # duplicate upload is rejected and logged; explicit re-screen allowed
    dup = run_screening(ctx, data, "cxr.png", pid, "auto")
    assert dup.rejection.code == "DUPLICATE_IMAGE"
    again = run_screening(ctx, data, "cxr.png", pid, "auto", allow_rescreen=True)
    assert again.stage == "screened" and again.duplicate_note

    for lang in ("en", "hi"):
        assert generate_report(report_from_outcome(o, lang), lang, tmp_path, o.explanation).exists()
        rec = ctx.screenings.get_result(o.result_id)
        assert generate_report(report_from_record(rec, lang), lang, tmp_path).exists()
    assert ctx.analytics.summary()["count_tb"] == 2
    assert {r["code"] for r in ctx.analytics.rejection_breakdown()} == {"DUPLICATE_IMAGE"}


def test_end_to_end_rejections_and_unavailable(make_ctx, artifacts):
    install_model(artifacts, "gatekeeper", logit=np.array([0, 0, 0, 6], dtype=np.float32))
    ctx = make_ctx()
    pid = ctx.patients.register()
    assert run_screening(ctx, b"garbage", "x.png", pid).rejection.code == "IMAGE_CORRUPTED"
    assert run_screening(ctx, png_bytes(colourful()), "c.png", pid).rejection.code == "UNSUPPORTED_MODALITY"
    # manual DR selection cannot override a gatekeeper "unsupported" decision
    assert run_screening(ctx, png_bytes(histo_like()), "h.png", pid, "dr").rejection.code == "UNSUPPORTED_MODALITY"
    assert ctx.analytics.summary()["rejections"] == 3


def test_end_to_end_manual_without_gatekeeper_and_missing_model(make_ctx, artifacts):
    install_model(artifacts, "oral", logit=logit_for(0.1))
    ctx = make_ctx()
    pid = ctx.patients.register()
    o = run_screening(ctx, png_bytes(histo_like()), "h.png", pid, "oral")
    assert o.result["prediction_status"] == "NO_FLAG_DETECTED" and "not verified" in o.modality.note
    assert run_screening(ctx, png_bytes(xray_like()), "x.png", pid, "tb").rejection.code == "MODEL_UNAVAILABLE"
    assert run_screening(ctx, png_bytes(xray_like(seed=5)), "x.png", pid, "auto").rejection.code == "UNCERTAIN_MODALITY"


def test_images_not_retained_by_default(make_ctx, artifacts, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RETAINED_IMAGES_DIR", tmp_path / "retained")
    install_model(artifacts, "oral")
    ctx = make_ctx()
    run_screening(ctx, png_bytes(histo_like()), "h.png", ctx.patients.register(), "oral")
    assert not (tmp_path / "retained").exists()


def test_patient_required(make_ctx):
    with pytest.raises(ValueError):
        run_screening(make_ctx(), png_bytes(xray_like()), "x.png", 999)


def test_packager_output_is_accepted_by_importer(artifacts, tmp_path, monkeypatch):
    """Contract test: scripts/package_artifacts.py → scripts/import_artifacts.py round trip."""
    from src.core.artifact_packager import package
    for key in ("tb", "dr", "oral"):
        d = dict(config.DISEASES[key])
        d.update(onnx_file=tmp_path / key / f"{key}.onnx", checkpoint=tmp_path / key / "best.pt",
                 threshold_file=tmp_path / key / "threshold.json", results_dir=tmp_path / "res" / key)
        make_onnx(d["onnx_file"], bias=logit_for(0.7))
        d["threshold_file"].write_text(json.dumps({"threshold": 0.42, "target_sensitivity": 0.95}))
        d["results_dir"].mkdir(parents=True)
        (d["results_dir"] / "metrics.json").write_text(json.dumps({"auc": "measured-elsewhere"}))
        monkeypatch.setitem(config.DISEASES, key, d)
    out, packed = package(tmp_path / "trained_outputs.zip", "test", keys=["tb", "dr", "oral"])
    assert packed == ["tb", "dr", "oral"]
    res = import_archive(out)
    assert sorted(res.installed) == ["dr", "oral", "tb"], res.lines()
    reg = ModelRegistry(models_dir=artifacts)
    md = reg.get("dr").metadata
    assert md["threshold"] == 0.42 and md["input_domain"] == "ben_graham_filtered"
    assert reg.get("tb").metadata["preprocessing"]["color_mode"] == "grayscale_to_rgb"
    assert (config.ARTIFACT_RESULTS_DIR / "tb" / "metrics.json").exists()
