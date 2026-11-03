"""
artifact_packager.py — build trained_outputs.zip from this repo's training outputs.

Input (written by src/train.py, src/evaluate.py, src/export_onnx.py):
    models/<key>/best.pt, models/<key>/<key>.onnx, models/<key>/threshold.json
    results/<key>/metrics.json, history.json, *.png

Output layout (what scripts/import_artifacts.py expects):
    trained_outputs/models/<key>/{model.onnx, best_model.pth, metadata.json}
    trained_outputs/results/<key>/...
    trained_outputs/manifests/manifest.json   (SHA-256 of every file)

Preprocessing in metadata.json is the eval transform of src/data.py — this
is only correct for models trained with this repository's pipeline.
"""
from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import config
from src.core.image_preprocessor import default_spec
from src.utils.image_hash import sha256_bytes, sha256_file

POSITIVE = {"tb": "Tuberculosis", "dr": "DR", "oral": "OSCC"}


def build_metadata(key: str, version: str, dr_input_domain: str | None = "ben_graham_filtered") -> dict:
    cfg = config.task_config(key)
    color = "grayscale_to_rgb" if cfg["image_mode"] == "grayscale" else "rgb"
    md = {
        "schema_version": "1.0",
        "model_key": key,
        "model_version": version,
        "architecture": "efficientnet_b0",
        "class_labels": list(cfg["classes"]),
        "preprocessing": default_spec(color, config.IMG_SIZE),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training": {"epochs_max": config.EPOCHS, "lr": config.LR, "batch_size": config.BATCH_SIZE,
                     "seed": config.SEED, "allow_hflip": cfg["allow_hflip"]},
    }
    if key == "gatekeeper":
        md.update(task="multiclass", output={"type": "logit", "activation": "softmax"})
    else:
        thr_file = cfg["threshold_file"]
        if not thr_file.exists():
            raise FileNotFoundError(f"{thr_file} missing — run `python -m src.evaluate --disease {key}` first")
        thr = json.loads(thr_file.read_text())
        md.update(task="binary", positive_class=POSITIVE[key],
                  output={"type": "logit", "activation": "sigmoid"},
                  threshold=float(thr["threshold"]),
                  threshold_source=f"validation split, target sensitivity {thr.get('target_sensitivity')}",
                  uncertainty_margin_logit=config.DEFAULT_UNCERTAINTY_MARGIN_LOGIT, calibration=None)
        if key == "dr" and dr_input_domain:
            md["input_domain"] = dr_input_domain
    metrics = cfg["results_dir"] / "metrics.json"
    if metrics.exists():
        md["metrics"] = json.loads(metrics.read_text())
        md["metrics_note"] = "Measured on this repository's held-out test split; small dataset, not clinical validation."
    return md


def package(out_zip: str | Path, version: str, keys: list[str] | None = None,
            dr_input_domain: str | None = "ben_graham_filtered") -> tuple[Path, list[str]]:
    out_zip = Path(out_zip)
    packed, files = [], {}
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key in keys or config.MODEL_KEYS:
            cfg = config.task_config(key)
            onnx, ckpt = cfg["onnx_file"], cfg["checkpoint"]
            if not onnx.exists():
                print(f"[{key}] skipped: {onnx} not found (run src.export_onnx)")
                continue
            md = build_metadata(key, f"{key}-{version}", dr_input_domain)
            md["sha256"] = {"model.onnx": sha256_file(onnx)}
            entries = {"model.onnx": onnx.read_bytes(), "metadata.json": json.dumps(md, indent=2).encode()}
            if ckpt.exists():
                md["sha256"]["best_model.pth"] = sha256_file(ckpt)
                entries["metadata.json"] = json.dumps(md, indent=2).encode()
                entries["best_model.pth"] = ckpt.read_bytes()
            for name, data in entries.items():
                rel = f"models/{key}/{name}"
                zf.writestr(f"trained_outputs/{rel}", data)
                files[rel] = sha256_bytes(data)
            res_dir = cfg["results_dir"]
            if res_dir.is_dir():
                for f in sorted(res_dir.iterdir()):
                    if f.is_file():
                        rel = f"results/{key}/{f.name}"
                        zf.write(f, f"trained_outputs/{rel}")
                        files[rel] = sha256_file(f)
            packed.append(key)
        manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "version": version, "files": files}
        zf.writestr("trained_outputs/manifests/manifest.json", json.dumps(manifest, indent=2))
    return out_zip, packed
