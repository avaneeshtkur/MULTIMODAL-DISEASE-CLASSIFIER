"""
external_adapter.py — convert an externally trained `trained_outputs/` folder
(e.g. the Gemini/Colab export) into MediScan's metadata schema v1.0.

The external metadata omits facts the app must not guess (resize/crop rule,
output activation, gatekeeper class order, hashes of model.onnx.data). Each
is either READ from the artifacts or DETERMINED from evidence, and the
evidence is written into metadata.json["conversion"]:

  * hashes        model.onnx / best_model.pth checked against the source manifest
  * architecture  checkpoint must load strictly into the expected torchvision model
  * output        read from the ONNX graph (1 logit → sigmoid; N logits → softmax)
  * weights       ONNX (incl. external .data) must reproduce the hash-verified
                  checkpoint on real images → verifies the unhashed .data file
  * preprocessing candidates are scored on the bundled labelled images; the
                  winner must beat the runner-up by a clear margin, else error
  * gatekeeper    class order = majority modality per output index, must be a
                  one-to-one mapping with ≥ 99 % agreement

Evidence uses the bundled datasets, which include training images, so the
numbers identify preprocessing only and are never reported as performance.
"""
from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

import config
from src.core.image_preprocessor import default_spec, preprocess
from src.utils.image_hash import sha256_file

EXTS = {".png", ".jpg", ".jpeg"}
MODALITY_OF = {"tb": "chest_xray", "dr": "fundus", "oral": "histopathology"}
POSITIVE = {"tb": "Tuberculosis", "dr": "DR", "oral": "OSCC"}
MIN_PREPROC_MARGIN = 0.01          # winner must beat runner-up accuracy by ≥ 1 point
MAX_ONNX_TORCH_DIFF = 1e-3
NO_OTHER_CLASS_MIN_CONFIDENCE = 0.95


class ConversionError(RuntimeError):
    pass


def candidate_specs(color: str) -> dict[str, dict]:
    a = default_spec(color)                                   # this repo's eval transform
    b = default_spec(color)
    b.update(resize=config.IMG_SIZE, resize_mode="exact", center_crop=None)
    return {"resize256_centercrop224": a, "resize224_exact": b}


def _labelled_images(key: str, stride: int = 1) -> list[tuple[Path, int]]:
    d = config.DISEASES[key]
    return [(p, i) for i, c in enumerate(d["classes"])
            for p in sorted((d["data_dir"] / c).glob("*"))[::stride] if p.suffix.lower() in EXTS]


def _logits(sess, name, img, spec):
    return np.asarray(sess.run(None, {name: preprocess(img, spec)})[0], dtype=np.float64).reshape(-1)


def _load_torch(key: str, ckpt: Path):
    import torch
    sd = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    if key == "gatekeeper":
        import torchvision
        n = sd["classifier.3.weight"].shape[0]
        model, arch = torchvision.models.mobilenet_v3_small(num_classes=n), "mobilenet_v3_small"
    else:
        from src.model import build_model
        model, arch = build_model(pretrained=False, num_outputs=1), "efficientnet_b0"
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        raise ConversionError(f"[{key}] checkpoint does not match {arch}: {str(e)[:200]}") from e
    return model.eval(), arch


def convert_model(src: Path, key: str, manifest: dict) -> tuple[dict, list[str]]:
    """Return (metadata, notes). Raises ConversionError if anything cannot be established."""
    import onnxruntime as ort
    import torch
    folder = src / "models" / key
    onnx_path, ckpt = folder / "model.onnx", folder / "best_model.pth"
    for f in (onnx_path, ckpt):
        if not f.exists():
            raise ConversionError(f"[{key}] {f.name} missing")
    m = manifest.get("models", {}).get(key, {})
    for f, field in ((onnx_path, "onnx_sha256"), (ckpt, "checkpoint_sha256")):
        if not m.get(field):
            raise ConversionError(f"[{key}] source manifest has no {field}")
        if sha256_file(f) != m[field].lower():
            raise ConversionError(f"[{key}] {f.name} SHA-256 does not match source manifest")
    src_md = json.loads((folder / "metadata.json").read_text()) if (folder / "metadata.json").exists() else None

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    n_out = out.shape[-1]
    model, arch = _load_torch(key, ckpt)
    color = "rgb"
    if src_md is not None:
        if src_md.get("input_color_order", "RGB").upper() != "RGB" or src_md.get("input_channels", 3) != 3:
            raise ConversionError(f"[{key}] unsupported input colour spec in source metadata")
        if list(src_md.get("input_size", [224, 224])) != [config.IMG_SIZE] * 2:
            raise ConversionError(f"[{key}] unexpected input_size {src_md.get('input_size')}")
        norm = (src_md.get("normalization_mean"), src_md.get("normalization_std"))
        if list(norm[0] or []) != config.MEAN or list(norm[1] or []) != config.STD:
            raise ConversionError(f"[{key}] normalisation differs from ImageNet; not supported automatically")

    disease_keys = ["tb", "dr", "oral"] if key == "gatekeeper" else [key]
    stride = 3 if key == "gatekeeper" else 1
    data = [(p, lab, dk) for dk in disease_keys for p, lab in _labelled_images(dk, stride)]
    imgs = [(Image.open(p).convert("RGB") if color == "rgb" else Image.open(p), lab, dk) for p, lab, dk in data]

    # 1. ONNX (with external weights) must equal the hash-verified checkpoint
    probe_spec = candidate_specs(color)["resize224_exact"]
    diff = 0.0
    for img, _, _ in imgs[:: max(1, len(imgs) // 12)]:
        x = preprocess(img, probe_spec)
        with torch.no_grad():
            diff = max(diff, float(np.abs(model(torch.from_numpy(x)).numpy().reshape(-1)
                                          - _logits(sess, inp.name, img, probe_spec)).max()))
    if diff > MAX_ONNX_TORCH_DIFF:
        raise ConversionError(f"[{key}] ONNX output differs from checkpoint (max {diff:.2e})")
    notes = [f"ONNX (incl. external weights) matches hash-verified checkpoint: max |Δlogit| {diff:.1e}"]

    # 2. Preprocessing candidates scored on labelled images
    scores = {}
    outs_by_spec = {}
    for name, spec in candidate_specs(color).items():
        outs = np.array([_logits(sess, inp.name, img, spec) for img, _, _ in imgs])
        outs_by_spec[name] = outs
        if key == "gatekeeper":
            mods = np.array([MODALITY_OF[dk] for _, _, dk in imgs])
            pred = outs.argmax(1)
            mapping = {}
            for idx in range(n_out):
                sel = mods[pred == idx]
                mapping[idx] = max(set(sel), key=list(sel).count) if len(sel) else None
            agree = float(np.mean([mapping[i] == m for i, m in zip(pred, mods)]))
            scores[name] = (agree, mapping)
        else:
            thr = float(src_md["threshold"])
            y = np.array([lab for _, lab, _ in imgs])
            acc = float(((1 / (1 + np.exp(-outs[:, 0])) >= thr) == y).mean())
            scores[name] = (acc, None)
    ranked = sorted(scores.items(), key=lambda kv: kv[1][0], reverse=True)
    (best, (best_acc, mapping)), (second, (second_acc, _)) = ranked[0], ranked[1]
    tie_note = ""
    if best_acc - second_acc < MIN_PREPROC_MARGIN:
        if not (key == "gatekeeper" and best_acc >= 0.99 and second_acc >= 0.99):
            raise ConversionError(f"[{key}] preprocessing ambiguous: {best} {best_acc:.3f} vs {second} {second_acc:.3f}")
        # Both candidates route every labelled image correctly, so either is safe; prefer the more confident one.
        conf = {n: float(np.mean([np.max(np.exp(o - o.max()) / np.exp(o - o.max()).sum()) for o in outs_by_spec[n]]))
                for n in (best, second)}
        if conf[second] > conf[best]:
            best, second, mapping = second, best, scores[second][1]
            best_acc, second_acc = second_acc, best_acc
        tie_note = f"; tie broken by mean top confidence {conf[best]:.4f} vs {conf[second]:.4f}"
    spec = candidate_specs(color)[best]
    notes.append(f"preprocessing '{best}' selected{tie_note}: {best_acc:.3f} vs '{second}' {second_acc:.3f} "
                 f"({'routing agreement' if key == 'gatekeeper' else 'accuracy at shipped threshold'}, "
                 f"{len(imgs)} bundled images incl. training data — identification only, not performance)")

    md = {"schema_version": "1.0", "model_key": key, "architecture": arch, "preprocessing": spec,
          "conversion": {"source": "external trained_outputs (Gemini/Colab)", "source_metadata": src_md,
                         "converted_at": datetime.now(timezone.utc).isoformat(), "evidence": notes,
                         "preprocessing_scores": {k: v[0] for k, v in scores.items()}}}

    if key == "gatekeeper":
        labels = [str(mapping[i]) if mapping[i] is not None else None for i in range(n_out)]
        if None in labels or len(set(labels)) != n_out or best_acc < 0.99:
            raise ConversionError(f"[{key}] class order not determinable (mapping {mapping}, agreement {best_acc:.3f})")
        top = np.array([np.max(np.exp(o - o.max()) / np.exp(o - o.max()).sum()) for o in outs_by_spec[best]])
        md.update(model_version=f"gatekeeper-{m['onnx_sha256'][:8]}", task="multiclass", class_labels=labels,
                  output={"type": "logit", "activation": "softmax"})
        notes.append(f"class order {labels} determined from {len(imgs)} labelled images ({best_acc:.1%} agreement)")
        if "other" not in labels:
            frac = float(np.mean(top >= NO_OTHER_CLASS_MIN_CONFIDENCE))
            md["routing"] = {"min_confidence": NO_OTHER_CLASS_MIN_CONFIDENCE,
                             "evidence": f"no 'other' class; {frac:.1%} of valid images have top score ≥ "
                                         f"{NO_OTHER_CLASS_MIN_CONFIDENCE} (min {top.min():.3f})"}
            notes.append(md["routing"]["evidence"])
    else:
        if n_out != 1:
            raise ConversionError(f"[{key}] expected 1 output logit, ONNX has {n_out}")
        labels = list(src_md["classes"])
        if labels[1] != POSITIVE[key]:
            raise ConversionError(f"[{key}] expected positive class {POSITIVE[key]!r} at index 1, got {labels}")
        md.update(model_version=f"{key}-{m['onnx_sha256'][:8]}", task="binary", class_labels=labels,
                  positive_class=POSITIVE[key], output={"type": "logit", "activation": "sigmoid"},
                  threshold=float(src_md["threshold"]),
                  threshold_source=f"source metadata ({src_md.get('threshold_selection_method', 'unspecified')})",
                  uncertainty_margin_logit=config.DEFAULT_UNCERTAINTY_MARGIN_LOGIT, calibration=None)
        if key == "dr":
            md["input_domain"] = "ben_graham_filtered"
            notes.append("DR trained on the bundled Gaussian-filtered images → input_domain ben_graham_filtered")
        metrics = src / "results" / key / "metrics.json"
        if metrics.exists():
            md["metrics"] = json.loads(metrics.read_text())
            md["metrics_note"] = ("Reported by the external training pipeline on its own held-out split; "
                                  "not re-measured by MediScan; small test set, not clinical validation.")
    return md, notes


def convert(src: str | Path, out_zip: str | Path, keys: list[str] | None = None) -> tuple[Path, dict]:
    src, out_zip = Path(src), Path(out_zip)
    mpath = next(iter(sorted((src / "manifests").glob("*.json"))), None)
    if mpath is None:
        raise ConversionError("source has no manifests/*.json")
    manifest = json.loads(mpath.read_text())
    report, files = {}, {}
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key in keys or config.MODEL_KEYS:
            if not (src / "models" / key).is_dir():
                report[key] = {"ok": False, "error": "not present"}
                continue
            try:
                md, notes = convert_model(src, key, manifest)
            except ConversionError as e:
                report[key] = {"ok": False, "error": str(e)}
                continue
            folder = src / "models" / key
            payload = {n: folder / n for n in ("model.onnx", "model.onnx.data", "best_model.pth") if (folder / n).exists()}
            md["sha256"] = {n: sha256_file(p) for n, p in payload.items()}
            for n, p in payload.items():
                zf.write(p, f"trained_outputs/models/{key}/{n}")
                files[f"models/{key}/{n}"] = md["sha256"][n]
            mbytes = json.dumps(md, indent=2).encode()
            zf.writestr(f"trained_outputs/models/{key}/metadata.json", mbytes)
            from src.utils.image_hash import sha256_bytes
            files[f"models/{key}/metadata.json"] = sha256_bytes(mbytes)
            res = src / "results" / key
            if res.is_dir():
                for f in sorted(res.iterdir()):
                    if f.is_file():
                        zf.write(f, f"trained_outputs/results/{key}/{f.name}")
                        files[f"results/{key}/{f.name}"] = sha256_file(f)
            report[key] = {"ok": True, "notes": notes}
        zf.writestr("trained_outputs/manifests/manifest.json",
                    json.dumps({"files": files, "converted_from": str(src)}, indent=2))
    return out_zip, report
