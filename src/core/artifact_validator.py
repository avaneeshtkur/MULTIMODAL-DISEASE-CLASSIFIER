"""
artifact_validator.py — decide whether a model folder is safe to use.

A model folder (artifacts/models/<key>/) must contain:
    model.onnx       required — used for inference
    metadata.json    required — labels, threshold, preprocessing (see docs/MODEL_DOCUMENTATION.md)
    best_model.pth   optional — PyTorch weights, used only for Grad-CAM

Validation is strict: anything missing or inconsistent is reported as an
error and the model is marked unavailable. Nothing is guessed.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.core.image_preprocessor import PreprocessingError, validate_spec
from src.utils.image_hash import sha256_file

ONNX_NAME = "model.onnx"
CHECKPOINT_NAMES = ("best_model.pth", "best.pt", "model.pth")
METADATA_NAME = "metadata.json"
SUPPORTED_SCHEMA = {"1.0"}

# Labels the gatekeeper may use for each routable modality. Anything else is
# treated as "unsupported" and never routed to a specialist.
GATEKEEPER_LABEL_ALIASES = {
    "chest_xray": {"chest_xray", "chest-xray", "xray", "cxr", "chest x-ray", "chest_x_ray"},
    "fundus": {"fundus", "retina", "retinal_fundus", "fundus_photo"},
    "histopathology": {"histopathology", "histo", "oral_histopathology", "histopath"},
}

# Expected positive-class names per specialist (case-insensitive) — guards against swapped labels.
EXPECTED_POSITIVE = {
    "tb": {"tuberculosis", "tb", "abnormal_tb"},
    "dr": {"dr", "referable", "referable_dr", "diabetic_retinopathy", "dr_present"},
    "oral": {"oscc", "oral_cancer", "malignant", "cancer"},
}

FIELD_ALIASES = {"classes": "class_labels", "labels": "class_labels", "version": "model_version"}


@dataclass
class ValidationReport:
    key: str
    ok: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict | None = None
    onnx_path: Path | None = None
    checkpoint_path: Path | None = None
    onnx_sha256: str | None = None
    input_name: str | None = None

    def error(self, msg: str):
        self.errors.append(msg)

    def summary(self) -> str:
        if self.ok:
            return f"[{self.key}] OK" + (f" ({len(self.warnings)} warning(s))" if self.warnings else "")
        return f"[{self.key}] INVALID: " + "; ".join(self.errors)


def canonical_modality(label: str) -> str | None:
    l = label.strip().lower()
    for modality, aliases in GATEKEEPER_LABEL_ALIASES.items():
        if l in aliases:
            return modality
    return None


def normalise_metadata(raw: dict) -> dict:
    """Apply documented field aliases only (no value guessing)."""
    md = dict(raw)
    for alias, canon in FIELD_ALIASES.items():
        if alias in md and canon not in md:
            md[canon] = md.pop(alias)
    return md


def validate_metadata(md: dict, key: str, report: ValidationReport) -> None:
    sv = str(md.get("schema_version", ""))
    if sv not in SUPPORTED_SCHEMA:
        report.error(f"schema_version {sv!r} not supported (expected one of {sorted(SUPPORTED_SCHEMA)})")
    if md.get("model_key") not in (None, key):
        report.error(f"metadata model_key {md.get('model_key')!r} does not match folder {key!r}")
    if not md.get("model_version"):
        report.error("model_version missing")

    labels = md.get("class_labels")
    if not isinstance(labels, list) or len(labels) < 2 or not all(isinstance(x, str) for x in labels):
        report.error("class_labels must be a list of ≥2 strings")
        labels = None

    out = md.get("output")
    if not isinstance(out, dict) or out.get("activation") not in ("sigmoid", "softmax", "none"):
        report.error("output.activation must be 'sigmoid', 'softmax' or 'none'")
    elif out.get("type") not in ("logit", "probability"):
        report.error("output.type must be 'logit' or 'probability'")
    elif out["type"] == "probability" and out["activation"] != "none":
        report.error("output.type='probability' requires activation='none'")
    elif out["type"] == "logit" and out["activation"] == "none":
        report.error("output.type='logit' requires sigmoid or softmax activation")

    task = md.get("task")
    if key == "gatekeeper":
        if task != "multiclass":
            report.error("gatekeeper task must be 'multiclass'")
        if labels:
            mapped = {canonical_modality(l) for l in labels} - {None}
            missing = set(GATEKEEPER_LABEL_ALIASES) - mapped
            if missing:
                report.error(f"gatekeeper class_labels do not cover modalities {sorted(missing)}")
            if all(canonical_modality(l) for l in labels):
                report.warnings.append("gatekeeper has no 'unsupported/other' class; non-medical images rely on "
                                       "confidence limits and deterministic consistency checks")
        routing = md.get("routing")
        if routing is not None:
            ok = isinstance(routing, dict) and set(routing) <= {"min_confidence", "min_margin", "evidence"} and all(
                isinstance(routing.get(k, 0.5), (int, float)) and 0 < routing.get(k, 0.5) < 1
                for k in ("min_confidence", "min_margin"))
            if not ok:
                report.error("routing must be {'min_confidence': (0,1), 'min_margin': (0,1)}")
    else:
        if task != "binary":
            report.error(f"{key}: only binary specialist models are supported (task={task!r}). "
                         "Multi-grade outputs are not inferred from binary models or vice versa.")
        if labels and len(labels) != 2:
            report.error(f"{key}: binary model must have exactly 2 class_labels, got {len(labels)}")
        pos = md.get("positive_class")
        if labels and pos not in labels:
            report.error(f"positive_class {pos!r} not in class_labels")
        elif labels and pos and pos.lower() not in EXPECTED_POSITIVE.get(key, {pos.lower()}):
            report.error(f"positive_class {pos!r} is not a recognised positive label for {key} "
                         f"(expected one of {sorted(EXPECTED_POSITIVE[key])})")
        elif labels and isinstance(out, dict) and out.get("activation") in ("sigmoid", "none") \
                and labels.index(pos) != 1:
            report.error("single-output models must have positive_class at index 1 of class_labels")
        thr = md.get("threshold")
        if not isinstance(thr, (int, float)) or not (0.0 < float(thr) < 1.0) or math.isnan(float(thr)):
            report.error("threshold must be a number strictly between 0 and 1")
        margin = md.get("uncertainty_margin")
        if margin is not None and not (isinstance(margin, (int, float)) and 0 <= margin < 0.5):
            report.error("uncertainty_margin must be in [0, 0.5)")
        lmargin = md.get("uncertainty_margin_logit")
        if lmargin is not None and not (isinstance(lmargin, (int, float)) and 0 <= lmargin <= 3):
            report.error("uncertainty_margin_logit must be in [0, 3]")

    cal = md.get("calibration")
    if cal is not None:
        if not isinstance(cal, dict) or cal.get("method") != "temperature" \
                or not isinstance(cal.get("temperature"), (int, float)) or cal["temperature"] <= 0:
            report.error("calibration must be null or {'method': 'temperature', 'temperature': >0}")

    try:
        md["preprocessing"] = validate_spec(md.get("preprocessing"))
    except PreprocessingError as e:
        report.error(f"preprocessing: {e}")
        return
    if key == "tb" and md["preprocessing"]["color_mode"] != "grayscale_to_rgb":
        report.warnings.append("TB metadata declares RGB input; src/data.py trains X-rays as grayscale→RGB")


def _expected_output_width(md: dict) -> int:
    if md.get("task") == "multiclass":
        return len(md["class_labels"])
    return 2 if md["output"]["activation"] == "softmax" else 1


def validate_onnx(path: Path, md: dict, report: ValidationReport, run_test: bool = True) -> None:
    try:
        import onnx
        onnx.checker.check_model(str(path))
    except ImportError:
        report.warnings.append("onnx package not installed; structural check skipped")
    except Exception as e:  # noqa: BLE001 — any checker failure means the file is unusable
        report.error(f"ONNX structural check failed: {e}")
        return

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as e:  # noqa: BLE001
        report.error(f"ONNX Runtime could not load model: {e}")
        return

    inputs = sess.get_inputs()
    if len(inputs) != 1:
        report.error(f"model must have exactly 1 input, has {len(inputs)}")
        return
    inp = inputs[0]
    report.input_name = inp.name
    shape = list(inp.shape)
    size = md["preprocessing"]["input_size"]
    if len(shape) != 4:
        report.error(f"input must be 4-D NCHW, got shape {shape}")
        return
    for dim, want in zip(shape[1:], (3, size, size)):
        if isinstance(dim, int) and dim != want:
            report.error(f"input shape {shape} incompatible with preprocessing (expected [N, 3, {size}, {size}])")
            return
    if "float" not in inp.type:
        report.error(f"input dtype must be float32, got {inp.type}")
        return

    if run_test:
        try:
            out = sess.run(None, {inp.name: np.zeros((1, 3, size, size), dtype=np.float32)})[0]
        except Exception as e:  # noqa: BLE001
            report.error(f"test inference failed: {e}")
            return
        out = np.asarray(out).reshape(1, -1)
        want = _expected_output_width(md)
        if out.shape[1] != want:
            report.error(f"model outputs {out.shape[1]} value(s) per image; metadata implies {want}")
        elif not np.all(np.isfinite(out)):
            report.error("test inference produced non-finite outputs")


def find_checkpoint(folder: Path) -> Path | None:
    for n in CHECKPOINT_NAMES:
        if (folder / n).exists():
            return folder / n
    return None


def validate_model_dir(folder: Path, key: str, expected_hashes: dict[str, str] | None = None,
                       run_test: bool = True) -> ValidationReport:
    """Full validation of one model folder. expected_hashes maps filename → sha256."""
    report = ValidationReport(key=key)
    folder = Path(folder)
    if not folder.is_dir():
        report.error(f"folder {folder} does not exist")
        return report

    md_path, onnx_path = folder / METADATA_NAME, folder / ONNX_NAME
    if not onnx_path.exists():
        report.error(f"{ONNX_NAME} not found")
    if not md_path.exists():
        report.error(f"{METADATA_NAME} not found")
    if report.errors:
        return report

    try:
        md = normalise_metadata(json.loads(md_path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        report.error(f"metadata.json is not valid JSON: {e}")
        return report
    validate_metadata(md, key, report)
    report.metadata = md
    if report.errors:
        return report

    report.onnx_path = onnx_path
    report.onnx_sha256 = sha256_file(onnx_path)
    # Every model-graph file with a declared hash is verified (incl. external weights model.onnx.data).
    for source, table in (("metadata", md.get("sha256") or {}), ("manifest", expected_hashes or {})):
        for name, want in table.items():
            if name in CHECKPOINT_NAMES or not want:
                continue
            f = folder / name
            if not f.is_file():
                report.error(f"{name} listed in {source} but missing")
            elif (report.onnx_sha256 if name == ONNX_NAME else sha256_file(f)) != want.lower():
                report.error(f"{name} SHA-256 mismatch against {source}")
    data_file = folder / (ONNX_NAME + ".data")
    if data_file.exists() and (ONNX_NAME + ".data") not in (md.get("sha256") or {}) | (expected_hashes or {}):
        report.warnings.append("external weights model.onnx.data have no declared SHA-256")
    if report.errors:
        return report

    ckpt = find_checkpoint(folder)
    if ckpt is not None:
        want = (expected_hashes or {}).get(ckpt.name) or (md.get("sha256") or {}).get(ckpt.name)
        if want and want.lower() != sha256_file(ckpt):
            report.warnings.append(f"{ckpt.name} SHA-256 mismatch — checkpoint ignored (Grad-CAM disabled)")
            ckpt = None
    report.checkpoint_path = ckpt

    validate_onnx(onnx_path, md, report, run_test=run_test)
    report.ok = not report.errors
    return report
