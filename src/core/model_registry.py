"""
model_registry.py — single source of truth for which models are usable.

On startup each folder in artifacts/models/<key>/ is validated
(artifact_validator). Valid models get one ONNX Runtime session each, created
lazily on first use and reused for every request. Invalid or missing models
are reported with their errors instead of crashing the app.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config
from src.core.artifact_validator import ValidationReport, validate_model_dir
from src.utils.logging_config import get_logger

log = get_logger("registry")

STATUS_AVAILABLE = "available"
STATUS_MISSING = "missing"
STATUS_INVALID = "invalid"


@dataclass
class ModelHandle:
    key: str
    status: str
    folder: Path
    report: ValidationReport | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    _session: object = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def available(self) -> bool:
        return self.status == STATUS_AVAILABLE

    @property
    def metadata(self) -> dict:
        return (self.report.metadata if self.report else None) or {}

    @property
    def version(self) -> str:
        return self.metadata.get("model_version", "—")

    @property
    def sha256(self) -> str | None:
        return self.report.onnx_sha256 if self.report else None

    @property
    def checkpoint_path(self) -> Path | None:
        return self.report.checkpoint_path if self.report else None

    def session(self):
        if not self.available:
            raise RuntimeError(f"model {self.key} is not available")
        with self._lock:
            if self._session is None:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                if config.ORT_INTRA_OP_THREADS:
                    opts.intra_op_num_threads = config.ORT_INTRA_OP_THREADS
                t0 = time.perf_counter()
                self._session = ort.InferenceSession(str(self.report.onnx_path), sess_options=opts,
                                                     providers=["CPUExecutionProvider"])
                log.info("Loaded ONNX session for %s in %.0f ms", self.key, (time.perf_counter() - t0) * 1000)
            return self._session

    def run(self, x: np.ndarray) -> np.ndarray:
        sess = self.session()
        return np.asarray(sess.run(None, {self.report.input_name: x})[0])

    def as_row(self) -> dict:
        md = self.metadata
        return {
            "key": self.key,
            "status": self.status,
            "version": self.version,
            "architecture": md.get("architecture", "—"),
            "classes": ", ".join(md.get("class_labels", [])) or "—",
            "threshold": md.get("threshold"),
            "gradcam": bool(self.checkpoint_path),
            "sha256": (self.sha256 or "")[:12],
            "errors": "; ".join(self.errors),
        }


def _installed_hashes() -> dict:
    p = config.INSTALLED_MANIFEST
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text()).get("models", {})
    except (json.JSONDecodeError, OSError):
        log.warning("installed_manifest.json unreadable; skipping manifest hash checks")
        return {}


class ModelRegistry:
    def __init__(self, models_dir: Path | None = None, keys: list[str] | None = None):
        self.models_dir = Path(models_dir or config.MODELS_DIR)
        self.keys = keys or list(config.MODEL_KEYS)
        self.handles: dict[str, ModelHandle] = {}
        self.refresh()

    def refresh(self) -> None:
        hashes = _installed_hashes() if self.models_dir == config.MODELS_DIR else {}
        for key in self.keys:
            folder = self.models_dir / key
            if not (folder / "model.onnx").exists() and not (folder / "metadata.json").exists():
                self.handles[key] = ModelHandle(key, STATUS_MISSING, folder,
                                                errors=["no model artifacts installed"])
                continue
            report = validate_model_dir(folder, key, expected_hashes=hashes.get(key))
            status = STATUS_AVAILABLE if report.ok else STATUS_INVALID
            self.handles[key] = ModelHandle(key, status, folder, report, list(report.errors), list(report.warnings))
            log.info("Model %s: %s", key, report.summary())

    def get(self, key: str) -> ModelHandle:
        return self.handles.get(key) or ModelHandle(key, STATUS_MISSING, self.models_dir / key,
                                                    errors=["unknown model key"])

    def is_available(self, key: str) -> bool:
        return self.get(key).available

    def status_rows(self) -> list[dict]:
        return [self.handles[k].as_row() for k in self.keys]

    def any_specialist_available(self) -> bool:
        return any(self.is_available(k) for k in config.SPECIALIST_KEYS)
