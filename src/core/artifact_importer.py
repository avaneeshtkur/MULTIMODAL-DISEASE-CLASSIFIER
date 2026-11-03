"""
artifact_importer.py — safely install trained_outputs.zip into artifacts/.

Steps
  1. Inspect the archive without extracting: reject absolute paths, '..',
     symlinks, oversized/zip-bomb members.
  2. Extract to artifacts/.staging-<ts>/ (same filesystem as the target).
  3. Verify SHA-256 checksums from the archive manifest.
  4. Validate each model folder (metadata schema, preprocessing, ONNX check,
     test inference) with src/core/artifact_validator.
  5. Install each *valid* model by atomic directory rename; the previous
     version is moved to artifacts/.backup/<ts>/. Invalid models leave the
     currently installed version untouched.
  6. Record installed file hashes in artifacts/manifests/installed_manifest.json
     (checked again by the registry at every app start).

Accepted manifest formats (in trained_outputs/manifests/ or the archive root):
  * JSON {"files": {"models/tb/model.onnx": "<sha256>", ...}}
  * JSON {"artifacts": [{"path": "...", "sha256": "..."}, ...]}
  * text  "<sha256>  <relative/path>" per line (sha256sum format)
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import config
from src.core.artifact_validator import ValidationReport, validate_model_dir
from src.utils.image_hash import sha256_file

MAX_MEMBERS = 5000
MAX_TOTAL_BYTES = 4 * 1024 ** 3
MAX_RATIO = 200          # uncompressed / compressed, for members > 1 MB
ROOT_NAME = "trained_outputs"


class ImportErrorUnsafe(RuntimeError):
    pass


@dataclass
class ImportResult:
    archive: Path
    dry_run: bool
    root: str = ""
    members: list[str] = field(default_factory=list)
    manifest_found: bool = False
    checksum_errors: dict[str, list[str]] = field(default_factory=dict)   # key → errors
    global_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reports: dict[str, ValidationReport] = field(default_factory=dict)
    installed: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    backup_dir: Path | None = None

    @property
    def ok(self) -> bool:
        return not self.global_errors and bool(self.installed or (self.dry_run and self.valid_keys))

    @property
    def valid_keys(self) -> list[str]:
        return [k for k, r in self.reports.items() if r.ok and not self.checksum_errors.get(k)]

    def lines(self) -> list[str]:
        out = [f"Archive: {self.archive}", f"Members: {len(self.members)}  root: {self.root or '(archive root)'}",
               f"Manifest: {'found' if self.manifest_found else 'NOT found'}"]
        out += [f"ERROR: {e}" for e in self.global_errors]
        out += [f"WARNING: {w}" for w in self.warnings]
        for key in config.MODEL_KEYS:
            if key in self.reports:
                r = self.reports[key]
                errs = self.checksum_errors.get(key, []) + r.errors
                state = "VALID" if not errs else "INVALID"
                out.append(f"  [{key}] {state}" + ("" if not errs else ": " + "; ".join(errs)))
                out += [f"      warning: {w}" for w in r.warnings]
            elif key in self.skipped:
                out.append(f"  [{key}] {self.skipped[key]}")
        if self.dry_run:
            out.append("Dry run: nothing installed.")
        else:
            out.append(f"Installed: {', '.join(self.installed) or 'none'}")
            if self.backup_dir:
                out.append(f"Previous versions backed up to: {self.backup_dir}")
        return out


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(info.external_attr >> 16)


def check_member_safety(infos: list[zipfile.ZipInfo]) -> None:
    if len(infos) > MAX_MEMBERS:
        raise ImportErrorUnsafe(f"archive has {len(infos)} members (limit {MAX_MEMBERS})")
    total = 0
    for i in infos:
        name = i.filename
        p = PurePosixPath(name)
        if name.startswith(("/", "\\")) or (len(name) > 1 and name[1] == ":") or "\\" in name:
            raise ImportErrorUnsafe(f"unsafe absolute or Windows path: {name!r}")
        if any(part == ".." for part in p.parts):
            raise ImportErrorUnsafe(f"path traversal in member: {name!r}")
        if _is_symlink(i):
            raise ImportErrorUnsafe(f"symlink member not allowed: {name!r}")
        if i.flag_bits & 0x1:
            raise ImportErrorUnsafe(f"encrypted member not supported: {name!r}")
        total += i.file_size
        if i.file_size > 1_000_000 and i.compress_size and i.file_size / i.compress_size > MAX_RATIO:
            raise ImportErrorUnsafe(f"suspicious compression ratio for {name!r}")
    if total > MAX_TOTAL_BYTES:
        raise ImportErrorUnsafe(f"archive expands to {total / 1e9:.1f} GB (limit {MAX_TOTAL_BYTES / 1e9:.0f} GB)")


def detect_root(names: list[str]) -> str:
    files = [n for n in names if not n.endswith("/") and not n.startswith("__MACOSX/")]
    if files and all(n.startswith(ROOT_NAME + "/") for n in files):
        return ROOT_NAME + "/"
    if any(n.startswith("models/") for n in files):
        return ""
    raise ImportErrorUnsafe(f"archive must contain '{ROOT_NAME}/models/...' or 'models/...'")


def parse_manifest(root: Path) -> dict[str, str] | None:
    """Return {relative/path: sha256} or None if no manifest file exists."""
    candidates = sorted((root / "manifests").glob("*")) if (root / "manifests").is_dir() else []
    candidates += [root / n for n in ("manifest.json", "SHA256SUMS", "sha256sums.txt") if (root / n).exists()]
    found: dict[str, str] = {}
    any_manifest = False
    for path in candidates:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix == ".json":
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            entries = {}
            if isinstance(data.get("files"), dict):
                entries = data["files"]
            elif isinstance(data.get("artifacts"), list):
                entries = {a.get("path"): a.get("sha256") for a in data["artifacts"] if isinstance(a, dict)}
            entries = {k: v for k, v in entries.items() if isinstance(k, str) and isinstance(v, str)}
            if entries:
                any_manifest = True
                found.update(entries)
        else:
            for line in text.splitlines():
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2 and len(parts[0]) == 64:
                    any_manifest = True
                    found[parts[1].lstrip("*").strip()] = parts[0]
    if not any_manifest:
        return None
    norm = {}
    for k, v in found.items():
        k = k.replace("\\", "/").lstrip("./")
        if k.startswith(ROOT_NAME + "/"):
            k = k[len(ROOT_NAME) + 1:]
        norm[k] = v.lower()
    return norm


def _model_hashes(manifest: dict[str, str], key: str) -> dict[str, str]:
    prefix = f"models/{key}/"
    return {k[len(prefix):]: v for k, v in manifest.items() if k.startswith(prefix) and "/" not in k[len(prefix):]}


def _record_installed(keys: list[str], archive_sha: str) -> None:
    path = config.INSTALLED_MANIFEST
    data = {"models": {}}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    data.setdefault("models", {})
    for key in keys:
        folder = config.MODELS_DIR / key
        data["models"][key] = {f.name: sha256_file(f) for f in sorted(folder.iterdir())
                               if f.is_file() and not f.name.startswith(".")}
    data["last_import"] = {"at": datetime.now(timezone.utc).isoformat(), "archive_sha256": archive_sha,
                           "models": keys}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def import_archive(archive: str | Path, dry_run: bool = False, only: list[str] | None = None,
                   allow_missing_manifest: bool = False) -> ImportResult:
    archive = Path(archive)
    res = ImportResult(archive, dry_run)
    if not archive.is_file():
        res.global_errors.append(f"archive not found: {archive}")
        return res
    keys = [k for k in config.MODEL_KEYS if not only or k in only]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    staging = config.ARTIFACTS_DIR / f".staging-{ts}"

    try:
        with zipfile.ZipFile(archive) as zf:
            infos = zf.infolist()
            res.members = [i.filename for i in infos]
            check_member_safety(infos)
            res.root = detect_root(res.members)
            staging.mkdir(parents=True)
            for info in infos:
                if info.is_dir() or not info.filename.startswith(res.root) or info.filename.startswith("__MACOSX/"):
                    continue
                rel = info.filename[len(res.root):]
                dest = (staging / rel).resolve()
                if staging.resolve() not in dest.parents:
                    raise ImportErrorUnsafe(f"member escapes staging dir: {info.filename!r}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
    except (zipfile.BadZipFile, ImportErrorUnsafe, OSError) as e:
        res.global_errors.append(f"{type(e).__name__}: {e}")
        shutil.rmtree(staging, ignore_errors=True)
        return res

    try:
        manifest = parse_manifest(staging)
        res.manifest_found = manifest is not None
        if manifest is None:
            msg = "no checksum manifest in archive"
            if not allow_missing_manifest:
                res.global_errors.append(msg + " (re-run with --allow-missing-manifest to import anyway)")
                return res
            res.warnings.append(msg + "; checksums NOT verified")
            manifest = {}
        else:
            for rel, want in manifest.items():
                f = staging / rel
                key = rel.split("/")[1] if rel.startswith("models/") and rel.count("/") >= 2 else None
                if not f.is_file():
                    msg = f"manifest lists missing file {rel}"
                elif sha256_file(f) != want:
                    msg = f"SHA-256 mismatch for {rel}"
                else:
                    continue
                if key:
                    res.checksum_errors.setdefault(key, []).append(msg)
                else:
                    res.warnings.append(msg)

        for key in keys:
            folder = staging / "models" / key
            if not folder.is_dir():
                res.skipped[key] = "not present in archive (installed version unchanged)"
                continue
            if manifest:
                unlisted = [f.name for f in folder.iterdir() if f.is_file() and f"models/{key}/{f.name}" not in manifest]
                if unlisted:
                    res.checksum_errors.setdefault(key, []).append(f"files not covered by manifest: {unlisted}")
            res.reports[key] = validate_model_dir(folder, key, expected_hashes=_model_hashes(manifest, key))

        if dry_run:
            return res

        valid = res.valid_keys
        if valid:
            res.backup_dir = config.ARTIFACTS_DIR / ".backup" / ts
        for key in valid:
            dest = config.MODELS_DIR / key
            src = staging / "models" / key
            backup = res.backup_dir / key
            moved = False
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(dest, backup)
                    moved = True
                os.replace(src, dest)
                res.installed.append(key)
            except OSError as e:
                if moved and not dest.exists():
                    os.replace(backup, dest)       # roll back
                res.reports[key].errors.append(f"install failed: {e}")

        if res.installed:
            for sub in ("results", "manifests"):
                src_dir = staging / sub
                if src_dir.is_dir():
                    target = config.ARTIFACT_RESULTS_DIR if sub == "results" else config.MANIFESTS_DIR / f"imported-{ts}"
                    shutil.copytree(src_dir, target, dirs_exist_ok=True)
            _record_installed(res.installed, sha256_file(archive))
        return res
    finally:
        shutil.rmtree(staging, ignore_errors=True)
