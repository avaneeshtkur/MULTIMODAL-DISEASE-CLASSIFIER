"""
prepare_gatekeeper.py — assemble data/gatekeeper/<class>/ for the modality model.

    python -m src.prepare_gatekeeper --other-dir /path/to/non_medical_images

chest_xray, fundus and histopathology are copied from data/tb, data/dr and
data/oral (both classes each). The "other" class must come from YOU: a folder
of unsupported images (natural photos, documents, other medical modalities
such as CT/MRI/skin photos). Without it the gatekeeper cannot learn to reject
unsupported inputs, and this script refuses to continue.

Note: the bundled DR images are Gaussian-filtered 224 px crops, so a
gatekeeper trained only on them may not recognise raw fundus photographs.
"""
import argparse
import shutil
from pathlib import Path

import config

SOURCES = {"chest_xray": "tb", "fundus": "dr", "histopathology": "oral"}
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def prepare(other_dir: Path, per_class_limit: int | None = None) -> dict:
    out = config.GATEKEEPER["data_dir"]
    other = sorted(p for p in Path(other_dir).rglob("*") if p.suffix.lower() in EXTS)
    if len(other) < 50:
        raise SystemExit(f"Need ≥50 'other' images in {other_dir}, found {len(other)}.")
    counts = {}
    for cls, disease in SOURCES.items():
        dst = out / cls
        dst.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in config.DISEASES[disease]["data_dir"].rglob("*") if p.suffix.lower() in EXTS)
        for f in files[:per_class_limit]:
            shutil.copy2(f, dst / f"{disease}_{f.parent.name}_{f.name}")
        counts[cls] = len(files[:per_class_limit])
    dst = out / "other"
    dst.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(other[:per_class_limit]):
        shutil.copy2(f, dst / f"other_{i:05d}{f.suffix.lower()}")
    counts["other"] = len(other[:per_class_limit])
    splits = config.GATEKEEPER["splits_csv"]
    if splits.exists():
        splits.unlink()      # force a fresh stratified split
    print("Gatekeeper dataset:", counts, "→", out)
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--other-dir", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None, help="max images per class")
    a = ap.parse_args()
    prepare(a.other_dir, a.limit)
