"""
Create data/samples/ for the demo.

  sample_chest_xray.png, sample_fundus.png, sample_histopathology.jpg
      copied from the TEST split of the bundled datasets (never used for training),
      so demo predictions are not on training images. Redistribution follows the
      original dataset licences — see docs/DATASET_DOCUMENTATION.md.
  invalid_blurry.png          SYNTHETIC smooth image (rejected as too blurry)
  unsupported_synthetic.png   SYNTHETIC non-medical shapes (unsupported / mismatched modality)

    python scripts/make_demo_assets.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def test_split_files(disease: str, label: int = 1) -> list[Path]:
    """Reproduce src/data.build_splits' test assignment without writing splits.csv."""
    d = config.DISEASES[disease]
    if d["splits_csv"].exists():
        from src.data import resolve_paths
        df = resolve_paths(pd.read_csv(d["splits_csv"]), d["data_dir"])
        return [Path(p) for p in df[(df.split == "test") & (df.label == label)]["path"]]
    rows = [{"path": str(p), "label": i} for i, c in enumerate(d["classes"])
            for p in sorted((d["data_dir"] / c).glob("*")) if p.suffix.lower() in EXTS]
    df = pd.DataFrame(rows)
    sub = df[df.label == label].sample(frac=1, random_state=config.SEED).reset_index(drop=True)
    n_test = max(1, int(len(sub) * config.TEST_FRAC))
    return [Path(p) for p in sub["path"][:n_test]]


def main():
    out = config.SAMPLES_DIR
    out.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for disease, name in (("tb", "sample_chest_xray"), ("dr", "sample_fundus"), ("oral", "sample_histopathology")):
        src = test_split_files(disease)[0]
        dst = out / f"{name}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        provenance[dst.name] = {"source": str(src.relative_to(config.ROOT)), "split": "test",
                                "class_folder": src.parent.name, "synthetic": False}

    # Synthetic blurry image: smooth gradient + blob, then heavy blur → no edges at all
    y, x = np.mgrid[0:384, 0:384]
    blob = 120 + 60 * np.exp(-(((x - 190) ** 2 + (y - 200) ** 2) / (2 * 90.0 ** 2)))
    img = Image.fromarray(blob.astype(np.uint8)).filter(ImageFilter.GaussianBlur(12))
    img.save(out / "invalid_blurry.png")
    provenance["invalid_blurry.png"] = {"synthetic": True, "purpose": "IMAGE_TOO_BLURRY rejection"}

    # Synthetic non-medical image: colourful geometric shapes
    rng = np.random.default_rng(config.SEED)
    canvas = Image.new("RGB", (512, 384), (250, 248, 240))
    draw = ImageDraw.Draw(canvas)
    for _ in range(40):
        x0, y0 = rng.integers(0, 460), rng.integers(0, 330)
        w, h = rng.integers(20, 120, size=2)
        colour = tuple(int(c) for c in rng.integers(0, 255, size=3))
        (draw.ellipse if rng.random() < 0.5 else draw.rectangle)([x0, y0, x0 + w, y0 + h], fill=colour)
    canvas.save(out / "unsupported_synthetic.png")
    provenance["unsupported_synthetic.png"] = {"synthetic": True, "purpose": "unsupported / mismatched modality"}

    (out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2))
    print("Wrote samples to", out)
    for k, v in provenance.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
