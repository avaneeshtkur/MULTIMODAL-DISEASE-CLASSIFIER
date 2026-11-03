# Model documentation

## Current state

**No trained model artifacts are installed in this repository.** Every metric is therefore *missing*.
The application, tests and demo run without them; see the README for the demo and simulation modes.

## Architecture

The same backbone is used for every model (`src/model.py::build_model`): torchvision **EfficientNet-B0**,
pretrained on ImageNet for training. The head is `Dropout(0.3) → Linear(1280, n)`.

| Model | n outputs | Activation | Loss | Threshold |
|---|---|---|---|---|
| TB / DR / oral | 1 logit | sigmoid | BCE with `pos_weight` | highest threshold with sensitivity ≥ 0.95 **on the validation split** (`src/evaluate.py`) |
| Gatekeeper | 4 logits | softmax | class-weighted cross-entropy | routing limits in `config.py` (0.80 confidence, 0.20 margin) |

Training settings (`config.py`): 224 px, batch size 32, up to 15 epochs, AdamW (lr 3e-4, wd 1e-4), cosine
schedule, early stopping on validation AUC (patience 4), and seed 42. Augmentation uses a random resized crop
(0.85–1.0), rotation ±10°, and brightness/contrast jitter 0.15. Horizontal flips are used for fundus and
histopathology only, never for chest X-rays.

## Artifact contract (`trained_outputs.zip`)

```
trained_outputs/
  models/<tb|dr|oral|gatekeeper>/
      model.onnx        required — single input [N,3,S,S] float32
      metadata.json     required — schema below
      best_model.pth    optional — state_dict for Grad-CAM (also accepts {"state_dict": ...} / {"model_state_dict": ...})
  results/<key>/...     optional — copied to artifacts/results/
  manifests/manifest.json   SHA-256 of every file: {"files": {"models/tb/model.onnx": "<sha256>", ...}}
```

`scripts/package_artifacts.py` writes exactly this from the repository's training outputs.

### `metadata.json` schema (v1.0)

```json
{
  "schema_version": "1.0",
  "model_key": "tb",
  "model_version": "tb-2026-11-02",
  "architecture": "efficientnet_b0",
  "task": "binary",
  "class_labels": ["Normal", "Tuberculosis"],
  "positive_class": "Tuberculosis",
  "output": {"type": "logit", "activation": "sigmoid"},
  "threshold": 0.41,
  "threshold_source": "validation split, target sensitivity 0.95",
  "uncertainty_margin": 0.05,
  "calibration": null,
  "input_domain": null,
  "preprocessing": {
    "color_mode": "grayscale_to_rgb",
    "resize": 256, "resize_mode": "shorter_side",
    "center_crop": 224, "input_size": 224,
    "interpolation": "bilinear", "scale": "0_1",
    "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "fundus_border_crop": false
  },
  "sha256": {"model.onnx": "...", "best_model.pth": "..."},
  "metrics": { "...": "as written by src/evaluate.py" }
}
```

Validation rules (anything else → model `invalid`, with the reason shown in the UI and the importer):

- `schema_version` must be `"1.0"`. `model_key`, if present, must match the folder.
- Specialists: `task: "binary"`, two labels, and `positive_class` must be a recognised positive name
  (TB: Tuberculosis/TB, DR: DR/Referable…, oral: OSCC/…). For single-output models it must be at index 1.
  Multi-grade DR models are **not** accepted. The app never turns a binary output into grades, or the reverse.
- `threshold` must be strictly between 0 and 1. `uncertainty_margin` must be in [0, 0.5).
- `output.type` is `logit` (with `sigmoid`/`softmax`) or `probability` (with `none`).
- `calibration` is null or `{"method": "temperature", "temperature": T > 0}`. The threshold is compared with
  the score *after* calibration.
- `preprocessing`: all required keys are present, unknown keys are rejected, and the pipeline must end at
  `input_size`, which must match the ONNX input shape.
- Gatekeeper: `task: "multiclass"`, softmax, and labels must cover `chest_xray`, `fundus` and
  `histopathology` (documented aliases accepted). Any other label is treated as *unsupported*.
- `input_domain: "ben_graham_filtered"` (DR) makes raw fundus photographs with a dark surround return
  `INCONCLUSIVE / OUT_OF_DISTRIBUTION`.
- Documented field aliases only: `classes`/`labels` → `class_labels`, `version` → `model_version`.

## Importing

```bash
python scripts/import_artifacts.py --archive trained_outputs.zip --dry-run
python scripts/import_artifacts.py --archive trained_outputs.zip [--only tb,dr] [--allow-missing-manifest]
```

Safety checks:

- Rejects absolute or Windows paths, `..`, symlinks, encrypted members, more than 5000 members,
  more than 4 GB expanded, and suspicious compression ratios.
- Rejects an archive with no manifest unless explicitly allowed. Rejects files missing from the manifest,
  or files the manifest does not cover.
- Installs each valid model by atomic rename. The previous version goes to `artifacts/.backup/<timestamp>/`.
- Records installed file hashes in `artifacts/manifests/installed_manifest.json`.

## Grad-CAM

Grad-CAM uses the last convolutional block (`model.features[-1]`) and targets the positive-class logit.
It is shown only if the architecture is `efficientnet_b0`, a hash-valid checkpoint is present, and the
checkpoint's score matches the ONNX score within 0.02. There is **no ONNX-only fallback**, because
gradient-based maps need the PyTorch graph. A heatmap shows where the model looked, not whether it is right.

## Metrics

| | TB | DR | Oral | Gatekeeper |
|---|---|---|---|---|
| Measured test metrics | **missing** | **missing** | **missing** | **missing** |
| Calibration | none | none | none | none |
| External validation | none | none | none | none |

After import, metrics shipped inside `metadata.json` appear verbatim on the *About & models* tab, labelled
as reported by the training pipeline.
