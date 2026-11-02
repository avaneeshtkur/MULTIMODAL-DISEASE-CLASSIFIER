"""
config.py — shared settings for the multi-disease medical image classifier.

Three separate binary models, one per disease. Each has its own data folder,
class names, and augmentation rules. Everything else (architecture, image size,
training hyper-parameters) is shared so we can reuse the same training loop.
"""
from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────
SEED = 42

# ── Shared training hyper-parameters ─────────────────────────────
IMG_SIZE = 224          # EfficientNet-B0 native input
BATCH_SIZE = 32
EPOCHS = 15
LR = 3e-4              # AdamW learning rate
WEIGHT_DECAY = 1e-4
PATIENCE = 4            # early-stopping patience (epochs without val-AUC improvement)
VAL_FRAC = 0.15
TEST_FRAC = 0.15
TARGET_SENSITIVITY = 0.95   # screening threshold: catch ≥95% of positives

# ── ImageNet normalisation (used by all EfficientNet models) ─────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ── Project root (everything is relative to this) ───────────────
ROOT = Path(__file__).resolve().parent

# ── Per-disease configuration ────────────────────────────────────
# Each disease gets its own binary model.
# "allow_hflip" controls whether horizontal flip is used in training augmentation.
#   - Chest X-rays: NO flip (heart is on the left; flipping is anatomically wrong)
#   - Fundus photos: YES (eyes are roughly symmetrical)
#   - Histopathology: YES (no fixed orientation under a microscope)

DISEASES = {
    "tb": {
        "name": "Tuberculosis (Chest X-ray)",
        "classes": ["Normal", "Tuberculosis"],
        "data_dir": ROOT / "data" / "tb",
        "allow_hflip": False,
        "image_mode": "grayscale",   # X-rays are single-channel
    },
    "dr": {
        "name": "Diabetic Retinopathy (Fundus Photo)",
        "classes": ["No_DR", "DR"],
        "data_dir": ROOT / "data" / "dr",
        "allow_hflip": True,
        "image_mode": "rgb",
    },
    "oral": {
        "name": "Oral Cancer (Histopathology)",
        "classes": ["Normal", "OSCC"],
        "data_dir": ROOT / "data" / "oral",
        "allow_hflip": True,
        "image_mode": "rgb",
    },
}

# ── Per-disease derived paths ────────────────────────────────────
# models/<disease>/best.pt, models/<disease>/threshold.json, etc.
for key, d in DISEASES.items():
    d["model_dir"]      = ROOT / "models" / key
    d["checkpoint"]     = d["model_dir"] / "best.pt"
    d["threshold_file"] = d["model_dir"] / "threshold.json"
    d["onnx_file"]      = d["model_dir"] / f"{key}.onnx"
    d["results_dir"]    = ROOT / "results" / key
    d["splits_csv"]     = d["data_dir"] / "splits.csv"

# ── Backward-compatible shortcuts (used by old single-disease code) ──
# These point to TB by default so nothing breaks while we migrate.
CLASSES        = DISEASES["tb"]["classes"]
DATA_DIR       = DISEASES["tb"]["data_dir"]
MODEL_DIR      = DISEASES["tb"]["model_dir"]
CHECKPOINT     = DISEASES["tb"]["checkpoint"]
THRESHOLD_FILE = DISEASES["tb"]["threshold_file"]
ONNX_FILE      = DISEASES["tb"]["onnx_file"]
RESULTS_DIR    = DISEASES["tb"]["results_dir"]
SPLITS_CSV     = DISEASES["tb"]["splits_csv"]
