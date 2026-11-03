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

# ── Gatekeeper (modality classifier) training config ─────────────
# Multi-class model that decides which specialist (if any) should see an image.
# data/gatekeeper/<class>/ must be populated (see src/prepare_gatekeeper.py);
# "other" needs non-medical / unsupported images supplied by you.
GATEKEEPER = {
    "name": "Gatekeeper (image modality)",
    "classes": ["chest_xray", "fundus", "histopathology", "other"],
    "data_dir": ROOT / "data" / "gatekeeper",
    "allow_hflip": True,
    "image_mode": "rgb",
    "model_dir": ROOT / "models" / "gatekeeper",
    "checkpoint": ROOT / "models" / "gatekeeper" / "best.pt",
    "threshold_file": ROOT / "models" / "gatekeeper" / "threshold.json",
    "onnx_file": ROOT / "models" / "gatekeeper" / "gatekeeper.onnx",
    "results_dir": ROOT / "results" / "gatekeeper",
    "splits_csv": ROOT / "data" / "gatekeeper" / "splits.csv",
}


def task_config(key: str) -> dict:
    """Training config for a disease key or 'gatekeeper'."""
    return GATEKEEPER if key == "gatekeeper" else DISEASES[key]


# ═════════════════════════════════════════════════════════════════
#  MediScan AI application settings
#  (everything below is used by app.py and src/core, src/gatekeeper, ...)
# ═════════════════════════════════════════════════════════════════
import os as _os

APP_NAME = "MediScan AI"
APP_VERSION = "1.0.0"

# Installed, validated model artifacts (populated by scripts/import_artifacts.py)
ARTIFACTS_DIR = Path(_os.environ.get("MEDISCAN_ARTIFACTS_DIR", ROOT / "artifacts"))
MODELS_DIR = ARTIFACTS_DIR / "models"
ARTIFACT_RESULTS_DIR = ARTIFACTS_DIR / "results"
MANIFESTS_DIR = ARTIFACTS_DIR / "manifests"
INSTALLED_MANIFEST = MANIFESTS_DIR / "installed_manifest.json"
MODEL_KEYS = ["tb", "dr", "oral", "gatekeeper"]
SPECIALIST_KEYS = ["tb", "dr", "oral"]

# Local storage
DATA_ROOT = ROOT / "data"
SAMPLES_DIR = DATA_ROOT / "samples"
LOG_DIR = Path(_os.environ.get("MEDISCAN_LOG_DIR", DATA_ROOT / "logs"))
DB_PATH = Path(_os.environ.get("MEDISCAN_DB", DATA_ROOT / "mediscan.db"))
REPORTS_DIR = Path(_os.environ.get("MEDISCAN_REPORTS_DIR", DATA_ROOT / "reports"))
RETAINED_IMAGES_DIR = DATA_ROOT / "retained_images"
FONTS_DIR = ROOT / "assets" / "fonts"

# Privacy: uploaded images are NOT kept after screening unless this is enabled.
RETAIN_IMAGES = _os.environ.get("MEDISCAN_RETAIN_IMAGES", "0") == "1"

# Simulated inference for UI rehearsal when no trained models are installed.
# Every simulated result is labelled SIMULATED in the UI, DB and PDF.
SIMULATION_MODE = _os.environ.get("MEDISCAN_SIMULATION", "0") == "1"

# Modality names used by the gatekeeper and routing
MODALITIES = {
    "chest_xray": "tb",
    "fundus": "dr",
    "histopathology": "oral",
}
UNSUPPORTED_MODALITY = "other"

# Gatekeeper decision rules (trained model)
GATEKEEPER_MIN_CONFIDENCE = 0.80   # top softmax score required to route
GATEKEEPER_MIN_MARGIN = 0.20       # top-1 minus top-2 score required to route

# ── Image quality thresholds (Stage A, deterministic) ────────────
# Calibrated on the bundled datasets so valid images pass. See docs/ARCHITECTURE.md.
QUALITY = {
    "max_pixels": 16_000_000,       # bound decoded memory before img.load()
    "allowed_extensions": [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"],
    "allowed_formats": ["PNG", "JPEG", "BMP", "TIFF", "WEBP", "MPO"],
    "max_file_mb": 25,
    "min_side_px": 128,            # DR training images are 224 px
    "max_aspect_ratio": 3.0,
    # Blur = variance of the 4-neighbour Laplacian on a grey copy (longest side ≤ 512 px).
    # Global floor is low because chest X-rays are naturally smooth (dataset min ≈ 5.0).
    "blur_min_global": 3.0,
    # Stricter per-modality floors applied after routing (dataset minima: fundus ≈ 473, histo ≈ 148).
    "blur_min_by_modality": {"chest_xray": 3.0, "fundus": 30.0, "histopathology": 30.0},
    # Exposure on grey percentiles (0–255). Dark backgrounds alone never trigger rejection.
    "underexposed_p99_max": 40,    # brightest 1% still below this → no usable signal
    "overexposed_p1_min": 235,     # darkest 1% above this → washed out
    "overexposed_white_frac": 0.85,
}

# Duplicate policy: "reject" (default) or "warn". Exact SHA-256 file matches only.
DUPLICATE_POLICY = _os.environ.get("MEDISCAN_DUPLICATE_POLICY", "reject")

# Uncertainty band around the threshold. Metadata may give "uncertainty_margin" (probability
# units) or "uncertainty_margin_logit". With neither, ±0.25 in LOGIT space is used, so the band
# scales with the threshold (t=0.5 → 0.44–0.56; t=0.01 → 0.008–0.013; t=0.93 → 0.91–0.94).
DEFAULT_UNCERTAINTY_MARGIN = 0.05
DEFAULT_UNCERTAINTY_MARGIN_LOGIT = 0.25

# ONNX Runtime CPU settings
ORT_INTRA_OP_THREADS = int(_os.environ.get("MEDISCAN_ORT_THREADS", "0"))  # 0 = ORT default
# Max |torch − onnx| score difference for Grad-CAM to be considered consistent
GRADCAM_MAX_SCORE_DIFF = 0.02

DISCLAIMER_KEY = "disclaimer"

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
