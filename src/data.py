"""
data.py — dataset loading, splitting, and augmentation for any of the 3 diseases.

Key idea: same pipeline, different settings per disease.
  - TB chest X-rays are grayscale and must NOT be horizontally flipped.
  - DR fundus photos and Oral histopathology slides are RGB and CAN be flipped.

Usage:
    from src.data import get_loaders
    loaders, df = get_loaders(disease="tb")   # or "dr" or "oral"
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config


def seed_everything(seed: int = config.SEED):
    """Lock every random number generator so results are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_image(path: str, mode: str = "grayscale") -> Image.Image:
    """
    Load an image and convert to RGB (EfficientNet always needs 3 channels).

    For grayscale (X-rays): open → convert to grayscale L → expand to 3-channel RGB.
    For rgb (fundus/histopath): open → ensure RGB.

    Why not just open as RGB directly for X-rays? Because some X-ray files are
    stored as palette or RGBA images, and converting through L first guarantees
    we get a clean single-channel image before tripling it.
    """
    img = Image.open(path)
    if mode == "grayscale":
        img = img.convert("L").convert("RGB")
    else:
        img = img.convert("RGB")
    return img


def get_transforms(train: bool, allow_hflip: bool = False):
    """
    Build the image transform pipeline.

    Training transforms add small random changes (augmentation) so the model
    sees slightly different versions of each image every epoch — this prevents
    overfitting on 150 images.

    Eval transforms are deterministic — same input always gives same output —
    so our metrics are reproducible.
    """
    if train:
        aug = [
            transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.85, 1.0)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ]
        if allow_hflip:
            aug.append(transforms.RandomHorizontalFlip())
        aug += [
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD),
        ]
        return transforms.Compose(aug)
    else:
        return transforms.Compose([
            transforms.Resize(config.IMG_SIZE + 32),   # resize a bit larger...
            transforms.CenterCrop(config.IMG_SIZE),     # ...then crop centre (no black borders)
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD),
        ])


# Keep a simple alias for evaluate.py / app.py that need eval transforms only
def eval_tfms(img: Image.Image) -> torch.Tensor:
    """Apply eval transforms to a single PIL image. Disease-agnostic (same for all)."""
    return get_transforms(train=False)(img)


def build_splits(disease: str = "tb") -> pd.DataFrame:
    """
    Scan the disease's data folder, assign each image to train/val/test.

    Folder structure expected:  data/<disease>/<ClassName>/image.png
    The split is STRATIFIED — each class gets the same train/val/test ratio —
    which matters a lot with only 150 images per class.

    The split is saved to data/<disease>/splits.csv so it's locked in:
    re-running this gives the same split (deterministic via SEED).
    """
    dcfg = config.task_config(disease)
    data_dir = dcfg["data_dir"]
    classes = dcfg["classes"]
    splits_csv = dcfg["splits_csv"]

    rows = []
    for cls_idx, cls_name in enumerate(classes):
        cls_dir = data_dir / cls_name
        if not cls_dir.exists():
            raise FileNotFoundError(f"Expected folder {cls_dir} — did you run the data download step?")
        for p in sorted(cls_dir.glob("*")):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
                rows.append({"path": p.relative_to(config.ROOT).as_posix(), "label": cls_idx, "class_name": cls_name})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found in {data_dir}. Check your folder structure.")

    # Stratified split: shuffle within each class, then assign
    seed_everything()
    parts = []
    for label in df["label"].unique():
        sub = df[df["label"] == label].sample(frac=1, random_state=config.SEED).reset_index(drop=True)
        n = len(sub)
        n_val = max(1, int(n * config.VAL_FRAC))
        n_test = max(1, int(n * config.TEST_FRAC))
        sub["split"] = "train"
        sub.loc[sub.index[:n_test], "split"] = "test"
        sub.loc[sub.index[n_test:n_test + n_val], "split"] = "val"
        parts.append(sub)

    df = pd.concat(parts, ignore_index=True)
    splits_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(splits_csv, index=False)
    print(f"[{disease}] Split: {dict(df['split'].value_counts())} — saved to {splits_csv}")
    return df


class MedicalImageDataset(Dataset):
    """A PyTorch Dataset that loads images on-the-fly with the right transforms."""

    def __init__(self, df: pd.DataFrame, transform, image_mode: str = "grayscale"):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.image_mode = image_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = load_image(row["path"], self.image_mode)
        return self.transform(img), torch.tensor(row["label"], dtype=torch.float32)


def resolve_paths(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """
    Make split paths usable on this machine without changing the split itself.

    Older splits.csv files contain absolute paths from another computer
    (e.g. C:\\Users\\...\\data\\tb\\Normal\\x.png). Relative paths are resolved
    against the project root; missing absolute paths are remapped to
    data/<disease>/<class_name>/<filename>.
    """
    def fix(row):
        raw = str(row["path"])
        p = Path(raw)
        if not p.is_absolute():
            p = config.ROOT / p
        if p.exists():
            return str(p)
        name = raw.replace("\\", "/").rsplit("/", 1)[-1]
        alt = data_dir / row["class_name"] / name
        if alt.exists():
            return str(alt)
        raise FileNotFoundError(f"Image in splits.csv not found: {raw}")
    df = df.copy()
    df["path"] = df.apply(fix, axis=1)
    return df


def get_loaders(disease: str = "tb", num_workers: int = 2):
    """
    One-call setup: build (or reload) splits, create DataLoaders for train/val/test.

    Returns:
        loaders: dict with keys "train", "val", "test"
        df: the full splits DataFrame (needed by evaluate.py)
    """
    dcfg = config.task_config(disease)
    splits_csv = dcfg["splits_csv"]

    if splits_csv.exists():
        df = pd.read_csv(splits_csv)
        print(f"[{disease}] Loaded existing split from {splits_csv}")
    else:
        df = build_splits(disease)
    df = resolve_paths(df, dcfg["data_dir"])

    allow_hflip = dcfg["allow_hflip"]
    image_mode = dcfg["image_mode"]

    loaders = {}
    for split in ("train", "val", "test"):
        is_train = (split == "train")
        ds = MedicalImageDataset(
            df[df["split"] == split],
            get_transforms(train=is_train, allow_hflip=allow_hflip if is_train else False),
            image_mode=image_mode,
        )
        loaders[split] = DataLoader(
            ds, batch_size=config.BATCH_SIZE,
            shuffle=is_train, num_workers=num_workers, pin_memory=True,
        )
    return loaders, df
