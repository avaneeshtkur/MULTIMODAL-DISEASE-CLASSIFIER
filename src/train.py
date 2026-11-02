"""
train.py — training loop for any of the 3 diseases.

Usage:
    python -m src.train              # trains TB (default)
    python -m src.train --disease dr   # trains DR
    python -m src.train --disease oral

What happens during training (plain English):

    Load images, split into train/val/test.
    For each epoch (pass through all training images):
        a. Show the model batches of 32 images.
        b. Model guesses "disease" or "normal" for each.
        c. Compare guess to truth → compute loss (how wrong it was).
        d. Adjust model weights to reduce loss (backpropagation).
    After each epoch, check val-set AUC. Save the model if AUC improved.
    If AUC hasn't improved for 4 epochs, stop early (overfitting risk).
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import config
from src.data import get_loaders, seed_everything
from src.model import build_model


def predict(model: nn.Module, loader, device: str):
    """Run the model on a DataLoader. Return (true_labels, predicted_probabilities)."""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            logit = model(x.to(device)).squeeze(1)
            ps.append(torch.sigmoid(logit).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train(disease: str = "tb", num_workers: int = 2):
    seed_everything()
    dcfg = config.DISEASES[disease]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training [{disease}] {dcfg['name']} on {device}")

    loaders, df = get_loaders(disease=disease, num_workers=num_workers)
    model = build_model().to(device)

    # --- Class imbalance handling ---
    # pos_weight tells the loss function "positive cases (disease) matter more"
    # If we have 150 normal and 150 disease, pos_weight = 1.0 (balanced).
    # If imbalanced (e.g., 200 normal, 100 disease), pos_weight = 2.0,
    # so missing a disease case is penalised 2x more than a false alarm.
    train_labels = df[df["split"] == "train"]["label"].values
    n_pos = train_labels.sum()
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimiser = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config.EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_auc, patience_counter = 0.0, 0
    history = []

    dcfg["model_dir"].mkdir(parents=True, exist_ok=True)
    dcfg["results_dir"].mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.EPOCHS + 1):
        # --- Training phase ---
        model.train()
        losses = []
        t0 = time.time()
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                loss = criterion(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()

        # --- Validation phase ---
        yv, pv = predict(model, loaders["val"], device)
        val_auc = roc_auc_score(yv, pv) if len(set(yv)) > 1 else 0.0
        elapsed = time.time() - t0
        print(f"  [{disease}] Epoch {epoch:2d}/{config.EPOCHS} — "
              f"loss {np.mean(losses):.4f} — val AUC {val_auc:.4f} — {elapsed:.1f}s")

        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_auc": val_auc})

        # --- Save best model ---
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), dcfg["checkpoint"])
            print(f"    ✓ New best model saved (AUC {val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"    Early stopping — no improvement for {config.PATIENCE} epochs.")
                break

    # Save training history for plotting later
    (dcfg["results_dir"] / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[{disease}] Training complete. Best val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", default="tb", choices=["tb", "dr", "oral"])
    args = parser.parse_args()
    train(args.disease)
