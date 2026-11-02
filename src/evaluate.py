"""
evaluate.py — honest scores on the untouched TEST set, per disease.

Usage:
    python -m src.evaluate              # evaluates TB (default)
    python -m src.evaluate --disease dr

1. Pick the decision THRESHOLD on the VALIDATION set (never on test).
2. Apply that threshold ONCE to the test set and report everything.

Outputs per disease: results/<disease>/metrics.json, confusion_matrix.png,
roc_curve.png, training_curve.png, gradcam_examples.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

import config
from src.data import get_loaders, seed_everything
from src.model import load_trained
from src.train import predict


def choose_threshold(y, p, target=config.TARGET_SENSITIVITY) -> float:
    """Highest threshold whose sensitivity (recall) is still >= target."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target)[0]
    t = float(thr[ok[0]]) if len(ok) else 0.5
    return float(min(max(t, 1e-6), 1.0))


def metrics_at(y, p, t) -> dict:
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    return {"threshold": t, "accuracy": (tp + tn) / len(y), "sensitivity": sens, "specificity": spec,
            "precision": prec, "f1": f1, "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def bootstrap_auc_ci(y, p, n=1000, seed=config.SEED):
    """95% confidence interval via 1000 bootstrap resamples."""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) > 1:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def plot_confusion(m, path, disease_name):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Pred Negative", "Pred Positive"])
    ax.set_yticks([0, 1], ["True Negative", "True Positive"])
    ax.set_title(f"Confusion matrix — {disease_name}")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_roc(y, p, auc, path, disease_name):
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Random guess")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("Sensitivity")
    ax.set_title(f"ROC curve — {disease_name}"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_history(path, disease):
    hist_file = config.DISEASES[disease]["results_dir"] / "history.json"
    if not hist_file.exists():
        return
    h = json.loads(hist_file.read_text())
    ep = [r["epoch"] for r in h]
    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    ax1.plot(ep, [r["train_loss"] for r in h], "o-", label="Train loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax2 = ax1.twinx(); ax2.plot(ep, [r["val_auc"] for r in h], "s-", color="tab:orange", label="Val AUC"); ax2.set_ylabel("AUC")
    fig.legend(loc="center right"); ax1.set_title("Training progress")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def gradcam_grid(model, df, probs, path, disease, n=4):
    """Heatmaps for the most confident correct positive predictions."""
    from src.data import eval_tfms, load_image
    from src.gradcam import GradCAM, overlay
    dcfg = config.DISEASES[disease]
    test = df[df["split"] == "test"].reset_index(drop=True).copy()
    test["prob"] = probs
    picks = test[test["label"] == 1].sort_values("prob", ascending=False).head(n)
    if picks.empty:
        return
    cam = GradCAM(model.cpu())
    fig, axes = plt.subplots(2, len(picks), figsize=(3 * len(picks), 6), squeeze=False)
    for j, (_, r) in enumerate(picks.iterrows()):
        img = load_image(r["path"], dcfg["image_mode"])
        p, heat = cam(eval_tfms(img).unsqueeze(0))
        arr = np.array(img.resize((384, 384)))
        axes[0, j].imshow(arr); axes[0, j].set_title(f"p={p:.2f}", fontsize=9)
        axes[1, j].imshow(overlay(arr, heat))
        for ax in axes[:, j]:
            ax.axis("off")
    fig.suptitle(f"Grad-CAM — {dcfg['name']} (red = evidence for disease)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def evaluate(disease: str = "tb", num_workers: int = 2):
    seed_everything()
    dcfg = config.DISEASES[disease]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating [{disease}] {dcfg['name']} on {device}")

    loaders, df = get_loaders(disease=disease, num_workers=num_workers)
    model = load_trained(device, disease=disease)

    yv, pv = predict(model, loaders["val"], device)
    t = choose_threshold(yv, pv)
    dcfg["threshold_file"].parent.mkdir(parents=True, exist_ok=True)
    dcfg["threshold_file"].write_text(json.dumps({"threshold": t, "target_sensitivity": config.TARGET_SENSITIVITY}))

    yt, pt = predict(model, loaders["test"], device)
    auc = roc_auc_score(yt, pt)
    lo, hi = bootstrap_auc_ci(yt, pt)
    tuned, default = metrics_at(yt, pt, t), metrics_at(yt, pt, 0.5)
    out = {"disease": disease, "name": dcfg["name"], "test_size": int(len(yt)),
           "auc": auc, "auc_95ci": [lo, hi], "tuned_threshold": tuned, "default_0.5": default}

    results_dir = dcfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    plot_confusion(tuned, results_dir / "confusion_matrix.png", dcfg["name"])
    plot_roc(yt, pt, auc, results_dir / "roc_curve.png", dcfg["name"])
    plot_history(results_dir / "training_curve.png", disease)
    gradcam_grid(model, df, pt, results_dir / "gradcam_examples.png", disease)

    print(f"[{disease}] Test AUC {auc:.4f} (95% CI {lo:.3f}–{hi:.3f})  |  threshold {t:.3f}")
    for k in ("accuracy", "sensitivity", "specificity", "precision", "f1"):
        print(f"  {k:12s} tuned {tuned[k]:.4f}   @0.5 {default[k]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", default="tb", choices=["tb", "dr", "oral"])
    args = parser.parse_args()
    evaluate(args.disease)
