"""
train.py
========
Training loop with validation-based early stopping and checkpointing.

HARD RULE enforced here: this module only ever builds the *train* and
*validation* loaders.  The test split is never touched during training,
hyper-parameter choice, or model selection - it is only read by ``evaluate.py``
for the single final evaluation of each trained model.

Runnable stand-alone (one experiment / one seed at a time)::

    python train.py --exp exp2_modelB_aug --seed 42
    python train.py --all            # every experiment x every seed, sequential
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config as C
from data import build_dataloaders, class_weights_from_train, load_splits
from models import (build_model, freeze_backbone, model_size_mb, trainable_parameters,
                    unfreeze_all)
from utils import apply_plot_style, human_hms, set_seed, timer

try:
    from sklearn.metrics import f1_score
except ImportError:  # pragma: no cover
    f1_score = None


# =========================================================================== #
# One epoch                                                                   #
# =========================================================================== #
def run_epoch(model, loader, criterion, optimizer, device, train: bool,
              scaler=None) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Train or evaluate for one epoch; returns (loss, accuracy, probs, labels)."""
    model.train(train)
    # ``model.train(True)`` re-enables BatchNorm training mode, which would undo
    # the frozen-BN statistics of Model B stage 1.  Restore eval mode for BN
    # whenever the backbone is flagged as frozen.
    if train and getattr(model, "_freeze_bn", False):
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    total_loss, total_n = 0.0, 0
    probs_all, labels_all = [], []

    use_amp = scaler is not None and device.type == "cuda"
    grad_ctx = torch.enable_grad() if train else torch.no_grad()

    with grad_ctx:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(x)
                    loss = criterion(logits, y)
            else:
                logits = model(x)
                loss = criterion(logits, y)

            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += float(loss.detach()) * y.size(0)
            total_n += y.size(0)
            probs_all.append(torch.softmax(logits.detach().float(), dim=1).cpu().numpy())
            labels_all.append(y.detach().cpu().numpy())

    probs = np.concatenate(probs_all) if probs_all else np.zeros((0, C.NUM_CLASSES), np.float32)
    labels = np.concatenate(labels_all) if labels_all else np.zeros((0,), np.int64)
    acc = float((probs.argmax(1) == labels).mean()) if len(labels) else 0.0
    return total_loss / max(total_n, 1), acc, probs, labels


def macro_f1(labels: np.ndarray, probs: np.ndarray) -> float:
    preds = probs.argmax(1)
    return float(f1_score(labels, preds, average="macro", zero_division=0))


# =========================================================================== #
# Schedule helper                                                             #
# =========================================================================== #
def _build_stages(model_key: str, epochs_scale: float) -> list[dict]:
    """Describe the training stages for a model as a list of dicts."""
    if model_key == "modelA":
        return [{
            "name": "scratch",
            "epochs": max(1, int(round(C.EPOCHS_MODEL_A * epochs_scale))),
            "lr": C.LR_MODEL_A,
            "eta_min": 1e-5,
            "freeze": False,
        }]
    return [
        {
            "name": "stage1_head",
            "epochs": max(1, int(round(C.EPOCHS_B_STAGE1 * epochs_scale))),
            "lr": C.LR_HEAD,
            "eta_min": C.LR_HEAD,
            "freeze": True,
        },
        {
            "name": "stage2_finetune",
            "epochs": max(1, int(round(C.EPOCHS_B_STAGE2 * epochs_scale))),
            "lr": C.LR_FINETUNE,
            "eta_min": C.LR_FINETUNE_MIN,
            "freeze": False,
        },
    ]


# =========================================================================== #
# Full experiment                                                             #
# =========================================================================== #
def train_experiment(exp_id: str, seed: int, device=None,
                     epochs_scale: float = 1.0, patience: int | None = None,
                     num_workers: int | None = None, verbose: bool = True) -> dict:
    """Train one (experiment, seed) pair and return a summary dict.

    Model selection uses VALIDATION macro-F1 only; the best checkpoint is saved
    to ``checkpoints/<exp>_seed<seed>_best.pt``.
    """
    if exp_id not in C.EXPERIMENTS:
        raise KeyError(f"unknown experiment {exp_id}; known: {list(C.EXPERIMENTS)}")
    spec = C.EXPERIMENTS[exp_id]
    model_key, augment = spec["model"], spec["augment"]
    device = device or C.get_device()
    patience = C.EARLY_STOP_PATIENCE if patience is None else patience

    set_seed(seed)                      # CPU/Python/NumPy/torch + cudnn determinism

    df = load_splits()
    # ``include_test=False`` -> the test split is never referenced during
    # training (not even wrapped in a Dataset).  The HARD RULE holds structurally.
    train_loader, val_loader, _ = build_dataloaders(
        df, augment=augment, seed=seed, num_workers=num_workers, include_test=False)

    weights = torch.tensor(class_weights_from_train(df), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = build_model(model_key, pretrained=(model_key == "modelB")).to(device)
    total_p, train_p = trainable_parameters(model)

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history: list[dict] = []
    best = {"val_macro_f1": -1.0, "epoch": -1, "val_acc": 0.0, "val_loss": float("inf")}
    ckpt_path = C.CKPT_DIR / f"{exp_id}_seed{seed}_best.pt"

    stages = _build_stages(model_key, epochs_scale)
    global_epoch = 0
    epochs_without_improvement = 0
    stop = False
    t_start = time.perf_counter()

    if verbose:
        print(f"\n{'=' * 86}\n[train] {exp_id} | seed={seed} | model={model_key} | "
              f"augment={augment} | device={device.type}\n"
              f"[train] params total={total_p:,} trainable={train_p:,} "
              f"size={model_size_mb(model):.2f} MB\n"
              f"[train] stages={[(s['name'], s['epochs'], s['lr']) for s in stages]}\n"
              f"{'=' * 86}")

    for stage in stages:
        if stop:
            break
        # A new stage is a new optimisation regime (frozen→full, different LR),
        # so the patience counter restarts at the stage boundary.  The *best*
        # checkpoint still persists across stages, so nothing good is lost.
        epochs_without_improvement = 0
        if stage["freeze"]:
            freeze_backbone(model)
        else:
            unfreeze_all(model)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=stage["lr"], weight_decay=C.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, stage["epochs"]), eta_min=stage["eta_min"])

        for _ in range(stage["epochs"]):
            if stop:
                break
            global_epoch += 1
            ep_start = time.perf_counter()

            tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion,
                                              optimizer, device, train=True, scaler=scaler)
            va_loss, va_acc, va_probs, va_labels = run_epoch(
                model, val_loader, criterion, None, device, train=False, scaler=None)
            va_f1 = macro_f1(va_labels, va_probs)
            scheduler.step()
            ep_sec = time.perf_counter() - ep_start

            row = {
                "exp": exp_id, "seed": seed, "epoch": global_epoch, "stage": stage["name"],
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": tr_loss, "train_acc": tr_acc,
                "val_loss": va_loss, "val_acc": va_acc, "val_macro_f1": va_f1,
                "epoch_seconds": ep_sec,
            }
            history.append(row)

            improved = va_f1 > best["val_macro_f1"] + C.EARLY_STOP_MIN_DELTA
            if improved:
                best = {"val_macro_f1": va_f1, "epoch": global_epoch,
                        "val_acc": va_acc, "val_loss": va_loss}
                torch.save({
                    "model_state": model.state_dict(),
                    "exp": exp_id, "seed": seed, "model_key": model_key,
                    "augment": augment, "epoch": global_epoch,
                    "val_macro_f1": va_f1, "val_acc": va_acc,
                    "classes": C.CLASSES, "img_size": C.IMG_SIZE,
                }, ckpt_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if verbose:
                print(f"[train] {exp_id} s{seed} epoch {global_epoch:3d} "
                      f"[{stage['name']:<15s}] lr={row['lr']:.2e} "
                      f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
                      f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} "
                      f"val_f1={va_f1:.4f}{'  *best' if improved else ''} "
                      f"({ep_sec:.1f}s)")

            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"[train] early stopping after {patience} epochs without "
                          f"validation macro-F1 improvement (best epoch {best['epoch']})")
                stop = True
                break

    total_seconds = time.perf_counter() - t_start

    hist_df = pd.DataFrame(history)
    hist_path = C.LOG_DIR / f"{exp_id}_seed{seed}_history.csv"
    hist_df.to_csv(hist_path, index=False)

    meta = {
        "exp": exp_id,
        "seed": seed,
        "model_key": model_key,
        "augment": augment,
        "label": spec["label"],
        "device": device.type,
        "epochs_run": int(len(history)),
        "best_epoch": best["epoch"],
        "best_val_macro_f1": best["val_macro_f1"],
        "best_val_acc": best["val_acc"],
        "best_val_loss": best["val_loss"],
        "total_params": int(total_p),
        "trainable_params": int(train_p),
        "size_mb": round(model_size_mb(model), 3),
        "train_seconds": total_seconds,
        "train_time_hms": human_hms(total_seconds),
        "epoch_seconds_mean": float(hist_df["epoch_seconds"].mean()) if len(hist_df) else 0.0,
        "class_weights": [float(w) for w in class_weights_from_train(df)],
        "batch_size": C.BATCH_SIZE,
        "img_size": C.IMG_SIZE,
        "weight_decay": C.WEIGHT_DECAY,
        "early_stop_patience": patience,
        "epochs_scale": epochs_scale,
        "checkpoint": str(ckpt_path),
        "history_csv": str(hist_path),
        "n_train": int((df["split"] == "train").sum()),
        "n_val": int((df["split"] == "val").sum()),
    }
    meta_path = C.RESULTS_DIR / f"run_{exp_id}_seed{seed}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if verbose:
        print(f"[train] finished {exp_id} seed {seed}: best val macro-F1="
              f"{best['val_macro_f1']:.4f} @ epoch {best['epoch']} in "
              f"{human_hms(total_seconds)}")

    return meta


# =========================================================================== #
# Curves                                                                      #
# =========================================================================== #
def plot_training_curves(exp_id: str, seeds: list[int] | None = None,
                         out: Path | None = None) -> Path:
    """Train/val loss, accuracy and macro-F1 vs epoch (mean over seeds + band)."""
    import matplotlib.pyplot as plt

    seeds = seeds or C.SEEDS
    frames = [pd.read_csv(C.LOG_DIR / f"{exp_id}_seed{s}_history.csv")
              for s in seeds if (C.LOG_DIR / f"{exp_id}_seed{s}_history.csv").exists()]
    if not frames:
        raise FileNotFoundError(f"no history CSVs for {exp_id}")
    all_hist = pd.concat(frames, ignore_index=True)

    metrics = [("train_loss", "val_loss", "cross-entropy loss", "Loss"),
               ("train_acc", "val_acc", "accuracy", "Accuracy"),
               ("val_macro_f1", None, "macro-F1", "Macro-F1")]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    for ax, (tr_col, va_col, name, title) in zip(axes, metrics):
        if va_col is None:
            grp = all_hist.groupby("epoch")[tr_col].agg(["mean", "std"]).fillna(0.0)
            ax.plot(grp.index, grp["mean"], marker="o", ms=3, color="#1f77b4",
                    label=f"val {name}")
            ax.fill_between(grp.index, grp["mean"] - grp["std"], grp["mean"] + grp["std"],
                            alpha=0.18, color="#1f77b4")
        else:
            for col, style, lab in ((tr_col, "-", "train"), (va_col, "--", "val")):
                grp = all_hist.groupby("epoch")[col].agg(["mean", "std"]).fillna(0.0)
                ax.plot(grp.index, grp["mean"], style, marker="o", ms=3, label=f"{lab} {name}")
                ax.fill_between(grp.index, grp["mean"] - grp["std"], grp["mean"] + grp["std"],
                                alpha=0.15)
        ax.set_xlabel("epoch")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs epoch")
        ax.legend()
    fig.suptitle(f"{C.EXPERIMENTS[exp_id]['label']} - training curves "
                 f"(mean over seeds {seeds})", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return __import__("utils").save_figure(fig, out or (C.FIGURES_DIR / f"training_curves_{exp_id}.png"),
                                           C.FIGURE_DPI)


# =========================================================================== #
# CLI                                                                         #
# =========================================================================== #
def configure_threads(n: int | None = None) -> int:
    n = n or int(os.environ.get("AIDER_THREADS", 0)) or (os.cpu_count() or 1)
    torch.set_num_threads(max(1, n))
    return max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train one AIDER experiment/seed.")
    ap.add_argument("--exp", default=None, help="experiment id, e.g. exp2_modelB_aug")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--all", action="store_true", help="run every experiment x every seed")
    ap.add_argument("--epochs-scale", type=float, default=float(os.environ.get("AIDER_EPOCH_SCALE", 1.0)))
    ap.add_argument("--patience", type=int, default=C.EARLY_STOP_PATIENCE)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()

    apply_plot_style()
    nth = configure_threads(args.threads)
    print(f"[train] torch threads = {nth}")
    print(f"[train] {C.describe_device()}")

    jobs = ([(e, s) for e in C.EXPERIMENTS for s in C.SEEDS]
            if args.all else [(args.exp or C.MAIN_EXPERIMENT, args.seed)])

    for exp_id, seed in jobs:
        train_experiment(exp_id, seed, epochs_scale=args.epochs_scale,
                         patience=args.patience, num_workers=args.num_workers)

    for exp_id in sorted({e for e, _ in jobs}):
        try:
            plot_training_curves(exp_id)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
