"""
evaluate.py
===========
Final TEST-set evaluation, metrics, plots, error analysis and Grad-CAM.

The test split is read exactly once per trained model, here.  Nothing in this
module feeds back into training or model selection.

Stand-alone::

    python evaluate.py                 # evaluate every trained checkpoint
    python evaluate.py --exp exp2_modelB_aug --seed 42
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import config as C
from data import build_transforms, class_weights_from_train, load_splits
from models import build_model, measure_inference_time, model_size_mb, trainable_parameters
from utils import apply_plot_style, save_figure, set_seed

try:
    from sklearn.metrics import (auc, classification_report, confusion_matrix,
                                 f1_score, precision_recall_fscore_support, roc_auc_score,
                                 roc_curve)
except ImportError:  # pragma: no cover
    pass


# =========================================================================== #
# Which (experiment, seed) pairs actually exist on disk                        #
# =========================================================================== #
# Discovering the seeds from disk (instead of trusting C.SEEDS) means that
# ``run_all.py --seeds ...`` and partially-completed runs are handled correctly,
# and evaluation never tries to open a checkpoint/metrics file that is absent.
_SEED_RE = re.compile(r"_seed(\d+)")


def _discover_seeds(exp_id: str, folder: Path, name_pattern: str) -> list[int]:
    """Sorted seed numbers for which a matching artefact exists in ``folder``."""
    seeds = []
    for p in folder.glob(name_pattern):
        m = _SEED_RE.search(p.name)
        if m:
            seeds.append(int(m.group(1)))
    return sorted(set(seeds))


def checkpoint_seeds(exp_id: str) -> list[int]:
    """Seeds with a trained checkpoint for ``exp_id``."""
    return _discover_seeds(exp_id, C.CKPT_DIR, f"{exp_id}_seed*_best.pt")


def metric_seeds(exp_id: str) -> list[int]:
    """Seeds with a saved metrics JSON for ``exp_id``."""
    return _discover_seeds(exp_id, C.RESULTS_DIR, f"metrics_{exp_id}_seed*.json")


# =========================================================================== #
# Loading + inference                                                         #
# =========================================================================== #
def load_checkpoint_model(ckpt_path: Path, device=None):
    device = device or C.get_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_key"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


class _NumpyListDataset(Dataset):
    def __init__(self, records, data_dir, transform):
        self.records, self.data_dir, self.transform = records, data_dir, transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        from PIL import Image
        with Image.open(self.data_dir / self.records[i][0]) as im:
            img = im.convert("RGB")
        return self.transform(img), self.records[i][1]


def build_test_loader(df: pd.DataFrame, batch_size: int = C.BATCH_SIZE,
                      num_workers: int | None = None) -> DataLoader:
    nw = C.NUM_WORKERS if num_workers is None else num_workers
    _, eval_tf = build_transforms(augment=False)
    sub = df[df["split"] == "test"]
    records = list(zip(sub["path"].tolist(), sub["label"].tolist()))
    ds = _NumpyListDataset(records, C.DATA_DIR, eval_tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=nw,
                      pin_memory=torch.cuda.is_available())


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (probabilities, labels, relative_paths) for a loader."""
    model.eval().to(device)
    probs, labels, paths = [], [], []
    idx = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        labels.append(y.numpy())
        idx += x.size(0)
    # relative paths in the same order DataLoader consumed them
    ds = loader.dataset
    paths = [r[0] for r in ds.records]
    return (np.concatenate(probs), np.concatenate(labels), paths)


# =========================================================================== #
# Metrics                                                                     #
# =========================================================================== #
def disaster_binary_view(labels: np.ndarray, preds: np.ndarray) -> dict:
    """Collapse the 4 disaster classes vs 'normal' and report the practical rates."""
    normal_idx = C.CLASSES.index(C.NORMAL_CLASS)
    y_dis = (labels != normal_idx).astype(int)     # 1 = disaster
    p_dis = (preds != normal_idx).astype(int)

    tp = int(((y_dis == 1) & (p_dis == 1)).sum())
    fn = int(((y_dis == 1) & (p_dis == 0)).sum())
    tn = int(((y_dis == 0) & (p_dis == 0)).sum())
    fp = int(((y_dis == 0) & (p_dis == 1)).sum())

    miss_rate = fn / (fn + tp) if (fn + tp) else float("nan")      # disaster -> normal
    false_alarm = fp / (fp + tn) if (fp + tn) else float("nan")    # normal -> disaster
    return {
        "true_disaster": fn + tp, "true_normal": fp + tn,
        "tp_disaster_correct": tp, "fn_disaster_missed": fn,
        "tn_normal_correct": tn, "fp_normal_false_alarm": fp,
        "miss_rate": miss_rate, "false_alarm_rate": false_alarm,
        "disaster_recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "normal_recall": tn / (tn + fp) if (tn + fp) else float("nan"),
        "disaster_precision": tp / (tp + fp) if (tp + fp) else float("nan"),
    }


def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    preds = probs.argmax(1)
    acc = float((preds == labels).mean())
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
    report_txt = classification_report(labels, preds, target_names=C.CLASSES,
                                       digits=4, zero_division=0)
    report_dict = classification_report(labels, preds, target_names=C.CLASSES,
                                        digits=4, zero_division=0, output_dict=True)
    cm = confusion_matrix(labels, preds, labels=list(range(C.NUM_CLASSES)))

    # macro one-vs-rest ROC-AUC
    try:
        onehot = np.eye(C.NUM_CLASSES)[labels]
        roc_auc = float(roc_auc_score(onehot, probs, average="macro", multi_class="ovr"))
    except ValueError:
        roc_auc = float("nan")

    prec, rec, f1, sup = precision_recall_fscore_support(
        labels, preds, labels=list(range(C.NUM_CLASSES)), zero_division=0)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_roc_auc_ovr": roc_auc,
        "report_text": report_txt,
        "report_dict": report_dict,
        "confusion_matrix": cm,
        "per_class": {
            C.CLASSES[i]: {"precision": float(prec[i]), "recall": float(rec[i]),
                           "f1": float(f1[i]), "support": int(sup[i])}
            for i in range(C.NUM_CLASSES)
        },
        "binary_disaster": disaster_binary_view(labels, preds),
        "preds": preds,
    }


# =========================================================================== #
# Figures                                                                     #
# =========================================================================== #
def figure_confusion_matrix(cm: np.ndarray, title: str, out: Path,
                            normalize: bool = False) -> Path:
    import matplotlib.pyplot as plt
    import seaborn as sns

    if normalize:
        data = cm.astype(float)
        row = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row, out=np.zeros_like(data), where=row != 0)
        fmt = ".2f"
    else:
        data = cm.astype(int)
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", cbar=True,
                xticklabels=C.CLASSES, yticklabels=C.CLASSES, ax=ax,
                annot_kws={"size": 11}, linewidths=0.5, linecolor="white")
    ax.set_xlabel("predicted label")
    ax.set_ylabel("true label")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    return save_figure(fig, out, C.FIGURE_DPI)


def figure_roc_curves(labels: np.ndarray, probs: np.ndarray, title: str, out: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, cls in enumerate(C.CLASSES):
        y = (labels == i).astype(int)
        fpr, tpr, _ = roc_curve(y, probs[:, i])
        ax.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save_figure(fig, out, C.FIGURE_DPI)


def figure_training_curves_from_meta(exp_id: str, seeds) -> Path:
    """Thin wrapper so run_all.py can regenerate curves from saved histories."""
    from train import plot_training_curves
    return plot_training_curves(exp_id, seeds)


# =========================================================================== #
# Error analysis                                                              #
# =========================================================================== #
def top_confusions(cm: np.ndarray, k: int = 5) -> list[dict]:
    pairs = []
    for i in range(C.NUM_CLASSES):
        for j in range(C.NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": C.CLASSES[i], "pred": C.CLASSES[j],
                              "count": int(cm[i, j]),
                              "rate_of_true_class": float(cm[i, j] / max(cm[i].sum(), 1))})
    pairs.sort(key=lambda d: -d["count"])
    return pairs[:k]


def figure_misclassified(labels, probs, paths, out: Path,
                         per_pair: int = 4, n_pairs: int = 3) -> Path:
    """Grid of misclassified test images for the most common confusion pairs."""
    import matplotlib.pyplot as plt
    from PIL import Image

    preds = probs.argmax(1)
    wrong = np.where(preds != labels)[0]
    cm = confusion_matrix(labels, preds, labels=list(range(C.NUM_CLASSES)))
    pairs = top_confusions(cm, k=n_pairs)
    if not wrong.size or not pairs:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.set_title("No misclassified test images")
        return save_figure(fig, out, C.FIGURE_DPI)

    ncols = per_pair
    nrows = len(pairs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.5 * nrows),
                             squeeze=False)
    for r, pair in enumerate(pairs):
        ti = C.CLASSES.index(pair["true"])
        pi = C.CLASSES.index(pair["pred"])
        sel = [i for i in wrong if labels[i] == ti and preds[i] == pi][:per_pair]
        for c in range(ncols):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(sel):
                k = int(sel[c])
                with Image.open(C.DATA_DIR / paths[k]) as im:
                    ax.imshow(im.convert("RGB"))
                conf = probs[k, preds[k]]
                ax.set_title(f"true: {C.CLASSES[labels[k]]}\n"
                             f"pred: {C.CLASSES[preds[k]]} ({conf:.2f})", fontsize=10)
        axes[r][0].set_ylabel(f"{pair['true']} -> {pair['pred']}\n(n={pair['count']})",
                              fontsize=11, fontweight="bold")
    fig.suptitle("Main model: misclassified test images for the top confusion pairs",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, out, C.FIGURE_DPI)


# =========================================================================== #
# Grad-CAM (implemented from scratch with forward/backward hooks)             #
# =========================================================================== #
class GradCAM:
    """Minimal Grad-CAM on any target sub-module (no external library).

    grad = d(score_class) / d(A) ;  weights = mean over spatial dims of grad
    cam  = ReLU( sum_channels(weights * A) ), then upsampled to the input size.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._h1 = target_layer.register_forward_hook(self._fwd)
        self._h2 = target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, module, inp, out):
        self.activations = out.detach()

    def _bwd(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self._h1.remove()
        self._h2.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        if class_idx is None:
            class_idx = int(out.argmax(1).item())
        score = out[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)     # GAP over HxW
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy(), out.detach()


def get_layer_by_name(model: nn.Module, dotted: str) -> nn.Module:
    obj = model
    for part in dotted.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def figure_gradcam(model, device, df: pd.DataFrame, out: Path,
                   n_per_group: int = 4, gradcam_layer: str | None = None) -> Path:
    """Grad-CAM on correctly- and incorrectly-classified test images.

    Top row: the plain image.  Bottom row: the same image with the Grad-CAM
    heat-map overlaid.  Positive examples (correct) and negative examples
    (misclassified) are shown side by side.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    if gradcam_layer is None:
        gradcam_layer = (C.GRADCAM_LAYER_B if model.__class__.__name__.lower().startswith("mobilenet")
                         else C.GRADCAM_LAYER_A)
    target = get_layer_by_name(model, gradcam_layer)
    print(f"[eval] Grad-CAM target layer: {gradcam_layer} ({target.__class__.__name__})")

    _, eval_tf = build_transforms(augment=False)
    sub = df[df["split"] == "test"].reset_index(drop=True)
    records = list(zip(sub["path"].tolist(), sub["label"].tolist()))

    model.eval()
    correct_idx: list[int] = []
    wrong_idx: list[int] = []

    # 1) find which test images are classified correctly / incorrectly
    with torch.no_grad():
        for i, (rel, lab) in enumerate(records):
            with Image.open(C.DATA_DIR / rel) as im:
                img = im.convert("RGB")
            x = eval_tf(img).unsqueeze(0).to(device)
            pred = int(model(x).argmax(1).item())
            (correct_idx if pred == lab else wrong_idx).append(i)
            if len(correct_idx) >= n_per_group and len(wrong_idx) >= n_per_group:
                break

    picks = ([("correct", i) for i in correct_idx[:n_per_group]]
             + [("wrong", i) for i in wrong_idx[:n_per_group]])
    if not picks:
        picks = [("wrong", i) for i in wrong_idx[:n_per_group]]

    # 2) compute Grad-CAM for the chosen images (one engine, reused)
    engine = GradCAM(model, target)
    n = len(picks)
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 6.6), squeeze=False)

    for c, (kind, i) in enumerate(picks):
        rel, lab = records[i]
        with Image.open(C.DATA_DIR / rel) as im:
            img = im.convert("RGB").resize((C.IMG_SIZE, C.IMG_SIZE))
        arr = np.asarray(img).astype(np.float32) / 255.0
        x = eval_tf(img).unsqueeze(0).to(device)
        with torch.enable_grad():
            heat, logits = engine(x)
        pred = int(logits.argmax(1).item())
        conf = float(torch.softmax(logits, 1)[0, pred])

        axes[0][c].imshow(arr)
        axes[0][c].set_title(f"true: {C.CLASSES[lab]}", fontsize=10)
        axes[0][c].axis("off")
        axes[1][c].imshow(arr)
        axes[1][c].imshow(heat, cmap="jet", alpha=0.45)
        axes[1][c].set_title(f"{kind}: pred {C.CLASSES[pred]} ({conf:.2f})", fontsize=10)
        axes[1][c].axis("off")

    engine.remove()
    fig.suptitle("Main model Grad-CAM (top: image, bottom: heat-map overlay) - "
                 "left half correct, right half misclassified",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save_figure(fig, out, C.FIGURE_DPI)


# =========================================================================== #
# Per-experiment evaluation                                                   #
# =========================================================================== #
def evaluate_checkpoint(exp_id: str, seed: int, device=None,
                        num_workers: int | None = None) -> dict:
    device = device or C.get_device()
    set_seed(seed)
    ckpt_path = C.CKPT_DIR / f"{exp_id}_seed{seed}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint {ckpt_path} (train first)")

    model, ckpt = load_checkpoint_model(ckpt_path, device)
    df = load_splits()
    test_loader = build_test_loader(df, num_workers=num_workers)

    probs, labels, paths = predict(model, test_loader, device)
    metrics = compute_metrics(labels, probs)

    prefix = f"{exp_id}_seed{seed}"
    # text + csv classification reports
    (C.RESULTS_DIR / f"classification_report_{prefix}.txt").write_text(
        f"Experiment: {exp_id}  seed: {seed}  split: TEST\n"
        f"Model: {ckpt['model_key']}  augment={ckpt['augment']}\n"
        f"Accuracy: {metrics['accuracy']:.4f}   macro-F1: {metrics['macro_f1']:.4f}   "
        f"weighted-F1: {metrics['weighted_f1']:.4f}   macro ROC-AUC(OvR): "
        f"{metrics['macro_roc_auc_ovr']:.4f}\n\n{metrics['report_text']}\n",
        encoding="utf-8")
    pd.DataFrame(metrics["report_dict"]).T.to_csv(
        C.RESULTS_DIR / f"classification_report_{prefix}.csv")
    pd.DataFrame(metrics["confusion_matrix"], index=C.CLASSES, columns=C.CLASSES).to_csv(
        C.RESULTS_DIR / f"confusion_matrix_{prefix}.csv")
    norm = metrics["confusion_matrix"] / np.maximum(
        metrics["confusion_matrix"].sum(1, keepdims=True), 1)
    pd.DataFrame(norm, index=C.CLASSES, columns=C.CLASSES).round(4).to_csv(
        C.RESULTS_DIR / f"confusion_matrix_normalized_{prefix}.csv")

    figure_confusion_matrix(
        metrics["confusion_matrix"],
        f"Confusion matrix (counts) - {C.EXPERIMENTS[exp_id]['label']} (seed {seed})",
        C.FIGURES_DIR / f"confusion_matrix_{prefix}.png")
    figure_confusion_matrix(
        metrics["confusion_matrix"],
        f"Confusion matrix (row-normalized) - {C.EXPERIMENTS[exp_id]['label']} (seed {seed})",
        C.FIGURES_DIR / f"confusion_matrix_normalized_{prefix}.png", normalize=True)
    figure_roc_curves(labels, probs,
                      f"Per-class ROC curves - {exp_id} (seed {seed})",
                      C.FIGURES_DIR / f"roc_curves_{prefix}.png")

    # inference timing with the actual trained model
    timing = measure_inference_time(model, test_loader, device, n_batches=15)

    # scenario-weighted accuracy: the 4 disaster classes are the mission-critical ones
    payload = {
        "exp": exp_id, "seed": seed, "split": "test",
        "model_key": ckpt["model_key"], "augment": bool(ckpt["augment"]),
        "best_epoch": int(ckpt["epoch"]), "val_macro_f1_at_selection": float(ckpt["val_macro_f1"]),
        "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "macro_roc_auc_ovr": metrics["macro_roc_auc_ovr"],
        "per_class": metrics["per_class"],
        "binary_disaster": metrics["binary_disaster"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "top_confusions": top_confusions(metrics["confusion_matrix"], k=6),
        "inference_ms_per_image": timing["ms_per_image"],
        "n_test": int(len(labels)),
        "device": device.type,
    }
    (C.RESULTS_DIR / f"metrics_{prefix}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    # cache raw predictions for reuse (error analysis / Grad-CAM)
    np.savez_compressed(C.RESULTS_DIR / f"preds_{prefix}.npz",
                        probs=probs.astype(np.float32), labels=labels,
                        paths=np.array(paths, dtype=object), allow_pickle=True)

    print(f"[eval] {exp_id} seed {seed}: acc={metrics['accuracy']:.4f} "
          f"macroF1={metrics['macro_f1']:.4f} AUC={metrics['macro_roc_auc_ovr']:.4f} "
          f"| miss={metrics['binary_disaster']['miss_rate']:.4f} "
          f"false-alarm={metrics['binary_disaster']['false_alarm_rate']:.4f} "
          f"| {timing['ms_per_image']:.2f} ms/img")
    return payload


# =========================================================================== #
# Aggregation + comparison figures                                            #
# =========================================================================== #
def aggregate_experiments(exp_ids: list[str] | None = None) -> pd.DataFrame:
    """Collect all per-seed metrics into results/test_metrics.csv."""
    exp_ids = exp_ids or list(C.EXPERIMENTS)
    rows = []
    for exp_id in exp_ids:
        for seed in metric_seeds(exp_id):
            p = C.RESULTS_DIR / f"metrics_{exp_id}_seed{seed}.json"
            if not p.exists():
                continue
            m = json.loads(p.read_text(encoding="utf-8"))
            row = {
                "exp": exp_id, "label": C.EXPERIMENTS[exp_id]["label"], "seed": seed,
                "model": m["model_key"], "augment": m["augment"],
                "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"], "macro_roc_auc_ovr": m["macro_roc_auc_ovr"],
                "miss_rate": m["binary_disaster"]["miss_rate"],
                "false_alarm_rate": m["binary_disaster"]["false_alarm_rate"],
                "inference_ms_per_image": m["inference_ms_per_image"],
                "n_test": m["n_test"],
            }
            for cls in C.CLASSES:
                row[f"precision_{cls}"] = m["per_class"][cls]["precision"]
                row[f"recall_{cls}"] = m["per_class"][cls]["recall"]
                row[f"f1_{cls}"] = m["per_class"][cls]["f1"]
            rows.append(row)
    if not rows:
        raise RuntimeError("no metrics_*.json found - train and evaluate first")
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / "test_metrics.csv", index=False)

    named = {
        "n_seeds": ("seed", "count"),
        "accuracy_mean": ("accuracy", "mean"), "accuracy_std": ("accuracy", "std"),
        "macro_f1_mean": ("macro_f1", "mean"), "macro_f1_std": ("macro_f1", "std"),
        "weighted_f1_mean": ("weighted_f1", "mean"),
        "roc_auc_mean": ("macro_roc_auc_ovr", "mean"),
        "miss_rate_mean": ("miss_rate", "mean"), "miss_rate_std": ("miss_rate", "std"),
        "false_alarm_rate_mean": ("false_alarm_rate", "mean"),
        "false_alarm_rate_std": ("false_alarm_rate", "std"),
        "inference_ms_mean": ("inference_ms_per_image", "mean"),
    }
    # per-class means are needed by the per-class comparison figure
    for cls in C.CLASSES:
        for metric in ("precision", "recall", "f1"):
            named[f"{metric}_{cls}"] = (f"{metric}_{cls}", "mean")

    agg = (df.groupby(["exp", "label", "model", "augment"])
             .agg(**named)
             .reset_index())
    agg.to_csv(C.RESULTS_DIR / "test_metrics_aggregated.csv", index=False)
    return agg


def figure_accuracy_f1_comparison(agg: pd.DataFrame,
                                  out: Path = C.FIGURES_DIR / "comparison_accuracy_macrof1.png") -> Path:
    import matplotlib.pyplot as plt

    labels = [C.EXPERIMENTS[e]["label"].replace(": ", ":\n") for e in agg["exp"]]
    x = np.arange(len(agg))
    w = 0.36
    fig, ax = plt.subplots(figsize=(max(9, 3.2 * len(agg)), 6))
    ax.bar(x - w / 2, agg["accuracy_mean"], w, yerr=agg["accuracy_std"].fillna(0),
           capsize=5, label="test accuracy", color="#3b6ea5", edgecolor="black", linewidth=0.6)
    ax.bar(x + w / 2, agg["macro_f1_mean"], w, yerr=agg["macro_f1_std"].fillna(0),
           capsize=5, label="test macro-F1", color="#e08a3c", edgecolor="black", linewidth=0.6)
    for xi, (a, f) in enumerate(zip(agg["accuracy_mean"], agg["macro_f1_mean"])):
        ax.text(xi - w / 2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=10)
        ax.text(xi + w / 2, f, f"{f:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Test accuracy and macro-F1 by experiment (error bars = std over seeds)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save_figure(fig, out, C.FIGURE_DPI)


def figure_per_class_comparison(agg: pd.DataFrame,
                                out: Path = C.FIGURES_DIR / "per_class_f1_recall_comparison.png") -> Path:
    import matplotlib.pyplot as plt

    exps = list(agg["exp"])
    n = len(exps)
    w = 0.8 / max(n, 1)
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    for ax, metric in zip(axes, ("f1", "recall")):
        x = np.arange(C.NUM_CLASSES)
        for i, exp_id in enumerate(exps):
            vals = [agg.loc[agg["exp"] == exp_id, f"{metric}_{c}"].mean() for c in C.CLASSES]
            ax.bar(x + i * w - 0.4 + w / 2, vals, w, edgecolor="black", linewidth=0.5,
                   label=C.EXPERIMENTS[exp_id]["label"])
        ax.set_xticks(x)
        ax.set_xticklabels(C.CLASSES, rotation=18, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric)
        ax.set_title(f"Per-class {metric.upper()} by experiment")
        ax.legend(fontsize=9)
    fig.suptitle("Per-class F1 and recall across experiments (mean over seeds)",
                 fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save_figure(fig, out, C.FIGURE_DPI)


def run_full_evaluation(exp_ids: list[str] | None = None, num_workers: int | None = None) -> pd.DataFrame:
    """Evaluate every available checkpoint, aggregate, and draw comparison plots."""
    exp_ids = exp_ids or list(C.EXPERIMENTS)
    device = C.get_device()
    print(f"[eval] device={device.type}")

    for exp_id in exp_ids:
        for seed in checkpoint_seeds(exp_id):
            evaluate_checkpoint(exp_id, seed, device=device, num_workers=num_workers)

    agg = aggregate_experiments(exp_ids)
    figure_accuracy_f1_comparison(agg)
    figure_per_class_comparison(agg)

    # ---- error analysis + Grad-CAM on the main experiment -------------------
    main = C.MAIN_EXPERIMENT
    main_seeds = checkpoint_seeds(main)
    main_seed = main_seeds[0] if main_seeds else C.SEED
    pred_file = C.RESULTS_DIR / f"preds_{main}_seed{main_seed}.npz"
    if pred_file.exists():
        data = np.load(pred_file, allow_pickle=True)
        probs, labels, paths = data["probs"], data["labels"], list(data["paths"])
        figure_misclassified(labels, probs, paths,
                             C.FIGURES_DIR / "misclassified_examples_main_model.png")
        try:
            df = load_splits()
            model, _ = load_checkpoint_model(
                C.CKPT_DIR / f"{main}_seed{main_seed}_best.pt", device)
            figure_gradcam(model, device, df, C.FIGURES_DIR / "gradcam_main_model.png")
            print("[eval] wrote Grad-CAM figure")
        except Exception as exc:                       # noqa: BLE001
            print(f"[eval] Grad-CAM figure failed: {exc}")

    print("[eval] wrote test_metrics.csv, test_metrics_aggregated.csv and comparison figures")
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()

    apply_plot_style()
    print(C.describe_device())
    if args.exp and args.seed is not None:
        evaluate_checkpoint(args.exp, args.seed, num_workers=args.num_workers)
    else:
        run_full_evaluation(num_workers=args.num_workers)


if __name__ == "__main__":
    main()
