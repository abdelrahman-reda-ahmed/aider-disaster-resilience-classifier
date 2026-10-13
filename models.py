"""
models.py
=========
The two architectures, plus utilities for parameter counts, model size,
layer-by-layer summaries, inference timing and architecture diagrams.

Model A - ``SmallCNN``      : compact 4-block CNN trained from random init.
Model B - ``MobileNetV2``   : torchvision MobileNetV2 (ImageNet pretrained) with
                              a new Dropout(0.3) + Linear(1280, 5) classifier head.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config as C
from utils import human_hms, save_figure


# =========================================================================== #
# Model A: compact CNN from scratch                                           #
# =========================================================================== #
def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv 3x3 -> BatchNorm -> ReLU -> MaxPool(2)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class SmallCNN(nn.Module):
    """A compact CNN: 4 conv blocks 32-64-128-256, GAP, dropout, 128-d FC, 5-way."""

    def __init__(self, num_classes: int = C.NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(C.DROPOUT_MODEL_A),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(C.DROPOUT_MODEL_A_FC),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# =========================================================================== #
# Model B: MobileNetV2 transfer learning                                      #
# =========================================================================== #
def build_mobilenet_v2(num_classes: int = C.NUM_CLASSES,
                       pretrained: bool = True) -> nn.Module:
    """ImageNet-pretrained MobileNetV2 with a fresh classification head."""
    from torchvision import models
    from torchvision.models import MobileNet_V2_Weights

    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[-1].in_features      # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(C.DROPOUT_MODEL_B_HEAD),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_model(key: str, num_classes: int = C.NUM_CLASSES,
                pretrained: bool = True) -> nn.Module:
    key = key.lower()
    if key == "modela":
        return SmallCNN(num_classes)
    if key == "modelb":
        return build_mobilenet_v2(num_classes, pretrained=pretrained)
    raise ValueError(f"unknown model key: {key}")


def freeze_backbone(model: nn.Module) -> None:
    """Freeze everything except the classifier head (Model B stage 1).

    Only the ``classifier`` parameters keep gradients, and every BatchNorm
    layer is switched to eval mode so it keeps its pre-trained running
    statistics.  The ``_freeze_bn`` flag is honoured by ``train.run_epoch``,
    which would otherwise re-enable BN training mode via ``model.train(True)``
    and let those running statistics drift - i.e. the backbone would keep
    "learning" even though it is supposed to be frozen.
    """
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("classifier")
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()      # keep frozen BN statistics
    model._freeze_bn = True


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True
    model._freeze_bn = False


def trainable_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model: nn.Module) -> float:
    """Approximate on-disk size of the weights (fp32) in MB."""
    bytes_ = sum(p.numel() * p.element_size() for p in model.parameters())
    bytes_ += sum(b.numel() * b.element_size() for b in model.buffers())
    return bytes_ / (1024 ** 2)


# =========================================================================== #
# Layer-by-layer summary                                                      #
# =========================================================================== #
def layer_summary(model: nn.Module, input_size=(C.IMG_SIZE, C.IMG_SIZE), device="cpu") -> str:
    """Forward a dummy tensor and record every leaf module's output shape.

    Returns a printable, markdown-friendly table (used in results/).
    """
    model = model.to(device).eval()
    rows: list[dict] = []
    handles = []

    def make_hook(name: str):
        def hook(module, inp, out):
            shape = tuple(out.shape) if isinstance(out, torch.Tensor) else None
            rows.append({
                "layer": name,
                "type": module.__class__.__name__,
                "output_shape": str(tuple(shape[1:])) if shape else "-",
                "params": sum(p.numel() for p in module.parameters(recurse=False)),
            })
        return hook

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            handles.append(module.register_forward_hook(make_hook(name or "input")))

    with torch.no_grad():
        model(torch.zeros(1, 3, *input_size, device=device))

    for h in handles:
        h.remove()
    model.train()

    header = f"{'layer':<26}{'type':<22}{'out shape':<22}{'params':>12}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(f"{r['layer']:<26}{r['type']:<22}{r['output_shape']:<22}{r['params']:>12,d}")
    return "\n".join(lines)


def save_layer_summaries(device=None) -> pd.DataFrame:
    """Write results/architecture_modelA.txt and ..._modelB.txt + a CSV table."""
    device = device or C.get_device()
    out_rows = []

    for key, fname in (("modelA", "architecture_modelA.txt"),
                       ("modelB", "architecture_modelB.txt")):
        model = build_model(key, pretrained=(key == "modelB"))
        model.eval()
        total, trainable = trainable_parameters(model)
        size = model_size_mb(model)
        summary = layer_summary(model, device=device)
        text = (
            f"Model: {'SmallCNN (from scratch)' if key == 'modelA' else 'MobileNetV2 (ImageNet pretrained) + new head'}\n"
            f"Input: 3 x {C.IMG_SIZE} x {C.IMG_SIZE}\n"
            f"Total parameters    : {total:,}\n"
            f"Trainable parameters: {trainable:,}\n"
            f"Approx size (fp32)  : {size:.2f} MB\n"
            f"{'-' * 70}\n\n{summary}\n"
        )
        (C.RESULTS_DIR / fname).write_text(text, encoding="utf-8")
        out_rows.append({
            "model": key,
            "description": "SmallCNN from scratch" if key == "modelA" else "MobileNetV2 transfer",
            "total_params": total,
            "trainable_params": trainable,
            "size_mb": round(size, 3),
        })
        print(f"[models] wrote {fname} ({total:,} params, {size:.2f} MB)")

    df = pd.DataFrame(out_rows)
    df.to_csv(C.RESULTS_DIR / "model_summary.csv", index=False)
    return df


# =========================================================================== #
# Inference timing                                                            #
# =========================================================================== #
@torch.no_grad()
def measure_inference_time(model: nn.Module, loader, device, n_batches: int = 20) -> dict:
    """Average ms per image on the available device, after warm-up."""
    model = model.to(device).eval()
    n_imgs = 0
    warmup = 3
    t0 = time.perf_counter()
    for i, (x, _) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i + 1 >= warmup:
            break
    del t0

    start = time.perf_counter()
    for i, (x, _) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        _ = model(x)
        n_imgs += x.size(0)
        if i + 1 >= n_batches:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "ms_per_image": 1000.0 * elapsed / max(n_imgs, 1),
        "images_timed": n_imgs,
        "seconds": elapsed,
        "device": device.type,
    }


def benchmark_all(device=None, batch_size: int = C.BATCH_SIZE, n_batches: int = 20) -> pd.DataFrame:
    """Time both architectures on the same test loader; save results/inference_time.csv."""
    from data import AiderDataset, build_transforms
    from torch.utils.data import DataLoader

    device = device or C.get_device()
    df = pd.read_csv(C.RESULTS_DIR / "splits.csv")
    test = df[df["split"] == "test"]
    records = list(zip(test["path"].tolist(), test["label"].tolist()))

    _, eval_tf = build_transforms(augment=False)
    loader = DataLoader(AiderDataset(records, C.DATA_DIR, eval_tf), batch_size=batch_size,
                        shuffle=False, num_workers=C.NUM_WORKERS)

    rows = []
    for key in ("modelA", "modelB"):
        model = build_model(key, pretrained=(key == "modelB"))
        res = measure_inference_time(model, loader, device, n_batches=n_batches)
        rows.append({"model": key, **res})
        print(f"[models] {key}: {res['ms_per_image']:.2f} ms/image on {res['device']} "
              f"({res['images_timed']} images)")

    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS_DIR / "inference_time.csv", index=False)
    return out


# =========================================================================== #
# Architecture diagram                                                        #
# =========================================================================== #
def _box(ax, x, y, w, h, text, color, fontsize=10):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                                linewidth=1.2, edgecolor="black", facecolor=color))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, wrap=True)


def figure_architecture(out: Path = C.FIGURES_DIR / "architecture_diagram.png") -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(13, 11))

    # --- Model A ------------------------------------------------------------
    ax = axes[0]
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.2); ax.axis("off")
    ax.set_title("Model A: SmallCNN (trained from scratch)", fontsize=16, fontweight="bold")
    labels = ["Input\n3x224x224", "Conv block 1\n3->32\n224->112", "Conv block 2\n32->64\n112->56",
              "Conv block 3\n64->128\n56->28", "Conv block 4\n128->256\n28->14",
              "Global avg\npool 256", "Dropout\n0.4", "FC 256->128\nReLU, Dropout 0.3",
              "Output\n5 classes"]
    colors = ["#dfe7f1", "#9fc0e2", "#9fc0e2", "#9fc0e2", "#9fc0e2",
              "#f3d9a4", "#f5b7b1", "#c8e6c9", "#f9e79f"]
    for i, (lab, col) in enumerate(zip(labels, colors)):
        _box(ax, 0.15 + i * 1.42, 0.9, 1.28, 1.3, lab, col, fontsize=8.5)
        if i < len(labels) - 1:
            ax.annotate("", xy=(0.15 + (i + 1) * 1.42, 1.55), xytext=(0.15 + i * 1.42 + 1.28, 1.55),
                        arrowprops=dict(arrowstyle="->", lw=1.3))
    ax.text(0.15, 2.7, "Each conv block = Conv3x3 -> BatchNorm -> ReLU -> MaxPool2x2",
            fontsize=9.5, style="italic")

    # --- Model B ------------------------------------------------------------
    ax = axes[1]
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.2); ax.axis("off")
    ax.set_title("Model B: MobileNetV2 transfer learning (ImageNet pretrained)", fontsize=16,
                 fontweight="bold")
    labels = ["Input\n3x224x224", "MobileNetV2\nfeatures 0-17\n(inverted residual\nblocks)",
              "features[18]\nConv 320->1280\n7x7", "Global avg\npool 1280",
              "NEW HEAD\nDropout 0.3", "NEW HEAD\nLinear 1280->5", "Output\n5 classes"]
    colors = ["#dfe7f1", "#b9cfe8", "#b9cfe8", "#f3d9a4", "#f5b7b1", "#c8e6c9", "#f9e79f"]
    for i, (lab, col) in enumerate(zip(labels, colors)):
        _box(ax, 0.15 + i * 1.83, 0.9, 1.65, 1.4, lab, col, fontsize=8.5)
        if i < len(labels) - 1:
            ax.annotate("", xy=(0.15 + (i + 1) * 1.83, 1.6), xytext=(0.15 + i * 1.83 + 1.65, 1.6),
                        arrowprops=dict(arrowstyle="->", lw=1.3))
    ax.text(0.15, 2.75, "Stage 1: backbone frozen, only the new head trains (lr=1e-3).  "
                        "Stage 2: all layers fine-tuned (lr=1e-4, cosine -> 3e-5).",
            fontsize=9.5, style="italic")

    fig.tight_layout()
    return save_figure(fig, out, C.FIGURE_DPI)


def main() -> None:
    from utils import apply_plot_style
    apply_plot_style()
    print(C.describe_device())
    save_layer_summaries()
    benchmark_all()
    figure_architecture()


if __name__ == "__main__":
    main()
