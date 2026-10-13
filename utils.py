"""
utils.py
========
Small shared helpers: deterministic seeding, a cross-platform figure style, and
timing utilities.  Kept dependency-light so it works identically on Windows,
Linux and macOS.

(This module is an addition to the file list in the project brief; it holds code
that would otherwise be duplicated across data.py / train.py / evaluate.py.)
"""

from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np


# --------------------------------------------------------------------------- #
# Reproducibility                                                             #
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed every RNG we use and ask the backend for deterministic kernels."""
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # MPS: there is no deterministic switch; seeding above is what we can do.
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


def worker_init_fn(worker_id: int) -> None:
    """Give every DataLoader worker a distinct, reproducible NumPy/Python seed."""
    import torch

    base = torch.initial_seed() % (2 ** 31)
    s = (base + worker_id) % (2 ** 31)
    np.random.seed(s)
    random.seed(s)


# --------------------------------------------------------------------------- #
# Timing                                                                      #
# --------------------------------------------------------------------------- #
@contextmanager
def timer() -> Iterator[dict]:
    """``with timer() as t: ...`` -> ``t['seconds']`` after the block."""
    box: dict = {}
    start = time.perf_counter()
    try:
        yield box
    finally:
        box["seconds"] = time.perf_counter() - start


def human_hms(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{sec:02d}s"


# --------------------------------------------------------------------------- #
# Plotting style (white background, readable fonts, no grid tricks)           #
# --------------------------------------------------------------------------- #
def apply_plot_style() -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless / file output only -> works on servers
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.30,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 17,
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig, path: str | Path, dpi: int = 300) -> Path:
    """Save a matplotlib figure and close it, creating parent folders."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Misc                                                                        #
# --------------------------------------------------------------------------- #
def relpath_posix(path: Path, base: Path) -> str:
    """Portable relative path string (always forward slashes)."""
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def fmt_float(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"
