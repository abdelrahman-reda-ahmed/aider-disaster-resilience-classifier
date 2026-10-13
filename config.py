"""
config.py
=========
Single source of truth for every path, hyper-parameter, and random seed used in
this project (EE 3701 - Project 02, Track 1: Disaster Resilience).

Everything here is cross-platform: all paths are built with ``pathlib`` and are
resolved relative to the location of this file, so the project runs unchanged on
Windows, Linux and macOS.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #
# PROJECT_ROOT = the folder that contains this file (i.e. the ``project/`` dir).
PROJECT_ROOT: Path = Path(__file__).resolve().parent

DATA_DIR: Path = PROJECT_ROOT / "data"     # holds AIDER.zip + extracted AIDER/
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
CKPT_DIR: Path = PROJECT_ROOT / "checkpoints"
LOG_DIR: Path = RESULTS_DIR / "logs"

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, CKPT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Dataset                                                                     #
# --------------------------------------------------------------------------- #
# AIDER (Aerial Image Dataset for Emergency Response Applications)
# Kyrkou & Theocharides, Zenodo record 3888300, DOI 10.5281/zenodo.3888300
AIDER_RECORD_API = "https://zenodo.org/api/records/3888300"
AIDER_URL = "https://zenodo.org/api/records/3888300/files/AIDER.zip/content"
AIDER_MD5 = "1ad4eb02ed156e8dfa19986ff382e58b"
AIDER_ZIP_SIZE = 275_701_096  # bytes, as reported by the Zenodo API

ZIP_PATH: Path = DATA_DIR / "AIDER.zip"

# The five classes, in a fixed order. Index == label used everywhere.
CLASSES = [
    "collapsed_building",
    "fire",
    "flood",
    "traffic_accident",
    "normal",
]
NUM_CLASSES = len(CLASSES)

# Folder names as they appear inside AIDER.zip, mapped to the canonical class
# names above.  The archive calls two folders differently from the brief:
# "flooded_areas" -> "flood" and "traffic_incident" -> "traffic_accident".
CLASS_DIR_ALIASES = {
    "collapsed_building": "collapsed_building",
    "fire": "fire",
    "flooded_areas": "flood",
    "flood": "flood",
    "traffic_incident": "traffic_accident",
    "traffic_accident": "traffic_accident",
    "normal": "normal",
}

# Disaster classes = everything that is not "normal" (used for the clinically
# important "disaster vs normal" binary view).
NORMAL_CLASS = "normal"
DISASTER_CLASSES = [c for c in CLASSES if c != NORMAL_CLASS]

# Accepted image extensions when scanning the dataset.
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# --------------------------------------------------------------------------- #
# Reproducibility                                                             #
# --------------------------------------------------------------------------- #
SEED = 42
# Seeds used to repeat experiments 1-3 so we can report mean +/- std.
SEEDS = [42, 43, 44]

# --------------------------------------------------------------------------- #
# Pre-processing / data loading                                               #
# --------------------------------------------------------------------------- #
IMG_SIZE = 224
BATCH_SIZE = 32
# ImageNet statistics (used for BOTH models so the comparison is apples-to-apples).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Augmentation (training split only).
AUG_HFLIP_P = 0.5
AUG_ROTATION_DEG = 15
AUG_CROP_SCALE = (0.7, 1.0)
AUG_JITTER = 0.2  # brightness & contrast

# Split fractions (70 / 15 / 15, stratified, seed = SEED).
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def _default_num_workers() -> int:
    """DataLoader worker count that behaves well on both POSIX and Windows.

    On Windows every worker is a fresh ``spawn``ed interpreter which is slow and
    memory-hungry, so we keep the count low there.  The value can always be
    overridden with the ``AIDER_NUM_WORKERS`` environment variable.
    """
    env = os.environ.get("AIDER_NUM_WORKERS")
    if env is not None:
        return int(env)
    cpu = os.cpu_count() or 1
    if os.name == "nt":
        return min(2, cpu)
    return min(4, cpu)


NUM_WORKERS: int = _default_num_workers()

# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #
WEIGHT_DECAY = 1e-4          # AdamW
LR_HEAD = 1e-3               # Model B stage 1 (frozen backbone, new head only)
LR_FINETUNE = 1e-4           # Model B stage 2 (full network)
LR_FINETUNE_MIN = 3e-5       # cosine annealing floor for stage 2
LR_MODEL_A = 1e-3            # Model A (trained from scratch)

EPOCHS_MODEL_A = 30
EPOCHS_B_STAGE1 = 4
EPOCHS_B_STAGE2 = 16

# Early stopping on VALIDATION macro-F1.
EARLY_STOP_PATIENCE = 6
EARLY_STOP_MIN_DELTA = 1e-4

# Dropout
DROPOUT_MODEL_A = 0.4
DROPOUT_MODEL_A_FC = 0.3
DROPOUT_MODEL_B_HEAD = 0.3

# Grad-CAM / figures
GRADCAM_LAYER_A = "features.3.0"  # last conv of Model A's 4th convolutional block
GRADCAM_LAYER_B = "features.18"   # last conv layer of the MobileNetV2 backbone

FIGURE_DPI = 300

# --------------------------------------------------------------------------- #
# Experiment registry                                                         #
# --------------------------------------------------------------------------- #
# Each experiment: id -> (model key, use_augmentation, human label)
EXPERIMENTS = {
    "exp1_modelA_aug":   {"model": "modelA", "augment": True,  "label": "Exp 1: Small CNN (scratch) + aug"},
    "exp2_modelB_aug":   {"model": "modelB", "augment": True,  "label": "Exp 2: MobileNetV2 transfer + aug"},
    "exp3_modelB_noaug": {"model": "modelB", "augment": False, "label": "Exp 3: MobileNetV2 transfer, NO aug"},
    "exp4_modelA_noaug": {"model": "modelA", "augment": False, "label": "Exp 4: Small CNN (scratch), NO aug"},
}

# The "main model" whose test predictions drive the error analysis / Grad-CAM.
MAIN_EXPERIMENT = "exp2_modelB_aug"

# --------------------------------------------------------------------------- #
# Device                                                                      #
# --------------------------------------------------------------------------- #
def get_device():
    """Best available device: CUDA -> Apple MPS -> CPU.  Cross-platform."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device() -> str:
    import platform

    import torch

    dev = get_device()
    py = platform.python_version()
    return (
        f"python={py} torch={torch.__version__} device={dev.type} "
        f"cuda_available={torch.cuda.is_available()} "
        f"mps_available={bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())} "
        f"cpu_count={os.cpu_count()} platform={platform.platform()}"
    )
