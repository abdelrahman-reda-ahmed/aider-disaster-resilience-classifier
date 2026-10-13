"""
data.py
=======
Everything data-related:

* download + checksum + unzip of the AIDER dataset (pure Python, no ``wget`` /
  ``unzip`` shell calls, so it also runs on Windows),
* integrity scan (unreadable / zero-byte files) and exact-duplicate removal,
* resolution statistics,
* reproducible stratified 70/15/15 train/val/test split saved to CSV,
* ``torch.utils.data.Dataset`` classes + transforms + DataLoaders,
* class weights computed from the TRAIN split only,
* exploratory figures (class distribution, sample grid, augmentation examples).

Nothing in this file ever looks at the test split except to build a loader for
the *final* evaluation performed in ``evaluate.py``.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import socket
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config as C
from utils import relpath_posix, save_figure, set_seed, worker_init_fn


# =========================================================================== #
# 1. Download                                                                 #
# =========================================================================== #
def _md5_of_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download_file(url: str, dest: Path, expected_size: int | None = None,
                  expected_md5: str | None = None) -> Path:
    """Stream ``url`` to ``dest`` with progress, resume, and verification.

    Pure standard library so it behaves identically on Windows and Linux.
    Resumes a partial ``dest.part`` file using an HTTP Range request.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    if dest.exists():
        if expected_md5 and _md5_of_file(dest) == expected_md5:
            print(f"[data] already downloaded and verified: {dest}")
            return dest
        print("[data] existing file failed verification, re-downloading")

    socket.setdefaulttimeout(120)
    downloaded = part.stat().st_size if part.exists() else 0

    req = urllib.request.Request(url, headers={"User-Agent": "aider-project/1.0"})
    if downloaded:
        req.add_header("Range", f"bytes={downloaded}-")
        print(f"[data] resuming download at {downloaded / 1e6:.1f} MB")

    mode = "ab" if downloaded else "wb"
    with urllib.request.urlopen(req) as resp:
        total = expected_size or 0
        if resp.status == 206 and "Content-Range" in resp.headers:
            total = int(resp.headers["Content-Range"].split("/")[-1])
        elif resp.headers.get("Content-Length"):
            total = downloaded + int(resp.headers["Content-Length"])
        with open(part, mode) as fh:
            last = 0
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                fh.write(block)
                downloaded += len(block)
                if total and downloaded - last > 20e6:
                    last = downloaded
                    print(f"[data]   {downloaded / 1e6:7.1f} / {total / 1e6:.1f} MB "
                          f"({100 * downloaded / total:5.1f}%)")

    if expected_size and part.stat().st_size != expected_size:
        raise RuntimeError(
            f"size mismatch for {part.name}: got {part.stat().st_size}, "
            f"expected {expected_size}")

    if expected_md5:
        got = _md5_of_file(part)
        if got != expected_md5:
            raise RuntimeError(f"md5 mismatch: got {got}, expected {expected_md5}")

    part.replace(dest)
    print(f"[data] downloaded {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def unzip_to(zip_path: Path, out_dir: Path) -> None:
    """Extract an archive with the stdlib ``zipfile`` (Windows-friendly)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        print(f"[data] extracting {len(names)} entries -> {out_dir}")
        zf.extractall(out_dir)


def locate_class_dirs(root: Path) -> dict[str, Path]:
    """Find one directory per AIDER class anywhere under ``root``.

    The archive names two of its folders differently from our canonical labels
    (``flooded_areas`` -> ``flood`` and ``traffic_incident`` ->
    ``traffic_accident``); ``config.CLASS_DIR_ALIASES`` holds that mapping, so
    the same code also accepts the canonical names if a mirror uses them.
    """
    found: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        key = path.name.lower().replace(" ", "_").replace("-", "_")
        canonical = C.CLASS_DIR_ALIASES.get(key)
        if canonical is not None:
            found.setdefault(canonical, path)
    return found


def ensure_dataset(force: bool = False) -> dict[str, Path]:
    """Download + extract AIDER if needed and return {class_name: folder}.

    Tries the primary Zenodo link and, if that fails, a list of documented
    mirrors.  Prints exactly which source was used.
    """
    if force:
        # ``force`` must remove BOTH the archive and everything already
        # extracted, otherwise ``locate_class_dirs`` would still find the old
        # class folders and the function would return early (a silent no-op).
        if C.ZIP_PATH.exists():
            C.ZIP_PATH.unlink()
        for name, folder in locate_class_dirs(C.DATA_DIR).items():
            if folder.exists() and C.DATA_DIR in folder.parents:
                shutil.rmtree(folder, ignore_errors=True)
                print(f"[data] --force-data: removed extracted folder {folder}")

    class_dirs = locate_class_dirs(C.DATA_DIR)
    if len(class_dirs) == len(C.CLASSES):
        print(f"[data] using already-extracted dataset in {C.DATA_DIR}")
        return class_dirs

    sources = [
        ("zenodo", C.AIDER_URL, C.AIDER_MD5),
        # Documented mirrors (see README). md5 may differ for re-packed mirrors,
        # so they are accepted on size/integrity alone.
        ("github-kyrkou", "https://github.com/ChristosKyrkou/AIDER/raw/master/AIDER.zip", None),
    ]
    last_err: Exception | None = None
    for name, url, md5 in sources:
        try:
            print(f"[data] downloading AIDER from source '{name}': {url}")
            download_file(url, C.ZIP_PATH, C.AIDER_ZIP_SIZE if md5 else None, md5)
            print(f"[data] SUCCESS: dataset source used = {name}")
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"[data] source '{name}' failed: {exc}")
    else:
        raise RuntimeError(f"all AIDER download sources failed; last error: {last_err}")

    unzip_to(C.ZIP_PATH, C.DATA_DIR)

    class_dirs = locate_class_dirs(C.DATA_DIR)
    missing = [c for c in C.CLASSES if c not in class_dirs]
    if missing:
        raise RuntimeError(
            f"could not find folders for classes {missing} under {C.DATA_DIR}. "
            f"Found: {sorted(class_dirs)}")
    return class_dirs


# =========================================================================== #
# 2. Integrity scan + duplicate removal                                       #
# =========================================================================== #
@dataclass
class CleaningReport:
    n_scanned: int = 0
    n_removed_unreadable: int = 0
    n_removed_zero_size: int = 0
    n_removed_duplicate: int = 0
    removed: dict = field(default_factory=dict)     # reason -> per-class counts

    def add(self, reason: str, cls: str) -> None:
        self.removed.setdefault(reason, Counter())[cls] += 1

    @property
    def n_removed_total(self) -> int:
        return (self.n_removed_unreadable + self.n_removed_zero_size
                + self.n_removed_duplicate)


def _probe_image(path: Path) -> tuple[int, int]:
    """Return (width, height); raise if the file is not a readable image."""
    from PIL import Image

    with Image.open(path) as im:
        im.verify()                       # detects truncated / corrupt files
    with Image.open(path) as im:          # verify() invalidates the handle
        im = im.convert("RGB")
        w, h = im.size
        im.load()
    return w, h


def scan_and_clean(class_dirs: dict[str, Path]) -> tuple[pd.DataFrame, CleaningReport]:
    """Walk every image, drop invalid/zero-byte files and exact duplicates.

    Exact duplicates are found by MD5 over file bytes and removed *globally*
    (keeping the first occurrence), which guarantees an identical image cannot
    appear in two different splits.
    """
    report = CleaningReport()
    rows: list[dict] = []
    seen_md5: dict[str, Path] = {}

    for cls in C.CLASSES:
        folder = class_dirs[cls]
        files = sorted(p for p in folder.rglob("*")
                       if p.is_file() and p.suffix.lower() in C.IMG_EXTS)
        print(f"[data] scanning {cls:<20s} {len(files):5d} files")
        for path in files:
            report.n_scanned += 1

            if path.stat().st_size == 0:
                report.n_removed_zero_size += 1
                report.add("zero_size", cls)
                continue

            try:
                w, h = _probe_image(path)
            except Exception:                              # noqa: BLE001
                report.n_removed_unreadable += 1
                report.add("unreadable", cls)
                continue

            md5 = _md5_of_file(path)
            if md5 in seen_md5:
                report.n_removed_duplicate += 1
                report.add("duplicate", cls)
                continue
            seen_md5[md5] = path

            rows.append({
                "path": relpath_posix(path, C.DATA_DIR),
                "class_name": cls,
                "label": C.CLASSES.index(cls),
                "md5": md5,
                "width": w,
                "height": h,
                "bytes": path.stat().st_size,
            })

    df = pd.DataFrame(rows).sort_values(["label", "path"]).reset_index(drop=True)
    return df, report


def resolution_stats(df: pd.DataFrame) -> pd.DataFrame:
    """min/median/max width and height, overall and per class."""
    out = []
    for name, group in [("ALL", df)] + list(df.groupby("class_name")):
        out.append({
            "scope": name,
            "n": len(group),
            "width_min": int(group["width"].min()),
            "width_median": float(group["width"].median()),
            "width_max": int(group["width"].max()),
            "height_min": int(group["height"].min()),
            "height_median": float(group["height"].median()),
            "height_max": int(group["height"].max()),
        })
    return pd.DataFrame(out)


# =========================================================================== #
# 3. Splits                                                                   #
# =========================================================================== #
def make_splits(df: pd.DataFrame, seed: int = C.SEED) -> pd.DataFrame:
    """Stratified 70/15/15 split.  Returns a copy of ``df`` with a 'split' column."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(df))
    labels = df["label"].to_numpy()

    idx_trainval, idx_test = train_test_split(
        idx, test_size=C.TEST_FRAC, stratify=labels, random_state=seed)

    val_share = C.VAL_FRAC / (C.TRAIN_FRAC + C.VAL_FRAC)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_share,
        stratify=labels[idx_trainval], random_state=seed)

    split = np.empty(len(df), dtype=object)
    split[idx_train] = "train"
    split[idx_val] = "val"
    split[idx_test] = "test"

    out = df.copy()
    out["split"] = split
    return out


def assert_no_overlap(df: pd.DataFrame) -> None:
    """Hard guarantee that train/val/test share no file and no image content."""
    sets = {s: set(df.loc[df["split"] == s, "path"]) for s in ("train", "val", "test")}
    md5s = {s: set(df.loc[df["split"] == s, "md5"]) for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (sets[a] & sets[b]), f"path overlap between {a} and {b}: {sets[a] & sets[b]}"
        assert not (md5s[a] & md5s[b]), f"content overlap between {a} and {b}"
    assert sum(len(v) for v in sets.values()) == len(df), "split did not cover every row"
    print("[data] overlap check passed: train/val/test are disjoint by path AND by md5")


def split_table(df: pd.DataFrame) -> pd.DataFrame:
    tab = (df.groupby(["class_name", "split"]).size()
             .unstack(fill_value=0)
             .reindex(C.CLASSES))
    tab = tab.reindex(columns=["train", "val", "test"], fill_value=0)
    tab["total"] = tab.sum(axis=1)
    return tab


# =========================================================================== #
# 4. Transforms / Dataset / DataLoaders                                       #
# =========================================================================== #
def build_transforms(augment: bool, img_size: int = C.IMG_SIZE):
    """Train transforms (optional augmentation) and eval transforms (resize+norm)."""
    from torchvision import transforms

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])

    if not augment:
        return eval_tf, eval_tf

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=C.AUG_CROP_SCALE),
        transforms.RandomHorizontalFlip(p=C.AUG_HFLIP_P),
        transforms.RandomRotation(degrees=C.AUG_ROTATION_DEG),
        transforms.ColorJitter(brightness=C.AUG_JITTER, contrast=C.AUG_JITTER),
        transforms.ToTensor(),
        transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])
    return train_tf, eval_tf


class AiderDataset(Dataset):
    """Minimal image-classification dataset (``torch.utils.data.Dataset``).

    Implemented from scratch (rather than ``ImageFolder``) so the exact file
    list coming out of ``splits.csv`` is what gets used.  Defined at module
    level (not inside a factory function) so that DataLoader worker processes
    can pickle it on Windows, where workers are ``spawn``ed.
    """

    def __init__(self, records: Sequence[tuple[str, int]], data_dir: Path, transform):
        self.records = list(records)
        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        from PIL import Image

        rel_path, label = self.records[i]
        with Image.open(self.data_dir / rel_path) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    # Kept tiny so DataLoader workers can pickle it cheaply on Windows.
    def __getstate__(self):
        return {"records": self.records, "data_dir": str(self.data_dir),
                "transform": self.transform}

    def __setstate__(self, state):
        self.records = state["records"]
        self.data_dir = Path(state["data_dir"])
        self.transform = state["transform"]


def build_dataloaders(df: pd.DataFrame, augment: bool, batch_size: int = C.BATCH_SIZE,
                      num_workers: int | None = None, seed: int = C.SEED,
                      include_test: bool = True):
    """Return (train_loader, val_loader, test_loader) for one experiment.

    With ``include_test=False`` the test split is never even wrapped in a
    dataset, and the third element of the tuple is ``None``.  ``train.py`` uses
    that so the test paths are not referenced at all while a model is being
    trained; only ``evaluate.py`` builds the real test loader.
    """
    from torch.utils.data import DataLoader

    nw = C.NUM_WORKERS if num_workers is None else num_workers
    train_tf, eval_tf = build_transforms(augment)

    def recs(split):
        sub = df[df["split"] == split]
        return list(zip(sub["path"].tolist(), sub["label"].tolist()))

    train_ds = AiderDataset(recs("train"), C.DATA_DIR, train_tf)
    val_ds = AiderDataset(recs("val"), C.DATA_DIR, eval_tf)

    pin = torch.cuda.is_available()
    common = dict(num_workers=nw, pin_memory=pin, persistent_workers=nw > 0,
                  worker_init_fn=worker_init_fn)
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=g, drop_last=False, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = None
    if include_test:
        test_ds = AiderDataset(recs("test"), C.DATA_DIR, eval_tf)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def class_weights_from_train(df: pd.DataFrame) -> "np.ndarray":
    """Class weights = N / (K * n_c), computed on the TRAIN split only."""
    counts = (df[df["split"] == "train"]["label"].value_counts()
                .reindex(range(C.NUM_CLASSES), fill_value=0).to_numpy(dtype=np.float64))
    if (counts == 0).any():
        raise RuntimeError(f"a class has zero training images: counts={counts}")
    weights = counts.sum() / (C.NUM_CLASSES * counts)
    print("[data] train class counts:", counts.astype(int).tolist())
    print("[data] class weights      :", np.round(weights, 4).tolist())
    return weights.astype(np.float32)


# =========================================================================== #
# 5. Figures                                                                  #
# =========================================================================== #
def figure_class_distribution(df: pd.DataFrame, report: CleaningReport,
                              out: Path = C.FIGURES_DIR / "class_distribution.png") -> Path:
    import matplotlib.pyplot as plt

    counts = df["class_name"].value_counts().reindex(C.CLASSES)
    removed_per_class = [sum(report.removed.get(r, Counter()).get(c, 0)
                             for r in ("duplicate", "unreadable", "zero_size"))
                         for c in C.CLASSES]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(C.CLASSES))
    bars = ax.bar(x, counts.to_numpy(), color="#3b6ea5", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(C.CLASSES, rotation=15, ha="right")
    ax.set_ylabel("number of images")
    ax.set_xlabel("class")
    ax.set_title(f"AIDER class distribution after cleaning (n = {len(df)})")
    for b, v, r in zip(bars, counts.to_numpy(), removed_per_class):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}", ha="center", va="bottom",
                fontsize=11)
        if r:
            ax.text(b.get_x() + b.get_width() / 2, v * 0.03, f"-{r}\nremoved",
                    ha="center", va="bottom", fontsize=8, color="crimson")
    ax.set_ylim(0, float(counts.max()) * 1.18)
    fig.tight_layout()
    return save_figure(fig, out, C.FIGURE_DPI)


def figure_sample_grid(df: pd.DataFrame, per_class: int = 4,
                       out: Path = C.FIGURES_DIR / "samples_grid.png") -> Path:
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, axes = plt.subplots(len(C.CLASSES), per_class,
                             figsize=(3.0 * per_class, 3.0 * len(C.CLASSES)))
    for r, cls in enumerate(C.CLASSES):
        sub = df[df["class_name"] == cls].sample(
            n=min(per_class, int((df["class_name"] == cls).sum())),
            random_state=C.SEED)
        for c in range(per_class):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(sub):
                with Image.open(C.DATA_DIR / sub.iloc[c]["path"]) as im:
                    ax.imshow(im.convert("RGB"))
                ax.set_title(f"{sub.iloc[c]['width']}x{sub.iloc[c]['height']}", fontsize=9)
        axes[r, 0].text(-0.18, 0.5, cls, transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=13, fontweight="bold")
    fig.suptitle("AIDER: sample images per class (4 per class)", fontsize=17,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(fig, out, C.FIGURE_DPI)


def figure_augmentation_examples(record: tuple[str, int], n: int = 6,
                                 out: Path = C.FIGURES_DIR / "augmentation_examples.png") -> Path:
    """Same image seen through several independent random augmentations."""
    import matplotlib.pyplot as plt
    from PIL import Image

    tf, _ = build_transforms(augment=True)
    with Image.open(C.DATA_DIR / record[0]) as im:
        base = im.convert("RGB")

    mean = np.array(C.IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(C.IMAGENET_STD).reshape(3, 1, 1)

    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.1))
    axes[0].imshow(base)
    axes[0].set_title("original", fontsize=11)
    axes[0].axis("off")
    for i in range(1, n):
        arr = tf(base).numpy()
        vis = np.clip(arr * std + mean, 0, 1).transpose(1, 2, 0)
        axes[i].imshow(vis)
        axes[i].set_title(f"aug sample {i}", fontsize=11)
        axes[i].axis("off")
    fig.suptitle("Training augmentation: one image under random augmentations",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save_figure(fig, out, C.FIGURE_DPI)


# =========================================================================== #
# 6. High level pipeline helpers used by run_all.py                           #
# =========================================================================== #
def load_splits() -> pd.DataFrame:
    return pd.read_csv(C.RESULTS_DIR / "splits.csv")


def prepare_dataset(force_download: bool = False) -> pd.DataFrame:
    """Download -> clean -> split -> persist CSVs.  Returns the split DataFrame."""
    class_dirs = ensure_dataset(force=force_download)

    clean_report_path = C.RESULTS_DIR / "cleaning_report.json"
    records_path = C.RESULTS_DIR / "records.csv"

    if records_path.exists() and clean_report_path.exists() and not force_download:
        df = pd.read_csv(records_path)
        print(f"[data] reusing cached record list ({len(df)} images)")
    else:
        df, report = scan_and_clean(class_dirs)
        df.to_csv(records_path, index=False)

        res = resolution_stats(df)
        res.to_csv(C.RESULTS_DIR / "resolution_stats.csv", index=False)

        # class distribution (raw + cleaned)
        dist = df["class_name"].value_counts().reindex(C.CLASSES).fillna(0).astype(int)
        dist.rename("count").to_frame().to_csv(C.RESULTS_DIR / "class_distribution.csv")

        report_json = {
            "n_scanned": report.n_scanned,
            "n_kept": int(len(df)),
            "n_removed_unreadable": report.n_removed_unreadable,
            "n_removed_zero_size": report.n_removed_zero_size,
            "n_removed_duplicate": report.n_removed_duplicate,
            "n_removed_total": report.n_removed_total,
            "removed_per_class_per_reason": {
                reason: dict(counts) for reason, counts in report.removed.items()
            },
        }
        clean_report_path.write_text(json.dumps(report_json, indent=2), encoding="utf-8")
        print(f"[data] removed {report.n_removed_total} files "
              f"(dup={report.n_removed_duplicate}, "
              f"unreadable={report.n_removed_unreadable}, "
              f"zero_size={report.n_removed_zero_size})")
        # keep the report around for the figure
        globals()["_LAST_REPORT"] = report

    report = _report_from_disk(clean_report_path)

    if not (C.RESULTS_DIR / "splits.csv").exists() or force_download:
        set_seed(C.SEED)
        df = make_splits(df, seed=C.SEED)
        assert_no_overlap(df)
        df.to_csv(C.RESULTS_DIR / "splits.csv", index=False)
        split_table(df).to_csv(C.RESULTS_DIR / "split_table.csv")
        print("[data] wrote splits.csv and split_table.csv")
    else:
        df = load_splits()
        assert_no_overlap(df)
        print("[data] reusing existing splits.csv")

    return df


def _report_from_disk(path: Path) -> CleaningReport:
    rep = CleaningReport()
    if not path.exists():
        return rep
    obj = json.loads(path.read_text(encoding="utf-8"))
    rep.n_scanned = obj.get("n_scanned", 0)
    rep.n_removed_unreadable = obj.get("n_removed_unreadable", 0)
    rep.n_removed_zero_size = obj.get("n_removed_zero_size", 0)
    rep.n_removed_duplicate = obj.get("n_removed_duplicate", 0)
    per = obj.get("removed_per_class_per_reason", {})
    for reason, counts in per.items():
        rep.removed[reason] = Counter(counts)
    return rep


def main() -> None:
    """Stand-alone entry point: ``python data.py`` runs download+EDA+splits."""
    df = prepare_dataset()
    rep = _report_from_disk(C.RESULTS_DIR / "cleaning_report.json")
    figure_class_distribution(df, rep)
    figure_sample_grid(df)
    # pick a random training image from each class for the augmentation figure
    row = df[df["split"] == "train"].sample(n=1, random_state=C.SEED).iloc[0]
    figure_augmentation_examples((row["path"], int(row["label"])), n=6)
    print(split_table(df))
    print("[data] done")


if __name__ == "__main__":
    main()
