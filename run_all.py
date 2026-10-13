"""
run_all.py
==========
End-to-end pipeline:  download -> clean -> split -> EDA -> models -> train ->
evaluate -> summarise.

Usage (identical on Windows, Linux and macOS)::

    python run_all.py                     # full pipeline
    python run_all.py --quick             # tiny smoke test (few epochs)
    python run_all.py --force-data        # re-download and re-clean the dataset
    python run_all.py --jobs 3            # how many trainings run in parallel

The test split is used exclusively by ``evaluate.py`` for the final evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from utils import apply_plot_style, human_hms, timer

# --------------------------------------------------------------------------- #
# Console tee -> results/logs/run_all.log                                     #
# --------------------------------------------------------------------------- #
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def banner(msg: str) -> None:
    print("\n" + "#" * 92)
    print(f"# {msg}")
    print("#" * 92, flush=True)


# =========================================================================== #
# Training orchestration                                                       #
# =========================================================================== #
def _run_is_done(exp_id: str, seed: int) -> bool:
    return ((C.CKPT_DIR / f"{exp_id}_seed{seed}_best.pt").exists()
            and (C.LOG_DIR / f"{exp_id}_seed{seed}_history.csv").exists())


def _spawn_train(exp_id: str, seed: int, epochs_scale: float, threads: int,
                 patience: int) -> tuple[str, int, int, str]:
    env = dict(os.environ)
    env["AIDER_THREADS"] = str(threads)
    env["AIDER_EPOCH_SCALE"] = str(epochs_scale)
    cmd = [sys.executable, str(C.PROJECT_ROOT / "train.py"),
           "--exp", exp_id, "--seed", str(seed),
           "--epochs-scale", str(epochs_scale), "--patience", str(patience),
           "--threads", str(threads)]
    log_path = C.LOG_DIR / f"train_{exp_id}_seed{seed}.log"
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=str(C.PROJECT_ROOT), env=env,
                              stdout=fh, stderr=subprocess.STDOUT)
    return exp_id, seed, proc.returncode, human_hms(time.perf_counter() - t0)


def run_training(exp_ids: list[str], seeds: list[int], epochs_scale: float,
                 max_jobs: int = 3, patience: int = C.EARLY_STOP_PATIENCE,
                 resume: bool = True) -> list[dict]:
    """Train every (experiment, seed) pair, ``max_jobs`` processes in parallel."""
    cpu = os.cpu_count() or 4
    threads = max(1, cpu // max(1, max_jobs))
    # Keep the extra process count small: this box is memory-bandwidth bound and
    # extra DataLoader processes mostly add contention (measured, see README).
    workers = max(0, min(1, threads // 3))
    os.environ["AIDER_NUM_WORKERS"] = str(workers)

    jobs = [(e, s) for e in exp_ids for s in seeds]
    if resume:
        jobs = [j for j in jobs if not _run_is_done(*j)]
    print(f"[run_all] {len(jobs)} training job(s); {max_jobs} in parallel; "
          f"{threads} torch threads and {workers} DataLoader worker(s) each")

    results = []
    if not jobs:
        print("[run_all] all trainings already present - skipping")
    else:
        with ThreadPoolExecutor(max_workers=max_jobs) as pool:
            futs = [pool.submit(_spawn_train, e, s, epochs_scale, threads, patience)
                    for e, s in jobs]
            for fut in futs:
                exp_id, seed, rc, took = fut.result()
                status = "OK" if rc == 0 else f"FAILED(rc={rc})"
                print(f"[run_all] {exp_id} seed {seed}: {status} in {took}")
                results.append({"exp": exp_id, "seed": seed, "rc": rc, "wall": took})

    # collect metadata written by each training process
    metas = []
    for exp_id in exp_ids:
        for seed in seeds:
            p = C.RESULTS_DIR / f"run_{exp_id}_seed{seed}.json"
            if p.exists():
                metas.append(json.loads(p.read_text(encoding="utf-8")))
    return metas


# =========================================================================== #
# summary.md                                                                  #
# =========================================================================== #
def _md_table(df: pd.DataFrame, index: bool = False, floatfmt: str = "{:.4f}") -> str:
    d = df.copy()
    for col in d.columns:
        if pd.api.types.is_float_dtype(d[col]):
            d[col] = d[col].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    return d.to_markdown(index=index)


def write_summary() -> Path:
    """Assemble results/summary.md purely from files that actually exist."""
    clean = json.loads((C.RESULTS_DIR / "cleaning_report.json").read_text(encoding="utf-8"))
    dist = pd.read_csv(C.RESULTS_DIR / "class_distribution.csv", index_col=0)
    res = pd.read_csv(C.RESULTS_DIR / "resolution_stats.csv")
    split_tab = pd.read_csv(C.RESULTS_DIR / "split_table.csv", index_col=0)
    model_sum = pd.read_csv(C.RESULTS_DIR / "model_summary.csv")
    test_metrics = pd.read_csv(C.RESULTS_DIR / "test_metrics.csv")
    agg = pd.read_csv(C.RESULTS_DIR / "test_metrics_aggregated.csv")
    infer = pd.read_csv(C.RESULTS_DIR / "inference_time.csv")

    metas = [json.loads(p.read_text(encoding="utf-8"))
             for p in sorted(C.RESULTS_DIR.glob("run_*_seed*.json"))]
    meta_df = pd.DataFrame(metas)

    L: list[str] = []
    add = L.append

    add("# AIDER Disaster-Response Image Classification - Results Summary\n")
    add("_Generated automatically by `run_all.py` from the CSV/JSON artefacts in "
        "`results/`. Every number below comes from an actual run._\n")

    # ---------------------------------------------------------------- dataset
    add("## 1. Dataset\n")
    add("- **Name:** AIDER (Aerial Image Dataset for Emergency Response Applications)")
    add("- **Authors:** Christos Kyrkou, Theocharis Theocharides")
    add("- **Source / DOI:** Zenodo record [3888300](https://zenodo.org/record/3888300), "
        "DOI [10.5281/zenodo.3888300](https://doi.org/10.5281/zenodo.3888300)")
    add(f"- **Download URL used:** `{C.AIDER_URL}`")
    add(f"- **Archive MD5:** `{C.AIDER_MD5}` (verified after download)")
    add("- **Licence:** CC-BY-4.0")
    add(f"- **Files scanned:** {clean['n_scanned']}")
    add(f"- **Removed:** {clean['n_removed_duplicate']} exact duplicates, "
        f"{clean['n_removed_unreadable']} unreadable, {clean['n_removed_zero_size']} zero-byte "
        f"(total {clean['n_removed_total']})")
    add(f"- **Images retained:** {clean['n_kept']}\n")
    add("Per-class counts after cleaning:\n")
    add(_md_table(dist.reset_index().rename(columns={"class_name": "class", "count": "images"}),
                  index=False))
    add("")
    rem = clean["removed_per_class_per_reason"]
    if rem:
        add("Removed per class and reason:\n")
        rdf = pd.DataFrame({k: pd.Series(v) for k, v in rem.items()}).fillna(0).astype(int)
        add(_md_table(rdf.reset_index().rename(columns={"index": "class"}), index=False))
        add("")
    add("Resolution statistics (original files, before resizing to "
        f"{C.IMG_SIZE}x{C.IMG_SIZE}):\n")
    add(_md_table(res, index=False, floatfmt="{:.1f}"))
    add("")

    # ---------------------------------------------------------------- splits
    add("## 2. Train / validation / test split\n")
    add(f"Stratified {int(C.TRAIN_FRAC*100)}/{int(C.VAL_FRAC*100)}/"
        f"{int(C.TEST_FRAC*100)} split, `random_state={C.SEED}`. "
        "Overlap is asserted to be zero by *path* and by *file content (MD5)*.\n")
    add(_md_table(split_tab.reset_index().rename(columns={"class_name": "class"}), index=False))
    add("")
    add(f"Totals: train={int(split_tab['train'].sum())}, "
        f"val={int(split_tab['val'].sum())}, test={int(split_tab['test'].sum())}.\n")

    # ------------------------------------------------------------ preprocessing
    add("## 3. Pre-processing\n")
    add(f"- Resize to {C.IMG_SIZE}x{C.IMG_SIZE}, tensor conversion, ImageNet "
        f"normalisation (mean={C.IMAGENET_MEAN}, std={C.IMAGENET_STD}) for **both** models.")
    add(f"- **Training augmentation only:** RandomResizedCrop(scale={C.AUG_CROP_SCALE}), "
        f"RandomHorizontalFlip(p={C.AUG_HFLIP_P}), RandomRotation(+-{C.AUG_ROTATION_DEG}deg), "
        f"ColorJitter(brightness={C.AUG_JITTER}, contrast={C.AUG_JITTER}).")
    add("- Validation and test: resize + normalise only (no augmentation, no random crop).")
    add("- **Class imbalance** handled with class-weighted cross-entropy, weights computed "
        "from the TRAIN split only (see per-run JSON for the exact values).")
    add("")

    # ------------------------------------------------------------- models
    add("## 4. Architectures\n")
    add(_md_table(model_sum, index=False))
    add("")
    if meta_df.empty:
        add("_No training runs were recorded._")
    else:
        add(f"Device used for all runs: `{meta_df['device'].iloc[0]}`.\n")

    # ------------------------------------------------------------- hyperparams
    add("## 5. Hyper-parameters actually used\n")
    if not meta_df.empty:
        hp = (meta_df.groupby(["batch_size", "img_size", "weight_decay", "early_stop_patience"])
              .size().reset_index(name="runs"))
        add(_md_table(hp, index=False, floatfmt="{:.0f}"))
        add("")
    add(f"- Optimiser: AdamW, weight decay {C.WEIGHT_DECAY}.")
    add(f"- Model A: {C.LR_MODEL_A} cosine-annealed to 1e-5, up to {C.EPOCHS_MODEL_A} epochs.")
    add(f"- Model B stage 1: head only, lr {C.LR_HEAD}, {C.EPOCHS_B_STAGE1} epochs; "
        f"stage 2: full fine-tune, lr {C.LR_FINETUNE} cosine-annealed to {C.LR_FINETUNE_MIN}, "
        f"up to {C.EPOCHS_B_STAGE2} epochs.")
    add(f"- Early stopping on **validation macro-F1**, patience {C.EARLY_STOP_PATIENCE}; "
        "the best checkpoint by validation macro-F1 is the one evaluated on test.")
    add(f"- Mixed precision (AMP) enabled only when running on CUDA "
        f"(`device={meta_df['device'].iloc[0] if not meta_df.empty else 'cpu'}`).\n")

    # ---------------------------------------------------------- training time
    if not meta_df.empty:
        add("## 6. Training time and epochs per run\n")
        tt = meta_df[["exp", "seed", "epochs_run", "best_epoch", "best_val_macro_f1",
                      "epoch_seconds_mean", "train_time_hms"]].copy()
        tt.columns = ["experiment", "seed", "epochs run", "best epoch",
                      "best val macro-F1", "s per epoch", "wall clock"]
        add(_md_table(tt.sort_values(["experiment", "seed"]), index=False))
        add("")

    # ----------------------------------------------------------- test metrics
    add("## 7. Final TEST metrics (evaluated once per trained model)\n")
    add("Per-seed results:\n")
    cols = ["exp", "seed", "accuracy", "macro_f1", "weighted_f1", "macro_roc_auc_ovr",
            "miss_rate", "false_alarm_rate"]
    add(_md_table(test_metrics[cols].sort_values(["exp", "seed"]), index=False))
    add("")
    add("Mean +/- std over seeds:\n")
    agg_show = agg[["exp", "n_seeds", "accuracy_mean", "accuracy_std",
                    "macro_f1_mean", "macro_f1_std", "roc_auc_mean",
                    "miss_rate_mean", "false_alarm_rate_mean"]].copy()
    agg_show.columns = ["experiment", "n seeds", "accuracy mean", "accuracy std",
                        "macro-F1 mean", "macro-F1 std", "ROC-AUC mean",
                        "miss-rate mean", "false-alarm mean"]
    add(_md_table(agg_show, index=False))
    add("")

    # ------------------------------------------------- disaster vs normal
    add("## 8. Disaster-vs-normal decision view (the operationally important one)\n")
    add("All four disaster classes are collapsed into one `disaster` class and compared "
        "against `normal`.\n")
    add("- **Miss rate** = fraction of true disasters classified as `normal` "
        "(a missed emergency).")
    add("- **False-alarm rate** = fraction of `normal` images flagged as a disaster.\n")
    dv = test_metrics[["exp", "seed", "miss_rate", "false_alarm_rate"]].copy()
    add(_md_table(dv.sort_values(["exp", "seed"]), index=False))
    add("")

    # ------------------------------------------------ per-class for main model
    main = C.MAIN_EXPERIMENT
    per_class_rows = []
    for _, r in test_metrics.iterrows():
        for cls in C.CLASSES:
            per_class_rows.append({"exp": r["exp"], "seed": r["seed"], "class": cls,
                                   "precision": r[f"precision_{cls}"],
                                   "recall": r[f"recall_{cls}"], "f1": r[f"f1_{cls}"]})
    pc = pd.DataFrame(per_class_rows)
    pc_mean = (pc.groupby(["exp", "class"])[["precision", "recall", "f1"]]
                 .mean().reset_index())
    add("## 9. Per-class precision / recall / F1 (mean over seeds)\n")
    add(_md_table(pc_mean, index=False))
    add("")

    main_pc = pc_mean[pc_mean["exp"] == main].sort_values("f1")
    if not main_pc.empty:
        hardest = main_pc.iloc[0]
        add(f"**Most difficult class (main model `{main}`):** `{hardest['class']}` with "
            f"mean F1 = {hardest['f1']:.4f}, recall = {hardest['recall']:.4f}, "
            f"precision = {hardest['precision']:.4f}.\n")
        easiest = main_pc.iloc[-1]
        add(f"**Easiest class:** `{easiest['class']}` with mean F1 = {easiest['f1']:.4f}.\n")

    # ---------------------------------------------------- confusions
    conf_rows = []
    for _, r in test_metrics.iterrows():
        mp = C.RESULTS_DIR / f"metrics_{r['exp']}_seed{r['seed']}.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        for c in m.get("top_confusions", []):
            conf_rows.append({"exp": r["exp"], "seed": r["seed"], "true": c["true"],
                              "pred": c["pred"], "count": c["count"]})
    if conf_rows:
        cdf = pd.DataFrame(conf_rows)
        top = (cdf[cdf["exp"] == main].groupby(["true", "pred"])["count"]
               .mean().reset_index().sort_values("count", ascending=False).head(8))
        add("## 10. Top confusion pairs for the main model (mean count over seeds)\n")
        add(_md_table(top, index=False, floatfmt="{:.1f}"))
        add("")

    # ---------------------------------------------------- ablation
    add("## 11. Augmentation ablation\n")
    def _ablation(order: list[tuple[str, str, str]]) -> None:
        for noaug, aug, name in order:
            if noaug in set(agg["exp"]) and aug in set(agg["exp"]):
                a = agg[agg["exp"] == aug].iloc[0]
                b = agg[agg["exp"] == noaug].iloc[0]
                add(f"- **{name}:** with augmentation accuracy = {a['accuracy_mean']:.4f} "
                    f"(macro-F1 {a['macro_f1_mean']:.4f}); without augmentation accuracy = "
                    f"{b['accuracy_mean']:.4f} (macro-F1 {b['macro_f1_mean']:.4f}). "
                    f"Delta accuracy = {a['accuracy_mean'] - b['accuracy_mean']:+.4f}, "
                    f"delta macro-F1 = {a['macro_f1_mean'] - b['macro_f1_mean']:+.4f}.")
    _ablation([("exp3_modelB_noaug", "exp2_modelB_aug", "Model B (MobileNetV2 transfer)"),
               ("exp4_modelA_noaug", "exp1_modelA_aug", "Model A (Small CNN from scratch)")])
    add("")

    # ---------------------------------------------------- inference time
    add("## 12. Inference cost (edge-deployment relevance)\n")
    add(_md_table(infer, index=False, floatfmt="{:.2f}"))
    add("")
    add("Trained-model inference time measured on the TEST set "
        "(per-experiment values are in `test_metrics.csv`).\n")

    # ---------------------------------------------------- observations
    add("## 13. Observations\n")
    for obs in _observations(test_metrics, agg, pc_mean, infer, clean, split_tab):
        add(f"- {obs}")
    add("")

    # ---------------------------------------------------- reproduction
    add("## 14. Reproducing\n")
    add("```bash\npip install -r requirements.txt\npython run_all.py\n```\n")
    add("Deterministic seeds (Python/NumPy/PyTorch + cuDNN deterministic kernels) and a "
        f"fixed data split (`results/splits.csv`) are used everywhere. "
        f"Repeated experiments use seeds {C.SEEDS}.\n")

    out = C.RESULTS_DIR / "summary.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"[run_all] wrote {out}")
    return out


def _observations(test_metrics, agg, pc_mean, infer, clean, split_tab) -> list[str]:
    """Factual observations computed from the saved numbers (nothing invented)."""
    obs: list[str] = []
    main = C.MAIN_EXPERIMENT
    best = agg.sort_values("macro_f1_mean", ascending=False).iloc[0]
    obs.append(
        f"Best mean test macro-F1 was **{best['macro_f1_mean']:.4f} "
        f"(+/- {0 if pd.isna(best['macro_f1_std']) else best['macro_f1_std']:.4f})** for "
        f"`{best['exp']}`, against {agg['macro_f1_mean'].min():.4f} for the weakest "
        f"experiment - transfer learning beats the from-scratch CNN by "
        f"{agg[agg['exp']==main]['macro_f1_mean'].max() - agg[agg['exp']=='exp1_modelA_aug']['macro_f1_mean'].mean():+.4f} "
        f"macro-F1."
        if main in set(agg["exp"]) and "exp1_modelA_aug" in set(agg["exp"]) else
        f"Best mean test macro-F1 was {best['macro_f1_mean']:.4f} for `{best['exp']}`.")

    imbalance = split_tab["total"].max() / split_tab["total"].min()
    obs.append(
        f"The dataset is imbalanced: the largest class (`{split_tab['total'].idxmax()}`) has "
        f"{imbalance:.1f}x the images of the smallest "
        f"(`{split_tab['total'].idxmin()}`), which is why class-weighted cross-entropy is used.")

    # hardest class for main model
    m = pc_mean[pc_mean["exp"] == main].sort_values("f1")
    if len(m):
        obs.append(
            f"For the main model the hardest class is `{m.iloc[0]['class']}` "
            f"(F1 {m.iloc[0]['f1']:.4f}, recall {m.iloc[0]['recall']:.4f}); the easiest is "
            f"`{m.iloc[-1]['class']}` (F1 {m.iloc[-1]['f1']:.4f}).")

    # miss vs false alarm
    d = test_metrics[test_metrics["exp"] == main]
    if len(d):
        obs.append(
            f"Collapsing to disaster-vs-normal, the main model misses "
            f"{100 * d['miss_rate'].mean():.2f}% of true disasters (classified as `normal`) "
            f"while falsely alarming on {100 * d['false_alarm_rate'].mean():.2f}% of normal "
            f"images - the two error types are asymmetric, which matters for triage.")

    # augmentation
    if {"exp2_modelB_aug", "exp3_modelB_noaug"} <= set(agg["exp"]):
        a = agg[agg["exp"] == "exp2_modelB_aug"].iloc[0]
        b = agg[agg["exp"] == "exp3_modelB_noaug"].iloc[0]
        obs.append(
            f"Augmentation {'helped' if a['macro_f1_mean'] > b['macro_f1_mean'] else 'did not help'} "
            f"for MobileNetV2: macro-F1 {a['macro_f1_mean']:.4f} with vs "
            f"{b['macro_f1_mean']:.4f} without (delta "
            f"{a['macro_f1_mean'] - b['macro_f1_mean']:+.4f}).")

    # seed stability
    for exp_id in sorted(set(test_metrics["exp"])):
        sub = test_metrics[test_metrics["exp"] == exp_id]
        if len(sub) > 1:
            obs.append(
                f"Across {len(sub)} seeds, `{exp_id}` macro-F1 spread was "
                f"{sub['macro_f1'].std():.4f} (min {sub['macro_f1'].min():.4f}, "
                f"max {sub['macro_f1'].max():.4f}), so differences smaller than that are "
                f"not meaningful.")

    # inference cost
    a_ms = float(infer.loc[infer["model"] == "modelA", "ms_per_image"].iloc[0])
    b_ms = float(infer.loc[infer["model"] == "modelB", "ms_per_image"].iloc[0])
    obs.append(
        f"MobileNetV2 costs {b_ms:.2f} ms/image versus {a_ms:.2f} ms/image for the small CNN "
        f"on `{infer['device'].iloc[0]}` - a {b_ms / a_ms:.1f}x cost for the accuracy gain, "
        f"the central edge-deployment trade-off for a drone payload.")

    # cleaning
    if clean["n_removed_total"]:
        obs.append(
            f"Cleaning removed {clean['n_removed_total']} of {clean['n_scanned']} files "
            f"({100 * clean['n_removed_total'] / clean['n_scanned']:.2f}%), of which "
            f"{clean['n_removed_duplicate']} were exact duplicates; removing them before "
            f"splitting prevents identical images leaking across train/test.")
    else:
        obs.append(
            "No corrupt or duplicate files were found, so no samples were removed.")
    return obs[:9]


# =========================================================================== #
# figures_index.md                                                            #
# =========================================================================== #
FIGURE_CAPTIONS = {
    "class_distribution.png": "Number of images per class after cleaning (removed files annotated).",
    "samples_grid.png": "Four example images per class with their original resolution.",
    "augmentation_examples.png": "One training image rendered under six independent random augmentations.",
    "architecture_diagram.png": "Layer diagram of Model A (SmallCNN) and the MobileNetV2 head modification of Model B.",
    "comparison_accuracy_macrof1.png": "Test accuracy and macro-F1 per experiment, error bars = std over seeds.",
    "per_class_f1_recall_comparison.png": "Per-class F1 and recall for every experiment.",
    "misclassified_examples_main_model.png": "Misclassified test images for the main model's most common confusion pairs.",
    "gradcam_main_model.png": "Grad-CAM heat-maps of correctly and incorrectly classified test images (main model).",
}


def write_figures_index() -> Path:
    lines = ["# Figure index\n",
             "All figures are 300 dpi PNGs in `figures/`.\n",
             "| file | caption |", "|---|---|"]
    for p in sorted(C.FIGURES_DIR.glob("*.png")):
        cap = FIGURE_CAPTIONS.get(p.name)
        if cap is None:
            if p.name.startswith("training_curves_"):
                cap = ("Train/val loss, accuracy and validation macro-F1 per epoch "
                       "(mean over seeds) for " + p.stem.replace("training_curves_", "") + ".")
            elif p.name.startswith("confusion_matrix_normalized_"):
                cap = "Row-normalized confusion matrix for " + p.stem.replace("confusion_matrix_normalized_", "") + "."
            elif p.name.startswith("confusion_matrix_"):
                cap = "Confusion matrix (raw counts) for " + p.stem.replace("confusion_matrix_", "") + "."
            elif p.name.startswith("roc_curves_"):
                cap = "One-vs-rest ROC curves for " + p.stem.replace("roc_curves_", "") + "."
            else:
                cap = ""
        lines.append(f"| `{p.name}` | {cap} |")
    out = C.RESULTS_DIR / "figures_index.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[run_all] wrote {out} ({len(list(C.FIGURES_DIR.glob('*.png')))} figures)")
    return out


# =========================================================================== #
# Final terminal report                                                       #
# =========================================================================== #
def final_report(metas: list[dict], agg: pd.DataFrame, elapsed: float) -> None:
    banner("FINAL REPORT")
    print("Generated files:")
    for folder in (C.RESULTS_DIR, C.FIGURES_DIR, C.CKPT_DIR):
        files = sorted(p for p in folder.rglob("*") if p.is_file())
        print(f"  {folder.name}/  ({len(files)} files)")
        for f in files[:200]:
            print(f"    - {f.relative_to(C.PROJECT_ROOT)}")

    print("\nHeadline test numbers (mean over seeds):")
    for _, r in agg.sort_values("macro_f1_mean", ascending=False).iterrows():
        std = "" if pd.isna(r["accuracy_std"]) else f" +/- {r['accuracy_std']:.4f}"
        print(f"  {r['exp']:<20s} acc={r['accuracy_mean']:.4f}{std}  "
              f"macroF1={r['macro_f1_mean']:.4f}  miss={r['miss_rate_mean']:.4f}  "
              f"false-alarm={r['false_alarm_rate_mean']:.4f}")

    print(f"\nTotal pipeline wall-clock: {human_hms(elapsed)}")
    print("\nDeviations from the brief (if any) are listed in results/summary.md and "
          "README.md (device, epochs, seeds).")


# =========================================================================== #
# Main                                                                        #
# =========================================================================== #
def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full AIDER project pipeline.")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 2-3 epochs per stage, seed 42 only")
    ap.add_argument("--force-data", action="store_true", help="re-download + re-clean data")
    ap.add_argument("--jobs", type=int, default=3, help="parallel training processes")
    ap.add_argument("--epochs-scale", type=float,
                    default=float(os.environ.get("AIDER_EPOCH_SCALE", 1.0)))
    ap.add_argument("--seeds", type=str, default=None,
                    help="comma separated seeds, default from config")
    ap.add_argument("--experiments", type=str, default=None,
                    help="comma separated experiment ids, default all")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    log_path = C.LOG_DIR / "run_all.log"
    log_fh = open(log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_fh)

    t0 = time.perf_counter()
    try:
        apply_plot_style()
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(C.SEEDS)
        exp_ids = [e.strip() for e in args.experiments.split(",")] if args.experiments \
            else list(C.EXPERIMENTS)
        epochs_scale = args.epochs_scale
        patience = C.EARLY_STOP_PATIENCE
        if args.quick:
            epochs_scale = args.epochs_scale if args.epochs_scale != 1.0 else 0.1
            seeds = [C.SEED]
            patience = 2

        banner("STEP 0 - environment")
        print(C.describe_device())
        print(f"[run_all] experiments = {exp_ids}")
        print(f"[run_all] seeds = {seeds}, epochs_scale = {epochs_scale}, "
              f"parallel jobs = {args.jobs}")

        import data as data_mod
        import evaluate as eval_mod
        import models as models_mod

        banner("STEP 1-3 - download, integrity check, cleaning, split, EDA figures")
        df = data_mod.prepare_dataset(force_download=args.force_data)
        rep = data_mod._report_from_disk(C.RESULTS_DIR / "cleaning_report.json")
        data_mod.figure_class_distribution(df, rep)
        data_mod.figure_sample_grid(df)
        row = df[df["split"] == "train"].sample(n=1, random_state=C.SEED).iloc[0]
        data_mod.figure_augmentation_examples((row["path"], int(row["label"])), n=6)
        print(data_mod.split_table(df))

        banner("STEP 4 - model summaries, architecture diagram, inference benchmark")
        models_mod.save_layer_summaries()
        models_mod.figure_architecture()
        models_mod.benchmark_all(n_batches=10)

        if not args.skip_train:
            banner("STEP 5-6 - training all experiments")
            with timer() as t:
                metas = run_training(exp_ids, seeds, epochs_scale,
                                     max_jobs=args.jobs, patience=patience)
            print(f"[run_all] training block finished in {human_hms(t['seconds'])}")
        else:
            metas = [json.loads(p.read_text(encoding="utf-8"))
                     for p in sorted(C.RESULTS_DIR.glob("run_*_seed*.json"))]

        banner("STEP 5 - training curves")
        import train as train_mod
        for exp_id in exp_ids:
            try:
                train_mod.plot_training_curves(exp_id, seeds)
            except FileNotFoundError as exc:
                print(f"[run_all] no curves for {exp_id}: {exc}")

        banner("STEP 7 - final TEST evaluation, comparison figures, error analysis, Grad-CAM")
        agg = eval_mod.run_full_evaluation(exp_ids)

        banner("STEP 8 - deliverables")
        write_summary()
        write_figures_index()

        elapsed = time.perf_counter() - t0
        final_report(metas, agg, elapsed)
    finally:
        sys.stdout = original_stdout
        log_fh.close()
        print(f"[run_all] full log: {log_path}")


if __name__ == "__main__":
    main()
