#!/usr/bin/env python3
"""Random F1 baselines for Orion evaluation splits."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

CLASS_NAMES = {
    0: "CD4+ T",
    1: "CD8+ T",
    2: "Treg",
    3: "B cells",
    4: "Monocytes / Macrophages",
    5: "Stromal Cells",
    6: "Smooth Muscle",
    7: "Tumor Cells",
    8: "Vasculature",
    9: "Granulocytes",
}

CLASS5_NAMES = {
    0: "Tumor/Epithelial",
    1: "Lymphocytes",
    2: "Myeloid",
    3: "Stromal/Mesenchymal",
    4: "Vasculature",
}

MAP10_TO_5 = np.array([1, 1, 1, 1, 2, 3, 3, 0, 4, 2], dtype=np.int64)

ORION_LABELS = {
    "CD4_T": 0,
    "CD8_T": 1,
    "Treg": 2,
    "B_cell": 3,
    "Mono_Macro": 4,
    "Stromal": 5,
    "Smooth_Muscle": 6,
    "Tumor": 7,
    "Vasculature": 8,
    "Granulocyte": 9,
}

FULL_DATASET_COUNTS = np.array(
    [747029, 812100, 3275429, 627760, 1214472, 1175956, 2213625, 3139062, 2917044, 160749],
    dtype=np.float64,
)


def npz_first(path: Path) -> np.ndarray:
    data = np.load(path)
    return np.asarray(data[data.files[0]])


def orion_samples(root: Path, split: str, val_fold: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    folders = np.array(glob.glob("CRC*", root_dir=root))
    folders = folders[rng.permutation(len(folders))]
    if split == "heldout":
        return folders[32:]
    train_val = folders[:32]
    folds = np.split(train_val, 4)
    if split == "val":
        return folds[val_fold]
    return np.concatenate([folds[i] for i in range(4) if i != val_fold])


def labels_from_orion(root: Path, samples: np.ndarray) -> np.ndarray:
    labels = []
    for sample in samples:
        for path in sorted((root / sample).glob("meta_*.csv")):
            df = pd.read_csv(path, usecols=["orion_label"])
            df = df[df["orion_label"] != "Unassigned"]
            labels.extend(df["orion_label"].map(ORION_LABELS).dropna().astype(np.int64).tolist())
    return np.asarray(labels, dtype=np.int64)


def collapse_counts(counts10: np.ndarray) -> np.ndarray:
    counts5 = np.zeros(len(CLASS5_NAMES), dtype=np.float64)
    for old, new in enumerate(MAP10_TO_5):
        counts5[new] += counts10[old]
    return counts5


def prepare_scheme(y_true10: np.ndarray, prior_counts10: np.ndarray, scheme: str):
    if scheme == "5class":
        y_true = MAP10_TO_5[y_true10] if y_true10.max(initial=0) > 4 else y_true10
        return y_true, collapse_counts(prior_counts10), CLASS5_NAMES
    return y_true10, prior_counts10.astype(np.float64), CLASS_NAMES


def score(y_true: np.ndarray, y_pred: np.ndarray, names: dict[int, str]) -> tuple[pd.DataFrame, dict]:
    labels = np.array(sorted(names), dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "label": labels,
            "cell_type": [names[int(i)] for i in labels],
            "support": support.astype(int),
            "accuracy": recall,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
    )
    summary = {
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_accuracy": float(np.nanmean(recall)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }
    return per_class, summary


def random_baseline(y_true: np.ndarray, probs: np.ndarray, names: dict[int, str], n_runs: int, seed: int):
    rng = np.random.default_rng(seed)
    per_runs, summaries = [], []
    classes = np.arange(len(probs))
    for run in range(n_runs):
        y_pred = rng.choice(classes, size=y_true.size, p=probs)
        per_class, summary = score(y_true, y_pred, names)
        per_class.insert(0, "run", run)
        summary["run"] = run
        per_runs.append(per_class)
        summaries.append(summary)
    per_all = pd.concat(per_runs, ignore_index=True)
    avg_per = (
        per_all.groupby(["label", "cell_type"], as_index=False)
        .agg(
            support=("support", "first"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
        )
        .sort_values("label")
    )
    summary_df = pd.DataFrame(summaries).drop(columns=["run"])
    avg_summary = {f"{c}_mean": float(summary_df[c].mean()) for c in summary_df}
    avg_summary.update({f"{c}_std": float(summary_df[c].std()) for c in summary_df})
    return avg_per, avg_summary


def main() -> None:
    # python src/orion_random_f1_baseline.py --labels /taiga/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/experiment_results/evaluation_results/orion_fold_testing_results/results_fold1_orion_testing_stuff/list_of_labels.npz --scheme 10class --prior full --n-runs 1000 --out-prefix src/data/figure_images/results/orion_random_full_prior
    # python src/orion_random_f1_baseline.py --config src/config_files/config_new_more_cts_testing_ablation.json --split heldout --scheme 10class --prior full --n-runs 1000 --out-prefix src/data/figure_images/results/orion_random_full_prior_10class
    # python src/orion_random_f1_baseline.py --config src/config_files/config_new_more_cts_testing_ablation.json --split heldout --scheme 5class --prior full --n-runs 1000 --out-prefix src/data/figure_images/results/orion_random_full_prior

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, help="Config JSON with Orion root_dir.")
    p.add_argument("--root-dir", type=Path, help="Orion cell_patches root; overrides config root_dir.")
    p.add_argument("--labels", type=Path, help="Optional list_of_labels.npz to exactly match an eval run.")
    p.add_argument("--split", choices=["heldout", "train", "val"], default="heldout")
    p.add_argument("--val-fold", type=int, default=2, help="Evaluation fold index; evaluate.py uses 2, contrastive_runner uses 3.")
    p.add_argument("--prior", choices=["full", "eval", "uniform", "majority"], default="full")
    p.add_argument("--scheme", choices=["10class", "5class"], default="10class")
    p.add_argument("--n-runs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-prefix", type=Path)
    args = p.parse_args()

    config = json.loads(args.config.read_text()) if args.config else {}
    root = args.root_dir or Path(config["root_dir"])
    y_true10 = npz_first(args.labels).astype(np.int64).reshape(-1) if args.labels else labels_from_orion(root, orion_samples(root, args.split, args.val_fold))
    y_true, full_counts, names = prepare_scheme(y_true10, FULL_DATASET_COUNTS, args.scheme)

    if args.prior == "eval":
        probs = np.bincount(y_true, minlength=len(names)).astype(np.float64)
        probs /= probs.sum()
    elif args.prior == "uniform":
        probs = np.ones(len(names)) / len(names)
    elif args.prior == "majority":
        probs = np.eye(len(names))[full_counts.argmax()]
    else:
        probs = full_counts / full_counts.sum()

    per_class, summary = random_baseline(y_true, probs, names, args.n_runs, args.seed)
    print(per_class.to_string(index=False))
    print(json.dumps(summary, indent=2))

    if args.out_prefix:
        args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
        per_class.to_csv(args.out_prefix.with_suffix(".per_class.csv"), index=False)
        args.out_prefix.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
