#!/usr/bin/env python3
"""Compute per-cell-type metrics from saved logits/labels npz files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


# BROAD_CLASS_NAMES = {
#     0: "Tumor/Epithelial",
#     1: "Lymphocytes",
#     2: "Myeloid",
#     3: "Stromal/Mesenchymal",
#     4: "Vasculature",
# }

# TEN_TO_FIVE = {
#     0: 1,  # CD4+ T -> Lymphocytes
#     1: 1,  # CD8+ T -> Lymphocytes
#     2: 1,  # Treg -> Lymphocytes
#     3: 1,  # B cells -> Lymphocytes
#     4: 2,  # Monocytes / Macrophages -> Myeloid
#     5: 3,  # Stromal Cells -> Stromal/Mesenchymal
#     6: 3,  # Smooth Muscle -> Stromal/Mesenchymal
#     7: 0,  # Tumor Cells -> Tumor/Epithelial
#     8: 4,  # Vasculature -> Vasculature
#     9: 2,  # Granulocytes -> Myeloid
# }

# def convert_10_to_5(labels):
#     labels = np.asarray(labels)
#     out = np.full(labels.shape, fill_value=-1, dtype=np.int64)

#     for old_label, new_label in TEN_TO_FIVE.items():
#         out[labels == old_label] = new_label

#     return out

CLASS10 = {
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

CLASS5 = {
    0: "Tumor/Epithelial",
    1: "Lymphocytes",
    2: "Myeloid",
    3: "Stromal/Mesenchymal",
    4: "Vasculature",
}

MAP10_TO_5 = np.array([1, 1, 1, 1, 2, 3, 3, 0, 4, 2], dtype=np.int64)


def npz_first(path: Path) -> np.ndarray:
    data = np.load(path)
    return np.asarray(data[data.files[0]])


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=1, keepdims=True)


def collapse_probs(probs10: np.ndarray) -> np.ndarray:
    probs5 = np.zeros((probs10.shape[0], len(CLASS5)), dtype=probs10.dtype)
    for old, new in enumerate(MAP10_TO_5):
        probs5[:, new] += probs10[:, old]
    return probs5


def score_npz(logits_path: Path, labels_path: Path, scheme: str, root_path = "") -> tuple[pd.DataFrame, dict]:
    full_logits_path = Path(root_path) / logits_path
    full_labels_path = Path(root_path) / labels_path
    logits = npz_first(full_logits_path)
    y_true10 = npz_first(full_labels_path).astype(np.int64).reshape(-1)
    probs10 = softmax(logits)

    if scheme == "5class":
        y_true = MAP10_TO_5[y_true10]
        y_pred = collapse_probs(probs10).argmax(axis=1)
        names = CLASS5
    else:
        y_true = y_true10
        y_pred = probs10.argmax(axis=1)
        names = CLASS10

    return summarize(y_true, y_pred, names)


def summarize(y_true: np.ndarray, y_pred: np.ndarray, names: dict[int, str]) -> tuple[pd.DataFrame, dict]:
    labels = np.array(sorted(names), dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    rows = []
    for i, label in enumerate(labels):
        is_class = y_true == label
        class_acc = float((y_pred[is_class] == label).mean()) if is_class.any() else np.nan
        rows.append(
            {
                "label": int(label),
                "cell_type": names[int(label)],
                "support": int(support[i]),
                "accuracy": class_acc,
                "f1": float(f1[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
            }
        )
    per_class = pd.DataFrame(rows)
    avg = {
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_accuracy": float(per_class["accuracy"].mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }
    return per_class, avg


def average_folds(per_fold: list[pd.DataFrame], summaries: list[dict]) -> tuple[pd.DataFrame, dict]:
    per_class_all = pd.concat(per_fold, ignore_index=True)
    metric_cols = ["accuracy", "f1", "precision", "recall"]
    averaged = (
        per_class_all.groupby(["label", "cell_type"], as_index=False)
        .agg(
            support_total=("support", "sum"),
            support_mean=("support", "mean"),
            **{f"{col}_mean": (col, "mean") for col in metric_cols},
            **{f"{col}_std": (col, "std") for col in metric_cols},
        )
        .sort_values("label")
    )

    summary_df = pd.DataFrame(summaries)
    avg = {}
    for col in summary_df.columns:
        avg[f"{col}_mean"] = float(summary_df[col].mean())
        avg[f"{col}_std"] = float(summary_df[col].std())
    return averaged, avg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logits", type=Path, nargs="+", required=True, help="One or more list_of_logits.npz files")
    p.add_argument("--labels", type=Path, nargs="+", required=True, help="Matching list_of_labels.npz files")
    p.add_argument("--scheme", choices=["10class", "5class"], default="10class")
    p.add_argument("--out-prefix", type=Path, default=None)
    args = p.parse_args()

    if len(args.logits) != len(args.labels):
        raise ValueError("--logits and --labels must have the same number of files.")

    per_fold, summaries = [], []
    for fold, (logits_path, labels_path) in enumerate(zip(args.logits, args.labels), start=1):
        per_class, avg = score_npz(logits_path, labels_path, args.scheme, root_path="/taiga/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/experiment_results/evaluation_results/orion_fold_testing_results/")
        per_class.insert(0, "fold", fold)
        avg["fold"] = fold
        per_fold.append(per_class)
        summaries.append(avg)

    per_class_all = pd.concat(per_fold, ignore_index=True)
    summary_all = pd.DataFrame(summaries)
    print(per_class_all.to_string(index=False))
    print(summary_all.to_string(index=False))

    averaged = averaged_summary = None
    if len(per_fold) > 1:
        averaged, averaged_summary = average_folds(per_fold, summary_all.drop(columns=["fold"]).to_dict("records"))
        print("\nAveraged across folds")
        print(averaged.to_string(index=False))
        print(json.dumps(averaged_summary, indent=2))

    if args.out_prefix:
        args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
        
        if len(per_fold) == 1:
            per_class_all.drop(columns=["fold"]).to_csv(args.out_prefix.with_suffix(".per_class.csv"), index=False)
            args.out_prefix.with_suffix(".summary.json").write_text(
                json.dumps(summary_all.drop(columns=["fold"]).iloc[0].to_dict(), indent=2)
            )
            
        per_class_all.to_csv(args.out_prefix.with_suffix(".per_fold_per_class.csv"), index=False)
        summary_all.to_csv(args.out_prefix.with_suffix(".per_fold_summary.csv"), index=False)
        if averaged is not None:
            averaged.to_csv(args.out_prefix.with_suffix(".avg_per_class.csv"), index=False)
            args.out_prefix.with_suffix(".avg_summary.json").write_text(json.dumps(averaged_summary, indent=2))


if __name__ == "__main__":
    main()
