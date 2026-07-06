import argparse
import json
import os
from pathlib import Path
from typing import Mapping, OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from typing import Dict, Tuple, Any, List
import matplotlib as mpl
import glob
from collections import Counter
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import torch
import pandas as pd
import sys

src_dir = Path(__file__).parents[1]
sys.path.insert(0, str(src_dir))

# Local imports
from train import load_config
from data.utils import load_samples, create_training_transform, create_validation_transform, build_optimizer_stage1
from data.data import CellCropsDataset
from data.orion_data_processing import load_cell_crops_from_orion


def average_celltype_roc_across_folds(
    y_true_folds,
    y_score_folds,
    class_names,
    n_points=101,
):
    n_classes = len(class_names)
    class_ids = np.arange(n_classes)
    mean_fpr = np.linspace(0, 1, n_points)

    roc_data = {}

    for class_idx, class_name in enumerate(class_names):
        fold_tprs = []
        fold_aucs = []
        fold_curves = []

        for y_true, y_score in zip(y_true_folds, y_score_folds):
            y_true_bin = label_binarize(y_true, classes=class_ids)

            positives = y_true_bin[:, class_idx].sum()
            negatives = len(y_true_bin) - positives

            if positives == 0 or negatives == 0:
                continue

            fpr, tpr, thresholds = roc_curve(
                y_true_bin[:, class_idx],
                y_score[:, class_idx],
            )

            fold_auc = auc(fpr, tpr)
            fold_aucs.append(fold_auc)
            fold_curves.append({
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
                "auc": fold_auc,
            })

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            fold_tprs.append(interp_tpr)

        if not fold_tprs:
            continue

        fold_tprs = np.vstack(fold_tprs)

        mean_tpr = fold_tprs.mean(axis=0)
        std_tpr = fold_tprs.std(axis=0, ddof=1) if fold_tprs.shape[0] > 1 else np.zeros_like(mean_tpr)

        mean_tpr[-1] = 1.0

        roc_data[class_name] = {
            "mean_fpr": mean_fpr,
            "mean_tpr": mean_tpr,
            "std_tpr": std_tpr,
            "mean_auc": auc(mean_fpr, mean_tpr),
            "fold_aucs": np.array(fold_aucs),
            "fold_curves": fold_curves,
        }

    return roc_data


def plot_average_celltype_roc(
    roc_data,
    class_names,
    output_path: str | Path = "average_celltype_roc",
    show_std_band: bool = True,
) -> plt.Figure:
    """
    Plot one-vs-rest ROC curves for each cell type across folds.

    Parameters
    ----------
    roc_data:
        Output from average_celltype_roc_across_folds function.

    class_names:
        Cell type names in the same order as prediction columns.

    output_path:
        Output path without extension.
    """
    
    
    n_classes = len(class_names)
    class_ids = np.arange(n_classes)
    # mean_fpr = np.linspace(0, 1, 101)

    colors = [
        "#4E79A7", "#F28E2B", "#59A14F", "#E15759",
        "#76B7B2", "#B07AA1", "#9C755F", "#BAB0AC",
        "#86BCB6", "#A0CBE8", "#FFBE7D", "#8CD17D",
    ]

    with mpl.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
    }):
        fig, ax = plt.subplots(figsize=(70 / 25.4, 58 / 25.4))

        for class_idx, (class_name, data) in enumerate(roc_data.items()):
            mean_fpr = np.asarray(data["mean_fpr"])
            mean_tpr = np.asarray(data["mean_tpr"])
            mean_auc = data.get("mean_auc", np.nan)

            color = colors[class_idx % len(colors)]

            ax.plot(
                mean_fpr,
                mean_tpr,
                linewidth=1.2,
                color=color,
                label=f"{class_name} ({mean_auc:.2f})",
            )

            if show_std_band and "std_tpr" in data:
                std_tpr = np.asarray(data["std_tpr"])
                lower = np.clip(mean_tpr - std_tpr, 0, 1)
                upper = np.clip(mean_tpr + std_tpr, 0, 1)

                ax.fill_between(
                    mean_fpr,
                    lower,
                    upper,
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=0.7,
            color="#BDBDBD",
            zorder=0,
        )

        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal", adjustable="box")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)

        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.03, 0.5),
            fontsize=6,
            handlelength=1.6,
            title="Cell type AUC",
            title_fontsize=6.5,
        )

        fig.tight_layout(pad=0.4)

        output_path = Path(output_path)
        fig.savefig(f"{output_path}.svg", bbox_inches="tight")
        fig.savefig(f"{output_path}.pdf", bbox_inches="tight")
        fig.savefig(f"{output_path}.png", dpi=600, bbox_inches="tight")

    return fig


def plot_celltype_roc_by_fold(
    y_true_folds: List[np.ndarray],
    y_score_folds: List[np.ndarray],
    class_names: List[str],
    output_path: str | Path = "celltype_roc_by_fold",
) -> plt.Figure:
    """
    Plot one-vs-rest ROC curves for each cell type across folds.

    Parameters
    ----------
    y_true_folds:
        List of arrays, one per fold, containing integer class labels.
        Example: [y_true_fold1, y_true_fold2, ...]

    y_score_folds:
        List of arrays, one per fold, shape = (n_cells, n_classes),
        containing predicted probabilities or decision scores.

    class_names:
        Cell type names in the same order as prediction columns.

    output_path:
        Output path without extension.
    """
    n_folds = len(y_true_folds)
    n_classes = len(class_names)
    class_ids = np.arange(n_classes)

    if len(y_score_folds) != n_folds:
        raise ValueError("y_true_folds and y_score_folds must have the same length.")

    colors = [
        "#4E79A7", "#F28E2B", "#59A14F", "#E15759",
        "#76B7B2", "#B07AA1", "#9C755F", "#BAB0AC",
        "#86BCB6", "#A0CBE8", "#FFBE7D", "#8CD17D",
    ]

    with mpl.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
    }):
        fig_width = max(89 / 25.4, 42 * n_folds / 25.4)
        fig, axes = plt.subplots(
            1,
            n_folds,
            figsize=(fig_width, 45 / 25.4),
            sharex=True,
            sharey=True,
        )

        if n_folds == 1:
            axes = [axes]

        fold_macro_aucs = []

        for fold_idx, (ax, y_true, y_score) in enumerate(
            zip(axes, y_true_folds, y_score_folds),
            start=1,
        ):
            y_true = np.asarray(y_true)
            y_score = np.asarray(y_score)

            if y_score.shape[1] != n_classes:
                raise ValueError(
                    f"Fold {fold_idx}: y_score has {y_score.shape[1]} columns, "
                    f"but {n_classes} class names were provided."
                )

            y_true_bin = label_binarize(y_true, classes=class_ids)
            class_aucs = []

            for class_idx, class_name in enumerate(class_names):
                positives = y_true_bin[:, class_idx].sum()
                negatives = len(y_true_bin) - positives

                if positives == 0 or negatives == 0:
                    continue

                fpr, tpr, thresholds = roc_curve(y_true_bin[:, class_idx], y_score[:, class_idx])
                roc_auc = auc(fpr, tpr)
                class_aucs.append(roc_auc)

                ax.plot(
                    fpr,
                    tpr,
                    linewidth=1.0,
                    color=colors[class_idx % len(colors)],
                    label=f"{class_name} ({roc_auc:.2f})",
                )

            macro_auc = float(np.mean(class_aucs)) if class_aucs else np.nan
            fold_macro_aucs.append(macro_auc)

            ax.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                linewidth=0.7,
                color="#BDBDBD",
                zorder=0,
            )

            ax.set_title(f"Fold {fold_idx}\nmacro AUC = {macro_auc:.2f}", pad=3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.set_aspect("equal", adjustable="box")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            # ax.grid(linewidth=0.35, color="#D0D0D0", alpha=0.6)
            ax.set_axisbelow(True)

        axes[0].set_ylabel("True positive rate")
        for ax in axes:
            ax.set_xlabel("False positive rate")

        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=6,
            handlelength=1.6,
        )

        fig.tight_layout(pad=0.4)

        output_path = Path(output_path)
        fig.savefig(f"{output_path}.svg", bbox_inches="tight")
        fig.savefig(f"{output_path}.pdf", bbox_inches="tight")
        fig.savefig(f"{output_path}.png", dpi=600, bbox_inches="tight")

    return fig


def dataset_figure(
    dataset_name: str,
    num_cells: int,
    num_classes: int,
    labels: List,
    class_names: list[str] | None = None,
    output_dir: str | Path = ".",
) -> plt.Figure:
    dataset_title_name = dataset_name.replace("_", " ").title()

    label_counts = Counter()

    for i in range(len(labels)):
        label = labels[i]
        label_counts[int(label)] += 1

    labels_sorted = sorted(label_counts)
    counts = np.array([label_counts[label] for label in labels_sorted], dtype=float)

    if class_names is None:
        pie_labels = [f"Class {label}" for label in labels_sorted]
    else:
        pie_labels = [
            class_names[label] if label < len(class_names) else f"Class {label}"
            for label in labels_sorted
        ]

    def autopct_with_counts(values):
        total = np.sum(values)

        def _autopct(percent):
            count = int(round(percent * total / 100.0))
            return f"{percent:.1f}%\n" # (n={count:,})

        return _autopct

    colors = [
        "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
        "#B279A2", "#FF9DA6", "#CF5635", "#BAB0AC", "#8CD17D",
    ]

    with mpl.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
    }):
        fig, ax = plt.subplots(figsize=(89 / 25.4, 89 / 25.4))

        wedges, texts, autotexts = ax.pie(
            counts,
            labels=pie_labels,
            colors=colors[:len(counts)],
            startangle=90,
            counterclock=False,
            autopct=autopct_with_counts(counts),
            pctdistance=1.05,
            labeldistance=1.20,
            wedgeprops={
                "linewidth": 0.6,
                "edgecolor": "white",
            },
            textprops={
                "fontsize": 6.5,
                "color": "#222222",
            },
        )

        for autotext in autotexts:
            autotext.set_fontsize(6)
            autotext.set_color("#222222")

        ax.set_title(
            f"{dataset_title_name}\n{num_cells:,} cells, {num_classes} classes",
            pad=4,
        )
        ax.set_aspect("equal")

        fig.tight_layout(pad=0.4)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(output_dir / f"{dataset_name}_composition.png", bbox_inches="tight")
        fig.savefig(output_dir / f"{dataset_name}_composition.svg", bbox_inches="tight")

    return fig


def dataset_bar_figure(
    dataset_name: str,
    labels: List,
    class_names: List[str] | None = None,
    output_dir: str | Path = ".",
    sort_desc: bool = True,
    color: str = "#4E79A7",
    width_mm: float = 89,
) -> plt.Figure:
    label_counts = Counter(int(label) for label in labels)
    labels_sorted = np.array(sorted(label_counts), dtype=int)
    counts = np.array([label_counts[label] for label in labels_sorted], dtype=float)

    if class_names is None:
        names = np.array([f"Class {label}" for label in labels_sorted], dtype=object)
    else:
        names = np.array(class_names,
            dtype=object,
        )

    if sort_desc and len(counts):
        order = np.argsort(counts)[::-1]
        counts = counts[order]
        names = names[order]

    total = counts.sum()
    percentages = counts / total * 100 if total > 0 else np.zeros_like(counts)

    with mpl.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
    }):
        fig_height = max(45 / 25.4, (5.0 * max(len(names), 1) + 12) / 25.4)
        fig, ax = plt.subplots(figsize=(width_mm / 25.4, fig_height))

        y = np.arange(len(names))
        bars = ax.barh(
            y,
            counts,
            height=0.62,
            color=color,
            edgecolor="none",
        )

        xmax = counts.max() if len(counts) else 1
        ax.set_xlim(0, xmax * 1.24)

        for bar, count, pct in zip(bars, counts, percentages):
            ax.text(
                bar.get_width() + xmax * 0.025,
                bar.get_y() + bar.get_height() / 2,
                f"{int(count):,}", # ({pct:.1f}%)
                ha="left",
                va="center",
                fontsize=5.2,
                color="#222222",
            )

        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()

        ax.set_xlabel("Cell count")
        ax.set_ylabel("")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        # ax.grid(axis="x", linewidth=0.4, color="#D0D0D0", alpha=0.7)
        ax.set_axisbelow(True)

        fig.tight_layout(pad=0.35)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        basename = output_dir / f"{dataset_name}_composition_bar"

        fig.savefig(f"{basename}.svg", bbox_inches="tight")
        fig.savefig(f"{basename}.pdf", bbox_inches="tight")
        fig.savefig(f"{basename}.png", dpi=600, bbox_inches="tight")

    return fig


def fold_metric_bar_graphs(
    f1_scores: List,
    accuracies: List,
    random_baselines: List = None,
    output_path="fold_metrics",
):
    
    data = {
    "F1 score": f1_scores,
    "Accuracy": accuracies,
    }

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    labels = list(data.keys())
    values = [data[k] for k in labels]
    means = np.array([v.mean() for v in values])
    sems = np.array([v.std(ddof=1) / np.sqrt(len(v)) for v in values])

    colors = ["#578FD6", "#C22CA6"]
    x = np.arange(len(labels))
    
    fig, ax = plt.subplots(figsize=(2.2, 2.0))

    ax.bar(
        x,
        means,
        yerr=sems,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        capsize=3,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
        width=0.62,
        zorder=2,
    )

    # Show individual fold values
    colors_folds = ["#B0CFF1", "#F2B8D9", "#FF6161", "#B2F4AC"]
    rng = np.random.default_rng(4)
    for i, vals in enumerate(values):
        jitter = rng.normal(0, 0.035, size=len(vals))
        ax.scatter(
            np.full(len(vals), x[i]) + jitter,
            vals,
            s=8,
            color=colors_folds, # [i % len(colors_folds)]
            edgecolor="black",
            linewidth=0.3,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    # ax.text(
    #     0.98,
    #     0.04,
    #     # "n = 4 folds; mean +/- s.e.m.",
    #     transform=ax.transAxes,
    #     ha="right",
    #     va="bottom",
    #     fontsize=6,
    # )

    fig.tight_layout()

    # fig.savefig(f"{output_path}/fold_metric_summary.svg", bbox_inches="tight")
    # fig.savefig(f"{output_path}/fold_metric_summary.pdf", bbox_inches="tight")
    fig.savefig(f"{output_path}/fold_metric_summary.png", dpi=600, bbox_inches="tight")

    plt.show()
    
    return fig
    


def fold_metric_figure(
    f1: List,
    precision: List,
    recall: List,
    output_path="fold_metrics",
):
    metrics = {
        "F1": np.asarray(f1, dtype=float),
        "Precision": np.asarray(precision, dtype=float),
        "Recall": np.asarray(recall, dtype=float),
    }

    names = list(metrics)
    values = np.vstack([metrics[name] for name in names]).T
    x = np.arange(len(names))

    means = values.mean(axis=0)
    sds = values.std(axis=0, ddof=1)

    with mpl.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }):
        fig, ax = plt.subplots(figsize=(65 / 25.4, 50 / 25.4))

        for fold_values in values:
            ax.plot(x, fold_values, color="#BDBDBD", linewidth=0.7, alpha=0.8, zorder=1)
            ax.scatter(x, fold_values, s=13, color="#7A7A7A", alpha=0.85, zorder=2)

        ax.errorbar(
            x,
            means,
            yerr=sds,
            fmt="o",
            markersize=4.2,
            color="#1F4E79",
            ecolor="#1F4E79",
            elinewidth=1.0,
            capsize=2.5,
            zorder=3,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.02)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ax.grid(axis="y", linewidth=0.4, color="#D0D0D0", alpha=0.7)
        ax.set_axisbelow(True)

        fig.tight_layout(pad=0.4)
        fig.savefig(f"{output_path}.svg", bbox_inches="tight")
        fig.savefig(f"{output_path}.pdf", bbox_inches="tight")
        fig.savefig(f"{output_path}.png", dpi=600, bbox_inches="tight")

    return fig


def average_confusion_matrices(confusion_matrices, mode="sum"):
    """
    Combine confusion matrices across folds.

    Parameters
    ----------
    confusion_matrices:
        List/array of confusion matrices with shape:
        (n_folds, n_classes, n_classes)

    mode:
        "sum"  : add raw counts across folds. Best default.
        "mean" : average raw counts across folds.
        "true_normalized_mean" : row-normalize each fold, then average.

    Returns
    -------
    combined_cm:
        Combined confusion matrix.
    """
    cms = np.asarray(confusion_matrices, dtype=float)

    if cms.ndim != 3:
        raise ValueError("confusion_matrices must have shape (n_folds, n_classes, n_classes).")

    if cms.shape[1] != cms.shape[2]:
        raise ValueError("Each confusion matrix must be square.")

    if mode == "sum":
        combined_cm = cms.sum(axis=0)

    elif mode == "mean":
        combined_cm = cms.mean(axis=0)

    elif mode == "true_normalized_mean":
        row_sums = cms.sum(axis=2, keepdims=True)
        normalized = np.divide(
            cms,
            row_sums,
            out=np.zeros_like(cms),
            where=row_sums != 0,
        )
        combined_cm = normalized.mean(axis=0)

    else:
        raise ValueError("mode must be 'sum', 'mean', or 'true_normalized_mean'.")

    return combined_cm


def plot_confusion_matrix_nature(
    confusion_matrix,
    class_names=None,
    output_path="confusion_matrix",
    normalize="true",  # "true", "pred", "all", or None
    annotate=True,
):
    cm = np.asarray(confusion_matrix, dtype=float)
    n_classes = cm.shape[0]

    if cm.shape[0] != cm.shape[1]:
        raise ValueError("confusion_matrix must be square.")

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    if normalize == "true": # maybe try this one or Pred?
        denom = cm.sum(axis=1, keepdims=True)
        plot_cm = np.divide(cm, denom, out=np.zeros_like(cm), where=denom != 0)
        cbar_label = "Fraction of true class"
    elif normalize == "pred":
        denom = cm.sum(axis=0, keepdims=True)
        plot_cm = np.divide(cm, denom, out=np.zeros_like(cm), where=denom != 0)
        cbar_label = "Fraction of predicted class"
    elif normalize == "all":
        denom = cm.sum()
        plot_cm = cm / denom if denom else np.zeros_like(cm)
        cbar_label = "Fraction of all cells"
    elif normalize is None:
        plot_cm = cm
        cbar_label = "Cell count"
    else:
        raise ValueError("normalize must be 'true', 'pred', 'all', or None.")

    with mpl.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.linewidth": 0.6,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
    }):
        fig_size = max(70 / 25.4, 5.5 * n_classes / 25.4)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        im = ax.imshow(
            plot_cm,
            cmap="Blues",
            vmin=0,
            vmax=1 if normalize is not None else None,
            interpolation="nearest",
        )

        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(class_names)

        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xticks(np.arange(n_classes + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_classes + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=0.45)
        ax.tick_params(which="minor", bottom=False, left=False)

        if annotate:
            threshold = 0.55 * np.nanmax(plot_cm) if np.nanmax(plot_cm) > 0 else 0

            for i in range(n_classes):
                for j in range(n_classes):
                    value = plot_cm[i, j]
                    count = int(cm[i, j])

                    if normalize is None:
                        text = f"{count:,}" if count > 0 else "0"
                    else:
                        text = f"{value * 100:.0f}%" # if value >= 0.01 else ""

                    if not text:
                        continue

                    ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        fontsize=3.2,
                        color="white" if value > threshold else "#222222",
                    )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(cbar_label)
        cbar.ax.tick_params(labelsize=6, width=0.6, length=2.5)
        cbar.outline.set_linewidth(0.6)

        fig.tight_layout(pad=0.4)

        output_path = Path(output_path)
        fig.savefig(f"{output_path}.svg", bbox_inches="tight")
        fig.savefig(f"{output_path}.pdf", bbox_inches="tight")
        fig.savefig(f"{output_path}.png", dpi=600, bbox_inches="tight")

    return fig


def plot_donut_chart(
    class_counts,
    class_names=None,
    output_path="confusion_matrix",
    version=1,
    nested=False,
    broad_class_names=None,
    # normalize="true",  # "true", "pred", "all", or None
    # annotate=True,
):
    
    # Example data
    labels = class_names
    celltype_counts = class_counts

    # Nature-style figure settings
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
    })

    celltype_colors = [
        "#4E79A7",  # CD4_T
        "#7BA7D7",  # CD8_T
        "#2F5F8F",  # Treg
        "#76B7B2",  # B_cell
        "#59A14F",  # Mono_Macro
        "#A0C86F",  # Granulocyte
        "#C65413",  # Stromal
        "#D78228",  # Smooth_Muscle
        "#B07AA1",  # Tumor
        "#DE6668",  # Vasculature
    ]

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
    })

    values = np.array([celltype_counts[k] for k, label in enumerate(labels)])
    colors = [celltype_colors[k] for k, label in enumerate(labels)]
    
    inner_counts = OrderedDict()
    

    fig, ax = plt.subplots(figsize=(3.0, 3.0))

    wedges, _ = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        labels=None,
        wedgeprops={
            "width": 0.34,
            "edgecolor": "white",
            "linewidth": 2.0,
        },
    )

    # Radial / orthogonal labels
    label_radius = 1.22

    for wedge, label, value in zip(wedges, labels, values):
        angle = 0.5 * (wedge.theta1 + wedge.theta2)
        angle = angle % 360
        angle_rad = np.deg2rad(angle)

        x = label_radius * np.cos(angle_rad)
        y = label_radius * np.sin(angle_rad)

        # Rotate labels radially, but keep left-side labels readable
        rotation = angle
        ha = "left"
        if 90 < angle < 270:
            rotation = angle + 180
            ha = "right"

        percent = value / values.sum() * 100

        ax.text(
            x,
            y,
            f"{label}", # \n{percent:.1f}%
            ha=ha,
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            fontsize=6.5,
        )

        # Optional subtle leader line
        line_start = 1.02
        line_end = 1.14
        ax.plot(
            [line_start * np.cos(angle_rad), line_end * np.cos(angle_rad)],
            [line_start * np.sin(angle_rad), line_end * np.sin(angle_rad)],
            color="0.55",
            lw=0.5,
        )

    ax.text(
        0,
        0,
        f"n = {int(values.sum())}",
        ha="center",
        va="center",
        fontsize=7,
    )

    ax.set(aspect="equal")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)
    ax.axis("off")

    fig.savefig(f"{output_path}/celltype_donut_radial_labels_{version}.svg", bbox_inches="tight")
    fig.savefig(f"{output_path}/celltype_donut_radial_labels_{version}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_path}/celltype_donut_radial_labels_{version}.png", dpi=600, bbox_inches="tight")

    plt.show()
    
    return fig


def get_multiclass_ct_name(label):

    new_mapping = {
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

    class_name = new_mapping[label]

    return class_name


def load_labels_from_orion(
        cell_patches_path: str,
        labels_name: str,
        label_files,
        sample_fraction=None,
        max_samples=None,
        sample_seed: int = 42) -> List:
    """
    Load cell crops from the Orion dataset.
    
    Args:
        cell_patches_path: Path to the directory containing cell patches
        mask_name: Name of the mask files
        img_patch_name: Name of the image patch files
        labels_name: Name of the label files
    Returns:
        List of CellCrop objects
    """
    labels = []
    sample_id = os.path.basename(os.path.normpath(cell_patches_path))
     
    # Load labels
    # label_files = glob.glob(f"{cell_patches_path}/{labels_name}_*.csv")
    for label_file in label_files:
        # Extract case and image ID from the filename
        filename = os.path.basename(label_file)
        parts = filename.split('_')
        file_id = parts[1].split('.')[0]  # Assuming format: meta_fileID.csv
        
        # print(images.shape, masks.shape)
        
        # Load labels
        labels_df = pd.read_csv(label_file)
        # ignore rows with cell_type == Unassigned
        new_labels_df = labels_df[labels_df['orion_label'] != 'Unassigned'] 
        # new_labels_df = new_labels_df[new_labels_df['orion_label'] != 'Granulocyte'] 
        
        shard_indices = new_labels_df['index_in_shard'].values
        cell_ids = new_labels_df['cellpose_id'].values
        cell_labels = new_labels_df['orion_label'].values
        x_coords = new_labels_df['x'].values
        y_coords = new_labels_df['y'].values
        sample_name = new_labels_df['case'].values[0]
        
        num_shards = len(shard_indices)
        
        for i in np.arange(num_shards):
            cell_id = cell_ids[i]
            label = cell_labels[i]
            int_label = get_multiclass_ct_name(label)
            x = x_coords[i]
            y = y_coords[i]
            
            labels.append(int_label)
    
    print(f"{len(labels)} labels loaded from Orion dataset.")
    
    return labels


def create_orion_label_loader(config: Dict[str, Any]) -> List:
    """
    Create training and validation data loaders.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # In this case, we can get the image names by looping through the files instead for our situation: 
    cell_patches_path = config["root_dir"]
    use_mask = True  # config.get("use_mask", False)
    
    # set random seed for reproducibility
    np.random.seed(42)
    
    # First get the list of folders and shuffle them to ensure random distribution of samples across folds
    # /taiga/illinois/vetmed/cb/kwang222/mz_jason/orion_all_without_largest/_meta/cell_labeling/cell_patches_64_match5um_area50_3000
    folders = glob.glob("CRC*", root_dir=cell_patches_path)
    perm_indices = np.random.permutation(len(folders))
    
    folders_perm = np.array(folders)
    folders_perm = folders_perm[perm_indices]
    
    # Then split into folds based on this.
    test_crc_samples = folders_perm # [32:len(folders)]

    # The data is numbered 00000
    mask_name = "cell_masks"
    img_patch_name = "image_patches"
    labels_name = "meta"

    # count
    print("Loading testing data...")
    test_labels = []
    for sample in test_crc_samples:
        filelist = glob.glob(f"{cell_patches_path}/{sample}/{labels_name}_*.csv")
        labels = load_labels_from_orion(f"{cell_patches_path}/{sample}", labels_name, filelist)
        test_labels.extend(labels)
    print(f"Loaded {len(test_labels)} testing samples")
    
    return test_labels


if __name__ == "__main__":
    config_path = "/taiga/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/src/config_files/config_new_more_cts_testing.json"
    
    config = load_config(config_path)
    print(f"Loaded configuration from {config_path}")
    
    orion_labels = create_orion_label_loader(config)
    
    label_counts = Counter(int(label) for label in orion_labels)
    labels_sorted = np.array(sorted(label_counts), dtype=int)
    counts = np.array([label_counts[label] for label in labels_sorted], dtype=float)
    
    class_names=[
        "CD4+ T",
        "CD8+ T",
        "Treg",
        "B cell",
        "Mono/Macro",
        "Stromal",
        "Smooth Muscle",
        "Tumor",
        "Vasculature",
        "Granulocyte"
    ]
    
    reordered_class_names = [0, 1, 2, 3, 4, 9, 5, 6, 7, 8]
    class_names = [class_names[i] for i in reordered_class_names]
    counts = [counts[i] for i in reordered_class_names]
    
    # class_names = [
    #     "CD4+ T",
    #     "CD8+ T",
    #     "B cell",
    #     "Granulocyte"]
    
    # fig = dataset_figure(
    #     dataset_name="Orion Dataset",
    #     num_cells=len(orion_labels),
    #     num_classes=10,
    #     labels=orion_labels
    # )
    
    # fig = dataset_bar_figure(
    #     dataset_name="Orion_Dataset",
    #     labels=orion_labels,
    #     class_names=[
    #         "CD4+ T",
    #         "CD8+ T",
    #         "Treg",
    #         "B cell",
    #         "Mono/Macro",
    #         "Stromal",
    #         "Smooth Muscle",
    #         "Tumor",
    #         "Vasculature",
    #         "Granulocyte"
    #     ],
    #     sort_desc=True,
    # )
    output_path = "/taiga/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/src/data/test_outputs/figures_testing"
    # fold_result_path = "/taiga/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/experiment_results/evaluation_results/orion_fold_testing_results"
    
    # broad_class = {
    #     "Tumor": "Tumor/Epithelial",
    #     "CD4_T": "Lymphocytes",
    #     "CD8_T": "Lymphocytes",
    #     "Treg": "Lymphocytes",
    #     "B_cell": "Lymphocytes",
    #     "Mono_Macro": "Myeloid",
    #     "Granulocyte": "Myeloid",
    #     "Stromal": "Stromal/Mesenchymal",
    #     "Smooth_Muscle": "Stromal/Mesenchymal",
    #     "Vasculature": "Vasculature",
    # }
    
    fig = plot_donut_chart(
        class_counts=counts,
        class_names=class_names,
        output_path=output_path,
        version=4
    )
    
    # f1_vals = []
    # precision_vals = []
    # recall_vals = []
    # accuracy_vals = []
    # for fold in range(1, 5):
    #     fold_result_file = f"{fold_result_path}/fold{fold}_evaluation_results.json"
    #     with open(fold_result_file, "r") as f:
    #         fold_results = json.load(f)
    #         print(f"Fold {fold} results")
    #     f1_vals.append(fold_results["f1_avg"])
    #     precision_vals.append(fold_results["precision_avg"])
    #     recall_vals.append(fold_results["recall_avg"])
    #     accuracy_vals.append(fold_results["accuracy"])
    
    # Create bar graphs for them.
    # fig = fold_metric_figure(f1_vals, precision_vals, recall_vals, output_path=f"{output_path}/fold_metrics")
    # fig = fold_metric_bar_graphs(np.array(f1_vals), np.array(accuracy_vals), output_path=f"{output_path}/fold_metrics")
    
    # plot_celltype_roc_by_fold
    # Process the list_of_logits to prediction classes first for each fold:
    # y_prob_folds = []
    # y_true_folds = []
    # for fold in range(1, 5):
    #     fold_testing_folder = f"results_fold{fold}_orion_testing_stuff"
    #     fold_logits_file = f"{fold_result_path}/{fold_testing_folder}/list_of_logits.npz"
        
    #     fold_logits = np.load(fold_logits_file, allow_pickle=True)
    #     fold_logits = fold_logits["arr_0"]
    #     fold_probs = F.softmax(torch.from_numpy(fold_logits), dim=1).numpy()
    #     y_prob_folds.append(fold_probs)
        
    #     fold_labels_file = f"{fold_result_path}/{fold_testing_folder}/list_of_labels.npz"
    #     fold_labels = np.load(fold_labels_file, allow_pickle=True)
    #     fold_labels = fold_labels["arr_0"]
    #     y_true_folds.append(fold_labels)
        
    # fig = plot_celltype_roc_by_fold(
    #     y_true_folds=y_true_folds,
    #     y_score_folds=y_prob_folds,
    #     class_names=class_names,
    #     output_path=f"{output_path}/celltype_roc_by_fold",
    # )
    
    # roc_data = average_celltype_roc_across_folds(
    #     y_true_folds=y_true_folds,
    #     y_score_folds=y_prob_folds,
    #     class_names=class_names,
    # )
    
    # fig = plot_average_celltype_roc(
    #     roc_data=roc_data,
    #     class_names=class_names,
    #     output_path=f"{output_path}/average_celltype_roc",
    # )
    
    # fold_cms = []
    # for fold in range(1, 2):
    #     fold_result_file = "/taiga/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/results_4_class_testining/evaluation_results.json" # f"{fold_result_path}/fold{fold}_evaluation_results.json"
    #     with open(fold_result_file, "r") as f:
    #         fold_results = json.load(f)
    #         print(f"Fold {fold} results")
    #     cm = fold_results["confusion_matrix"]
    #     fold_cms.append(cm) 
        
    # single_cm_path = "/taiga/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/results_4_class_testining/evaluation_results.json"
    # with open(single_cm_path, "r") as f:
    #     fold_results = json.load(f)
    #     # print(f"Fold {fold} results")
        
    # single_cm = fold_results["confusion_matrix"]
        
    # # combined_cm = average_confusion_matrices(fold_cms, mode="sum")
    # fig = plot_confusion_matrix_nature(
    #     confusion_matrix=single_cm, # combined_cm,
    #     class_names=class_names,
    #     output_path=f"{output_path}/4class_testing_confusion_matrix",
    #     normalize="pred", # "true", "pred", "all", or None
    #     annotate=True,
    # )
        

