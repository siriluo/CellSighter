#!/usr/bin/env python3
"""
Create Nature-style bar figures from CellSighter evaluation JSON files.

The script expects the evaluation JSON format produced by the CellSighter
evaluation code, for example:

{
  "pannuke_fold": {
    "precision_per_class": [...],
    "recall_per_class": [...],
    "f1_per_class": [...],
    "precision_avg": 0.42,
    "recall_avg": 0.41,
    "f1_avg": 0.40,
    "auc": 0.75,
    "pr_auc": 0.52,
    "multi_class_aucs": [...],
    "multi_class_pr_aucs": [...],
    "class_names": [...]
  }
}

Manual comparison baselines are optional. Pass a JSON file with this shape:

{
  "Random baseline": {
    "pannuke_fold": {
      "average": {
        "roc_auc": 0.50,
        "f1": 0.33,
        "precision": 0.33,
        "recall": 0.33,
        "pr_auc": 0.33
      },
      "per_class": {
        "roc_auc": [0.50, 0.50, 0.50],
        "f1": [0.33, 0.33, 0.33],
        "precision": [0.33, 0.33, 0.33],
        "recall": [0.33, 0.33, 0.33],
        "pr_auc": [0.33, 0.33, 0.33]
      }
    }
  }
}

Examples:
    python src/eval_testing_scripts/plot_nature_metric_bars.py \
        --result "CellSighter=src/eval_testing_scripts/evaluation_results_ADA.json"

    python src/eval_testing_scripts/plot_nature_metric_bars.py \
        --result "CellSighter=src/eval_testing_scripts/evaluation_results_ADA.json" \
        --comparison "Baseline=src/eval_testing_scripts/evaluation_results_ADA_baseline.json" \
        --manual-baselines manual_baselines.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "figures" / "nature_metric_bars"

MM_TO_IN = 1 / 25.4

AVERAGE_METRICS = [
    ("roc_auc", "ROC AUC", "auc"),
    ("f1", "F1", "f1_avg"),
    ("precision", "Precision", "precision_avg"),
    ("recall", "Recall", "recall_avg"),
    ("pr_auc", "PR AUC", "pr_auc"),
]

PER_CLASS_METRICS = [
    ("roc_auc", "ROC AUC", "multi_class_aucs"),
    ("f1", "F1", "f1_per_class"),
    ("precision", "Precision", "precision_per_class"),
    ("recall", "Recall", "recall_per_class"),
    ("pr_auc", "PR AUC", "multi_class_pr_aucs"),
]

DATASET_LABELS = {
    "pannuke_fold": "PanNuke",
    "orion_test": "Orion",
}

PALETTE = [
    "#4C78A8",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#7F7F7F",
    "#E69F00",
]

GRID_COLOR = "#D8D8D8"
EDGE_COLOR = "#262626"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot average and per-cell-type CellSighter metrics as grouped "
            "Nature-style bar graphs."
        )
    )
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Primary result JSON. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Additional full evaluation JSON to compare against.",
    )
    parser.add_argument(
        "--manual-baselines",
        type=Path,
        default=None,
        help="Optional manually entered baseline metrics JSON.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset keys to plot. Defaults to all datasets found in the result JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "svg", "pdf"],
        choices=["png", "svg", "pdf", "tiff"],
        help="Figure formats to write.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="Raster output DPI.")
    parser.add_argument(
        "--percent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot scores as percentages instead of 0-1 fractions.",
    )
    return parser.parse_args()


def set_nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.4,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Missing label in argument: {value}")
    return label, Path(path.strip())


def is_dataset_metrics(value: object) -> bool:
    return isinstance(value, dict) and any(
        key in value
        for key in (
            "auc",
            "f1_avg",
            "precision_avg",
            "recall_avg",
            "pr_auc",
            "f1_per_class",
            "precision_per_class",
        )
    )


def normalize_result_json(data: dict) -> dict[str, dict]:
    if is_dataset_metrics(data):
        return {"results": data}
    return {key: value for key, value in data.items() if is_dataset_metrics(value)}


def normalize_methods(labeled_paths: Iterable[str]) -> dict[str, dict[str, dict]]:
    methods = {}
    for item in labeled_paths:
        label, path = parse_labeled_path(item)
        methods[label] = normalize_result_json(load_json(path))
    return methods


def normalize_manual_baselines(path: Path | None) -> dict[str, dict[str, dict]]:
    if path is None:
        return {}
    raw = load_json(path)
    normalized = {}
    for method, datasets in raw.items():
        normalized[method] = {}
        for dataset_key, dataset_metrics in datasets.items():
            normalized[method][dataset_key] = {
                "class_names": dataset_metrics.get("class_names", []),
                "_manual_average": dataset_metrics.get("average", {}),
                "_manual_per_class": dataset_metrics.get("per_class", {}),
            }
    return normalized


def dataset_display_name(dataset_key: str) -> str:
    return DATASET_LABELS.get(dataset_key, dataset_key.replace("_", " ").title())


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def metric_value(dataset: dict, canonical_metric: str, json_key: str, level: str) -> object:
    manual_key = "_manual_average" if level == "average" else "_manual_per_class"
    manual = dataset.get(manual_key, {})
    if canonical_metric in manual:
        return manual[canonical_metric]
    return dataset.get(json_key, np.nan)


def as_score_array(values: object, length: int | None = None) -> np.ndarray:
    if values is None:
        arr = np.asarray([], dtype=float)
    elif np.isscalar(values):
        arr = np.asarray([values], dtype=float)
    else:
        arr = np.asarray(values, dtype=float)

    if length is not None:
        padded = np.full(length, np.nan, dtype=float)
        padded[: min(length, arr.size)] = arr[:length]
        return padded
    return arr


def collect_dataset_keys(methods: dict[str, dict[str, dict]], requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    keys = []
    for datasets in methods.values():
        for dataset_key in datasets:
            if dataset_key not in keys:
                keys.append(dataset_key)
    return keys


def get_class_names(methods: dict[str, dict[str, dict]], dataset_key: str) -> list[str]:
    for datasets in methods.values():
        dataset = datasets.get(dataset_key)
        if dataset and dataset.get("class_names"):
            return [str(name) for name in dataset["class_names"]]

    max_classes = 0
    for datasets in methods.values():
        dataset = datasets.get(dataset_key)
        if not dataset:
            continue
        for canonical_metric, _, json_key in PER_CLASS_METRICS:
            values = metric_value(dataset, canonical_metric, json_key, "per_class")
            max_classes = max(max_classes, as_score_array(values).size)
    return [f"Class {idx}" for idx in range(max_classes)]


def scale_scores(values: np.ndarray, percent: bool) -> np.ndarray:
    return values * 100 if percent else values


def prettify_axis(ax: plt.Axes, ylabel: str, percent: bool) -> None:
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.45, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100 if percent else 1.0)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
    )


def grouped_bar_width(n_methods: int) -> float:
    return min(0.18, 0.76 / max(n_methods, 1))


def plot_grouped_bars(
    ax: plt.Axes,
    values_by_method: list[np.ndarray],
    group_labels: list[str],
    method_labels: list[str],
    title: str,
    ylabel: str,
    percent: bool,
) -> None:
    x = np.arange(len(group_labels))
    n_methods = len(method_labels)
    width = grouped_bar_width(n_methods)
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * width

    for idx, (values, method_label) in enumerate(zip(values_by_method, method_labels)):
        ax.bar(
            x + offsets[idx],
            scale_scores(values, percent),
            width=width * 0.9,
            color=PALETTE[idx % len(PALETTE)],
            edgecolor=EDGE_COLOR,
            linewidth=0.35,
            label=method_label,
        )

    ax.set_title(title, pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=35, ha="right")
    prettify_axis(ax, ylabel, percent)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str], dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        if fmt in {"png", "tiff"}:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")


def make_average_figure(
    methods: dict[str, dict[str, dict]],
    dataset_keys: list[str],
    out_dir: Path,
    formats: list[str],
    dpi: int,
    percent: bool,
) -> None:
    method_labels = list(methods.keys())
    n_datasets = len(dataset_keys)
    fig_width = max(85, 58 * n_datasets) * MM_TO_IN
    fig, axes = plt.subplots(
        1,
        n_datasets,
        figsize=(fig_width, 70 * MM_TO_IN),
        squeeze=False,
        constrained_layout=True,
    )

    metric_labels = [label for _, label, _ in AVERAGE_METRICS]
    for idx, dataset_key in enumerate(dataset_keys):
        ax = axes[0, idx]
        values_by_method = []
        for datasets in methods.values():
            dataset = datasets.get(dataset_key, {})
            values = [
                metric_value(dataset, canonical_metric, json_key, "average")
                for canonical_metric, _, json_key in AVERAGE_METRICS
            ]
            values_by_method.append(as_score_array(values, len(metric_labels)))

        plot_grouped_bars(
            ax,
            values_by_method,
            metric_labels,
            method_labels,
            dataset_display_name(dataset_key),
            "Score (%)" if percent else "Score",
            percent,
        )
        add_panel_label(ax, chr(ord("a") + idx))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
    save_figure(fig, out_dir, "average_metric_bars", formats, dpi)
    plt.close(fig)


def make_per_class_figures(
    methods: dict[str, dict[str, dict]],
    dataset_keys: list[str],
    out_dir: Path,
    formats: list[str],
    dpi: int,
    percent: bool,
) -> None:
    method_labels = list(methods.keys())
    for dataset_key in dataset_keys:
        class_names = get_class_names(methods, dataset_key)
        if not class_names:
            continue

        fig, axes = plt.subplots(
            1,
            len(PER_CLASS_METRICS),
            figsize=(190 * MM_TO_IN, 70 * MM_TO_IN),
            squeeze=False,
            constrained_layout=True,
        )

        for idx, (canonical_metric, label, json_key) in enumerate(PER_CLASS_METRICS):
            ax = axes[0, idx]
            values_by_method = []
            for datasets in methods.values():
                dataset = datasets.get(dataset_key, {})
                values = metric_value(dataset, canonical_metric, json_key, "per_class")
                values_by_method.append(as_score_array(values, len(class_names)))

            plot_grouped_bars(
                ax,
                values_by_method,
                class_names,
                method_labels,
                label,
                "Score (%)" if percent else "Score",
                percent,
            )
            add_panel_label(ax, chr(ord("a") + idx))

        fig.suptitle(dataset_display_name(dataset_key), y=1.08, fontsize=8)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
        save_figure(fig, out_dir, f"{slugify(dataset_key)}_per_cell_type_metric_bars", formats, dpi)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    set_nature_style()

    methods = normalize_methods([*args.result, *args.comparison])
    methods.update(normalize_manual_baselines(args.manual_baselines))

    dataset_keys = collect_dataset_keys(methods, args.datasets)
    if not dataset_keys:
        raise ValueError("No plottable dataset entries were found in the supplied JSON files.")

    make_average_figure(
        methods,
        dataset_keys,
        args.out_dir,
        args.formats,
        args.dpi,
        args.percent,
    )
    make_per_class_figures(
        methods,
        dataset_keys,
        args.out_dir,
        args.formats,
        args.dpi,
        args.percent,
    )
    print(f"Wrote metric bar figures to {args.out_dir}")


if __name__ == "__main__":
    main()
