#!/usr/bin/env python3
"""
Create publication-style figures comparing ADA and baseline evaluation results.

Example:
    python src/eval_testing_scripts/plot_ada_baseline_comparison.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_ADA = SCRIPT_DIR / "evaluation_results_ADA.json"
DEFAULT_BASELINE = SCRIPT_DIR / "evaluation_results_ADA_baseline.json"
DEFAULT_OUT_DIR = REPO_ROOT / "figures" / "ada_baseline_comparison"

MM_TO_IN = 1 / 25.4

DATASET_LABELS = {
    "pannuke_fold": "PanNuke external",
    "orion_test": "Orion original",
}

OVERALL_METRICS = [
    ("accuracy", "Accuracy"),
    ("f1_avg", "F1"),
    ("auc", "AUROC"),
    ("pr_auc", "AUPRC"),
]

COLORS = {
    "baseline": "#4C78A8",
    "ada": "#D55E00",
    "gain": "#0072B2",
    "loss": "#B03A2E",
    "neutral": "#404040",
    "grid": "#D8D8D8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ADA vs. baseline metrics from evaluation result JSON files."
    )
    parser.add_argument("--ada", type=Path, default=DEFAULT_ADA, help="ADA results JSON.")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="Baseline results JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for figure outputs.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "svg"],
        choices=["png", "pdf", "svg"],
        help=(
            "Output formats. SVG is the default vector format; PDF can be requested "
            "when the local matplotlib/fontTools PDF backend is healthy."
        ),
    )
    parser.add_argument("--dpi", type=int, default=600, help="PNG output resolution.")
    return parser.parse_args()


def set_nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
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


def pct(values: Iterable[float]) -> np.ndarray:
    return 100 * np.asarray(list(values), dtype=float)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
    )


def prettify_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)


def plot_overall_panel(
    ax: plt.Axes,
    baseline_dataset: dict,
    ada_dataset: dict,
    title: str,
) -> None:
    baseline = pct(baseline_dataset[key] for key, _ in OVERALL_METRICS)
    ada = pct(ada_dataset[key] for key, _ in OVERALL_METRICS)
    labels = [label for _, label in OVERALL_METRICS]

    x = np.arange(len(labels))
    width = 0.34
    ax.bar(
        x - width / 2,
        baseline,
        width=width,
        color=COLORS["baseline"],
        edgecolor="#222222",
        linewidth=0.45,
        label="Baseline",
    )
    ax.bar(
        x + width / 2,
        ada,
        width=width,
        color=COLORS["ada"],
        edgecolor="#222222",
        linewidth=0.45,
        label="ADA",
    )

    for xi, b_val, a_val in zip(x, baseline, ada):
        delta = a_val - b_val
        color = COLORS["gain"] if delta >= 0 else COLORS["loss"]
        y = max(b_val, a_val) + 2.0
        ax.text(
            xi,
            y,
            f"{delta:+.1f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
            color=color,
        )

    ax.set_title(title, pad=3)
    ax.set_ylabel("Metric value (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x, labels)
    prettify_axis(ax, "y")


def plot_delta_panel(
    ax: plt.Axes,
    baseline_dataset: dict,
    ada_dataset: dict,
    title: str,
    max_label_chars: int = 18,
) -> None:
    class_names = baseline_dataset["class_names"]
    baseline_f1 = pct(baseline_dataset["f1_per_class"])
    ada_f1 = pct(ada_dataset["f1_per_class"])
    delta = ada_f1 - baseline_f1

    y = np.arange(len(class_names))
    colors = [COLORS["gain"] if value >= 0 else COLORS["loss"] for value in delta]

    ax.axvline(0, color="#222222", linewidth=0.7)
    ax.barh(y, delta, height=0.58, color=colors, edgecolor="#222222", linewidth=0.35)

    labels = [
        name if len(name) <= max_label_chars else name.replace(" / ", "/\n")
        for name in class_names
    ]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("ADA - baseline F1 (percentage points)")
    ax.set_title(title, pad=3)

    lim = max(6, np.nanmax(np.abs(delta)) * 1.25)
    ax.set_xlim(-lim, lim)

    for yi, value in zip(y, delta):
        ha = "left" if value >= 0 else "right"
        offset = lim * 0.025 if value >= 0 else -lim * 0.025
        ax.text(
            value + offset,
            yi,
            f"{value:+.1f}",
            va="center",
            ha=ha,
            fontsize=6.1,
            color=COLORS["neutral"],
        )

    prettify_axis(ax, "x")


def plot_per_class_f1_panel(
    ax: plt.Axes,
    baseline_dataset: dict,
    ada_dataset: dict,
    title: str,
) -> None:
    class_names = baseline_dataset["class_names"]
    baseline_f1 = pct(baseline_dataset["f1_per_class"])
    ada_f1 = pct(ada_dataset["f1_per_class"])

    y = np.arange(len(class_names))
    ax.hlines(y, baseline_f1, ada_f1, color="#999999", linewidth=0.8, zorder=1)
    ax.scatter(
        baseline_f1,
        y,
        s=16,
        color=COLORS["baseline"],
        edgecolor="#222222",
        linewidth=0.35,
        label="Baseline",
        zorder=3,
    )
    ax.scatter(
        ada_f1,
        y,
        s=16,
        color=COLORS["ada"],
        edgecolor="#222222",
        linewidth=0.35,
        label="ADA",
        zorder=3,
    )

    ax.set_yticks(y, class_names)
    ax.invert_yaxis()
    ax.set_xlim(0, 90)
    ax.set_xlabel("F1 score (%)")
    ax.set_title(title, pad=3)
    prettify_axis(ax, "x")


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str], dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.03}
        if fmt == "png":
            kwargs["dpi"] = dpi
        try:
            fig.savefig(path, **kwargs)
        except Exception as exc:
            print(f"Warning: could not write {path}: {exc}")
            continue
        print(f"Wrote {path}")


def make_main_figure(baseline: dict, ada: dict, out_dir: Path, formats: list[str], dpi: int) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(183 * MM_TO_IN, 128 * MM_TO_IN),
        gridspec_kw={"height_ratios": [0.86, 1.14], "wspace": 0.38, "hspace": 0.52},
    )

    plot_overall_panel(
        axes[0, 0],
        baseline["pannuke_fold"],
        ada["pannuke_fold"],
        "PanNuke external",
    )
    plot_overall_panel(
        axes[0, 1],
        baseline["orion_test"],
        ada["orion_test"],
        "Orion original",
    )
    plot_delta_panel(
        axes[1, 0],
        baseline["pannuke_fold"],
        ada["pannuke_fold"],
        "Class-level F1 change, PanNuke",
    )
    plot_delta_panel(
        axes[1, 1],
        baseline["orion_test"],
        ada["orion_test"],
        "Class-level F1 change, Orion",
    )

    for ax, label in zip(axes.ravel(), ["a", "b", "c", "d"]):
        panel_label(ax, label)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        handlelength=1.3,
    )

    save_figure(fig, out_dir, "ada_vs_baseline_main", formats, dpi)
    plt.close(fig)


def make_f1_lollipop_figure(
    baseline: dict,
    ada: dict,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(183 * MM_TO_IN, 82 * MM_TO_IN),
        gridspec_kw={"wspace": 0.34},
    )

    plot_per_class_f1_panel(
        axes[0],
        baseline["pannuke_fold"],
        ada["pannuke_fold"],
        "PanNuke external",
    )
    plot_per_class_f1_panel(
        axes[1],
        baseline["orion_test"],
        ada["orion_test"],
        "Orion original",
    )

    for ax, label in zip(axes, ["a", "b"]):
        panel_label(ax, label)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handlelength=1.2,
    )

    save_figure(fig, out_dir, "ada_vs_baseline_per_class_f1", formats, dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_nature_style()

    ada = load_json(args.ada)
    baseline = load_json(args.baseline)

    make_main_figure(baseline, ada, args.out_dir, args.formats, args.dpi)
    make_f1_lollipop_figure(baseline, ada, args.out_dir, args.formats, args.dpi)


if __name__ == "__main__":
    main()
