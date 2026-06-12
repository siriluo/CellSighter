#!/usr/bin/env python3
"""
Template for assembling Figure 1 from separately produced panel images.

The script is intentionally light-touch: edit PANEL_PATHS below, or pass paths
on the command line, and it will arrange panels 1A-1E into a single
publication-style figure.

Example:
    python scripts/assemble_figure1_template.py \
        --panel-a figures/panels/model_architecture.svg \
        --panel-b figures/panels/data_overview.png \
        --panel-c figures/panels/cross_validation.png \
        --panel-d figures/panels/confusion_matrix.png \
        --panel-e figures/panels/conformal_performance.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "figures" / "assembled"

MM_TO_INCH = 1 / 25.4

# Edit these paths directly if you prefer not to use command-line arguments.
PANEL_PATHS = {
    "A": None,  # Model architecture: ResNet + contrastive learning + ADA + conformal prediction
    "B": None,  # Data
    "C": None,  # Cross validation
    "D": None,  # Confusion matrix
    "E": None,  # Performance with conformal prediction
}

PANEL_TITLES = {
    "A": "Model architecture",
    "B": "Data",
    "C": "Cross validation",
    "D": "Confusion matrix",
    "E": "Conformal prediction performance",
}

# Grid layout: (row_start, row_end, col_start, col_end)
# A is intentionally wide because architecture schematics usually need space.
PANEL_LAYOUT = {
    "A": (0, 1, 0, 4),
    "B": (0, 1, 4, 6),
    "C": (1, 2, 0, 2),
    "D": (1, 2, 2, 4),
    "E": (1, 2, 4, 6),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble a multi-panel Figure 1 from individual panel images."
    )
    parser.add_argument("--panel-a", type=Path, default=None, help="Path to panel 1A image.")
    parser.add_argument("--panel-b", type=Path, default=None, help="Path to panel 1B image.")
    parser.add_argument("--panel-c", type=Path, default=None, help="Path to panel 1C image.")
    parser.add_argument("--panel-d", type=Path, default=None, help="Path to panel 1D image.")
    parser.add_argument("--panel-e", type=Path, default=None, help="Path to panel 1E image.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for assembled figure output.",
    )
    parser.add_argument(
        "--stem",
        default="figure1_template",
        help="Output file stem.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "svg"],
        choices=["png", "svg", "pdf"],
        help="Output formats. SVG is recommended for vector editing.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution.")
    parser.add_argument(
        "--hide-titles",
        action="store_true",
        help="Hide panel titles while keeping panel letters.",
    )
    return parser.parse_args()


def set_manuscript_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )


def resolved_panel_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    cli_paths = {
        "A": args.panel_a,
        "B": args.panel_b,
        "C": args.panel_c,
        "D": args.panel_d,
        "E": args.panel_e,
    }

    paths = {}
    for panel, cli_path in cli_paths.items():
        path = cli_path if cli_path is not None else PANEL_PATHS[panel]
        paths[panel] = None if path is None else Path(path)
    return paths


def add_panel_label(ax: plt.Axes, panel: str) -> None:
    ax.text(
        -0.035,
        1.035,
        panel.lower(),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        clip_on=False,
    )


def draw_placeholder(ax: plt.Axes, panel: str, title: str) -> None:
    rect = patches.Rectangle(
        (0.015, 0.015),
        0.97,
        0.97,
        transform=ax.transAxes,
        facecolor="#F7F7F7",
        edgecolor="#999999",
        linewidth=0.7,
        linestyle=(0, (3, 2)),
    )
    ax.add_patch(rect)
    ax.text(
        0.5,
        0.53,
        f"Panel 1{panel}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="#333333",
    )
    ax.text(
        0.5,
        0.43,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.5,
        color="#555555",
        wrap=True,
    )


def read_raster(path: Path) -> np.ndarray:
    image = plt.imread(path)
    if image.ndim == 3 and image.shape[-1] == 4:
        rgb = image[..., :3]
        alpha = image[..., 3:]
        image = rgb * alpha + (1 - alpha)
    return image


def draw_panel_image(ax: plt.Axes, path: Path, panel: str, title: str) -> None:
    if not path.exists():
        draw_placeholder(ax, panel, f"Missing: {path}")
        return

    if path.suffix.lower() in {".svg", ".pdf"}:
        draw_placeholder(
            ax,
            panel,
            f"Vector input placeholder\n{path.name}\nExport this panel as PNG for preview.",
        )
        return

    image = read_raster(path)
    ax.imshow(image, aspect="equal")


def format_panel_axis(ax: plt.Axes, panel: str, show_title: bool) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_panel_label(ax, panel)
    if show_title:
        ax.set_title(PANEL_TITLES[panel], pad=2.5)


def build_figure(panel_paths: Mapping[str, Path | None], show_titles: bool) -> plt.Figure:
    fig = plt.figure(figsize=(183 * MM_TO_INCH, 126 * MM_TO_INCH))
    grid = fig.add_gridspec(
        2,
        6,
        height_ratios=[1.05, 1.0],
        wspace=0.28,
        hspace=0.34,
    )

    for panel in ["A", "B", "C", "D", "E"]:
        row_start, row_end, col_start, col_end = PANEL_LAYOUT[panel]
        ax = fig.add_subplot(grid[row_start:row_end, col_start:col_end])
        path = panel_paths[panel]
        if path is None:
            draw_placeholder(ax, panel, PANEL_TITLES[panel])
        else:
            draw_panel_image(ax, path, panel, PANEL_TITLES[panel])
        format_panel_axis(ax, panel, show_titles)

    return fig


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


def main() -> None:
    args = parse_args()
    set_manuscript_style()
    panel_paths = resolved_panel_paths(args)
    fig = build_figure(panel_paths, show_titles=not args.hide_titles)
    save_figure(fig, args.out_dir, args.stem, args.formats, args.dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
