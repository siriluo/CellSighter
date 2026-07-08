"""Combine image panels into a publication-style multi-panel figure.

Example
-------
python scripts/combine_images_nature.py ^
    --images figures/a.png figures/b.png figures/c.png figures/d.png ^
    --rows 2 --cols 2 ^
    --labels a b c d ^
    --titles "Input" "Prediction" "Ground truth" "Overlay" ^
    --output figures/combined_figure
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine multiple image files into one Nature-style figure."
    )
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Image paths to combine. PNG, JPG, and TIFF are supported by matplotlib/Pillow.",
    )
    parser.add_argument(
        "--output",
        default="combined_figure",
        help="Output path without extension, or with one extension when --formats has one value.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Number of figure rows. If omitted, inferred from --cols or image count.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Number of figure columns. If omitted, inferred from --rows or image count.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Panel labels, for example: a b c d. Defaults to lowercase letters.",
    )
    parser.add_argument(
        "--titles",
        nargs="*",
        default=None,
        help="Optional panel titles. Use one title per image.",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=7.0,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--panel-height",
        type=float,
        default=2.0,
        help="Approximate height in inches per row.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="DPI for raster export.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf", "svg"],
        choices=["png", "pdf", "svg", "tiff"],
        help="Output formats.",
    )
    parser.add_argument(
        "--wspace",
        type=float,
        default=0.03,
        help="Horizontal space between panels.",
    )
    parser.add_argument(
        "--hspace",
        type=float,
        default=0.08,
        help="Vertical space between panels.",
    )
    parser.add_argument(
        "--bg",
        default="white",
        help="Figure background color.",
    )
    parser.add_argument(
        "--hide-empty",
        action="store_true",
        help="Hide unused grid panels when rows x cols exceeds image count.",
    )
    return parser.parse_args()


def infer_grid(n_images: int, rows: int | None, cols: int | None) -> tuple[int, int]:
    if rows is None and cols is None:
        cols = math.ceil(math.sqrt(n_images))
        rows = math.ceil(n_images / cols)
    elif rows is None:
        rows = math.ceil(n_images / cols)
    elif cols is None:
        cols = math.ceil(n_images / rows)

    if rows <= 0 or cols <= 0:
        raise ValueError("--rows and --cols must be positive integers.")
    if rows * cols < n_images:
        raise ValueError(
            f"Grid has {rows * cols} slots but {n_images} images were provided."
        )
    return rows, cols


def default_labels(n_images: int) -> list[str]:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    labels = []
    for i in range(n_images):
        if i < len(alphabet):
            labels.append(alphabet[i])
        else:
            labels.append(f"{alphabet[i // len(alphabet) - 1]}{alphabet[i % len(alphabet)]}")
    return labels


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
        }
    )


def output_stem_and_formats(output: str, formats: list[str]) -> tuple[Path, list[str]]:
    output_path = Path(output)
    if output_path.suffix:
        suffix = output_path.suffix.lstrip(".").lower()
        if len(formats) > 1:
            return output_path.with_suffix(""), formats
        return output_path.with_suffix(""), [suffix]
    return output_path, formats


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.0,
        1.02,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )


def main() -> None:
    args = parse_args()
    image_paths = [Path(path) for path in args.images]
    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing image file(s): " + ", ".join(missing))

    rows, cols = infer_grid(len(image_paths), args.rows, args.cols)
    labels = args.labels if args.labels else default_labels(len(image_paths))
    titles = args.titles if args.titles else [""] * len(image_paths)

    if len(labels) != len(image_paths):
        raise ValueError("--labels must have the same number of entries as --images.")
    if len(titles) != len(image_paths):
        raise ValueError("--titles must have the same number of entries as --images.")

    configure_style()

    fig_height = max(args.panel_height * rows, 1.8)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(args.fig_width, fig_height),
        facecolor=args.bg,
        squeeze=False,
    )
    fig.patch.set_facecolor(args.bg)

    flat_axes = axes.ravel()
    for ax, image_path, label, title in zip(flat_axes, image_paths, labels, titles):
        image = mpimg.imread(image_path)
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(args.bg)
        add_panel_label(ax, label)
        if title:
            ax.set_title(title, fontsize=7, pad=3)

    for ax in flat_axes[len(image_paths) :]:
        ax.set_xticks([])
        ax.set_yticks([])
        if args.hide_empty:
            ax.axis("off")
        else:
            ax.set_facecolor(args.bg)

    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.02,
        top=0.96,
        wspace=args.wspace,
        hspace=args.hspace,
    )

    stem, formats = output_stem_and_formats(args.output, args.formats)
    stem.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        save_kwargs = {"bbox_inches": "tight", "facecolor": fig.get_facecolor()}
        if fmt in {"png", "tiff"}:
            save_kwargs["dpi"] = args.dpi
        fig.savefig(stem.with_suffix(f".{fmt}"), **save_kwargs)

    plt.close(fig)


if __name__ == "__main__":
    main()
