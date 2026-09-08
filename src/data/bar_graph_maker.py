# professional_celltype_barplot.py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------- Nature-style helpers ----------
MM_TO_INCH = 1 / 25.4

def set_nature_style():
    mpl.rcParams.update({
        # Font (Nature commonly uses sans-serif; Arial preferred if available)
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,

        # Lines/spines/ticks
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,

        # Export
        "pdf.fonttype": 42,  # editable text
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.transparent": False,
    })
    
    
def _compute_null_scores(metric_name, class_counts, n_classes):
    """
    Auto-compute per-class null baseline:
    - ROC AUC: 0.5 for all classes
    - PR AUC: prevalence p_c
    - F1 Score: prevalence-matched random baseline p_c (q_c = p_c)
    """
    m = metric_name.strip().lower()

    if ("roc" in m and "auc" in m) or m == "auc":
        return np.full(n_classes, 0.5, dtype=float)

    if class_counts is None:
        raise ValueError(
            "For this metric, provide null_scores manually or pass class_counts "
            "to auto-compute null baselines."
        )

    counts = np.array(class_counts, dtype=float)
    if np.any(counts < 0) or counts.sum() <= 0:
        raise ValueError("class_counts must be non-negative and sum to > 0.")
    p = counts / counts.sum()

    if "pr" in m and "auc" in m:
        return p  # PR AUC null baseline

    if "f1" in m:
        return p  # F1 prevalence-matched null baseline

    raise ValueError(
        "Unknown metric for auto null baseline. Provide null_scores explicitly."
    )
    
    
def plot_nature_bar(
    cell_types,
    scores,
    metric_name="F1 score",
    errors=None,                      # optional: std/SEM/CI half-width
    output_stem="nature_celltype_metric",
    title=None,
    sort_desc=True,
    horizontal=False,
    y_or_x_lim=(0.0, 1.0),
    annotate=False,                   # Nature often avoids on-bar numbers; keep False by default
    reference_line=None,              # e.g., macro average
    reference_label="Mean",
    single_column=True,               # True=89 mm wide, False=183 mm wide
):
    if len(cell_types) != len(scores):
        raise ValueError("cell_types and scores must have same length")
    if errors is not None and len(errors) != len(scores):
        raise ValueError("errors must have same length as scores")

    set_nature_style()

    cell_types = np.array(cell_types, dtype=object)
    scores = np.array(scores, dtype=float)
    errs = None if errors is None else np.array(errors, dtype=float)

    if sort_desc:
        order = np.argsort(scores)[::-1]
        cell_types, scores = cell_types[order], scores[order]
        if errs is not None:
            errs = errs[order]

    n = len(cell_types)
    fig_w_mm = 89 if single_column else 183
    if horizontal:
        fig_h_mm = max(45, 5.5 * n + 12)
    else:
        fig_h_mm = max(45, 3.2 * n + 24)

    fig, ax = plt.subplots(figsize=(fig_w_mm * MM_TO_INCH, fig_h_mm * MM_TO_INCH))

    # restrained, colorblind-safe single-color style
    bar_color = "#4C78A8"
    edge_color = "#333333"

    idx = np.arange(n)

    if horizontal:
        bars = ax.barh(
            idx, scores, xerr=errs, height=0.72,
            color=bar_color, edgecolor=edge_color, linewidth=0.5, capsize=2.0
        )
        ax.set_yticks(idx)
        ax.set_yticklabels(cell_types)
        ax.set_xlabel(metric_name)
        # ax.set_ylabel("Cell type")
        ax.set_xlim(*y_or_x_lim)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="-", linewidth=0.4, alpha=0.22)
        ax.grid(axis="y", visible=False)
    else:
        bars = ax.bar(
            idx, scores, yerr=errs, width=0.72,
            color=bar_color, edgecolor=edge_color, linewidth=0.5, capsize=2.0
        )
        ax.set_xticks(idx)
        ax.set_xticklabels(cell_types, rotation=35, ha="right")
        ax.set_ylabel(metric_name)
        # ax.set_xlabel("Cell type")
        ax.set_ylim(*y_or_x_lim)
        ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.22)
        ax.grid(axis="x", visible=False)

    if reference_line is not None:
        if horizontal:
            ax.axvline(reference_line, color="#666666", linestyle="--", linewidth=0.8,
                       label=f"{reference_label} ({reference_line:.3f})")
        else:
            ax.axhline(reference_line, color="#666666", linestyle="--", linewidth=0.8,
                       label=f"{reference_label} ({reference_line:.3f})")
        ax.legend(frameon=False, loc="best", fontsize=6.5, handlelength=1.4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title:
        ax.set_title(title, pad=3)

    if annotate:
        span = y_or_x_lim[1] - y_or_x_lim[0]
        for b, s in zip(bars, scores):
            if horizontal:
                ax.text(s + 0.01 * span, b.get_y() + b.get_height() / 2, f"{s:.3f}",
                        va="center", ha="left", fontsize=6.2)
            else:
                ax.text(b.get_x() + b.get_width() / 2, s + 0.01 * span, f"{s:.3f}",
                        va="bottom", ha="center", fontsize=6.2)

    fig.tight_layout(pad=0.5)
    fig.savefig(f"{output_stem}.pdf", dpi=600, bbox_inches="tight")
    fig.savefig(f"{output_stem}.svg", dpi=600, bbox_inches="tight")
    fig.savefig(f"{output_stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_stem}.pdf/.svg/.png")


def plot_celltype_metric_bar(
    cell_types,
    scores,
    metric_name="F1 Score",
    errors=None,                    # Optional error bars for actual scores
    null_scores=None,               # Optional per-class null baseline bars
    null_label="Null Baseline",
    class_counts=None,              # Optional counts to auto-build null_scores
    title=None,
    output_stem="celltype_metric",
    sort_desc=True,
    ylim=(0.0, 1.0),
    annotate=True,
    annotate_null=False,
    horizontal=False,
    reference_line=None,
    reference_label="Macro Avg",
):
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    def _compute_null_scores(metric_name_local, class_counts_local, n_classes):
        m = metric_name_local.strip().lower()

        if ("roc" in m and "auc" in m) or m == "auc":
            return np.full(n_classes, 0.5, dtype=float)

        if class_counts_local is None:
            raise ValueError(
                "For this metric, provide null_scores manually or pass class_counts "
                "to auto-compute null baselines."
            )

        counts = np.array(class_counts_local, dtype=float)
        if np.any(counts < 0) or counts.sum() <= 0:
            raise ValueError("class_counts must be non-negative and sum to > 0.")
        p = counts / counts.sum()

        if "pr" in m and "auc" in m:
            return p  # PR AUC null baseline
        if "f1" in m:
            return p  # F1 prevalence-matched null baseline

        raise ValueError(
            "Unknown metric for auto null baseline. Provide null_scores explicitly."
        )

    # ---- Validation ----
    if len(cell_types) != len(scores):
        raise ValueError("cell_types and scores must have same length.")
    if errors is not None and len(errors) != len(scores):
        raise ValueError("errors must have same length as scores.")
    if class_counts is not None and len(class_counts) != len(scores):
        raise ValueError("class_counts must have same length as scores.")

    cell_types = np.array(cell_types, dtype=object)
    scores = np.array(scores, dtype=float)
    errs = None if errors is None else np.array(errors, dtype=float)

    # Auto null baseline if not provided
    if null_scores is None and class_counts is not None:
        null_scores = _compute_null_scores(metric_name, class_counts, len(scores))
    print(null_scores)
    null_vals = None if null_scores is None else np.array(null_scores, dtype=float)
    if null_vals is not None and len(null_vals) != len(scores):
        raise ValueError("null_scores must have same length as scores.")

    # ---- Sort (apply same order to all arrays) ----
    if sort_desc:
        idx = np.argsort(scores)[::-1]
        cell_types, scores = cell_types[idx], scores[idx]
        if errs is not None:
            errs = errs[idx]
        if null_vals is not None:
            null_vals = null_vals[idx]

    # ---- Paper-style rcParams ----
    mpl.rcParams.update({
        # "font.family": "serif",
        # "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 12,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    # ---- Figure size adapts to category count ----
    n = len(cell_types)
    if horizontal:
        fig_h = max(3.2, 0.35 * n + 1.6)
        fig, ax = plt.subplots(figsize=(7.2, fig_h))
    else:
        fig_w = max(6.0, 0.58 * n + 1.8)
        fig, ax = plt.subplots(figsize=(fig_w, 4.3))

    # ---- Color policy: class-colored actual bars + single neutral null baseline ----
    # cell_type_colors = {
    #     "Tumor Cells": "#C73E3A",                         # muted red
    #     "T / Treg cells": "#7B61A8",                      # muted purple
    #     "B cells": "#3B6FB6",                             # professional blue
    #     "Myeloid cells": "#D98C2B",                       # muted orange
    #     "Stromal / Vascular / Smooth Muscle": "#4C9A5F", # muted green
    # }
    cell_type_colors = {
        "Tumor/Epithelial": "#C73E3A",                         # muted red
        "Lymphocytes": "#7B61A8",                      # muted purple
        "Myeloid": "#3B6FB6",                             # professional blue
        "Stromal/Mesenchymal": "#D98C2B",                       # muted orange
        "Vasculature": "#4C9A5F", # muted green
    }

    # cell_type_colors = {
    #     "CD4+ T": "#4E79A7",  # 
    #     "CD8+ T": "#7BA7D7",  # CD8_T
    #     "Treg": "#2F5F8F",  # Treg
    #     "B cell": "#76B7B2",  # B_cell
    #     "Mono/Macro": "#59A14F",  # Mono_Macro
    #     "Granulocyte": "#A0C86F",  # Granulocyte
    #     "Stromal": "#C65413",  # Stromal
    #     "Smooth Muscle": "#D78228",  # Smooth_Muscle
    #     "Tumor": "#B07AA1",  # Tumor
    #     "Vasculature": "#DE6668",  # Vasculature
    # }
    
    fallback_actual = "#3B6FB6"
    actual_colors = "#4C72B0" # [cell_type_colors.get(str(ct), fallback_actual) for ct in cell_types]

    null_color = "#BFC3C9"
    edge_color = "#1F1F1F"

    x = np.arange(n)
    has_null = null_vals is not None

    if has_null:
        bar_w = 0.36
        offset = bar_w / 2
    else:
        bar_w = 0.72
        offset = 0.0

    if horizontal:
        if has_null:
            bars_actual = ax.barh(
                x - offset, scores, xerr=errs,
                color=actual_colors, edgecolor=edge_color,
                linewidth=0.6, height=bar_w, capsize=2.8, label="Actual"
            )
            bars_null = ax.barh(
                x + offset, null_vals,
                color=null_color, edgecolor=edge_color,
                linewidth=0.6, height=bar_w, label=null_label # , hatch="//"
            )
        else:
            bars_actual = ax.barh(
                x, scores, xerr=errs,
                color=actual_colors, edgecolor=edge_color,
                linewidth=0.6, height=bar_w, capsize=2.8, label="Actual"
            )
            bars_null = None

        ax.set_yticks(x)
        ax.set_yticklabels(cell_types)
        ax.set_xlabel(metric_name)
        ax.set_ylabel("Cell Type", fontsize=14)
        ax.set_xlim(*ylim)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle=(0, (2, 2)), linewidth=0.6, alpha=0.45)
        ax.grid(axis="y", visible=False)
    else:
        if has_null:
            bars_actual = ax.bar(
                x - offset, scores, yerr=errs,
                color=actual_colors, edgecolor=edge_color,
                linewidth=0.6, width=bar_w, capsize=2.8, label="Actual"
            )
            bars_null = ax.bar(
                x + offset, null_vals,
                color=null_color, edgecolor=edge_color,
                linewidth=0.6, width=bar_w, label=null_label # , hatch="//"
            )
        else:
            bars_actual = ax.bar(
                x, scores, yerr=errs,
                color=actual_colors, edgecolor=edge_color,
                linewidth=0.6, width=bar_w, capsize=2.8, label="Actual"
            )
            bars_null = None

        ax.set_xticks(x)
        ax.set_xticklabels(cell_types, rotation=30, ha="right", fontsize=12)
        ax.set_ylabel(metric_name, fontsize=14)
        # ax.set_xlabel("Cell Type", fontsize=14)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", visible=False) # linestyle=(0, (2, 2)), linewidth=0.6, alpha=0.45
        ax.grid(axis="x", visible=False)

    # Reference line (optional)
    if reference_line is not None:
        if horizontal:
            ax.axvline(
                reference_line, color="#6E6E6E", linestyle="--", linewidth=1.0, alpha=0.9,
                label=f"{reference_label}: {reference_line:.3f}"
            )
        else:
            ax.axhline(
                reference_line, color="#6E6E6E", linestyle="--", linewidth=1.0, alpha=0.9,
                label=f"{reference_label}: {reference_line:.3f}"
            )

    if has_null or reference_line is not None:
        ax.legend(frameon=False, loc="best")

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title is None:
        title = f"{metric_name} by Cell Type"
    # ax.set_title(title, pad=8)

    # Numeric annotations
    if annotate or annotate_null:
        span = ylim[1] - ylim[0]
        if horizontal:
            if annotate:
                for b, s in zip(bars_actual, scores):
                    ax.text(
                        s + 0.012 * span, b.get_y() + b.get_height() / 2, f"{s:.3f}",
                        va="center", ha="left", fontsize=5, color="#222222"
                    )
            if annotate_null and bars_null is not None:
                for b, s in zip(bars_null, null_vals):
                    ax.text(
                        s + 0.012 * span, b.get_y() + b.get_height() / 2, f"{s:.3f}",
                        va="center", ha="left", fontsize=5, color="#444444"
                    )
        else:
            if annotate:
                for b, s in zip(bars_actual, scores):
                    ax.text(
                        b.get_x() + b.get_width() / 2, s + 0.012 * span, f"{s:.3f}",
                        va="bottom", ha="center", fontsize=5, color="#222222"
                    )
            if annotate_null and bars_null is not None:
                for b, s in zip(bars_null, null_vals):
                    ax.text(
                        b.get_x() + b.get_width() / 2, s + 0.012 * span, f"{s:.3f}",
                        va="bottom", ha="center", fontsize=5, color="#444444"
                    )

    fig.tight_layout()
    # fig.savefig(f"figure_images/{output_stem}.pdf", dpi=600, bbox_inches="tight")
    # fig.savefig(f"figure_images/{output_stem}.svg", dpi=600, bbox_inches="tight")
    # fig.savefig(f"figure_images/{output_stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: figure_images/{output_stem}.pdf, figure_images/{output_stem}.svg, figure_images/{output_stem}.png")


def plot_overall_accuracy_vs_random_baseline(
    accuracy,
    random_baseline,
    output_stem="overall_accuracy_vs_random_baseline",
    model_label="Model",
    baseline_label="Random baseline",
    title=None,
    ylim=(0.0, 1.0),
    annotate=True,
    single_column=True,
):
    """
    Plot overall model accuracy against a manually supplied random baseline.

    Parameters
    ----------
    accuracy : float
        Overall model accuracy.
    random_baseline : float
        Overall random-baseline accuracy.
    """
    mpl.rcParams.update({
        # "font.family": "serif",
        # "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    values = np.array([accuracy, random_baseline], dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("accuracy and random_baseline must be finite numbers.")

    fig_w_mm = 64 if single_column else 90
    fig, ax = plt.subplots(figsize=(fig_w_mm * MM_TO_INCH, 48 * MM_TO_INCH))

    labels = [model_label, baseline_label]
    colors = ["#4C78A8", "#BFC3C9"]
    bars = ax.bar(
        np.arange(2),
        values,
        width=0.58,
        color=colors,
        edgecolor="#333333",
        linewidth=0.5,
    )

    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("Overall accuracy", fontsize=9)
    ax.set_ylim(*ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title:
        ax.set_title(title, pad=3)

    if annotate:
        span = ylim[1] - ylim[0]
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015 * span,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#222222",
            )

    # fig.tight_layout(pad=0.5)
    # fig.savefig(f"figure_images/{output_stem}.pdf", dpi=600, bbox_inches="tight")
    # fig.savefig(f"figure_images/{output_stem}.svg", dpi=600, bbox_inches="tight")
    # fig.savefig(f"figure_images/{output_stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_stem}.pdf/.svg/.png")


if __name__ == "__main__":
    # -------- MANUAL INPUT --------
    metric_name = "PR AUC"  # PR AUC or "F1 Score" ROC AUC  Class-wise Accuracy
    # cell_types = [
    #     "Tumor Cells",
    #     "T / Treg cells",
    #     "B cells",
    #     "Myeloid cells",
    #     "Stromal / Vascular / Smooth Muscle",
    # ]
    
    # cell_types = [
    #     "Tumor/Epithelial",
    #     "Lymphocytes",
    #     "Myeloid",
    #     "Stromal/Mesenchymal",
    #     "Vasculature"
    # ]
    
    cell_types=[
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
    # cell_types = ["10 Cell Type Avg"]
    
    # scores = [0.9789543020004792,
    #     0.8732201105633748,
    #     0.837799886030762,
    #     0.8686055892403939,
    #     0.8770346009146741]
    # scores = [0.8567, 0.6200, 0.1372, 0.3638, 0.7411]
    # scores = [
    #     0.9279656832995415,
    #     0.6914809868196645,
    #     0.17452080771338802,
    #     0.43547955921805903,
    #     0.832469449624032
    # ]
    
    
    # PR AUC
    # 10 cell type scores
    # micro
    # scores = [0.117980337, 0.148079096, 0.503294865, 0.215328212, 0.284291536, 0.513979079, 0.476767444, 0.940647384, 0.542936631, 0.183677838]
    # macro
    scores = [0.117980337, 0.148079096, 0.503294866, 0.215328212, 0.284291536, 0.513979079, 0.476767444, 0.940647384, 0.542936632, 0.183677838]
    # 5 cell type scores
    # scores = [0.940647384, 0.73866939, 0.419002214, 0.76399035, 0.542936632]
    
    # f1 score macro 
    # 10 cell type
    # scores = [0.00333, 0.1099, 0.5315, 0.1103, 0.3521, 0.4271, 0.4154, 0.8630, 0.4853, 0.0069]
    # 5 cell type
    # scores = [0.8631, 0.6746, 0.3918, 0.6253, 0.4771]
    
    # Accuracy
    # mean 
    # 10 cell type
    # scores = [0.526]
    # 5 cell type
    # scores = [0.652]
    # individual
    # 10 cell type
    # scores = [0.001744847354292734, 0.06773264712634883, 0.6966013952001153, 0.06901397963552428, 0.37781013973056066, 0.3735248040085194, 0.3621078074617663, 0.8629908173016995, 0.5718202365737127, 0.0035267729311825903]
    # 5 cell type
    # scores = [0.854138031329206, 0.7617518117621256, 0.32372585748950683, 0.561166949899581, 0.48793328243881934]

    
    
    errors = None # [0.010, 0.012, 0.015, 0.017, 0.013, 0.020]  # optional; set None to disable

    macro_avg = None # float(np.mean(scores))
    
    class_counts = [81353, 74053, 318268, 90332, 130357, 138766, 314296, 375161, 369068, 39986] # None # [601675, 658525, 107442, 243968, 1016519]  # optional; set None to disable auto null baseline
    # class_counts = [375161, 564006, 170343, 453062, 369068]
    
    # null_scores_used = [0.2289, 0.2506, 0.0409, 0.0928, 0.3868] # PR AUC null baselines (prevalence) - computed from class counts
    # null_scores_used = [0.2135, 0.2224, 0.0679, 0.1268, 0.2637] # F1 Score null baselines (prevalence-matched random) - computed from class counts
    # null_scores_used = None # set to None to auto-compute null baselines from class_counts and metric_name
    
    # null_f1_10class = [0.03956178639506208, 0.050744109356905404, 0.2030631537779395, 0.0378247203719037, 0.07207235464605792, 0.07314186136277187, 0.13080401675816783, 0.2031141629331471, 0.1780799933669056, 0.009926518129265207]
    # null_f1_5class = [0.2031, 0.3320, 0.0820, 0.2041, 0.1781]
    # mean
    # null_acc_10class = [0.14770880754548069]
    # null_acc_5class = [0.2316918401609893]
    # independent
    # null_acc_10class = [0.045884434092034555, 0.049864336803648455, 0.20116410273159793, 0.03854768261380903, 0.07457393093045049, 0.07223302542205082, 0.13595143128201886, 0.19277417275302386, 0.17914964878786788, 0.009877702443392606]
    # null_acc_5class = [0.19277951567599041, 0.33546649743385215, 0.08444162970487896, 0.2081745430557702, 0.17915468643060406]
    
    plot_celltype_metric_bar(
        cell_types=cell_types,
        scores=scores,
        metric_name=metric_name,
        errors=errors,
        class_counts=class_counts,           # auto-computes PR AUC null per class class_counts
        null_scores=None,                    # leave None to auto-compute
        null_label="Random",
        title=None, # f"{metric_name} for Individual Cell Types"
        output_stem=f"{metric_name.lower().replace(' ', '_')}_celltype_bar_5class_fixednull_acc",
        sort_desc=True,
        ylim=(0.0, 1.0),
        annotate=False,  # Nature often avoids on-bar numbers; keep False by default
        annotate_null=False,  # Nature often avoids on-bar numbers; keep False by default
        horizontal=False,           # switch to True for many classes
        reference_line=macro_avg,
        reference_label="Mean",
    )
    # plot_overall_accuracy_vs_random_baseline(
    #     accuracy=0.652,
    #     random_baseline=0.2316918401609893,
    #     output_stem="orion_overall_accuracy_vs_random_5class",
    #     model_label="Model (5 class)",
    #     baseline_label="Random baseline",
    #     annotate=False,
    # )
    
    # cell_types=cell_types,
    # scores=scores,
    # metric_name=metric_name,
    # errors=errors,
    # class_counts=class_counts,           # auto-computes PR AUC null per class
    # null_scores=None,                    # leave None to auto-compute
    # null_label="Random (Prevalence)",
    # title=f"{metric_name} vs Null by Cell Type",
    # output_stem="pr_auc_celltype_with_null",
    # sort_desc=True,
    # ylim=(0.0, 1.0),
    # annotate=True,
    # annotate_null=False,
    # horizontal=False,
    # reference_line=float(np.mean(scores)),
    # reference_label="Mean Actual",
    
    
    # plot_nature_bar(
    #     cell_types=cell_types,
    #     scores=scores,
    #     metric_name=metric_name,
    #     errors=errors,
    #     output_stem="nature_f1score_celltypes",
    #     title=None,                     # Nature style often omits large titles
    #     sort_desc=False,
    #     horizontal=False,               # True recommended for many cell types
    #     y_or_x_lim=(0.0, 1.0),
    #     annotate=False,
    #     reference_line=float(np.mean(scores)),
    #     reference_label="Mean",
    #     single_column=True,             # 89 mm wide
    # )