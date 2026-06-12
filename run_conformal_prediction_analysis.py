#!/usr/bin/env python3
"""Conformal prediction analysis for ORION, PanNuke, and Xenium outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


CLASS_NAMES = [
    "CD4+ T",
    "CD8+ T",
    "Treg",
    "B cells",
    "Monocytes / Macrophages",
    "Stromal Cells",
    "Smooth Muscle",
    "Tumor Cells",
    "Vasculature",
    "Granulocytes",
]

PROB_COLS = [
    "prob_name_cd4plus_t",
    "prob_name_cd8plus_t",
    "prob_name_treg",
    "prob_name_b_cells",
    "prob_name_monocytes__macrophages",
    "prob_name_stromal_cells",
    "prob_name_smooth_muscle",
    "prob_name_tumor_cells",
    "prob_name_vasculature",
    "prob_name_granulocytes",
]

MERGE5_NAMES = [
    "Tumor/Epithelial",
    "Lymphocytes",
    "Myeloid",
    "Stromal/Mesenchymal",
    "Vasculature",
]

MERGE5_BY_CLASS = {
    0: 1,  # CD4+ T -> Lymphocytes
    1: 1,  # CD8+ T -> Lymphocytes
    2: 1,  # Treg -> Lymphocytes
    3: 1,  # B cells -> Lymphocytes
    4: 2,  # Monocytes / Macrophages -> Myeloid
    5: 3,  # Stromal Cells -> Stromal/Mesenchymal
    6: 3,  # Smooth Muscle -> Stromal/Mesenchymal
    7: 0,  # Tumor Cells -> Tumor/Epithelial
    8: 4,  # Vasculature
    9: 2,  # Granulocytes -> Myeloid
}


def parse_args() -> argparse.Namespace:
    base = Path("/taiga/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=base)
    parser.add_argument("--calibration-dir", type=Path, default=base / "con_classifier_testing_checkpoints_val_testing")
    parser.add_argument("--orion-test-dir", type=Path, default=base / "experiment_results/evaluation_results/orion_fold_testing_results/results_fold4_orion_testing_stuff")
    parser.add_argument(
        "--pannuke-parquet",
        type=Path,
        default=base
        / "pannuke_inference_outputs_x20_fold3_retrain"
        / "PanNuke_colon_x20_fold3_retrain"
        / "PanNuke_colon_x20_fold3_retrain_cell_predictions.parquet",
    )
    parser.add_argument(
        "--xenium-parquet",
        type=Path,
        default=base
        / "xenium_inference_outputs_patchseg_fold3_retrain"
        / "Xenium_SMURF_fold3_retrain"
        / "Xenium_SMURF_fold3_retrain_cell_predictions.parquet",
    )
    parser.add_argument("--output-dir", type=Path, default=base / "conformal_prediction_outputs_fold4")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.15, 0.10, 0.05])
    return parser.parse_args()


def coverage_label(alpha: float) -> str:
    coverage = 1.0 - float(alpha)
    return f"{int(round(coverage * 100)):02d}"


def iter_alphas(alphas: Iterable[float]) -> List[Tuple[float, str]]:
    return [(float(alpha), coverage_label(float(alpha))) for alpha in sorted(alphas, reverse=True)]


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32, copy=False)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def load_npz_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    return data[data.files[0]]


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return 1.0
    level = np.ceil((scores.size + 1) * (1.0 - alpha)) / scores.size
    level = min(float(level), 1.0)
    return float(np.quantile(scores, level, method="higher"))


def fit_thresholds(probs: np.ndarray, labels: np.ndarray, alpha: float, num_classes: int) -> Dict[str, np.ndarray | float]:
    true_scores = 1.0 - probs[np.arange(labels.size), labels]
    global_q = conformal_quantile(true_scores, alpha)
    class_q = np.zeros(num_classes, dtype=np.float64)
    for class_id in range(num_classes):
        class_scores = true_scores[labels == class_id]
        class_q[class_id] = conformal_quantile(class_scores, alpha)
    return {"global": global_q, "class_conditional": class_q}


def prediction_mask(probs: np.ndarray, threshold: np.ndarray | float) -> np.ndarray:
    scores = 1.0 - probs
    if np.isscalar(threshold):
        return scores <= float(threshold)
    return scores <= np.asarray(threshold, dtype=np.float64).reshape(1, -1)


def mask_to_strings(mask: np.ndarray, names: List[str]) -> List[str]:
    out: List[str] = []
    for row in mask:
        idx = np.flatnonzero(row)
        out.append("|".join(names[i] for i in idx))
    return out


def add_set_columns(
    df: pd.DataFrame,
    probs: np.ndarray,
    thresholds: Dict[str, Dict[float, Dict[str, np.ndarray | float]]],
    names: List[str],
    prefix: str,
    alphas: Iterable[float],
) -> pd.DataFrame:
    for alpha, label in iter_alphas(alphas):
        for method in ["global", "class_conditional"]:
            q = thresholds[prefix][alpha][method]
            mask = prediction_mask(probs, q)
            set_col = f"conformal_set_{label}_{method}_{prefix}"
            size_col = f"set_size_{label}_{method}_{prefix}"
            single_col = f"is_singleton_{label}_{method}_{prefix}"
            df[set_col] = mask_to_strings(mask, names)
            sizes = mask.sum(axis=1).astype(np.int16)
            df[size_col] = sizes
            df[single_col] = sizes == 1
    return df


def summarize_dataset(
    dataset: str,
    scheme: str,
    probs: np.ndarray,
    thresholds: Dict[float, Dict[str, np.ndarray | float]],
    labels: np.ndarray | None = None,
    alphas: Iterable[float] = (0.10, 0.05),
) -> List[Dict[str, float | str | int]]:
    rows: List[Dict[str, float | str | int]] = []
    for alpha, _ in iter_alphas(alphas):
        for method in ["global", "class_conditional"]:
            mask = prediction_mask(probs, thresholds[alpha][method])
            sizes = mask.sum(axis=1)
            row: Dict[str, float | str | int] = {
                "dataset": dataset,
                "scheme": scheme,
                "method": method,
                "target_coverage": 1.0 - alpha,
                "n": int(probs.shape[0]),
                "avg_set_size": float(sizes.mean()),
                "singleton_rate": float((sizes == 1).mean()),
                "empty_set_rate": float((sizes == 0).mean()),
            }
            if labels is not None:
                covered = mask[np.arange(labels.size), labels]
                row["empirical_coverage"] = float(covered.mean())
            else:
                row["empirical_coverage"] = np.nan
            rows.append(row)
    return rows


def per_class_orion(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: Dict[float, Dict[str, np.ndarray | float]],
    names: List[str],
    scheme: str,
    alphas: Iterable[float] = (0.10, 0.05),
) -> List[Dict[str, float | str | int]]:
    rows: List[Dict[str, float | str | int]] = []
    for alpha, _ in iter_alphas(alphas):
        for method in ["global", "class_conditional"]:
            mask = prediction_mask(probs, thresholds[alpha][method])
            sizes = mask.sum(axis=1)
            covered = mask[np.arange(labels.size), labels]
            for class_id, name in enumerate(names):
                is_class = labels == class_id
                if not np.any(is_class):
                    continue
                rows.append(
                    {
                        "scheme": scheme,
                        "method": method,
                        "target_coverage": 1.0 - alpha,
                        "class": name,
                        "support": int(is_class.sum()),
                        "coverage": float(covered[is_class].mean()),
                        "avg_set_size": float(sizes[is_class].mean()),
                        "singleton_rate": float((sizes[is_class] == 1).mean()),
                        "empty_set_rate": float((sizes[is_class] == 0).mean()),
                    }
                )
    return rows


def merge5_probs(probs10: np.ndarray) -> np.ndarray:
    out = np.zeros((probs10.shape[0], len(MERGE5_NAMES)), dtype=np.float32)
    for src, dst in MERGE5_BY_CLASS.items():
        out[:, dst] += probs10[:, src]
    return out


def merge5_labels(labels10: np.ndarray) -> np.ndarray:
    mapper = np.array([MERGE5_BY_CLASS[i] for i in range(10)], dtype=np.int64)
    return mapper[labels10]


def external_probs(path: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(path)
    missing = [col for col in PROB_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing probability columns in {path}: {missing}")
    probs = df[PROB_COLS].to_numpy(dtype=np.float32)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    return df, probs


def compact_external_df(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "case",
        "cell_id",
        "cell_index",
        "patch_stem",
        "instance_id",
        "x_centroid",
        "y_centroid",
        "pannuke_label_name",
        "celltypist_predicted_labels",
        "celltypist_conf_score",
        "qc_pass",
        "pred_label_id",
        "pred_label_name",
    ]
    return df[[col for col in keep if col in df.columns]].copy()


def set_class_frequency(mask: np.ndarray, names: Iterable[str], dataset: str, scheme: str, method: str, target: float) -> List[Dict[str, float | str | int]]:
    n = mask.shape[0]
    rows = []
    for idx, name in enumerate(names):
        rows.append(
            {
                "dataset": dataset,
                "scheme": scheme,
                "method": method,
                "target_coverage": target,
                "class": name,
                "set_inclusion_fraction": float(mask[:, idx].mean()) if n else 0.0,
                "set_inclusion_count": int(mask[:, idx].sum()),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cal_logits = load_npz_array(args.calibration_dir / "list_of_logits.npz")
    cal_labels = load_npz_array(args.calibration_dir / "list_of_labels.npz").astype(np.int64)
    test_logits = load_npz_array(args.orion_test_dir / "list_of_logits.npz")
    test_labels = load_npz_array(args.orion_test_dir / "list_of_labels.npz").astype(np.int64)

    cal_probs10 = softmax(cal_logits)
    test_probs10 = softmax(test_logits)
    cal_probs5 = merge5_probs(cal_probs10)
    test_probs5 = merge5_probs(test_probs10)
    cal_labels5 = merge5_labels(cal_labels)
    test_labels5 = merge5_labels(test_labels)

    thresholds = {"10class": {}, "5class": {}}
    for alpha in args.alphas:
        thresholds["10class"][alpha] = fit_thresholds(cal_probs10, cal_labels, alpha, len(CLASS_NAMES))
        thresholds["5class"][alpha] = fit_thresholds(cal_probs5, cal_labels5, alpha, len(MERGE5_NAMES))

    summary_rows: List[Dict[str, float | str | int]] = []
    per_class_rows: List[Dict[str, float | str | int]] = []
    set_frequency_rows: List[Dict[str, float | str | int]] = []

    summary_rows.extend(summarize_dataset("ORION test7", "10class", test_probs10, thresholds["10class"], test_labels, args.alphas))
    summary_rows.extend(summarize_dataset("ORION test7", "5class", test_probs5, thresholds["5class"], test_labels5, args.alphas))
    per_class_rows.extend(per_class_orion(test_probs10, test_labels, thresholds["10class"], CLASS_NAMES, "10class", args.alphas))
    per_class_rows.extend(per_class_orion(test_probs5, test_labels5, thresholds["5class"], MERGE5_NAMES, "5class", args.alphas))

    pred10 = test_probs10.argmax(axis=1)
    orion_df = pd.DataFrame(
        {
            "row_index": np.arange(test_labels.size, dtype=np.int64),
            "true_label_id": test_labels,
            "true_label_name": [CLASS_NAMES[i] for i in test_labels],
            "pred_label_id": pred10,
            "pred_label_name": [CLASS_NAMES[i] for i in pred10],
            "pred_prob": test_probs10[np.arange(test_labels.size), pred10],
        }
    )
    orion_df = add_set_columns(orion_df, test_probs10, thresholds, CLASS_NAMES, "10class", args.alphas)
    orion_df = add_set_columns(orion_df, test_probs5, thresholds, MERGE5_NAMES, "5class", args.alphas)
    orion_df.to_parquet(args.output_dir / "orion_test7_conformal_cell_predictions.parquet", index=False)

    # for dataset, parquet_path, output_name in [
    #     ("PanNuke", args.pannuke_parquet, "pannuke_conformal_cell_predictions.parquet"),
    #     ("Xenium", args.xenium_parquet, "xenium_conformal_cell_predictions.parquet"),
    # ]:
    #     ext_df, ext_probs10 = external_probs(parquet_path)
    #     ext_probs5 = merge5_probs(ext_probs10)
    #     summary_rows.extend(summarize_dataset(dataset, "10class", ext_probs10, thresholds["10class"], None, args.alphas))
    #     summary_rows.extend(summarize_dataset(dataset, "5class", ext_probs5, thresholds["5class"], None, args.alphas))
    #     ext_out = compact_external_df(ext_df)
    #     pred = ext_probs10.argmax(axis=1)
    #     ext_out["pred_label_name_from_probs"] = [CLASS_NAMES[i] for i in pred]
    #     ext_out["pred_prob"] = ext_probs10[np.arange(ext_probs10.shape[0]), pred]
    #     ext_out = add_set_columns(ext_out, ext_probs10, thresholds, CLASS_NAMES, "10class", args.alphas)
    #     ext_out = add_set_columns(ext_out, ext_probs5, thresholds, MERGE5_NAMES, "5class", args.alphas)
    #     ext_out.to_parquet(args.output_dir / output_name, index=False)

    #     for alpha, _ in iter_alphas(args.alphas):
    #         for method in ["global", "class_conditional"]:
    #             mask10 = prediction_mask(ext_probs10, thresholds["10class"][alpha][method])
    #             mask5 = prediction_mask(ext_probs5, thresholds["5class"][alpha][method])
    #             set_frequency_rows.extend(set_class_frequency(mask10, CLASS_NAMES, dataset, "10class", method, 1.0 - alpha))
    #             set_frequency_rows.extend(set_class_frequency(mask5, MERGE5_NAMES, dataset, "5class", method, 1.0 - alpha))

    summary_df = pd.DataFrame(summary_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    # set_freq_df = pd.DataFrame(set_frequency_rows)
    summary_df.to_csv(args.output_dir / "conformal_summary_metrics.csv", index=False)
    per_class_df.to_csv(args.output_dir / "conformal_per_class_orion.csv", index=False)
    # set_freq_df.to_csv(args.output_dir / "conformal_external_set_class_frequency.csv", index=False)

    thresholds_json = {}
    for scheme, by_alpha in thresholds.items():
        thresholds_json[scheme] = {}
        for alpha, vals in by_alpha.items():
            thresholds_json[scheme][str(alpha)] = {
                "global": float(vals["global"]),
                "class_conditional": [float(x) for x in vals["class_conditional"]],
            }
    (args.output_dir / "conformal_thresholds.json").write_text(json.dumps(thresholds_json, indent=2))

    make_summary_figure(summary_df, per_class_df, args.output_dir)
    print(f"[DONE] wrote conformal outputs to {args.output_dir}")


def make_summary_figure(summary_df: pd.DataFrame, per_class_df: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    main = summary_df[(summary_df["method"] == "class_conditional") & (summary_df["target_coverage"] == 0.9)]
    sns.barplot(data=main, x="avg_set_size", y="dataset", hue="scheme", ax=axes[0, 0], palette="Set2")
    axes[0, 0].set_title("Average conformal set size at 90% target coverage")
    axes[0, 0].set_xlabel("Average set size")
    axes[0, 0].set_ylabel("")

    orion = summary_df[summary_df["dataset"].eq("ORION test7")]
    sns.barplot(data=orion, x="target_coverage", y="empirical_coverage", hue="scheme", ax=axes[0, 1], palette="Set2")
    axes[0, 1].plot([0.85, 1.0], [0.85, 1.0], color="0.35", linestyle="--", linewidth=1)
    axes[0, 1].set_title("ORION empirical coverage")
    axes[0, 1].set_ylim(0.80, 1.0)
    axes[0, 1].set_xlabel("Target coverage")
    axes[0, 1].set_ylabel("Empirical coverage")

    per = per_class_df[
        per_class_df["scheme"].eq("10class")
        & per_class_df["method"].eq("class_conditional")
        & per_class_df["target_coverage"].eq(0.9)
    ]
    sns.barplot(data=per, x="coverage", y="class", ax=axes[1, 0], color="#7aa6c2")
    axes[1, 0].axvline(0.9, color="0.35", linestyle="--", linewidth=1)
    axes[1, 0].set_title("ORION 10-class per-class coverage, class-conditional 90%")
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_xlabel("Coverage")
    axes[1, 0].set_ylabel("")

    sns.barplot(data=main, x="singleton_rate", y="dataset", hue="scheme", ax=axes[1, 1], palette="Set2")
    axes[1, 1].set_title("Singleton prediction rate at 90% target coverage")
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_xlabel("Singleton rate")
    axes[1, 1].set_ylabel("")

    fig.suptitle("Conformal prediction summary using ORION calibration cohort", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "conformal_prediction_summary.png", dpi=220)
    fig.savefig(output_dir / "conformal_prediction_summary.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
