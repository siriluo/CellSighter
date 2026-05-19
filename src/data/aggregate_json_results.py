#!/usr/bin/env python3
"""
Aggregate fold/evaluation metrics stored in JSON files.

Examples:
  python scripts/aggregate_json_results.py \
      --glob "results_*/*eval*.json" \
      --out-json aggregated_summary.json \
      --out-csv aggregated_summary.csv

  python scripts/aggregate_json_results.py \
      --files fold1.json fold2.json fold3.json \
      --metrics accuracy macro_f1 auc \
      --weight-key n_samples

  python src/data/aggregate_json_results.py \
      --glob "/taiga/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/experiment_results/evaluation_results/orion_fold_testing_results/*evaluation_results.json" \
      --keys accuracy f1_avg auc pr_auc multi_class_aucs multi_class_pr_aucs \
      --expand-list-keys multi_class_aucs multi_class_pr_aucs
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
from typing import Dict, Iterable, List, Tuple, Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate numeric metrics from multiple JSON files."
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=[],
        help="Explicit JSON files to aggregate.",
    )
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        default=None,
        help='Glob pattern for JSON files, e.g. "results_*/*metrics*.json".',
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Optional metric names to include. If omitted, aggregate all numeric metrics found.",
    )
    parser.add_argument(
        "--keys",
        nargs="*",
        default=None,
        help=(
            "Optional keys to include (top-level or dot paths). "
            "If set, only these keys are considered."
        ),
    )
    parser.add_argument(
        "--expand-list-keys",
        nargs="*",
        default=None,
        help=(
            "Keys whose numeric list values should be expanded and aggregated per element "
            "(e.g., multi_class_aucs -> multi_class_aucs[0], ...)."
        ),
    )
    parser.add_argument(
        "--class-names-key",
        default=None,
        help=(
            "Optional key containing class names (list of strings). If provided and lengths match, "
            "expanded list metrics will use names like metric[class_name] instead of metric[idx]."
        ),
    )
    parser.add_argument(
        "--weight-key",
        default=None,
        help="Optional key used as sample weight (e.g., n_samples, num_examples).",
    )
    parser.add_argument(
        "--out-json",
        default="aggregated_metrics.json",
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--out-csv",
        default="aggregated_metrics.csv",
        help="Output CSV summary path.",
    )
    return parser.parse_args()


def is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def flatten_numeric(d: dict, parent_key: str = "") -> Dict[str, float]:
    """Flatten nested dict and keep only numeric leaves."""
    out: Dict[str, float] = {}
    for k, v in d.items():
        key = f"{parent_key}.{k}" if parent_key else str(k)
        if isinstance(v, dict):
            out.update(flatten_numeric(v, key))
        elif is_number(v):
            out[key] = float(v)
    return out


def get_by_path(d: Dict[str, Any], path: str):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def build_metrics_from_json(
    data: Dict[str, Any],
    keys: List[str] | None = None,
    expand_list_keys: List[str] | None = None,
    class_names_key: str | None = None,
) -> Dict[str, float]:
    """
    Build a metric dict from JSON with optional key filtering and per-list expansion.
    """
    metrics: Dict[str, float] = {}

    class_names = None
    if class_names_key:
        class_names_val = get_by_path(data, class_names_key)
        if isinstance(class_names_val, list) and all(isinstance(x, str) for x in class_names_val):
            class_names = class_names_val

    expand_list_keys = expand_list_keys or []
    keys_set = set(keys) if keys else None

    # Start from all numeric scalar leaves.
    flat_all = flatten_numeric(data)
    for k, v in flat_all.items():
        if keys_set is None or k in keys_set:
            metrics[k] = v

    # Expand selected list keys into per-index/per-class metrics.
    for list_key in expand_list_keys:
        list_val = get_by_path(data, list_key)
        if not isinstance(list_val, list):
            continue

        for idx, item in enumerate(list_val):
            if not is_number(item):
                continue
            if class_names is not None and len(class_names) == len(list_val):
                metric_name = f"{list_key}[{class_names[idx]}]"
            else:
                metric_name = f"{list_key}[{idx}]"

            if keys_set is None or list_key in keys_set or metric_name in keys_set:
                metrics[metric_name] = float(item)

    return metrics


def weighted_mean(values: List[float], weights: List[float]) -> float:
    denom = sum(weights)
    if denom == 0:
        return float("nan")
    return sum(v * w for v, w in zip(values, weights)) / denom


def summarize(values: List[float], weights: List[float] | None = None) -> Dict[str, float]:
    n = len(values)
    if n == 0:
        return {}

    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else 0.0
    ci95 = 1.96 * sem

    result = {
        "count": n,
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "ci95_low": mean - ci95,
        "ci95_high": mean + ci95,
    }
    if weights is not None and len(weights) == n:
        result["weighted_mean"] = weighted_mean(values, weights)
    return result


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_files(explicit_files: Iterable[str], glob_pattern: str | None) -> List[str]:
    files = list(explicit_files)
    if glob_pattern:
        files.extend(glob.glob(glob_pattern, recursive=True))
    files = sorted(set(files))
    return [f for f in files if os.path.isfile(f)]


def write_csv(path: str, summary: Dict[str, Dict[str, float]]) -> None:
    headers = [
        "metric",
        "count",
        "mean",
        "std",
        "min",
        "max",
        "ci95_low",
        "ci95_high",
        "weighted_mean",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for metric in sorted(summary.keys()):
            row = {"metric": metric}
            row.update(summary[metric])
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    files = resolve_files(args.files, args.glob_pattern)
    if not files:
        raise FileNotFoundError("No JSON files found. Provide --files and/or --glob.")

    per_file_metrics: List[Tuple[str, Dict[str, float]]] = []
    per_file_weights: List[Tuple[str, float | None]] = []

    for path in files:
        data = load_json(path)
        flat = build_metrics_from_json(
            data=data,
            keys=args.keys,
            expand_list_keys=args.expand_list_keys,
            class_names_key=args.class_names_key,
        )
        per_file_metrics.append((path, flat))
        weight = float(flat[args.weight_key]) if args.weight_key and args.weight_key in flat else None
        per_file_weights.append((path, weight))

    all_metric_names = sorted({k for _, metrics in per_file_metrics for k in metrics.keys()})
    if args.metrics:
        selected = set(args.metrics)
        metric_names = [m for m in all_metric_names if m in selected]
    else:
        metric_names = all_metric_names

    summary: Dict[str, Dict[str, float]] = {}
    per_metric_values: Dict[str, Dict[str, float]] = {}

    for metric in metric_names:
        values = []
        weights = []
        file_value_map = {}
        for (path, metrics), (_, weight) in zip(per_file_metrics, per_file_weights):
            if metric in metrics:
                v = metrics[metric]
                values.append(v)
                file_value_map[path] = v
                if weight is not None:
                    weights.append(weight)

        if not values:
            continue

        use_weights = len(weights) == len(values) and len(weights) > 0
        summary[metric] = summarize(values, weights if use_weights else None)
        per_metric_values[metric] = file_value_map

    output = {
        "files": files,
        "summary": summary,
        "per_metric_per_file": per_metric_values,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    write_csv(args.out_csv, summary)

    print(f"Aggregated {len(files)} files")
    print(f"Wrote JSON summary: {args.out_json}")
    print(f"Wrote CSV summary:  {args.out_csv}")
    print(f"Metrics aggregated: {len(summary)}")


if __name__ == "__main__":
    main()
