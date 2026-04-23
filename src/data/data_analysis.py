# hobbit long pytorch supcontrastive trainer
import json
import os
import sys
import argparse
import time
import math
import csv

# import tensorboard_logger as tb_logger
import torch
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, average_precision_score, precision_recall_curve, auc
from matplotlib import pyplot as plt
from PIL import Image


LABEL_PATH = "/taiga/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/con_classifier_testing_checkpoints_orion_topk/list_of_labels.npz"
LOGITS_PATH = "/taiga/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/con_classifier_testing_checkpoints_orion_topk/list_of_logits.npz"


def pr_auc_score(
    y_true,
    y_score,
    *,
    labels=None,
    average="macro",   # None, "macro", "weighted", "micro"
    method="ap",       # "ap" (Average Precision) or "trapezoid"
    return_per_class=False,
):
    """
    Precision-Recall AUC for binary/multiclass one-vs-rest.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True class labels.
    y_score : array-like, shape (n_samples, n_classes)
        Predicted scores/probabilities for each class.
    labels : array-like, optional
        Class order matching columns of y_score.
    average : {None, "macro", "weighted", "micro"}, default="macro"
    method : {"ap", "trapezoid"}, default="ap"
        "ap" uses average_precision_score (recommended by sklearn for PR).
        "trapezoid" uses auc(recall, precision).
    return_per_class : bool, default=False
        If True, returns (overall, {label: score}).

    Returns
    -------
    score : float or np.ndarray
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_score.ndim == 1:
        y_score = y_score.reshape(-1, 1)

    if labels is None:
        labels = np.unique(y_true)
    labels = np.asarray(labels)

    if y_score.shape[1] != len(labels):
        raise ValueError(
            f"y_score has {y_score.shape[1]} columns but labels has {len(labels)} classes."
        )

    y_true_bin = label_binarize(y_true, classes=labels)
    # label_binarize returns (n_samples, 1) for binary; make it 2-column OVR
    if y_true_bin.shape[1] == 1 and len(labels) == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    per_class = []
    for i in range(len(labels)):
        yt = y_true_bin[:, i]
        ys = y_score[:, i]

        if method == "ap":
            s = average_precision_score(yt, ys)
        elif method == "trapezoid":
            p, r, _ = precision_recall_curve(yt, ys)
            s = auc(r, p)
        else:
            raise ValueError("method must be 'ap' or 'trapezoid'")
        per_class.append(s)

    per_class = np.asarray(per_class, dtype=float)

    if average is None:
        overall = per_class
    elif average == "macro":
        overall = np.nanmean(per_class)
    elif average == "weighted":
        supports = y_true_bin.sum(axis=0)
        overall = np.average(per_class, weights=supports)
    elif average == "micro":
        if method == "ap":
            overall = average_precision_score(y_true_bin.ravel(), y_score.ravel())
        else:
            p, r, _ = precision_recall_curve(y_true_bin.ravel(), y_score.ravel())
            overall = auc(r, p)
    else:
        raise ValueError("average must be None, 'macro', 'weighted', or 'micro'")

    if return_per_class:
        return overall, dict(zip(labels.tolist(), per_class.tolist()))
    return overall


def simple_convert(array): # np.ndarray to np.ndarray
    
    lut = np.array([1, 1, 1, 2, 3, 4, 4, 0, 4, 3], dtype=np.uint8)
    
    return lut[array]


def probabilitiy_convert(array):
    
    lut = np.array([1, 1, 1, 2, 3, 4, 4, 0, 4, 3], dtype=np.uint8)
    
    new_probs_5class = np.zeros((array.shape[0], 5), dtype=array.dtype)
    
    for old_cls, new_cls in enumerate(lut):
        new_probs_5class[:, new_cls] += array[:, old_cls]
    
    return new_probs_5class


def evaluate(logits, labels, num_classes, simpler_labels=True):
    
    # convert labels to simpler labels and predictions to simpler version
    logits_tensor = torch.from_numpy(logits)
    labels_tensor = torch.from_numpy(labels)
    
    probs = torch.softmax(logits_tensor, dim=1)
    preds = torch.argmax(probs, dim=1)
    
    all_probs = probs.numpy()
    all_labels = labels_tensor.numpy()
    all_preds = preds.numpy()
    
    if simpler_labels:
        all_orig_labels = all_labels.copy()
        all_labels = simple_convert(all_orig_labels)
        all_orig_preds = all_preds.copy()
        all_preds = simple_convert(all_orig_preds)

    accuracy = accuracy_score(all_labels, all_preds)

    if num_classes <= 2:
        average_method = 'binary'
    else:
        average_method = 'weighted'

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )

    # Overall metrics
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=average_method, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # AUC
    try:
        if num_classes <= 2:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            # Fix the multi aucs
            if simpler_labels:
                # all_labels = all_orig_labels
                all_orig_probs = all_probs.copy()
                all_probs = probabilitiy_convert(all_probs)
                
            all_probs = np.vstack(all_probs)
            
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovo', average='weighted')
            multi_aucs = roc_auc_score(all_labels, all_probs, multi_class='ovr', average=None)
            
            pr_auc = pr_auc_score(all_labels, all_probs, average='weighted')
            multi_pr_aucs = pr_auc_score(all_labels, all_probs, average=None)
    except ValueError:
        auc = 0.0  # Handle case where only one class is present
        if num_classes > 2:
            multi_aucs = []

    results = {
        'accuracy': accuracy,
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1.tolist(),
        'support_per_class': support.tolist(),
        'precision_avg': precision_avg,
        'recall_avg': recall_avg,
        'f1_avg': f1_avg,
        'auc': auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm.tolist(),
    }

    if num_classes > 2:
        if isinstance(multi_aucs, list):
            multi_aucs_list = multi_aucs
            multi_pr_aucs_list = multi_pr_aucs
        else:
            multi_aucs_list = multi_aucs.tolist()
            multi_pr_aucs_list = multi_pr_aucs.tolist()
        results.update({'multi_class_aucs': multi_aucs_list, 'multi_class_pr_aucs': multi_pr_aucs_list})
        
    # save results to json
    with open("/taiga/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/con_classifier_testing_checkpoints_orion_topk/evaluation_results_simpler.json", "w") as f:
        json.dump(results, f, indent=4)
        
        
# null baselines
# Compute per-class null baselines from either:
# 1) labels (y_true), or
# 2) class counts

def baselines_from_labels(y_true, class_order=None):
    """
    y_true: iterable of class labels (e.g., [0,0,1,2,2,...] or ["Tumor", ...])
    class_order: optional list to force output order
    """
    counts = Counter(y_true)
    return baselines_from_counts(counts, class_order=class_order)


def baselines_from_counts(counts, class_order=None):
    """
    counts: dict-like {class_label: n_samples}
    """
    if class_order is None:
        class_order = list(counts.keys())

    total = sum(counts[c] for c in class_order)
    if total <= 0:
        raise ValueError("Total count must be > 0.")

    rows = []
    K = len(class_order)

    for c in class_order:
        n_c = counts[c]
        p_c = n_c / total  # prevalence

        # Null baselines
        roc_auc_null = 0.5               # random ranking
        pr_auc_null = p_c                # one-vs-rest prevalence baseline

        # Optional F1 null baselines for reference:
        f1_null_prev_matched = p_c       # q_c = p_c
        f1_null_uniform = (2 * p_c) / (K * p_c + 1)  # q_c = 1/K

        rows.append({
            "class": c,
            "count": n_c,
            "prevalence_p": p_c,
            "ROC_AUC_null": roc_auc_null,
            "PR_AUC_null": pr_auc_null,
            "F1_null_prev_matched": f1_null_prev_matched,
            "F1_null_uniform": f1_null_uniform,
        })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    # ----------------------------
    # OPTION A: Enter class counts
    # ----------------------------
    counts = {
        "Tumor Cells": 601675,
        "T / Treg cells": 658525,
        "B cells": 107442,
        "Myeloid cells": 243968,
        "Stromal / Vascular / Smooth Muscle": 1016519
    }

    df = baselines_from_counts(counts, class_order=list(counts.keys()))
    print("\nBaselines from counts:\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ----------------------------
    # OPTION B: Enter raw labels
    # ----------------------------
    # y_true = ["Tumor", "Tumor", "Stroma", "Lymphocyte", ...]
    # df = baselines_from_labels(y_true, class_order=["Tumor","Lymphocyte","Stroma","Macrophage","Endothelial","Necrosis"])
    # print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Optional: save for plotting pipelines
    # df.to_csv("null_baselines.csv", index=False)
 
    # logits_list = np.load(LOGITS_PATH, allow_pickle=True)["arr_0"]
    # labels_list = np.load(LABEL_PATH, allow_pickle=True)["arr_0"]

    # evaluate(logits_list, labels_list, num_classes=10, simpler_labels=True)
        