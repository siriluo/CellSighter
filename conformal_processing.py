import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

CLASS_NAMES = [
    "CD4+ T", "CD8+ T", "Treg", "B cells", "Monocytes / Macrophages",
    "Stromal Cells", "Smooth Muscle", "Tumor Cells", "Vasculature", "Granulocytes",
]
name_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}

df = pd.read_parquet(
    "/taiga/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/conformal_prediction_outputs_fold4/orion_test7_conformal_cell_predictions.parquet"
)

def singleton_metrics(df, set_col, prob_threshold=None):
    work = df.copy()

    # Keep only singleton conformal predictions.
    is_singleton = work[set_col].fillna("").str.contains("|", regex=False) == False
    is_nonempty = work[set_col].fillna("") != ""
    keep = is_singleton & is_nonempty

    # Optional extra confidence threshold using top-1 softmax probability.
    if prob_threshold is not None:
        keep &= work["pred_prob"] >= prob_threshold

    kept = work.loc[keep].copy()

    if len(kept) == 0:
        return {
            "n_total": len(work),
            "n_kept": 0,
            "retention": 0.0,
            "accuracy": np.nan,
            "precision_macro": np.nan,
            "recall_macro": np.nan,
            "f1_macro": np.nan,
        }

    # Use the singleton class itself as the prediction.
    y_true = kept["true_label_id"].to_numpy()
    y_pred = kept[set_col].map(name_to_id).to_numpy()

    return {
        "n_total": len(work),
        "n_kept": len(kept),
        "retention": len(kept) / len(work),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    
    
rows = []

set_cols = [
    "conformal_set_85_class_conditional_10class",
    "conformal_set_90_class_conditional_10class",
    "conformal_set_95_class_conditional_10class",
]

prob_thresholds = [None, 0.5, 0.7, 0.8, 0.9, 0.95]

for set_col in set_cols:
    for thr in prob_thresholds:
        metrics = singleton_metrics(df, set_col, prob_threshold=thr)
        metrics["set_col"] = set_col
        metrics["prob_threshold"] = thr if thr is not None else "none"
        rows.append(metrics)

singleton_results = pd.DataFrame(rows)
print(singleton_results)
singleton_results.to_csv("singleton_threshold_metrics.csv", index=False)

#                          precision    recall  f1-score   support

#                  CD4+ T       0.08      0.00      0.01    109895
#                  CD8+ T       0.32      0.07      0.11    104536
#                    Treg       0.42      0.75      0.54    444094
#                 B cells       0.31      0.07      0.12    107442
# Monocytes / Macrophages       0.32      0.42      0.37    191583
#           Stromal Cells       0.48      0.37      0.41    154471
#           Smooth Muscle       0.47      0.39      0.43    395724
#             Tumor Cells       0.86      0.88      0.87    601675
#             Vasculature       0.48      0.50      0.49    466324
#            Granulocytes       0.00      0.00      0.00     52385

#                accuracy                           0.53   2628129
#               macro avg       0.38      0.34      0.33   2628129
#            weighted avg       0.51      0.53      0.50   2628129