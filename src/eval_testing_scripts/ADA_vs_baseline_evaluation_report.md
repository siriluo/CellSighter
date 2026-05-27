# ADA vs. Baseline Evaluation Report

## Inputs

- `evaluation_results_ADA.json`: model evaluated with domain adaptation applied for the external PanNuke dataset.
- `evaluation_results_ADA_baseline.json`: baseline model evaluated without domain adaptation.
- Datasets:
  - `pannuke_fold`: external PanNuke test fold with 11,160 samples and 3 broad tissue/cell categories.
  - `orion_test`: original Orion test set with 2,628,129 samples and 10 Orion cell classes.

## Executive Summary

Domain adaptation improves performance on the external PanNuke dataset, but reduces performance on the original Orion dataset. On PanNuke, ADA increases top-1 accuracy from 36.14% to 38.34%, average F1 from 27.16% to 29.96%, ROC-AUC from 57.67% to 61.34%, and PR-AUC from 39.84% to 44.45%. The largest class-level PanNuke gain is for Connective/Stromal cells, whose F1 improves from 39.70% to 45.34%.

On Orion, the baseline remains stronger. Without ADA, Orion accuracy is 52.08% compared with 46.92% after ADA. Average F1 drops from 49.21% to 43.25%, ROC-AUC from 85.31% to 81.64%, and PR-AUC from 53.47% to 48.75%. The ADA model improves Stromal Cells on Orion, but most other classes decline, especially B cells, Monocytes / Macrophages, Smooth Muscle, Vasculature, and Treg.

Overall, ADA appears to trade original-domain performance for better external-domain generalization. This is expected if the adaptation procedure shifts the learned representation toward PanNuke-like morphology or staining distribution. The results support ADA as useful for cross-domain deployment, but not as a universal replacement for the original Orion model unless the target use case prioritizes PanNuke-like external data.

## Overall Metrics

### PanNuke External Dataset

| Metric | Baseline | ADA | Delta |
| --- | ---: | ---: | ---: |
| Accuracy / Top-1 | 36.14% | 38.34% | +2.20 pp |
| Avg. precision | 39.73% | 48.09% | +8.36 pp |
| Avg. recall | 36.14% | 38.34% | +2.20 pp |
| Avg. F1 | 27.16% | 29.96% | +2.81 pp |
| ROC-AUC | 57.67% | 61.34% | +3.67 pp |
| PR-AUC | 39.84% | 44.45% | +4.62 pp |

ADA improves every reported aggregate PanNuke metric. The strongest aggregate improvement is average precision, which increases by 8.36 percentage points. PR-AUC also improves meaningfully, suggesting ADA improves ranking quality under class imbalance, not just hard-label accuracy.

### Orion Original Dataset

| Metric | Baseline | ADA | Delta |
| --- | ---: | ---: | ---: |
| Accuracy / Top-1 | 52.08% | 46.92% | -5.17 pp |
| Top-3 accuracy | 82.67% | 78.03% | -4.64 pp |
| Top-5 accuracy | 92.88% | 89.90% | -2.98 pp |
| Avg. precision | 50.87% | 47.43% | -3.44 pp |
| Avg. recall | 52.08% | 46.92% | -5.17 pp |
| Avg. F1 | 49.21% | 43.25% | -5.96 pp |
| ROC-AUC | 85.31% | 81.64% | -3.67 pp |
| PR-AUC | 53.47% | 48.75% | -4.73 pp |

The Orion results show a consistent decline after ADA. The drop affects both classification accuracy and probability-ranking metrics. Top-5 accuracy remains high after ADA, but it still falls by almost 3 percentage points, indicating that the correct Orion class is less reliably retained among the model's highest-confidence predictions.

## Class-Level Findings

### PanNuke External Dataset

| Class | Baseline F1 | ADA F1 | Delta | Main Interpretation |
| --- | ---: | ---: | ---: | --- |
| Tumor/Epithelial | 0.72% | 4.06% | +3.34 pp | ADA improves detection, but recall remains very low. |
| Immune/Inflammatory | 50.13% | 48.95% | -1.17 pp | Slight decline; still the strongest PanNuke class by recall. |
| Connective/Stromal | 39.70% | 45.34% | +5.64 pp | Largest PanNuke F1 gain; both precision and recall improve. |

The PanNuke confusion matrices show that both models struggle heavily with Tumor/Epithelial recall. The baseline correctly identifies only 16 of 4,433 Tumor/Epithelial samples, while ADA identifies 93. This is a relative improvement, but the absolute recall remains only 2.10%, so Tumor/Epithelial remains the major failure mode.

For Connective/Stromal, ADA increases true positives from 1,741 to 1,961 and reduces misclassification into Immune/Inflammatory from 1,818 to 1,582, although misclassification into Tumor/Epithelial increases slightly from 17 to 33. The Immune/Inflammatory class remains recall-heavy in both settings, with a modest ADA decline from 72.23% to 70.61% recall.

### Orion Original Dataset

| Class | Baseline F1 | ADA F1 | Delta | Main Interpretation |
| --- | ---: | ---: | ---: | --- |
| CD4+ T | 0.09% | 0.01% | -0.08 pp | Both models nearly fail to recover this class. |
| CD8+ T | 10.52% | 9.80% | -0.73 pp | Small decline; recall remains low. |
| Treg | 54.79% | 48.99% | -5.80 pp | ADA raises recall but lowers precision enough to reduce F1. |
| B cells | 18.77% | 4.88% | -13.89 pp | Major ADA degradation, driven by recall collapse. |
| Monocytes / Macrophages | 29.97% | 18.31% | -11.66 pp | Large decline in recall and F1. |
| Stromal Cells | 41.00% | 46.67% | +5.68 pp | Main Orion class that benefits from ADA. |
| Smooth Muscle | 41.30% | 31.57% | -9.74 pp | Precision increases, but recall drops sharply. |
| Tumor Cells | 84.03% | 79.80% | -4.23 pp | Still the strongest Orion class, but ADA reduces performance. |
| Vasculature | 49.11% | 41.04% | -8.07 pp | Large decline across precision and recall. |
| Granulocytes | 0.03% | 0.00% | -0.03 pp | Essentially not recovered by either model. |

The Orion baseline is substantially better for most classes. ADA improves only Stromal Cells, increasing recall from 34.44% to 43.05% and F1 from 41.00% to 46.67%. This improvement may align with the PanNuke Connective/Stromal gain, suggesting ADA encourages stromal-like feature recognition across domains.

However, that benefit comes with broad degradation elsewhere. B cells drop from 18.77% to 4.88% F1, Monocytes / Macrophages from 29.97% to 18.31%, Smooth Muscle from 41.30% to 31.57%, and Vasculature from 49.11% to 41.04%. CD4+ T and Granulocytes remain near-zero F1 for both models, so neither result meaningfully solves the rare or difficult Orion classes.

## Interpretation

The results indicate that domain adaptation is helping the model adapt to external PanNuke data, especially for stromal/connective morphology and overall ranking metrics. This supports the core purpose of ADA: improving external-domain generalization when the test data distribution differs from Orion.

At the same time, the Orion decline shows a clear cost. ADA likely shifts the decision boundary away from the original Orion distribution. This produces better PanNuke generalization but weaker retention of Orion-specific cell-type discrimination. The cost is not limited to rare classes; several high-support Orion classes also decline, including Treg, Smooth Muscle, Tumor Cells, and Vasculature.

One important nuance is that PanNuke has only three broad classes, while Orion has ten more specific classes. The PanNuke improvement may therefore reflect better coarse-grained adaptation, while the Orion decline reflects loss of fine-grained original-domain specialization. The most consistent cross-dataset gain is stromal/connective recognition.

## Recommended Summary

Domain adaptation improved external PanNuke performance but reduced original Orion performance. On PanNuke, ADA increased accuracy by 2.20 percentage points, average F1 by 2.81 points, ROC-AUC by 3.67 points, and PR-AUC by 4.62 points. The most notable PanNuke gain was Connective/Stromal F1, which rose from 39.70% to 45.34%.

On Orion, the non-adapted baseline performed better overall. ADA reduced Orion accuracy by 5.17 percentage points, average F1 by 5.96 points, ROC-AUC by 3.67 points, and PR-AUC by 4.73 points. ADA improved Orion Stromal Cells, but most other classes declined, especially B cells, Monocytes / Macrophages, Smooth Muscle, and Vasculature.

These results suggest ADA is beneficial when the priority is external-domain robustness on PanNuke-like data, but the baseline is preferable when maintaining original Orion-domain performance is the primary goal.

## Practical Recommendation

Use the ADA model for external PanNuke-style evaluation or deployment where domain shift is expected. Use the baseline model for Orion-native analysis. If a single model is needed for both settings, consider a mixed-domain fine-tuning strategy, domain-balanced validation, or model selection based on a weighted objective that explicitly includes both PanNuke generalization and Orion retention.
