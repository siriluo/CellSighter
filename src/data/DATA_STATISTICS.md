# CellSighter Dataset Statistics

## Overview

This document provides detailed statistics and analysis of the CellSighter dataset used for tumor cell classification.

## Dataset Summary

### Overall Numbers
- Total cells: 239,792
- Training set: 233,854 (97.5%)
- Validation set: 5,938 (2.5%)

### Class Distribution
Training Set:
- Normal cells (label=0): 190,383 (81.4%)
- Tumor cells (label≠0): 43,471 (18.6%)

Validation Set:
- Normal cells (label=0): 4,427 (74.6%)
- Tumor cells (label≠0): 1,511 (25.4%)

## Detailed Data Distribution

```
Distribution of Tumor Cell Percentages:

0-10%   |██████████████████████ (40 images)
10-20%  |███████████ (22 images)
20-30%  |████████ (16 images)
30-40%  |████ (8 images)
40-50%  |████ (8 images)
50-60%  |██ (4 images)
60-70%  |██ (4 images)
70-80%  |█ (2 images)
80-90%  |██ (4 images)
```

## Notable Cases

High Tumor Concentration:
```
reg058_B: 86.5% (1371/1585 cells)
reg058_A: 83.8% (1865/2225 cells)
reg002_B: 83.5% (2160/2586 cells)
reg042_A: 72.2% (1797/2490 cells)
reg043_B: 68.2% (393/576 cells)
```

Low Tumor Concentration:
```
reg047_B: 0.2% (2/951 cells)
reg038_A: 0.7% (17/2535 cells)
reg050_A: 0.7% (7/1065 cells)
reg019_B: 0.9% (1/107 cells)
```

Cell Count Range:
```
Largest:  reg055_A: 3,731 cells (242 tumor)
Smallest: reg057_A: 80 cells (19 tumor)
```

## Regional Patterns (A vs B Regions)

Region A:
```
Average cells per image: 1,950
Tumor percentage range: 0.7% - 83.8%
Median tumor percentage: 11.5%
```

Region B:
```
Average cells per image: 1,680
Tumor percentage range: 0.2% - 86.5%
Median tumor percentage: 13.9%
```

## Key Observations

1. **Distribution Characteristics**:
   - Most images (62 out of 128) have less than 20% tumor cells
   - Only 10 images have more than 50% tumor cells
   - Extreme cases (>80% tumor) are rare but present

2. **Size Variations**:
   - Cell count varies significantly (80 to 3,731 cells)
   - Most images contain 1,500-2,500 cells
   - Median: ~2,000 cells per image

3. **Regional Balance**:
   - A and B regions show similar tumor distributions
   - B regions tend to have slightly higher tumor percentages
   - Cell density is generally higher in A regions

4. **Implications for Training**:
   - Strong class imbalance needs addressing
   - Wide range of tumor percentages provides diverse training scenarios
   - Consider weighted sampling based on tumor percentage

## Complete Image-Level Statistics

[See test_outputs/dataset_statistics.txt for complete per-image statistics]

## Visualizations

The following visualizations are available in the test_outputs directory:
- class_distribution.png: Pie charts showing class distribution
- balanced_cell_samples.png: Example images from both classes