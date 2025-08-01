# CellSighter

CellSighter is a deep learning project for automated tumor cell classification from H&E stained histopathology images using multiplex immunofluorescence (mIF) microscopy annotations as ground truth.

## Project Overview

This project implements an advanced pipeline for tumor cell classification that combines:

1. **Feature Extraction**: Utilizes a pretrained vision foundation model to extract patch-wise features from H&E stained histopathology images.

2. **Spatial Context Modeling**: Employs Graph Neural Networks (GNN) to:
   - Structure and aggregate patch embeddings
   - Model cell-cell interactions
   - Capture region-level relationships across tissue samples

3. **Representation Learning**: Implements contrastive learning techniques to:
   - Refine latent space representations
   - Enhance separation between different cell types
   - Improve discriminative power across tumor phenotypes

## Project Structure

```
CellSighter/
├── datasets/
│   └── CellTypes/
│       ├── data/
│       │   └── images/     # Raw images (HxWxC, npz/tiff format)
│       ├── cells/         # Cell segmentation masks (HxW, npz/tiff format)
│       └── cells2labels/  # Cell labels (npz/txt format)
├── src/            # Source code
├── notebooks/      # Jupyter notebooks for experiments
├── configs/        # Configuration files
└── results/        # Model outputs and evaluations
```

### Data Structure Details

The `datasets/CellTypes/` directory follows a specific structure for organizing image data, segmentation masks, and labels:

1. **Raw Images** (`data/images/`):
   - Format: `.npz` or `.tiff`
   - Shape: Height × Width × Channels (HxWxC)
   - C represents the number of protein channels

2. **Cell Segmentation** (`cells/`):
   - Format: `.npz` or `.tiff`
   - Shape: Height × Width (HxW)
   - Contains labeled object matrix where each cell has a unique ID (1 to N)

3. **Cell Labels** (`cells2labels/`):
   - Format: `.npz` or `.txt`
   - Each row corresponds to a cell ID
   - Unlabeled cells (e.g., test data) are marked as -1

## Getting Started

[Coming soon]

## License

[Coming soon]