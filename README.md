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

## Technical Support

This project is developed in collaboration with the National Center for Supercomputing Applications (NCSA) to accelerate model prototyping and implementation.

## Project Structure

```
CellSighter/
├── data/           # Dataset storage (not tracked in git)
├── src/            # Source code
├── notebooks/      # Jupyter notebooks for experiments
├── configs/        # Configuration files
└── results/        # Model outputs and evaluations
```

## Getting Started

[Coming soon]

## License

[Coming soon]