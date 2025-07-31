"""
CellSighter Data Module

This module provides tools for loading, processing, and augmenting cell image data
for the CellSighter project.

Main Components:
- CellCropsDataset: PyTorch dataset for cell images
- CellCrop: Base class for cell image crops
- Data loading and augmentation utilities
"""

from .data import CellCropsDataset
from .utils import (
    # Core classes
    CellCrop,
    ImageAugmentor,
    ShiftAugmentation,
    
    # Transform functions
    create_training_transform,
    create_validation_transform,
    
    # Data loading functions
    load_samples,
    load_data,
    load_image,
    create_slices
)

# Convenient aliases
train_transform = create_training_transform
val_transform = create_validation_transform

__all__ = [
    # Main classes
    'CellCropsDataset',
    'CellCrop',
    
    # Augmentation
    'ImageAugmentor',
    'ShiftAugmentation',
    'train_transform',
    'val_transform',
    
    # Data loading
    'load_samples',
    'load_data',
    'load_image',
    'create_slices'
]