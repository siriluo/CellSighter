"""
data.py - Dataset Implementation

This module implements the PyTorch Dataset class for cell images.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Callable
from .utils import CellCrop


class CellCropsDataset(Dataset):
    """
    Dataset class for cell crops from multiplexed images.
    
    This dataset handles:
    - Loading cell crops
    - Applying transformations
    - Combining image and mask data
    - Providing samples for training/validation
    """
    
    def __init__(self,
                 crops: List[CellCrop],
                 transform: Optional[Callable] = None,
                 mask: bool = False,
                 contrastive: bool = False,):
        """
        Initialize the dataset.
        
        Args:
            crops: List of CellCrop objects
            transform: Optional transform pipeline
            mask: If True, include mask data in samples
        """
        super().__init__()
        self._crops = crops
        self._transform = transform
        self._mask = mask
        self.contrastive = contrastive  # Set to True for contrastive learning

        self._labels = [self._crops[i]._label for i in range(len(self._crops))]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._crops)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing:
            - 'image': Cell image with protein channels
            - 'mask': Cell mask (if self._mask is True)
            - Other metadata (cell_id, image_id, etc.)
        """
        # Get raw sample from CellCrop
        sample = self._crops[idx].sample(self._mask)
        # print("break?")

        num_cell_samples = len(sample['image']) if isinstance(sample['image'], list) else 1
        # print(f"num_cell_samples: {num_cell_samples}")

        # see if I can construct an adjacency matrix here too?

        if self._mask:
            # Stack image with masks
            # consider only adding the single mask displaying the cell of interest
            stacked = np.dstack([
                sample['image'],
                sample['all_cells_mask'][:, :, np.newaxis],
                sample['mask'][:, :, np.newaxis]
            ])
            
            # Apply transforms
            if self._transform:
                transformed = self._transform(stacked) # .float()
                
                # Split back into image and mask
                if self.contrastive:
                    # For contrastive learning, return combined tensor
                    sample['image'] = [transformed[0][:-1, :, :], transformed[1][:-1, :, :]]
                    sample['mask'] = [transformed[0][[-1], :, :], transformed[1][[-1], :, :]]
                else:
                    sample['image'] = transformed[:-1, :, :]
                    sample['mask'] = transformed[[-1], :, :]
        else:
            # Just transform the image if no mask needed
            if self._transform:
                # print("breaking point")
                # print(sample['image'].shape)
                # print(self._transform(np.dstack([sample['image']])).float().shape)
                sample['image'] = self._transform(np.dstack([sample['image']])) # [image, image] for TwoCropTransform
        
        return sample