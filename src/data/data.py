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
                 mask: bool = False):
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
        
        if self._mask:
            # Stack image with masks
            stacked = np.dstack([
                sample['image'],
                sample['all_cells_mask'][:, :, np.newaxis],
                sample['mask'][:, :, np.newaxis]
            ])
            
            # Apply transforms
            if self._transform:
                transformed = self._transform(stacked).float()
                
                # Split back into image and mask
                sample['image'] = transformed[:-1, :, :]
                sample['mask'] = transformed[[-1], :, :]
        else:
            # Just transform the image if no mask needed
            if self._transform:
                sample['image'] = self._transform(sample['image']).float()
        
        return sample


def _test_cellcropsdataset():
    """Unit tests for CellCropsDataset class."""
    import unittest
    import torch.testing
    
    class TestCellCropsDataset(unittest.TestCase):
        def setUp(self):
            # Create mock CellCrop objects
            from .utils import CellCrop
            import numpy as np
            
            self.image = np.random.rand(100, 100, 3)
            self.cells = np.zeros((100, 100))
            self.cells[40:60, 40:60] = 1  # Cell ID 1
            
            self.mock_crop = CellCrop(
                cell_id=1,
                image_id="test_001",
                label=0,
                slices=(slice(30, 70), slice(30, 70)),
                cells=self.cells,
                image=self.image
            )
            
            # Simple transform function
            self.transform = lambda x: torch.from_numpy(x.transpose(2, 0, 1))
        
        def test_dataset_initialization(self):
            """Test dataset initialization."""
            dataset = CellCropsDataset([self.mock_crop])
            self.assertEqual(len(dataset), 1)
            self.assertIsNone(dataset._transform)
            self.assertFalse(dataset._mask)
        
        def test_getitem_no_mask(self):
            """Test getting an item without mask."""
            dataset = CellCropsDataset([self.mock_crop], transform=self.transform)
            sample = dataset[0]
            
            self.assertIn('image', sample)
            self.assertNotIn('mask', sample)
            self.assertEqual(sample['cell_id'], 1)
            self.assertEqual(sample['image_id'], "test_001")
            self.assertTrue(isinstance(sample['image'], torch.Tensor))
        
        def test_getitem_with_mask(self):
            """Test getting an item with mask."""
            dataset = CellCropsDataset([self.mock_crop], transform=self.transform, mask=True)
            sample = dataset[0]
            
            self.assertIn('image', sample)
            self.assertIn('mask', sample)
            self.assertIn('all_cells_mask', sample)
            self.assertTrue(isinstance(sample['mask'], torch.Tensor))
            self.assertEqual(sample['mask'].shape[0], 1)  # Single channel mask
        
        def test_transform_application(self):
            """Test that transforms are correctly applied."""
            def custom_transform(x):
                return torch.ones_like(torch.from_numpy(x.transpose(2, 0, 1)))
            
            dataset = CellCropsDataset([self.mock_crop], transform=custom_transform)
            sample = dataset[0]
            
            self.assertTrue(torch.all(sample['image'] == 1.0))
    
    # Run the tests
    if __name__ == '__main__':
        unittest.main()