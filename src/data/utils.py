"""
utils.py - Utility Functions and Classes for Cell Data Processing

This module contains all helper functions and classes for:
- Cell crop handling
- Data augmentation
- Image processing
- Data loading and preprocessing
"""
import os
import numpy as np
import torchvision
import scipy.ndimage as ndimage
import torch 
from typing import List, Tuple, Dict, Any, Optional, Callable
from torchvision.transforms import Lambda
import cv2
import pandas as pd


class CellCrop:
    """
    A class representing a cropped region around a cell in a multiplexed image.
    """
    
    def __init__(self, 
                 cell_id: int,
                 image_id: str,
                 label: int,
                 slices: Tuple[slice, ...],
                 cells: np.ndarray,
                 image: np.ndarray,
                 coords_path: str = None):
        """Initialize a CellCrop instance."""
        self._cell_id = cell_id
        self._image_id = image_id
        self._label = label
        self._slices = slices
        self._cells = cells
        self._image = image
        self._coordinates = None  # Placeholder for coordinates if needed
        # if coords_path, then access the coordinates located there, and grab the corresponding coordinates from 
        # the right image and right cell_id. Should be path/to/coords/{image_id}_cell_info.csv
        if coords_path:
            coordinates_df = pd.read_csv(coords_path)
            y = coordinates_df["Y"].values # 'centroid-0' y_coord_scaled
            x = coordinates_df["X"].values # 'centroid-1' x_coord_scaled\
            cts = coordinates_df["CellType"].values
            self._coordinates = np.array([y[self._cell_id - 1], x[self._cell_id - 1]]) # format of i, j

    def sample(self, mask: bool = False) -> Dict[str, Any]:
        """Extract a sample of the cell region with optional masks."""
        result = {
            'cell_id': self._cell_id,
            'image_id': self._image_id,
            'image': self._image[self._slices].astype(np.float32),
            'slice_x_start': self._slices[0].start,
            'slice_y_start': self._slices[1].start,
            'slice_x_end': self._slices[0].stop,
            'slice_y_end': self._slices[1].stop,
            'label': np.array(self._label, dtype=np.longlong),
            'coordinates': self._coordinates,
        }
        
        if mask:
            cells_crop = self._cells[self._slices]
            result.update({
                'mask': (cells_crop == self._cell_id).astype(np.float32),
                'all_cells_mask': (cells_crop > 0).astype(np.float32),
                'all_cells_mask_seperate': cells_crop.astype(np.float32)
            })
        
        return result


class ImageAugmentor:
    """Collection of image augmentation methods."""
    
    @staticmethod
    def poisson_sampling(x: np.ndarray) -> np.ndarray:
        """Simulate acquisition noise using Poisson sampling."""
        blur = cv2.GaussianBlur(x[:, :, :-2], (5, 5), 0)
        x[:, :, :-2] = np.random.poisson(lam=blur, size=x[:, :, :-2].shape)
        return x

    @staticmethod
    def augment_cell_shape(x: np.ndarray) -> np.ndarray:
        """Randomly dilate the cell mask."""
        if np.random.random() < 0.5:
            cell_mask = x[:, :, -1]
            kernel_size = np.random.choice([2, 3, 5])
            kernel = np.ones(kernel_size, np.uint8)
            x[:, :, -1] = cv2.dilate(cell_mask, kernel, iterations=1)
        return x

    @staticmethod
    def augment_environment_shape(x: np.ndarray) -> np.ndarray:
        """Randomly dilate the environment mask."""
        if np.random.random() < 0.5:
            env_mask = x[:, :, -2]
            kernel_size = np.random.choice([2, 3, 5])
            kernel = np.ones(kernel_size, np.uint8)
            x[:, :, -2] = cv2.dilate(env_mask, kernel, iterations=1)
        return x


class ShiftAugmentation:
    """Implements random shift augmentation."""
    
    def __init__(self, shift_max: int, n_size: int):
        self.shift_max = shift_max
        self.n_size = n_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shift_x = np.random.randint(-self.shift_max, self.shift_max + 1)
        shift_y = np.random.randint(-self.shift_max, self.shift_max + 1)
        
        x = torch.roll(x, shifts=(shift_y, shift_x), dims=(1, 2))
        return x


def create_validation_transform(crop_size: int) -> Callable:
    """Create transformation pipeline for validation data."""
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.CenterCrop((crop_size, crop_size))
    ])


def create_training_transform(crop_size: int, shift: int, mask: bool = True) -> Callable:
    """Create transformation pipeline for training data."""
    augmentor = ImageAugmentor()

    if mask:
        return torchvision.transforms.Compose([
            torchvision.transforms.Lambda(augmentor.poisson_sampling),
            torchvision.transforms.Lambda(augmentor.augment_cell_shape),
            torchvision.transforms.Lambda(augmentor.augment_environment_shape),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomRotation(degrees=(0, 360)),
            Lambda(lambda x: ShiftAugmentation(shift_max=shift, n_size=crop_size)(x) 
                if np.random.random() < 0.5 else x),
            torchvision.transforms.CenterCrop((crop_size, crop_size)),
            torchvision.transforms.RandomHorizontalFlip(p=0.75),
            torchvision.transforms.RandomVerticalFlip(p=0.75),
        ])
    else:
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomRotation(degrees=(0, 360)),
            Lambda(lambda x: ShiftAugmentation(shift_max=shift, n_size=crop_size)(x) 
                if np.random.random() < 0.5 else x),
            torchvision.transforms.CenterCrop((crop_size, crop_size)),
            torchvision.transforms.RandomHorizontalFlip(p=0.75),
            torchvision.transforms.RandomVerticalFlip(p=0.75),
            # torchvision.transforms.RandomApply([
            #     torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            # ], p=0.8),
        ])



def load_data(fname: str) -> np.ndarray:
    """Load image data from file."""
    if fname.endswith(".npz"):
        return np.load(fname, allow_pickle=True)['data']
    elif fname.endswith((".tif", ".tiff")):
        return cv2.imread(fname, -1)
    raise ValueError(f"Unsupported file format: {fname}")


def load_image(image_path: str,
              cells_path: str,
              cells2labels_path: str,
              to_pad: bool = False,
              crop_size: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess image data."""
    # Load image data
    image = load_data(image_path)
    
    # Load cell segmentation
    cells = load_data(cells_path).astype(np.int64)
    
    # Load cell labels
    if cells2labels_path.endswith(".npz"):
        cells2labels = np.load(cells2labels_path, allow_pickle=True)['data'].astype(np.int32)
    else:
        with open(cells2labels_path, "r") as f:
            cells2labels = np.array(f.read().strip().split('\n')).astype(float).astype(int)
    
    # Pad if needed
    if to_pad:
        pad_width = crop_size // 2
        image = np.pad(image, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), 'constant')
        cells = np.pad(cells, ((pad_width, pad_width), (pad_width, pad_width)), 'constant')
    
    return image, cells, cells2labels


def create_slices(slices, crop_size, bounds):
    """
    Create new slices that extend the given slices to a desired crop size,
    centered on the original region. If the crop would go out of bounds,
    it is clipped to fit within the image.

    Args:
        slices: Tuple of slices (from ndimage.find_objects)
        crop_size: Tuple of desired crop sizes per dimension
        bounds: Tuple of image dimensions

    Returns:
        A tuple of adjusted slices that fit within bounds.
    """

    # Modify code to use different crop sizes for extracting from image and when augmenting image
    new_slices = []
    for slc, cs, max_size in zip(slices, crop_size, bounds):
        current_size = slc.stop - slc.start
        d = cs - current_size

        start = slc.start - (d // 2)
        stop = slc.stop + (d + 1) // 2

        # Clip to image bounds
        start = max(0, start)
        stop = min(max_size, stop)

        # Ensure final slice is exactly the desired crop size
        if stop - start < cs:
            # Shift back if needed
            shift = cs - (stop - start)
            start = max(0, start - shift)
            stop = min(max_size, start + cs)

        new_slices.append(slice(start, stop))

    return tuple(new_slices)


def load_samples(config, images_names) -> Tuple[List[CellCrop], List[List[int]]]:
    """Load and process cell samples from images."""
    dataset_dir = os.path.join(config['root_dir'], "CellTypes")
    # images_names = config['train_set'] # will be val_set if for validation
    
    image_dir =  os.path.join(dataset_dir, "data", "images")
    cells_dir =  os.path.join(dataset_dir, "cells")
    cells2labels_dir =  os.path.join(dataset_dir, "cells2labels") 
    
    crops = []
    
    for image_id in images_names:
        # Find image files
        image_path = os.path.join(image_dir, image_id+'.tif')
        cells_path = os.path.join(cells_dir, image_id+'.npz')
        cells2labels_path = os.path.join(cells2labels_dir, image_id+'.npz')
        
        # Load data
        image, cells, cl2lbl = load_image(
            image_path=image_path,
            cells_path=cells_path,
            cells2labels_path=cells2labels_path,
            to_pad=config['to_pad'],
            crop_size=config['crop_size'] # Try using crop_input_size here later?
        )
        
        # Process each cell
        # add option to use cell coordinates instead of finding objects
        objs = ndimage.find_objects(cells)
        
        coords_crc_path = f"/projects/illinois/vetmed/cb/kwang222/mz_jason/crc_ffpe_csvs/{image_id}_cell_info.csv"

        for cell_id, obj in enumerate(objs, 1):
            if obj is None: continue
            
            slices = create_slices(obj, (config['crop_size'], config['crop_size']), cells.shape)
            label = cl2lbl[cell_id]

            crops.append(
                    CellCrop(cell_id=cell_id,
                            image_id=image_id,
                            label=label,
                            slices=slices,
                            cells=cells,
                            image=image,
                            coords_path=coords_crc_path))
    
    return crops