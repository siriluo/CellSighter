# Mainly for processing the sharded orion dataset into CellCrops 

import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Callable
from .utils import CellCrop
import os
import glob

# Just do CRC01

cell_patches_path = "/projects/illinois/vetmed/cb/kwang222/mz_jason/orion_all_without_largest/_meta/cell_labeling/cell_patches_64_crc01_match5um_area50_3000/CRC01"

# The data is numbered 00000
mask_name = "cell_masks"
img_patch_name = "image_patches"
labels_name = "meta"

# count
filelist = glob.glob(f"{cell_patches_path}/{labels_name}_*.csv")
print(len(filelist))
# for file in filelist:
#     print(file)


def get_multiclass_ct_name_test(label):

    new_mapping = {
        "CD4_T": 0,
        "CD8_T": 1,
        "B_cell": 2,
        "Granulocyte": 3,
    }

    class_name = new_mapping[label]

    return class_name


def get_multiclass_ct_name(label):

    new_mapping = {
        "CD4_T": 0,
        "CD8_T": 1,
        "Treg": 2,
        "B_cell": 3,
        "Mono_Macro": 4,
        "Stromal": 5,
        "Smooth_Muscle": 6,
        "Tumor": 7,
        "Vasculature": 8,
        "Granulocyte": 9,
    }

    class_name = new_mapping[label]

    return class_name


# Make a function for helping to make balanced sample loading for orion to increase contrastive learning effectiveness


def load_cell_crops_from_orion(
        cell_patches_path: str,
        mask_name: str,
        img_patch_name: str,
        labels_name: str,
        label_files,
        sample_fraction=None,
        max_samples=None,
        sample_seed: int = 42) -> List[CellCrop]:
    """
    Load cell crops from the Orion dataset.
    
    Args:
        cell_patches_path: Path to the directory containing cell patches
        mask_name: Name of the mask files
        img_patch_name: Name of the image patch files
        labels_name: Name of the label files
    Returns:
        List of CellCrop objects
    """
    if sample_fraction is not None or max_samples is not None:
        return load_sampled_cell_crops_from_orion(
            cell_patches_path=cell_patches_path,
            mask_name=mask_name,
            img_patch_name=img_patch_name,
            label_files=label_files,
            sample_fraction=sample_fraction,
            max_samples=max_samples,
            sample_seed=sample_seed,
        )

    cell_crops = []
    sample_id = os.path.basename(os.path.normpath(cell_patches_path))
     
    # Load labels
    # label_files = glob.glob(f"{cell_patches_path}/{labels_name}_*.csv")
    for label_file in label_files:
        # Extract case and image ID from the filename
        filename = os.path.basename(label_file)
        parts = filename.split('_')
        file_id = parts[1].split('.')[0]  # Assuming format: meta_fileID.csv
        
        images = np.load(f"{cell_patches_path}/{img_patch_name}_{file_id}.npy")
        masks = np.load(f"{cell_patches_path}/{mask_name}_{file_id}.npy")
        
        # print(images.shape, masks.shape)
        
        # Load labels
        labels_df = pd.read_csv(label_file)
        # ignore rows with cell_type == Unassigned
        cells_to_ignore = ["Unassigned", "Vasculature", "Tumor", "Smooth_Muscle", "Stromal", "Mono_Macro", "Treg"]
        new_labels_df = labels_df[labels_df['orion_label'] != 'Unassigned'] 
        # new_labels_df = new_labels_df[new_labels_df['orion_label'] != 'Granulocyte'] 
        # new_labels_df = labels_df[~labels_df['orion_label'].isin(cells_to_ignore)].copy()
        
        shard_indices = new_labels_df['index_in_shard'].values
        cell_ids = new_labels_df['cellpose_id'].values
        cell_labels = new_labels_df['orion_label'].values
        x_coords = new_labels_df['x'].values
        y_coords = new_labels_df['y'].values
        sample_name = new_labels_df['case'].values[0]
        
        num_shards = len(shard_indices)
        
        for i in np.arange(num_shards):
            cell_id = cell_ids[i]
            label = cell_labels[i]
            int_label = get_multiclass_ct_name(label) # get_multiclass_ct_name get_multiclass_ct_name_test
            x = x_coords[i]
            y = y_coords[i]
            
            img_patch = images[shard_indices[i]]
            # print(img_patch.max(), img_patch.min())
            mask_patch = masks[shard_indices[i]]
            
            cell_crop = CellCrop(cell_id=1, # cell id should correspond to the integer in the mask, which should only be 1 for orion
                        image_id=f"{sample_name}-{img_patch_name}_{file_id}_{i}",
                        label=int_label,
                        slices=None,
                        cells=mask_patch,
                        image=img_patch,
                        coords_path=None, #  None np.array([cell_id, x, y])
                        orion_format=False) # False True 
            
            cell_crops.append(cell_crop)
    
    print(f"{len(cell_crops)} cell crops loaded from Orion dataset.")
    
    return cell_crops


def load_sampled_cell_crops_from_orion(
        cell_patches_path: str,
        mask_name: str,
        img_patch_name: str,
        label_files,
        sample_fraction=None,
        max_samples=None,
        sample_seed: int = 42) -> List[CellCrop]:
    """
    Fast optional Orion sampling path.

    This samples metadata rows before constructing CellCrop objects, so debug runs
    avoid materializing the full CRC sample and then copying a subset in Python.
    """
    label_files = sorted(label_files)
    usecols = ["case", "index_in_shard", "cellpose_id", "orion_label", "x", "y"]
    sampled_frames = []
    total_cells = 0

    for label_file in label_files:
        filename = os.path.basename(label_file)
        file_id = filename.split('_')[1].split('.')[0]
        labels_df = pd.read_csv(label_file, usecols=usecols)
        labels_df = labels_df[labels_df['orion_label'] != 'Unassigned'].copy()
        if labels_df.empty:
            continue

        labels_df["file_id"] = file_id
        labels_df["row_index"] = np.arange(len(labels_df))
        total_cells += len(labels_df)
        sampled_frames.append(labels_df)

    if not sampled_frames:
        print("0 cell crops loaded from Orion dataset.")
        return []

    labels_df = pd.concat(sampled_frames, ignore_index=True)
    if sample_fraction is not None:
        sample_count = max(1, int(round(len(labels_df) * sample_fraction)))
    else:
        sample_count = int(max_samples)
    sample_count = min(len(labels_df), sample_count)

    labels_df = labels_df.sample(n=sample_count, random_state=sample_seed)
    labels_df = labels_df.sort_values(["file_id", "row_index"])

    cell_crops = []
    for file_id, file_df in labels_df.groupby("file_id", sort=True):
        images = np.load(f"{cell_patches_path}/{img_patch_name}_{file_id}.npy", mmap_mode="r")
        masks = np.load(f"{cell_patches_path}/{mask_name}_{file_id}.npy", mmap_mode="r")

        shard_indices = file_df['index_in_shard'].values
        cell_ids = file_df['cellpose_id'].values
        cell_labels = file_df['orion_label'].values
        x_coords = file_df['x'].values
        y_coords = file_df['y'].values
        row_indices = file_df['row_index'].values
        sample_name = file_df['case'].values[0]

        for i in np.arange(len(shard_indices)):
            cell_id = cell_ids[i]
            label = cell_labels[i]
            int_label = get_multiclass_ct_name(label)
            x = x_coords[i]
            y = y_coords[i]

            img_patch = np.asarray(images[shard_indices[i]]).copy()
            mask_patch = np.asarray(masks[shard_indices[i]]).copy()

            cell_crop = CellCrop(
                cell_id=1,
                image_id=f"{sample_name}-{img_patch_name}_{file_id}_{row_indices[i]}",
                label=int_label,
                slices=None,
                cells=mask_patch,
                image=img_patch,
                coords_path=None, # None np.array([cell_id, x, y])
                orion_format=False, # False True
            )

            cell_crops.append(cell_crop)

    print(f"{len(cell_crops)}/{total_cells} cell crops loaded from Orion dataset.")
    return cell_crops


# load_cell_crops_from_orion(cell_patches_path, mask_name, img_patch_name, labels_name, filelist)

# CellID from CRC33_01_option2_patch_overlay_1500_filtered_by_patchcsv_cells aligns with
# orion_cellid from meta_00000.csv, so we can use that to load the cell crops with the correct cell ID.
def load_cell_crops_from_orion_with_cellid(cell_patches_path: str, mask_name: str, img_patch_name: str) -> List[CellCrop]:
    """
    Load cell crops from the Orion dataset with cell ID.
    
    Args:
        cell_patches_path: Path to the directory containing cell patches
        mask_name: Name of the mask files
        img_patch_name: Name of the image patch files
        labels_name: Name of the label files
    Returns:
        List of CellCrop objects
    """
    cell_crops = []
    sample_id = os.path.basename(os.path.normpath(cell_patches_path))
     
    filtered_cells_labels = "/taiga/illinois/vetmed/cb/kwang222/mz_jason/orion_all_without_largest/_meta/small_patch_label_overlays/CRC33_01_option2_patch_overlay_1500_filtered_by_patchcsv_cells.csv"
    orig_cells_labels = "meta_00000.csv"
     
    # Load labels, first load all the csv files together.

    file_id = "00000"
        
    images = np.load(f"{cell_patches_path}/{img_patch_name}_{file_id}.npy")
    masks = np.load(f"{cell_patches_path}/{mask_name}_{file_id}.npy")
    
    # print(images.shape, masks.shape)
    
    # Load labels
    labels_df = pd.read_csv(f"{cell_patches_path}/{orig_cells_labels}")
    filtered_labels_df = pd.read_csv(filtered_cells_labels)
    new_labels_df = labels_df[labels_df['orion_label'] != 'Unassigned'] 
    new_labels_df = new_labels_df[new_labels_df["orion_cellid"].isin(filtered_labels_df["CellID"])].copy()

    
    shard_indices = new_labels_df['index_in_shard'].values
    cell_ids = new_labels_df['cellpose_id'].values
    cell_labels = new_labels_df['orion_label'].values
    x_coords = new_labels_df['x'].values
    y_coords = new_labels_df['y'].values
    sample_name = new_labels_df['case'].values[0]
    
    num_shards = len(shard_indices)
    
    coords_list = []
    for i in np.arange(num_shards):
        cell_id = cell_ids[i]
        label = cell_labels[i]
        int_label = get_multiclass_ct_name(label)
        x = x_coords[i]
        y = y_coords[i]
        
        img_patch = images[shard_indices[i]]
        mask_patch = masks[shard_indices[i]]
        
        coords = [cell_id, x, y]
        
        cell_crop = CellCrop(cell_id=1, # use the actual cell id from orion
                    image_id=f"{sample_name}_{img_patch_name}_{file_id}_{i}",
                    label=int_label,
                    slices=None,
                    cells=mask_patch,
                    image=img_patch,
                    coords_path=np.array([cell_id, x, y]),
                    orion_format=True)
        
        cell_crops.append(cell_crop)
        coords_list.append(coords)
        
    coords_list = np.array(coords_list)
    np.savez(f"cell_coords.npz", coords_list)
    
    print(f"{len(cell_crops)} cell crops loaded from Orion dataset with cell ID.")
    
    return cell_crops
