#!/usr/bin/env python3
"""
Simple script to read and analyze the cells2labels npz file
"""

import numpy as np
import sys
import os

def read_labels_file(file_path):
    """Read and analyze a cells2labels npz file."""
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    print(f"Reading file: {file_path}")
    
    try:
        # Load the npz file
        data = np.load(file_path, allow_pickle=True)
        
        print(f"Keys in npz file: {list(data.keys())}")
        
        # Get the data array (usually stored under 'data' key)
        if 'data' in data.keys():
            arr = data['data']
            key_name = 'data'
        else:
            # Get the first (and likely only) key
            key_name = list(data.keys())[0]
            arr = data[key_name]
        
        print(f"\nArray from key '{key_name}':")
        print(f"  Type: {type(arr)}")
        print(f"  Shape: {arr.shape}")
        print(f"  Dtype: {arr.dtype}")
        print(f"  Size: {arr.size}")
        
        # Basic statistics
        print(f"\nBasic Statistics:")
        print(f"  Min value: {np.min(arr)}")
        print(f"  Max value: {np.max(arr)}")
        print(f"  Mean: {np.mean(arr):.3f}")
        
        # Show first 20 values
        print(f"\nFirst 20 values: {arr[:20]}")
        
        # Unique values and their counts
        unique_vals, counts = np.unique(arr, return_counts=True)
        print(f"\nTotal unique values: {len(unique_vals)}")
        
        print(f"\nLabel distribution (first 15):")
        for i in range(min(15, len(unique_vals))):
            val = unique_vals[i]
            count = counts[i]
            percentage = (count / arr.size) * 100
            print(f"  Label {val}: {count} cells ({percentage:.2f}%)")
        
        if len(unique_vals) > 15:
            print(f"  ... and {len(unique_vals) - 15} more unique labels")
        
        # Binary classification statistics
        print(f"\nBinary Classification (0=non-tumor, >0=tumor):")
        non_tumor = np.sum(arr == 0)
        tumor = np.sum(arr > 0)
        total = arr.size
        
        print(f"  Non-tumor cells (label=0): {non_tumor} ({(non_tumor/total)*100:.2f}%)")
        print(f"  Tumor cells (label>0): {tumor} ({(tumor/total)*100:.2f}%)")
        print(f"  Total cells: {total}")
        
        # Close the file
        data.close()
        
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    file_path = "/u/shirui/bbka/CellSighter/datasets/Adaptive/cells2labels/reg001_A.npz"
    read_labels_file(file_path)