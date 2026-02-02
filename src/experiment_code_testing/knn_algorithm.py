import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn_count(data, labels, test_idx, k=5):
    """
    Perform K-Nearest Neighbors count.

    Parameters:
    - data: numpy array of shape (num_samples, num_features)
    - labels: numpy array of shape (num_samples,)
    - test_point: numpy array of shape (num_features,)
    - k: number of nearest neighbors to consider

    Returns:
    - predicted_label: the predicted label for the test_point
    """
    test_point = data[test_idx]

    # Calculate Cosine Similarity from the test point to all data points
    distances = F.cosine_similarity(test_point, torch.tensor(data), dim=1).numpy()

    new_dists = np.delete(distances, test_idx)
    new_labels = np.delete(labels, test_idx) 
    
    # Get the indices of the k nearest neighbors
    knn_indices = np.argsort(new_dists)[:k]
    
    # Get the labels of the k nearest neighbors
    knn_labels = new_labels[knn_indices]
    
    return knn_labels
