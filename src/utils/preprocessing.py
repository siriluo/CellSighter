import numpy as np
import cv2
from typing import Tuple, Optional
from pathlib import Path
import yaml

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def normalize_staining(img: np.ndarray,
                      target_means: Optional[np.ndarray] = None,
                      target_stds: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Normalize staining in H&E images using the method from:
    A method for normalizing histology slides for quantitative analysis.
    M. Macenko et al., ISBI 2009
    
    Args:
        img: Input image (RGB)
        target_means: Target means for each channel
        target_stds: Target standard deviations for each channel
        
    Returns:
        Normalized image
    """
    # Convert to optical density
    od = -np.log((img.astype(float) + 1) / 256)
    
    # Remove zeros
    od = od[od > 0]
    
    # SVD on optical density values
    _, _, V = np.linalg.svd(od)
    
    # Project on first two principal components
    That = np.dot(od, V[:2].T)
    
    # Fit plane to standardize staining
    phi = np.arctan2(That[:, 1], That[:, 0])
    
    # Find angular extremes (stain vectors)
    minPhi = np.percentile(phi, 1)
    maxPhi = np.percentile(phi, 99)
    
    # Convert back to RGB
    stain1 = np.dot(V[:2].T, np.array([np.cos(minPhi), np.sin(minPhi)]))
    stain2 = np.dot(V[:2].T, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    
    # Normalize
    norm_img = img.copy()
    for i in range(3):
        if target_means is not None and target_stds is not None:
            norm_img[:, :, i] = ((img[:, :, i] - img[:, :, i].mean()) / 
                                (img[:, :, i].std() + 1e-8) * target_stds[i] + 
                                target_means[i])
        else:
            norm_img[:, :, i] = (img[:, :, i] - img[:, :, i].mean()) / (img[:, :, i].std() + 1e-8)
    
    return norm_img

def register_images(moving_img: np.ndarray,
                   fixed_img: np.ndarray,
                   max_features: int = 1000,
                   good_match_percent: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Register two images using feature matching and homography.
    
    Args:
        moving_img: Image to be transformed
        fixed_img: Reference image
        max_features: Maximum number of features to detect
        good_match_percent: Percentage of good matches to use
        
    Returns:
        Tuple of (registered moving image, transformation matrix)
    """
    # Convert images to grayscale
    moving_gray = cv2.cvtColor(moving_img, cv2.COLOR_RGB2GRAY)
    fixed_gray = cv2.cvtColor(fixed_img, cv2.COLOR_RGB2GRAY)
    
    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(moving_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(fixed_gray, None)
    
    # Match features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    
    # Remove not so good matches
    num_good_matches = int(len(matches) * good_match_percent)
    matches = matches[:num_good_matches]
    
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    
    # Find homography
    h_matrix, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    
    # Use homography to warp image
    height, width = fixed_img.shape[:2]
    registered_img = cv2.warpPerspective(moving_img, h_matrix, (width, height))
    
    return registered_img, h_matrix

def extract_patches(img: np.ndarray,
                   patch_size: int,
                   overlap: float = 0.5) -> np.ndarray:
    """
    Extract overlapping patches from an image.
    
    Args:
        img: Input image
        patch_size: Size of patches to extract
        overlap: Overlap between patches (0-1)
        
    Returns:
        Array of patches
    """
    assert 0 <= overlap < 1, "Overlap must be in [0, 1)"
    
    stride = int(patch_size * (1 - overlap))
    height, width = img.shape[:2]
    
    patches = []
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = img[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
    
    return np.array(patches)

def reconstruct_from_patches(patches: np.ndarray,
                           original_shape: Tuple[int, int, int],
                           patch_size: int,
                           overlap: float = 0.5) -> np.ndarray:
    """
    Reconstruct an image from its patches.
    
    Args:
        patches: Array of image patches
        original_shape: Shape of the original image
        patch_size: Size of the patches
        overlap: Overlap between patches
        
    Returns:
        Reconstructed image
    """
    height, width = original_shape[:2]
    stride = int(patch_size * (1 - overlap))
    
    reconstructed = np.zeros(original_shape)
    counts = np.zeros(original_shape[:2])
    
    patch_idx = 0
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            reconstructed[y:y + patch_size, x:x + patch_size] += patches[patch_idx]
            counts[y:y + patch_size, x:x + patch_size] += 1
            patch_idx += 1
    
    # Average overlapping regions
    reconstructed = reconstructed / counts[:, :, np.newaxis]
    
    return reconstructed