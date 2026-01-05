import torch
from torch.utils.data import Sampler
import numpy as np


class BalancedLargeBatchSampler(Sampler):
    """
    Creates large batches with balanced class representation.
    Specifically designed for batch_size=2048 with 10 classes.
    """
    
    def __init__(self, labels, batch_size=2048, samples_per_class=None):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.n_classes = len(np.unique(labels))
        
        # Determine samples per class per batch
        if samples_per_class is None:
            # Equal representation: 2048 / 10 = 204.8 ≈ 204 per class
            self.samples_per_class = batch_size // self.n_classes
        else:
            self.samples_per_class = samples_per_class
        
        # Adjust batch size to be divisible
        self.actual_batch_size = self.samples_per_class * self.n_classes
        
        print(f"Batch size: {self.actual_batch_size} "
              f"({self.samples_per_class} samples × {self.n_classes} classes)")
        
        # Group indices by class
        self.class_indices = {}
        self.class_sizes = {}
        for class_id in range(self.n_classes):
            indices = np.where(labels == class_id)[0]
            self.class_indices[class_id] = indices
            self.class_sizes[class_id] = len(indices)
            print(f"  Class {class_id}: {len(indices)} samples")
        
        # Number of batches limited by smallest class
        min_class_size = min(self.class_sizes.values())
        self.n_batches = min_class_size // self.samples_per_class
        
        print(f"Total batches per epoch: {self.n_batches}")
        print(f"Samples per epoch: {self.n_batches * self.actual_batch_size}")
        
    def __iter__(self):
        # Shuffle indices within each class
        shuffled_indices = {}
        for class_id in range(self.n_classes):
            indices = self.class_indices[class_id].copy()
            np.random.shuffle(indices)
            
            # If class is smaller than needed, repeat samples
            n_needed = self.n_batches * self.samples_per_class
            if len(indices) < n_needed:
                # Repeat indices to meet requirement
                n_repeats = (n_needed // len(indices)) + 1
                indices = np.tile(indices, n_repeats)[:n_needed]
            
            shuffled_indices[class_id] = indices
        
        # Create batches
        for batch_idx in range(self.n_batches):
            batch = []
            
            for class_id in range(self.n_classes):
                start_idx = batch_idx * self.samples_per_class
                end_idx = start_idx + self.samples_per_class
                batch.extend(shuffled_indices[class_id][start_idx:end_idx])
            
            # Shuffle within batch (optional, but recommended)
            np.random.shuffle(batch)
            
            yield batch
    
    def __len__(self):
        return self.n_batches


class HybridBatchSampler(Sampler):
    """
    Hybrid approach: Part of batch is balanced, part is naturally sampled.
    Useful when you want some natural distribution while ensuring minority classes.
    """
    
    def __init__(self, labels, batch_size=2048, balance_ratio=0.7):
        """
        Args:
            balance_ratio: Fraction of batch that should be balanced (0.7 = 70% balanced)
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.balance_ratio = balance_ratio
        self.n_classes = len(np.unique(labels))
        
        # Split batch into balanced and natural parts
        self.balanced_size = int(batch_size * balance_ratio)
        self.natural_size = batch_size - self.balanced_size
        
        # Make balanced_size divisible by n_classes
        self.samples_per_class = self.balanced_size // self.n_classes
        self.balanced_size = self.samples_per_class * self.n_classes
        self.natural_size = batch_size - self.balanced_size
        
        print(f"Batch composition:")
        print(f"  Balanced portion: {self.balanced_size} ({self.samples_per_class} per class)")
        print(f"  Natural portion: {self.natural_size}")
        print(f"  Total: {self.balanced_size + self.natural_size}")
        
        # Group indices by class
        self.class_indices = {}
        for class_id in range(self.n_classes):
            self.class_indices[class_id] = np.where(labels == class_id)[0]
        
        # All indices for natural sampling
        self.all_indices = np.arange(len(labels))
        
        # Calculate number of batches
        min_class_size = min(len(indices) for indices in self.class_indices.values())
        self.n_batches = min_class_size // self.samples_per_class
    
    def __iter__(self):
        # Shuffle all indices for natural sampling
        natural_indices = self.all_indices.copy()
        np.random.shuffle(natural_indices)
        
        # Shuffle class-specific indices for balanced sampling
        balanced_indices = {}
        for class_id in range(self.n_classes):
            indices = self.class_indices[class_id].copy()
            np.random.shuffle(indices)
            
            n_needed = self.n_batches * self.samples_per_class
            if len(indices) < n_needed:
                n_repeats = (n_needed // len(indices)) + 1
                indices = np.tile(indices, n_repeats)[:n_needed]
            
            balanced_indices[class_id] = indices
        
        # Create batches
        natural_offset = 0
        
        for batch_idx in range(self.n_batches):
            batch = []
            
            # Add balanced portion
            for class_id in range(self.n_classes):
                start = batch_idx * self.samples_per_class
                end = start + self.samples_per_class
                batch.extend(balanced_indices[class_id][start:end])
            
            # Add natural portion
            batch.extend(natural_indices[natural_offset:natural_offset + self.natural_size])
            natural_offset += self.natural_size
            
            np.random.shuffle(batch)
            yield batch
    
    def __len__(self):
        return self.n_batches


class TwoStageBalancedSampler(Sampler):
    """
    Stage 1: Sample without replacement as much as possible
    Stage 2: When minority classes run out, continue with majority classes only
    
    This maximizes data usage while avoiding harmful repetition.
    """
    
    def __init__(self, labels, batch_size=2048, balance_threshold=0.5):
        """
        Args:
            balance_threshold: Fraction of epoch to maintain balance
                              0.5 = first 50% balanced, then majority classes only
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.n_classes = len(np.unique(labels))
        self.balance_threshold = balance_threshold
        
        # Group indices
        self.class_indices = {}
        self.class_sizes = {}
        for class_id in range(self.n_classes):
            indices = np.where(labels == class_id)[0]
            self.class_indices[class_id] = indices
            self.class_sizes[class_id] = len(indices)
        
        min_class_size = min(self.class_sizes.values())
        max_class_size = max(self.class_sizes.values())
        
        # Stage 1: Balanced batches (limited by minority classes)
        self.samples_per_class_balanced = batch_size // self.n_classes
        self.n_balanced_batches = int(
            (min_class_size / self.samples_per_class_balanced) * balance_threshold
        )
        
        # Stage 2: Continue with remaining majority class samples
        remaining_samples = sum(
            max(0, size - self.samples_per_class_balanced * self.n_balanced_batches)
            for size in self.class_sizes.values()
        )
        self.n_unbalanced_batches = remaining_samples // batch_size
        
        self.total_batches = self.n_balanced_batches + self.n_unbalanced_batches
        
        print("="*60)
        print("TWO-STAGE BALANCED SAMPLER")
        print("="*60)
        print(f"Stage 1 (Balanced):")
        print(f"  Batches: {self.n_balanced_batches}")
        print(f"  Batch size: {self.samples_per_class_balanced * self.n_classes}")
        print(f"  Samples: {self.n_balanced_batches * self.samples_per_class_balanced * self.n_classes}")
        
        print(f"\nStage 2 (Natural distribution):")
        print(f"  Batches: {self.n_unbalanced_batches}")
        print(f"  Batch size: {batch_size}")
        print(f"  Samples: {self.n_unbalanced_batches * batch_size}")
        
        print(f"\nTotal batches: {self.total_batches}")
        print(f"Total samples: {self.n_balanced_batches * self.samples_per_class_balanced * self.n_classes + self.n_unbalanced_batches * batch_size}")
        print("="*60)
    
    def __iter__(self):
        # Shuffle all indices
        shuffled_indices = {}
        for class_id in range(self.n_classes):
            indices = self.class_indices[class_id].copy()
            np.random.shuffle(indices)
            shuffled_indices[class_id] = indices
        
        batch_idx = 0
        
        # Stage 1: Balanced batches
        for i in range(self.n_balanced_batches):
            batch = []
            
            for class_id in range(self.n_classes):
                start = i * self.samples_per_class_balanced
                end = start + self.samples_per_class_balanced
                batch.extend(shuffled_indices[class_id][start:end])
            
            np.random.shuffle(batch)
            yield batch
            batch_idx += 1
        
        # Stage 2: Use remaining samples from majority classes
        remaining_indices = []
        for class_id in range(self.n_classes):
            start = self.n_balanced_batches * self.samples_per_class_balanced
            remaining_indices.extend(shuffled_indices[class_id][start:])
        
        np.random.shuffle(remaining_indices)
        
        # Create batches from remaining
        for i in range(self.n_unbalanced_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            batch = remaining_indices[start:end]
            yield batch
    
    def __len__(self):
        return self.total_batches
    

# Try and focus on cells from different slides when balancing the samples.