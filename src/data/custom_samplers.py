import torch
from torch.utils.data import Sampler
import numpy as np
import math


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
    """Fixed version with proper dtype handling."""
    
    def __init__(self, labels, batch_size=2048, balance_threshold=0.5):
        # Convert labels to numpy array with int64
        self.labels = np.array(labels, dtype=np.int64)
        self.batch_size = int(batch_size)  # Ensure int
        self.n_classes = len(np.unique(self.labels))
        self.balance_threshold = float(balance_threshold)
        
        # Group indices by class
        self.class_indices = {}
        self.class_sizes = {}
        
        for class_id in range(self.n_classes):
            # Get indices for this class
            indices = np.where(self.labels == class_id)[0]
            
            # Store as int64
            self.class_indices[class_id] = indices.astype(np.int64)
            self.class_sizes[class_id] = len(indices)
        
        min_class_size = min(self.class_sizes.values())
        max_class_size = max(self.class_sizes.values())
        
        # Calculate batch composition
        self.samples_per_class_balanced = self.batch_size // self.n_classes
        self.n_balanced_batches = int(
            (min_class_size / self.samples_per_class_balanced) * self.balance_threshold
        )
        
        remaining_samples = sum(
            max(0, size - self.samples_per_class_balanced * self.n_balanced_batches)
            for size in self.class_sizes.values()
        )
        self.n_unbalanced_batches = int(remaining_samples // self.batch_size)
        
        self.total_batches = self.n_balanced_batches + self.n_unbalanced_batches
        
        print(f"TwoStageBalancedSampler initialized:")
        print(f"  Balanced batches: {self.n_balanced_batches}")
        print(f"  Unbalanced batches: {self.n_unbalanced_batches}")
        print(f"  Total batches: {self.total_batches}")
    
    def __iter__(self):
        # Shuffle indices for each class
        shuffled_indices = {}
        for class_id in range(self.n_classes):
            indices = self.class_indices[class_id].copy()
            np.random.shuffle(indices)
            shuffled_indices[class_id] = indices
        
        # Stage 1: Balanced batches
        for batch_idx in range(self.n_balanced_batches):
            batch_indices = []
            
            for class_id in range(self.n_classes):
                start = batch_idx * self.samples_per_class_balanced
                end = start + self.samples_per_class_balanced
                
                # Get indices for this class
                class_batch = shuffled_indices[class_id][start:end]
                batch_indices.extend(class_batch)
            
            # Convert to int64 array
            batch_indices = np.array(batch_indices, dtype=np.int64)
            np.random.shuffle(batch_indices)
            
            # CRITICAL: Yield as list of native Python ints
            yield [int(idx) for idx in batch_indices]
        
        # Stage 2: Unbalanced batches
        remaining = []
        for class_id in range(self.n_classes):
            start = self.n_balanced_batches * self.samples_per_class_balanced
            remaining.extend(shuffled_indices[class_id][start:])
        
        # Shuffle remaining
        remaining = np.array(remaining, dtype=np.int64)
        np.random.shuffle(remaining)
        
        # Create unbalanced batches
        for batch_idx in range(self.n_unbalanced_batches):
            start = batch_idx * self.batch_size
            end = start + self.batch_size
            
            batch_indices = remaining[start:end]
            
            # CRITICAL: Yield as list of native Python ints
            yield [int(idx) for idx in batch_indices]
    
    def __len__(self):
        return self.total_batches
    

# Try and focus on cells from different slides when balancing the samples.
class ClassAwareSupConBatchSampler(Sampler):
    """
    Class-aware batch sampler for supervised contrastive learning.

    Each batch contains a fixed number of classes and a fixed number of samples
    per class, guaranteeing same-class positives inside the batch. Rare classes
    can be sampled with replacement.
    """

    def __init__(
        self,
        labels,
        batch_size,
        samples_per_class=64,
        classes_per_batch=None,
        oversample=True,
        num_batches=None,
        seed=42,
        drop_incomplete=True,
        fill=False,
        fill_type="random",
    ):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.samples_per_class = int(samples_per_class)
        self.oversample = bool(oversample)
        self.seed = int(seed)
        self.drop_incomplete = bool(drop_incomplete)
        self.epoch = 0
        self.will_fill = fill
        self.fill_type = fill_type

        self.classes = np.array(sorted(np.unique(self.labels).tolist()), dtype=np.int64)
        self.num_classes = len(self.classes)

        if classes_per_batch is None:
            classes_per_batch = min(
                self.num_classes,
                max(1, self.batch_size // self.samples_per_class),
            )

        self.classes_per_batch = int(classes_per_batch)
        if self.classes_per_batch > self.num_classes:
            raise ValueError(
                f"classes_per_batch={self.classes_per_batch} exceeds "
                f"available classes={self.num_classes}."
            )

        self.actual_batch_size = self.classes_per_batch * self.samples_per_class
        if self.actual_batch_size > self.batch_size:
            raise ValueError(
                f"classes_per_batch * samples_per_class = {self.actual_batch_size}, "
                f"which exceeds batch_size={self.batch_size}."
            )
        self.remainder_size = 0
        if self.will_fill:
            self.remainder_size = self.batch_size - self.actual_batch_size

        if self.remainder_size < 0:
            raise ValueError(
                f"Balanced core batch size={self.actual_batch_size} exceeds "
                f"batch_size={self.batch_size}."
            )

        self.all_indices = np.arange(len(self.labels), dtype=np.int64)

        self.class_indices = {}
        for class_id in self.classes:
            indices = np.where(self.labels == class_id)[0].astype(np.int64)
            if len(indices) == 0:
                raise ValueError(f"Class {class_id} has no samples.")
            self.class_indices[int(class_id)] = indices
        
        class_counts = {
            int(class_id): len(self.class_indices[int(class_id)])
            for class_id in self.classes
        }

        if self.fill_type == "inverse_frequency":
            weights = np.zeros(len(self.labels), dtype=np.float64)
            for class_id, count in class_counts.items():
                weights[self.labels == class_id] = 1.0 / count
            self.fill_probs = weights / weights.sum()
        else:
            self.fill_probs = None

        valid_fill_strategies = {"random", "inverse_frequency", "class_uniform"}
        if self.fill_type not in valid_fill_strategies:
            raise ValueError(
                f"Unknown fill_strategy={self.fill_type}. "
                f"Expected one of {sorted(valid_fill_strategies)}."
            )

        if num_batches is None:
            num_batches = math.ceil(len(self.labels) / self.actual_batch_size)
        self.num_batches = int(num_batches)

        print("ClassAwareSupConBatchSampler initialized:")
        print(f"  Classes: {self.classes.tolist()}")
        print(f"  Classes per batch: {self.classes_per_batch}")
        print(f"  Samples per class: {self.samples_per_class}")
        print(f"  Actual batch size: {self.actual_batch_size}")
        print(f"  Batches per epoch: {self.num_batches}")
        print(f"  Oversample rare classes: {self.oversample}")
        for class_id in self.classes:
            print(f"  Class {int(class_id)}: {len(self.class_indices[int(class_id)])} samples")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        class_order = self.classes.copy()
        rng.shuffle(class_order)
        class_cursor = 0

        shuffled_by_class = {}
        cursor_by_class = {}
        for class_id in self.classes:
            indices = self.class_indices[int(class_id)].copy()
            rng.shuffle(indices)
            shuffled_by_class[int(class_id)] = indices
            cursor_by_class[int(class_id)] = 0

        for _ in range(self.num_batches):
            if class_cursor + self.classes_per_batch > len(class_order):
                rng.shuffle(class_order)
                class_cursor = 0

            batch_classes = class_order[
                class_cursor : class_cursor + self.classes_per_batch
            ]
            class_cursor += self.classes_per_batch

            batch = []
            for class_id in batch_classes:
                class_id = int(class_id)
                indices = shuffled_by_class[class_id]
                cursor = cursor_by_class[class_id]
                end = cursor + self.samples_per_class

                if end <= len(indices):
                    selected = indices[cursor:end]
                    cursor_by_class[class_id] = end
                elif self.oversample:
                    needed = self.samples_per_class
                    selected = rng.choice(indices, size=needed, replace=True)
                    cursor_by_class[class_id] = len(indices)
                else:
                    selected = indices[cursor:]
                    cursor_by_class[class_id] = len(indices)

                batch.extend(selected.tolist())
                if self.will_fill and self.remainder_size > 0:
                    if self.fill_type == "random":
                        fill = rng.choice(
                            self.all_indices,
                            size=self.remainder_size,
                            replace=True,
                        )

                    elif self.fill_type == "inverse_frequency":
                        fill = rng.choice(
                            self.all_indices,
                            size=self.remainder_size,
                            replace=True,
                            p=self.fill_probs,
                        )

                    elif self.fill_type == "class_uniform":
                        fill = []
                        fill_classes = rng.choice(
                            self.classes,
                            size=self.remainder_size,
                            replace=True,
                        )
                        for fill_class in fill_classes:
                            class_pool = self.class_indices[int(fill_class)]
                            fill.append(int(rng.choice(class_pool)))
                        fill = np.asarray(fill, dtype=np.int64)

                    batch.extend(fill.tolist())

            expected_batch_size = self.batch_size if self.will_fill else self.actual_batch_size

            if len(batch) < expected_batch_size and self.drop_incomplete:
                continue

            rng.shuffle(batch)
            yield [int(idx) for idx in batch]

    def __len__(self):
        return self.num_batches

