f"""
train.py - Main Training Script for Cell Classification

This script handles the complete training pipeline:
- Data loading and preprocessing
- Model initialization
- Training and validation
- Model evaluation and saving
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Any
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Local imports
from models import create_model, get_model_info
from trainer import Trainer
from data.utils import load_samples, create_training_transform, create_validation_transform
from data.data import CellCropsDataset


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def convert_to_binary_label(labels, positive_label=0):
    new_labels = labels.clone()

    new_labels[labels != positive_label] = 0
    
    new_labels[labels == positive_label] = 1

    return new_labels 


def create_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    print("Loading training data...")
    train_crops = load_samples(config, config['train_set'])
    print(f"Loaded {len(train_crops)} training samples")
    
    print("Loading validation data...")
    val_crops = load_samples(config, config['val_set'])
    print(f"Loaded {len(val_crops)} validation samples")
    
    use_mask = False

    # Create transforms
    if config.get('aug', False):
        train_transform = create_training_transform(
            crop_size=config['crop_size'],
            shift=config.get('shift', 5),
            mask=use_mask,
        )
        print("Using data augmentation for training")
    else:
        train_transform = create_validation_transform(crop_size=config['crop_size'])
        print("No data augmentation applied")
    
    val_transform = create_validation_transform(crop_size=config['crop_size'])
    
    # Create datasets
    train_dataset = CellCropsDataset(
        crops=train_crops,
        transform=train_transform,
        mask=use_mask  # Set to True if you want to include mask information
    )
    
    val_dataset = CellCropsDataset(
        crops=val_crops,
        transform=val_transform,
        mask=use_mask
    )
    
    # Print dataset statistics
    print_dataset_stats(train_dataset, "Training")
    print_dataset_stats(val_dataset, "Validation")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader


def get_multiclass_ct_name(label):

    new_mapping = {
        # 0: "CD4+ T",
        # 1: "CD8+ T",
        # 2: "Treg",
        # 3: "B cells",
        # 4: "NK Cells",
        # 5: "Dendritic Cells",
        # 6: "Monocytes / Macrophages",
        # 7: "Stromal Cells",
        # 8: "Smooth Muscle",
        # 9: "Tumor Cells",
        # 10: "Vasculature",
        # 11: "Granulocytes",
        0: "CD4+ T",
        1: "CD8+ T",
        2: "Treg",
        3: "B cells",
        4: "Monocytes / Macrophages",
        5: "Stromal Cells",
        6: "Smooth Muscle",
        7: "Tumor Cells",
        8: "Vasculature",
        9: "Granulocytes",
    }

    class_name = new_mapping[label]

    return class_name


def print_dataset_stats(dataset: CellCropsDataset, dataset_name: str):
    """Print statistics about the dataset."""
    print(f"\n{dataset_name} Dataset Statistics:")
    print(f"Total samples: {len(dataset)}")
    
    # Count labels
    labels = []
    for i in range(len(dataset)):
        sample = dataset[i]
        label = sample['label'].item()
        binary_label = label #  1 if label > 0 else 0
        labels.append(binary_label)
    
    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    
    print(f"Class distribution:")
    for label, count in zip(unique, counts):
        class_name = get_multiclass_ct_name(label) # "Tumor" if label == 1 else "Non-tumor"
        percentage = (count / len(labels)) * 100
        print(f"  {class_name} (label {label}): {count} samples ({percentage:.1f}%)")
    
    # Get sample image shape
    sample_image = dataset[0]['image']
    print(f"Image shape: {sample_image.shape}")
    print(f"Image dtype: {sample_image.dtype}")


def calculate_class_weights(train_loader: DataLoader, num_classes: int, device: str) -> torch.Tensor:
    """Calculate class weights for balanced training."""
    print("Calculating class weights...")
    
    all_labels = []
    for batch in train_loader:
        labels = batch['label']
        # binary_labels = (labels > 0).long()

        if num_classes > 2:
            long_labels = labels.long() # (labels > 0)
        else:
            new_bin_labels = convert_to_binary_label(labels=labels, positive_label=7) # 
            long_labels = (new_bin_labels).long()

        all_labels.extend(long_labels.numpy())
    
    print(np.unique(all_labels))
    all_labels = np.array(all_labels)
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(all_labels), # np.unique(all_labels)
        y=all_labels
    )
    
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    
    array_str = np.array2string(weights_tensor.cpu().numpy(), precision=3, separator=', ')

    print(f"Class weights: {list(array_str)}") # Non-tumor= , Tumor={weights_tensor[1]:.3f}
    return weights_tensor


def create_optimizer_and_scheduler(model: nn.Module, config: Dict[str, Any]) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """Create optimizer and learning rate scheduler."""
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 1e-4) #  1e-5
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get('lr_step_size', 10),
        gamma=config.get('lr_gamma', 0.5)
    )
    
    return optimizer, scheduler


def main(config_path: str, model_type: str = 'cnn', resume_checkpoint: str = None):
    """
    Main training function.
    
    Args:
        config_path: Path to configuration file
        model_type: Type of model to use ('cnn' or 'resnet')
        resume_checkpoint: Path to checkpoint to resume from (optional)
    """
    # Load configuration
    config = load_config(config_path)
    print(f"Loaded configuration from {config_path}")
    print(f"Training on {len(config['train_set'])} images, validating on {len(config['val_set'])} images")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config)
    
    # Get input channels from a sample
    sample_batch = next(iter(train_loader))
    input_channels = sample_batch['image'].shape[1] #  5 #
    print(f"Input channels: {input_channels}")
    
    # Create model
    model = create_model(
        model_type=model_type,
        input_channels=input_channels,
        num_classes=config['num_classes'],
        dropout_rate=config.get('dropout_rate', 0.3)
    )
    
    # Print model information
    model_info = get_model_info(model)
    print(f"\nModel: {model_info['architecture']}")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    print(f"Model size: {model_info['model_size_mb']:.2f} MB")
    
    # Calculate class weights for balanced training
    class_weights = calculate_class_weights(train_loader, config['num_classes'], device)
    
    # Create loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    
    # Create save directory
    save_dir = config.get('save_dir', './checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save configuration
    config_save_path = os.path.join(save_dir, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_save_path}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_classes=config['num_classes'],
        scheduler=scheduler,
        device=device,
        save_dir=save_dir,
        log_interval=config.get('log_interval', 10)
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_checkpoint:
        if os.path.exists(resume_checkpoint):
            start_epoch = trainer.load_checkpoint(resume_checkpoint)
            print(f"Resumed training from epoch {start_epoch}")
        else:
            print(f"Checkpoint not found: {resume_checkpoint}")
    
    # Train the model
    print(f"\nStarting training...")
    history = trainer.train(
        num_epochs=config['epoch_max'],
        early_stopping_patience=config.get('early_stopping_patience', 15)
    )
    
    # Plot training curves
    plot_save_path = os.path.join(save_dir, 'training_curves.png')
    trainer.plot_training_curves(save_path=plot_save_path)
    
    # Detailed evaluation
    print("\nPerforming detailed evaluation...")
    eval_results = trainer.evaluate_detailed()
    
    # Save evaluation results
    eval_save_path = os.path.join(save_dir, 'evaluation_results.json')
    with open(eval_save_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"Evaluation results saved to {eval_save_path}")
    
    # Print final results
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Best validation F1 score: {trainer.best_val_f1:.4f}")
    print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
    print(f"Best epoch: {trainer.best_epoch}")
    print(f"Final evaluation accuracy: {eval_results['accuracy']:.4f}")
    print(f"Final evaluation F1 score: {eval_results['f1_avg']:.4f}")
    print(f"Final evaluation AUC: {eval_results['auc']:.4f}")
    
    print(f"\nConfusion Matrix:")

    if config['num_classes'] <= 2:
        cm = np.array(eval_results['confusion_matrix'])
        print(f"                Predicted")
        print(f"              Non-tumor  Tumor")
        print(f"Actual Non-tumor    {cm[0, 0]:5d}  {cm[0, 1]:5d}")
        print(f"       Tumor        {cm[1, 0]:5d}  {cm[1, 1]:5d}")
    
    print(f"\nModel and results saved to: {save_dir}")
    print(f"Best model: {os.path.join(save_dir, 'best_model.pth')}")
    
    return trainer, history, eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train cell classification model')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet'],
                       help='Type of model to use')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    try:
        trainer, history, eval_results = main(
            config_path=args.config,
            model_type=args.model,
            resume_checkpoint=args.resume
        )
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise