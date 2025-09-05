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
from evaluation_metrics import Evaluator
from data.utils import load_samples, create_training_transform, create_validation_transform
from data.data import CellCropsDataset


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def get_multiclass_ct_name(label):

    new_mapping = {
        0: "CD4+ T",
        1: "CD8+ T",
        2: "Treg",
        3: "B cells",
        4: "NK Cells",
        5: "Dendritic Cells",
        6: "Monocytes / Macrophages",
        7: "Stromal Cells",
        8: "Smooth Muscle",
        9: "Tumor Cells",
        10: "Vasculature",
        11: "Granulocytes",
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
    
    use_mask = True

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
        shuffle=False,
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


def main(config_path: str, checkpoint_path: str, model_type: str = 'resnet'):
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
    input_channels = 5 # sample_batch['image'].shape[1] # 
    print(f"Input channels: {input_channels}")
    
    # Create model
    model = create_model(
        model_type=model_type,
        input_channels=input_channels,
        num_classes=config['num_classes'],
        dropout_rate=config.get('dropout_rate', 0.3)
    )
    print(checkpoint_path)
    loaded_checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(loaded_checkpoint['model_state_dict'])

    # Print model information
    model_info = get_model_info(model)
    print(f"\nModel: {model_info['architecture']}")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    print(f"Model size: {model_info['model_size_mb']:.2f} MB")
    
    # Create save directory
    save_dir = config.get('save_dir', './eval_metrics')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save configuration
    config_save_path = os.path.join(save_dir, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_save_path}")
    
    # Create trainer
    evaluator = Evaluator(
        model=model,
        data_loader=train_loader,
        num_classes=config['num_classes'],
        device=device,
        save_dir=save_dir,
        log_interval=config.get('log_interval', 10)
    )
    
    
    # Train the model
    print(f"\nStarting training...")
    history = evaluator.get_training_metrics(save_dir=save_dir)
    print(f"Evaluation metrics results saved to {save_dir}")

    
    return evaluator, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train cell classification model')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet'],
                       help='Type of model to use')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    try:
        trainer, history = main(
            config_path=args.config,
            checkpoint_path=args.model_path,
            model_type=args.model,
        )
        print("Metrics calculated successfully!")
    except Exception as e:
        print(f"Calculation failed with error: {e}")
        raise

