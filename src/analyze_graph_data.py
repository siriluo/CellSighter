# hobbit long pytorch supcontrastive trainer
import os
import sys
import argparse
import time
import math
import csv

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from matplotlib import pyplot as plt
from torch_geometric.loader import HGTLoader, NeighborLoader


# Local imports
from models import create_model, get_model_info
from trainer import Trainer
from data.utils import load_samples, create_training_transform, create_validation_transform
from data.data import CellCropsDataset
from data.graph_data import *
from train import create_data_loaders, load_config, calculate_class_weights, create_optimizer_and_scheduler
from contrastive_learn_add import ContrastiveModel
import json


use_mask = False


def create_contrastive_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
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
    
    # Create transforms
    if config.get('aug', False):
        train_transform = create_training_transform(
            crop_size=config['crop_size'], # potentially replace with config['crop_input_size']
            shift=config.get('shift', 5),
            mask=use_mask,
        )
        print("Using data augmentation for training")
    else:
        train_transform = create_validation_transform(crop_size=config['crop_size'])
        print("No data augmentation applied")
    
    val_transform = create_validation_transform(crop_size=config['crop_size'])

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
    
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    ) 
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'], #  1
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader


def create_contrastive_model(encoder_kwargs, projection_head_kwargs, classification_head_kwargs, model_type: str = 'resnet') -> nn.Module:
    model = ContrastiveModel(
        base_model=model_type,
        encoder_kwargs=encoder_kwargs,
        projection_head_kwargs=projection_head_kwargs,
        classification_head_kwargs=classification_head_kwargs,
        norm_proj_head_input=False,
    )

    return model


def main(config_path: str, args=None):
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
    train_loader, val_loader = create_contrastive_data_loaders(config)

    encoder_kwargs = {
        'in_channel': 3, # 2*
        # 'num_classes': config['num_classes'],
    }
    projection_head_kwargs = {
        'feature_dims': (2048, 128), # resnet18 if resnet34   2048 512
        # 'activation': nn.ReLU(),
        'use_batch_norm': False,
        'normalize_output': True
    }
    classification_head_kwargs = {
        # 'input_dim': 512,
        'num_classes': config['num_classes'],
        'dropout_rate': 0.2,
        'name': 'resnet50',
    }

    model = create_contrastive_model(
        encoder_kwargs=encoder_kwargs,
        projection_head_kwargs=projection_head_kwargs,
        classification_head_kwargs=classification_head_kwargs,
        model_type='resnet'
    )

    graph_stuff = GraphDataConstructor(model)

    distance_val = 20

    node_num_neighbors, avg_node_neighbors = graph_stuff.analyze_dataset(train_loader, distance=distance_val)


    try:
        with open('output_num_neighbors.json', 'w') as f:
            json.dump(node_num_neighbors, f, indent=4)
        print("List of dictionaries successfully saved to output.json")
    except IOError as e:
        print(f"Error writing to file: {e}")

    df = pd.DataFrame(avg_node_neighbors)
    df.to_csv("/projects/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/src/avg_num_neighbors.csv")
    
    return 


main(config_path="/projects/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/src/config_new_more_cts_con_classifier_graph.json")