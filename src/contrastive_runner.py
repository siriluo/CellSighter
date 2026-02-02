#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, random_split
from typing import Dict, Tuple, Any
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Add src to Python path for imports
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

# Local imports
from models import create_model, get_model_info
from contrastive_learn_add import ContrastiveModel, ProjectionHead, ClassificationHead
from gat_model import GATv2ClassificationHead
from contrastive_trainer import ContrastiveTrainer
from contrastive_classifier_trainer import ConClassTrainer
from contrastive_gat_classifier_trainer import ConClassGraphTrainer
from data.utils import load_samples, create_training_transform, create_validation_transform
from data.data import CellCropsDataset
from train import get_multiclass_ct_name, load_config, create_data_loaders, calculate_class_weights
from data.custom_samplers import TwoStageBalancedSampler
from contrastive_losses import MultiPosConLoss, SupConLoss
from util.utils import TwoCropTransform


use_mask = True # False set to true if you want to include the mask info
cifar = False

model_dict = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'convnextv2_tiny': 768,
}

def create_contrastive_model(encoder_kwargs, projection_head_kwargs, classification_head_kwargs, model_type: str = 'resnet', model_name: str = 'resnet18') -> nn.Module:
    model = ContrastiveModel(
        base_model=model_type,
        encoder_kwargs=encoder_kwargs,
        projection_head_kwargs=projection_head_kwargs,
        classification_head_kwargs=classification_head_kwargs,
        norm_proj_head_input=False,
        model_name=model_name
    )

    return model


def create_optimizer_and_scheduler(model: nn.Module, config: Dict[str, Any]) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """Create optimizer and learning rate scheduler."""
    # Optimizer
    # if not cifar:
    #     optimizer = optim.Adam(
    #         model.parameters(),
    #         lr=config['lr'],
    #         weight_decay=config.get('weight_decay', 1e-4) #  1e-5
    #     )
    #     print("Adam")
    # else:

    if config['classifier']:
        optimizer = optim.SGD(model.parameters(),
                        lr=config['lr'],
                        momentum=0.9,
                        weight_decay=1e-4)
        print("SGD")
    else:
        ## Test optimizer with multiple LRs
        # optimizer = optim.SGD([
        #         {'params': model.encoder.parameters(), 'lr': config['lr'], 'name': 'encoder'},
        #         {'params': model.projection_head.parameters(), 'lr': config['proj_lr'], "weight_decay": 0.0, 'name': 'projection_head'},],
        #         momentum=0.9,
        #         weight_decay=1e-4)
        optimizer = optim.SGD(model.parameters(),
                lr=config['lr'],
                momentum=0.9,
                weight_decay=1e-4)
        print("SGD with different LRs")

    # # Learning rate scheduler
    # scheduler = optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=config.get('lr_step_size', 40), # 10 20
    #     gamma=config.get('lr_gamma', 0.2)  #  0.5
    # )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.get('epoch_max', 100), eta_min=1e-6
    )
    
    return optimizer, scheduler


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
    sample_image = dataset[0]['image'][0]
    print(f"Image crops (views): {len(dataset[0]['image'])}")
    print(f"Image shape: {sample_image.shape}")
    print(f"Image dtype: {sample_image.dtype}")


def set_loader(config: Dict[str, Any]):
    # construct data loader

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    normalize = torchvision.transforms.Normalize(mean=mean, std=std)

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        normalize,
    ])

    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        normalize,
    ])


    cifar_dataset = torchvision.datasets.CIFAR10(root='/projects/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/src/data/cifar_10',
                                        transform=train_transform,
                                        download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='/projects/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/src/data/cifar_10',
                                    train=False,
                                    transform=val_transform)
        
    train_size = int(0.8 * len(cifar_dataset))
    val_size = len(cifar_dataset) - train_size

    RANDOM_SEED = 42 
    train_dataset, val_dataset = random_split(cifar_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(RANDOM_SEED))

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=(train_sampler is None),
        num_workers=config['num_workers'], pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=config['num_workers'], pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False,
        num_workers=config['num_workers'], pin_memory=True)

    return train_loader, val_loader, test_loader


def create_contrastive_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # In this case, we can get the image names by looping through the files instead for our situation: 
    # folds = ["fold1"]
    # for fold in folds:
    #     fold_path = f"{patches_path}/{fold}"

    #     os.makedirs(f"{save_cs_path}/cells2labels/{fold}", exist_ok=True)

    #     fold_file_names = os.listdir(fold_path)
    #     fold_file_names = [f for f in fold_file_names if os.path.isfile(fold_path + "/" + f)]
    # image_names = fold_file_names
    use_xenium = config.get("xenium", False)

    if use_xenium:
        root_dir = config["root_dir"]

        folds = config.get("xenium_fold", [])
        val_folds = ["fold4"]
        
        train_image_names = []
        for fold in folds:
            fold_path = f"{root_dir}/CellTypes/cells2labels/{fold}"

            tr_fold_file_names = os.listdir(fold_path)
            tr_fold_file_names = [f"{fold}/{f.split('.')[0]}" for f in tr_fold_file_names if os.path.isfile(fold_path + "/" + f)]
            train_image_names.extend(tr_fold_file_names)


        val_image_names = []
        for fold in val_folds:
            fold_path = f"{root_dir}/CellTypes/cells2labels/{fold}"

            val_fold_file_names = os.listdir(fold_path)
            val_fold_file_names = [f"{fold}/{f.split('.')[0]}" for f in val_fold_file_names if os.path.isfile(fold_path + "/" + f)]
            val_image_names.extend(val_fold_file_names)
    else:
        train_image_names = config['train_set']
        val_image_names = config['val_set'] 

    print("Loading training data...")
    train_crops = load_samples(config, train_image_names, already_cropped=use_xenium, testing=use_xenium)
    print(f"Loaded {len(train_crops)} training samples")
    
    print("Loading validation data...")
    val_crops = load_samples(config, val_image_names, already_cropped=use_xenium, testing=use_xenium)
    print(f"Loaded {len(val_crops)} validation samples")
    
    # Create transforms
    if config.get('aug', False):
        train_transform = create_training_transform(
            crop_size=config['crop_input_size'], # crop_input_size crop_size potentially replace with config['crop_input_size']
            shift=config.get('shift', 5),
            mask=use_mask,
        )
        print("Using data augmentation for training")
    else:
        train_transform = create_validation_transform(crop_size=config['crop_size'])
        print("No data augmentation applied")
    
    val_transform = create_validation_transform(crop_size=config['crop_input_size'])

    # no_op_transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Lambda(lambda x: x)
    # ])
    
    # Create datasets
    if config['classifier']:
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
    else:
        train_dataset = CellCropsDataset(
            crops=train_crops,
            transform=TwoCropTransform(train_transform),
            mask=use_mask,  # Set to True if you want to include mask information
            contrastive=True,
        )
        
        val_dataset = CellCropsDataset(
            crops=val_crops,
            transform=TwoCropTransform(val_transform), # train_transform val_transform
            mask=use_mask,
            contrastive=True,
        )
    
    # Print dataset statistics
    print_dataset_stats(train_dataset, "Training")
    print_dataset_stats(val_dataset, "Validation")
    
    train_dataset_labels = train_dataset._labels

    # Create data loaders
    if config.get('cifar', False):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        normalize = torchvision.transforms.Normalize(mean=mean, std=std)

        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomApply([
                torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.ToTensor(),
            normalize,
        ])

        cifar_dataset = torchvision.datasets.CIFAR10(root='/projects/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/src/data/cifar_10',
                                    transform=TwoCropTransform(train_transform),
                                    download=True)
        
        train_size = int(0.8 * len(cifar_dataset))
        val_size = len(cifar_dataset) - train_size

        RANDOM_SEED = 42 
        train_dataset, val_dataset = random_split(cifar_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(RANDOM_SEED))
    
    use_graph = config.get('graph', False)
    if not use_graph:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            # sampler=torch.utils.data.RandomSampler(train_dataset, num_samples=len(train_dataset), replacement=True),
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True if torch.cuda.is_available() else False
        )

        # cust_sampler = TwoStageBalancedSampler(train_dataset_labels, batch_size=config['batch_size'], balance_threshold=0.6)
        # train_loader = DataLoader(
        #     train_dataset,
        #     batch_sampler=cust_sampler,
        #     num_workers=config['num_workers'],
        #     pin_memory=True if torch.cuda.is_available() else False
        # )
    else:
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


def main(config_path: str, model_type: str = 'cnn', resume_checkpoint: str = None, args=None):
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
    if (not args.cifar) or (not args.classifier):
        train_loader, val_loader = create_contrastive_data_loaders(config)
        test_loader = None
    else:
        train_loader, val_loader, test_loader = set_loader(config)
    # Get input channels from a sample
    # sample_batch = next(iter(train_loader))

    if use_mask:
        input_channels = 5
    else:
        input_channels = 3 # sample_batch['image'][0].shape[1] #  5 #

    if args.cifar:
        input_channels = 3
    print(f"Input channels: {input_channels}")
    
    # Create model
    # create_contrastive_model
    chosen_model = 'resnet34' # 'convnextv2_tiny' resnet18
    encoder_kwargs = {
        'in_channel': input_channels, # 2*
        # 'num_classes': config['num_classes'],
    }
    projection_head_kwargs = {
        'feature_dims': (model_dict[chosen_model], 128), # resnet18 if resnet34   2048 512 ConvNeXtV2: 768 256
        # 'activation': nn.ReLU(),
        'use_batch_norm': True, # True False
        'normalize_output': True
    }
    classification_head_kwargs = {
        # 'input_dim': 512,
        'num_classes': config['num_classes'],
        'dropout_rate': 0.7,
        'name': chosen_model, # resnet50 resnet18
    }
    model = create_contrastive_model(
        encoder_kwargs=encoder_kwargs,
        projection_head_kwargs=projection_head_kwargs,
        classification_head_kwargs=classification_head_kwargs,
        model_type='resnet',
        model_name=chosen_model
    )

    if args.classifier:
        use_graph = config.get('graph', False)
        if not use_graph:
            classifier = ClassificationHead(**classification_head_kwargs)
        else:
            classifier = GATv2ClassificationHead(**classification_head_kwargs)
        # state_dict = torch.load('/projects/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/results_conclass_mask_test_aug/best_model.pth')
        # classifier.load_state_dict(state_dict['model_state_dict'])
        # print("Loaded classifier weights from checkpoint")

    # Print model information
    model_info = get_model_info(model)
    print(f"\nModel: {model_info['architecture']}")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    print(f"Model size: {model_info['model_size_mb']:.2f} MB")
    
    # Calculate class weights for balanced training
    if args.cifar == False:
        class_weights = calculate_class_weights(train_loader, config['num_classes'], device)
    
    # Create loss function with class weights
    if not args.classifier:
        criterion = SupConLoss(temperature=0.13) # try default 0.07 #  temperature=0.07, 0.1, 0.13, 0.15, 0.2 25 
    else:
        if args.cifar == False:
            criterion = nn.CrossEntropyLoss(weight=class_weights) # 
        else:
            criterion = nn.CrossEntropyLoss()
    
    # Create optimizer and scheduler
    if not args.classifier:
        optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    else:
        optimizer, scheduler = create_optimizer_and_scheduler(classifier, config)
    
    # Create save directory
    save_dir = config.get('save_dir', './checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save configuration
    config_save_path = os.path.join(save_dir, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_save_path}")
    
    # Create trainer
    if not args.classifier:
        trainer = ContrastiveTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_classes=config['num_classes'],
            scheduler=scheduler,
            device=device,
            save_dir=save_dir,
            log_interval=config.get('log_interval', 50),
            args=args
        )
    else:
        use_graph = config.get('graph', False)
        if not use_graph:
            trainer = ConClassTrainer(
                model=model,
                encoder_ckpt_path=config["ckpt_path"],
                classifier=classifier,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                num_classes=config['num_classes'],
                scheduler=scheduler,
                device=device,
                save_dir=save_dir,
                log_interval=config.get('log_interval', 50),
                args=args
            )
        else:
            trainer = ConClassGraphTrainer(
                model=model,
                encoder_ckpt_path=config["ckpt_path"],
                classifier=classifier,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                num_classes=config['num_classes'],
                scheduler=scheduler,
                device=device,
                save_dir=save_dir,
                log_interval=config.get('log_interval', 50),
                args=args
            )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_checkpoint:
        if os.path.exists(resume_checkpoint):
            start_epoch = trainer.load_checkpoint(resume_checkpoint)
            print(f"Resumed training from epoch {start_epoch}")
        else:
            print(f"Checkpoint not found: {resume_checkpoint}")

    # history = {}
    # eval_results = {}

    # trainer.debug_gat()
    
    # Train the model
    print(f"\nStarting training...")
    print()
    history = trainer.train(
        num_epochs=config['epoch_max'],
        early_stopping_patience=config.get('early_stopping_patience', 20)
    )
    
    # Plot training curves
    plot_save_path = os.path.join(save_dir, 'training_curves.png')
    trainer.plot_training_curves(save_path=plot_save_path)
    
    eval_results = {}
    # Detailed evaluation
    print("\nPerforming detailed evaluation...")
    eval_results = trainer.evaluate_detailed(test_loader=test_loader)
    
    # Save evaluation results
    eval_save_path = os.path.join(save_dir, 'evaluation_results.json')
    with open(eval_save_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"Evaluation results saved to {eval_save_path}")
    
    # Print final results
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED")
    print(f"{'='*60}")
    # print(f"Best validation F1 score: {trainer.best_val_f1:.4f}")
    # print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
    print(f"Best epoch: {trainer.best_epoch}")
    # print(f"Final evaluation accuracy: {eval_results['accuracy']:.4f}")
    # print(f"Final evaluation F1 score: {eval_results['f1_avg']:.4f}")
    # print(f"Final evaluation AUC: {eval_results['auc']:.4f}")
    
    # print(f"\nConfusion Matrix:")

    # if config['num_classes'] <= 2:
    #     cm = np.array(eval_results['confusion_matrix'])
    #     print(f"                Predicted")
    #     print(f"              Non-tumor  Tumor")
    #     print(f"Actual Non-tumor    {cm[0, 0]:5d}  {cm[0, 1]:5d}")
    #     print(f"       Tumor        {cm[1, 0]:5d}  {cm[1, 1]:5d}")
    
    print(f"\nModel and results saved to: {save_dir}")
    print(f"Best model: {os.path.join(save_dir, 'best_model.pth')}")
    
    return trainer, history, eval_results