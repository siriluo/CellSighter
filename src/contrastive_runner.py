#!/usr/bin/env python3

import os
import sys
import argparse
import glob
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
from contrastive_learn_add import ClassificationHead2, ContrastiveModel, ProjectionHead, ClassificationHead
from gat_model import GATv2ClassificationHead
from adversarial_contrastive_trainer import AdversarialContrastiveTrainer
from contrastive_trainer import ContrastiveTrainer
from contrastive_classifier_trainer import ConClassTrainer
from contrastive_gat_classifier_trainer import ConClassGraphTrainer
from data.utils import load_samples, create_training_transform, create_validation_transform, build_optimizer_stage1
from data.data import CellCropsDataset
from data.orion_data_processing import load_cell_crops_from_orion
from train import get_multiclass_ct_name, load_config, create_data_loaders, calculate_class_weights
from data.custom_samplers import TwoStageBalancedSampler, ClassAwareSupConBatchSampler
from contrastive_losses import MultiPosConLoss, SupConLoss
from domain_adaptation import DomainDiscriminator
from util.utils import TwoCropTransform


use_mask = True # False set to true if you want to include the mask info
cifar = False

model_dict = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'convnextv2_tiny': 768,
    'new_fused': 512
}

def create_contrastive_model(encoder_kwargs, projection_head_kwargs, classification_head_kwargs, model_type: str = 'resnet', model_name: str = 'resnet18') -> nn.Module:
    model = ContrastiveModel(
        base_model=model_type,
        encoder_kwargs=encoder_kwargs,
        projection_head_kwargs=projection_head_kwargs,
        classification_head_kwargs=classification_head_kwargs,
        norm_proj_head_input=False,
        model_name=model_name,
        pretrained=True
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
        # optimizer = optim.SGD(model.parameters(),
        #                 lr=config['lr'],
        #                 momentum=0.9,
        #                 weight_decay=1e-4)
        # print("SGD")
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 1e-4) #  1e-5
        )
        print("Adam")
    else:
        # optimizer = optim.SGD(model.parameters(),
        #         lr=config['lr'],
        #         momentum=0.9,
        #         weight_decay=1e-4)
        # print("SGD with different LRs")
        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 1e-4) #  1e-5
        )
        print("Adam")
        
    # if config.get('pretrained_test', False):
    #     optimizer = build_optimizer_stage1(
    #         model,
    #         head_lr=config['proj_lr'],
    #         weight_decay=1e-4,
    #     )

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


def create_contrastive_data_loaders(config: Dict[str, Any], uni_transform=None) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # In this case, we can get the image names by looping through the files instead for our situation: 
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
            use_uni=config.get('use_uni', False),
            uni_transform=uni_transform
        )
        print("Using data augmentation for training")
    else:
        train_transform = create_validation_transform(crop_size=config['crop_size'], use_uni=config.get('use_uni', False), uni_transform=uni_transform)
        print("No data augmentation applied")
    
    val_transform = create_validation_transform(crop_size=config['crop_input_size'], use_uni=config.get('use_uni', False), uni_transform=uni_transform)
    
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
    
    # train_dataset_labels = train_dataset._labels

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
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True
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
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True
        ) 
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'], #  1
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True
    )
    
    return train_loader, val_loader


def create_adversarial_target_loader(config: Dict[str, Any], uni_transform=None) -> DataLoader:
    """
    Create an unlabeled target-domain loader for DANN.

    The existing crop loader expects labels to exist in the source data format, but
    this path ignores target labels and only uses the crops for domain alignment.
    """
    adv_config = config.get("adversarial", {})
    target_config = dict(config)

    target_config_path = adv_config.get("target_config_path")
    if target_config_path:
        with open(target_config_path, "r") as f:
            target_config.update(json.load(f))

    if adv_config.get("target_root_dir"):
        target_config["root_dir"] = adv_config["target_root_dir"]

    target_set = (
        adv_config.get("target_set")
        or target_config.get("target_set")
        or build_pannuke_target_set(target_config, adv_config)
        or target_config.get("train_set")
    )
    if not target_set:
        raise ValueError(
            "Adversarial training requires adversarial.target_set, "
            "adversarial.pannuke_folds, target_set, or a target_config_path with train_set."
        )

    use_xenium = target_config.get("xenium", False)
    target_already_cropped = adv_config.get(
        "target_already_cropped",
        target_config.get("target_already_cropped", True),
    )
    target_testing = adv_config.get(
        "target_testing",
        target_config.get("target_testing", True),
    )

    target_crops = load_samples(
        target_config,
        target_set,
        already_cropped=target_already_cropped,
        testing=target_testing,
        coords_path=None,  # Ensure coords_path is None to avoid loading coordinates for target domain
    )
    if len(target_crops) == 0:
        raise ValueError("Adversarial target loader found 0 target-domain crops.")
    print(f"Loaded {len(target_crops)} target-domain samples for adversarial adaptation")

    if config.get("aug", False):
        target_transform = create_training_transform(
            crop_size=config["crop_input_size"],
            shift=config.get("shift", 5),
            mask=use_mask,
            use_uni=config.get("use_uni", False),
            uni_transform=uni_transform,
        )
    else:
        target_transform = create_validation_transform(
            crop_size=config["crop_input_size"],
            use_uni=config.get("use_uni", False),
            uni_transform=uni_transform,
        )

    target_dataset = CellCropsDataset(
        crops=target_crops,
        transform=TwoCropTransform(target_transform),
        mask=use_mask,
        contrastive=True,
    )

    return DataLoader(
        target_dataset,
        batch_size=adv_config.get("target_batch_size", config["batch_size"]),
        shuffle=True,
        num_workers=adv_config.get("target_num_workers", config["num_workers"]),
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=adv_config.get("target_drop_last", False),
        persistent_workers=True
    )


def build_pannuke_target_set(target_config: Dict[str, Any], adv_config: Dict[str, Any]):
    pannuke_folds = adv_config.get("pannuke_folds") or target_config.get("pannuke_folds")
    if not pannuke_folds:
        return None

    cells_dir = os.path.join(target_config["root_dir"], "CellTypes", "cells")
    target_set = []
    for fold in pannuke_folds:
        fold_name = str(fold)
        if fold_name.startswith("f"):
            fold_name = fold_name[1:]
        pattern = os.path.join(cells_dir, f"pannuke_f{fold_name}_*.npz")
        fold_ids = [os.path.splitext(os.path.basename(path))[0] for path in glob.glob(pattern)]
        fold_ids = sorted(fold_ids, key=pannuke_image_sort_key)
        print(f"Found {len(fold_ids)} PanNuke target images for fold {fold_name}")
        target_set.extend(fold_ids)

    return target_set


def pannuke_image_sort_key(image_id: str):
    parts = image_id.split("_")
    try:
        return int(parts[-1])
    except ValueError:
        return image_id


def create_orion_data_loaders(config: Dict[str, Any], uni_transform = None) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # In this case, we can get the image names by looping through the files instead for our situation:
    cell_patches_path = config["root_dir"]

    # set random seed for reproducibility
    np.random.seed(42)

    # First get the list of folders and shuffle them to ensure random distribution of samples across folds
    folders = glob.glob("CRC*", root_dir=cell_patches_path)
    perm_indices = np.random.permutation(len(folders))

    folders_perm = np.array(folders)
    folders_perm = folders_perm[perm_indices]

    # Then split into folds based on this.
    test_crc_samples = folders_perm[32:len(folders)]
    print(test_crc_samples)
    print("fold 2")
    train_val_samples = folders_perm[:32]

    folds = 4
    splits = np.split(train_val_samples, folds)

    val_fold = 3  # fold1: 3, fold2: 0, fold3: 1, fold4: 2
    crc_samples = []
    for i in range(folds):
        if i != val_fold:
            crc_samples.append(splits[i])
    crc_samples = np.concatenate(crc_samples)

    val_crc_samples = splits[val_fold]

    # print(folders)

    # crc_samples = ["CRC01", "CRC02", "CRC04", "CRC05", "CRC06", "CRC09", "CRC10", "CRC11", "CRC12", "CRC13",
    #                "CRC14", "CRC15", "CRC16", "CRC17", "CRC18", "CRC19", "CRC20"] # , "CRC04"
    # val_crc_sample = "CRC07"

    # The data is numbered 00000
    mask_name = "cell_masks"
    img_patch_name = "image_patches"
    labels_name = "meta"

    # count
    print("Loading testing data...")
    sample_fraction = config.get("orion_sample_fraction", None)
    train_sample_fraction = config.get("orion_train_sample_fraction", sample_fraction)
    val_sample_fraction = config.get("orion_val_sample_fraction", sample_fraction)
    train_max_per_sample = config.get("orion_train_max_per_sample", None)
    val_max_per_sample = config.get("orion_val_max_per_sample", None)
    sample_seed = config.get("orion_sample_seed", 42)

    training_crops = []
    for sample_idx, sample in enumerate(crc_samples):
        filelist = glob.glob(f"{cell_patches_path}/{sample}/{labels_name}_*.csv")
        crops = load_cell_crops_from_orion(f"{cell_patches_path}/{sample}", mask_name, img_patch_name, labels_name,
                                           filelist,
                                           sample_fraction=train_sample_fraction,
                                           max_samples=train_max_per_sample,
                                           sample_seed=sample_seed + sample_idx)
        training_crops.extend(crops)
        print(sample)

    test_crops = []
    for sample_idx, sample in enumerate(val_crc_samples):
        validation_filelist = glob.glob(f"{cell_patches_path}/{sample}/{labels_name}_*.csv")
        val_crops = load_cell_crops_from_orion(f"{cell_patches_path}/{sample}", mask_name, img_patch_name, labels_name,
                                               validation_filelist,
                                               sample_fraction=val_sample_fraction,
                                               max_samples=val_max_per_sample,
                                               sample_seed=sample_seed + 10000 + sample_idx)
        test_crops.extend(val_crops)
        print(sample)

    print(f"Loaded {len(test_crops)} testing samples")

    # Create transforms
    if config.get('aug', False):
        train_transform = create_training_transform(
            crop_size=config['crop_input_size'],
            # crop_input_size crop_size potentially replace with config['crop_input_size']
            shift=config.get('shift', 5),
            mask=use_mask,
            use_uni=config.get('use_uni', False), uni_transform=uni_transform
        )
        print("Using data augmentation for training")
    else:
        train_transform = create_validation_transform(crop_size=config['crop_size'], use_uni=config.get('use_uni', False), uni_transform=uni_transform)
        print("No data augmentation applied")

    test_transform = create_validation_transform(crop_size=config['crop_input_size'], use_uni=config.get('use_uni', False), uni_transform=uni_transform)

    # Create datasets
    if config['classifier']:
        train_dataset = CellCropsDataset(
            crops=training_crops,
            transform=train_transform,
            mask=use_mask,
            contrastive=False,
        )
        test_dataset = CellCropsDataset(
            crops=test_crops,
            transform=test_transform,
            mask=use_mask,
            contrastive=False,
        )
    else:
        train_dataset = CellCropsDataset(
            crops=training_crops,
            transform=TwoCropTransform(train_transform), # train_transform, #
            mask=use_mask, 
            contrastive=True,
        )
        test_dataset = CellCropsDataset(
            crops=test_crops,
            transform=TwoCropTransform(test_transform), # test_transform, #
            mask=use_mask,
            contrastive=True,
        )

    # Create data loaders

    sampling_cfg = config.get("batch_sampling", {})
    use_class_aware_sampling = (
        sampling_cfg.get("enabled", False)
        and sampling_cfg.get("strategy", "class_aware_supcon") == "class_aware_supcon"
        and not config["classifier"]
    )
    
    print(f"Using class-aware batch sampling: {use_class_aware_sampling}")
    if use_class_aware_sampling:
        print("Using class-aware batch sampling for SupCon training")
        batch_sampler = ClassAwareSupConBatchSampler(
            labels=train_dataset._labels,
            batch_size=config["batch_size"],
            samples_per_class=sampling_cfg.get("samples_per_class", 64),
            classes_per_batch=sampling_cfg.get("classes_per_batch", config["num_classes"]),
            oversample=sampling_cfg.get("oversample", True),
            num_batches=sampling_cfg.get("num_batches", None),
            seed=sampling_cfg.get("seed", 42),
            drop_incomplete=sampling_cfg.get("drop_incomplete", True),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=config["num_workers"],
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True,
        )
    else: 
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True
        )
    
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],  # 1
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True
    )

    return train_loader, test_loader


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

    if use_mask:
        input_channels = 5
    else:
        input_channels = 3 # sample_batch['image'][0].shape[1] #  5 #

    if args.cifar:
        input_channels = 3
    print(f"Input channels: {input_channels}")
    
    # Create model
    # create_contrastive_model
    chosen_model = 'new_fused' # 'convnextv2_tiny' resnet34 resnet18 resnet50 new_fused
    encoder_kwargs = {
        'in_channel': input_channels, # 2*
        # 'num_classes': config['num_classes'],
    }
    if chosen_model == 'new_fused':
        encoder_kwargs['backbone'] = 'uni2h' # resnet50 dinov2_vitb14 uni2h
        encoder_kwargs['freeze_backbone'] = True # True False
        
    projection_head_kwargs = {
        'feature_dims': (model_dict[chosen_model], 128), # resnet18 if resnet34   2048 512 ConvNeXtV2: 768 256
        # 'activation': nn.ReLU(),
        'use_batch_norm': True, # True False
        'normalize_output': True
    }
    
    classification_head_kwargs = {
        # 'input_dim': 512,
        'num_classes': config['num_classes'],
        'dropout_rate': 0.2,
        'name': chosen_model, # resnet50 resnet18
    }
    
    model = create_contrastive_model(
        encoder_kwargs=encoder_kwargs,
        projection_head_kwargs=projection_head_kwargs,
        classification_head_kwargs=classification_head_kwargs,
        model_type='new_fused', # resnet
        model_name=chosen_model
    )

    if args.classifier:
        use_graph = config.get('graph', False)
        if not use_graph:
            classifier = ClassificationHead(**classification_head_kwargs)
            # classifier = ClassificationHead2(in_dim=model_dict[chosen_model], n_classes=config['num_classes'])
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
    
    uni_transform = model.encoder.uni_transform
    
    # Create data loaders
    if (not args.cifar) or (not args.classifier):
        if not config.get("orion", False):
            train_loader, val_loader = create_contrastive_data_loaders(config, uni_transform=uni_transform)
        else: 
            train_loader, val_loader = create_orion_data_loaders(config, uni_transform=uni_transform)
        test_loader = None
    else:
        train_loader, val_loader, test_loader = set_loader(config)

    use_adversarial = config.get("adversarial", {}).get("enabled", False)
    target_loader = None
    if use_adversarial:
        if args.classifier:
            raise ValueError("Adversarial adaptation is currently implemented for SupCon encoder training only.")
        if args.cifar:
            raise ValueError("Adversarial adaptation is intended for H&E cell crops, not the CIFAR debug path.")
        target_loader = create_adversarial_target_loader(config, uni_transform=uni_transform)

    # Get input channels from a sample
    # sample_batch = next(iter(train_loader))
    
    # Class weights are only used for classifier CE training; computing them
    # iterates the whole loader and is very slow for Orion SupCon/ADA debug runs.
    class_weights = None
    if args.classifier and args.cifar == False:
        class_weights = calculate_class_weights(train_loader, config['num_classes'], device)
    
    # Create loss function with class weights
    # 0.1 or 0.07 seems to perform best?
    # and try switching back to resnet50
    if not args.classifier:
        criterion = SupConLoss(temperature=0.15) # try default 0.07 #  temperature=0.07, 0.1, 0.13, 0.15, 0.2 25 
    else:
        if args.cifar == False:
            criterion = nn.CrossEntropyLoss(weight=class_weights) # 
        else:
            criterion = nn.CrossEntropyLoss()
    
    # Create optimizer and scheduler
    domain_discriminator = None
    if use_adversarial:
        adv_config = config.get("adversarial", {})
        domain_discriminator = DomainDiscriminator(
            in_dim=model_dict[chosen_model],
            hidden_dim=adv_config.get("domain_hidden_dim", 256),
            num_domains=adv_config.get("num_domains", 2),
            dropout=adv_config.get("domain_dropout", 0.2),
        )
        optimizer = optim.Adam(
            list(model.parameters()) + list(domain_discriminator.parameters()),
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 1e-4),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.get("epoch_max", 100), eta_min=1e-6
        )
        print("Adam with domain discriminator")
    elif not args.classifier:
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
    if use_adversarial:
        trainer = AdversarialContrastiveTrainer(
            model=model,
            domain_discriminator=domain_discriminator,
            target_loader=target_loader,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_classes=config["num_classes"],
            scheduler=scheduler,
            device=device,
            save_dir=save_dir,
            log_interval=config.get("log_interval", 50),
            args=args,
            config=config,
        )
    elif not args.classifier:
        trainer = ContrastiveTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader, # train_loader val_loader
            criterion=criterion,
            optimizer=optimizer,
            num_classes=config['num_classes'],
            scheduler=scheduler,
            device=device,
            save_dir=save_dir,
            log_interval=config.get('log_interval', 50),
            args=args,
            config=config
        )
    else:
        use_graph = config.get('graph', False)
        if not use_graph:
            trainer = ConClassTrainer(
                model=model,
                encoder_ckpt_path=config["ckpt_path"],
                classifier=classifier,
                train_loader=train_loader,
                val_loader=val_loader, # train_loader val_loader
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

    # trainer.debug_gat()
    
    # Train the model
    print(f"\nStarting training...")
    if not config.get('pretrained_test', False):
        history = trainer.train(
            num_epochs=config['epoch_max'],
            early_stopping_patience=config.get('early_stopping_patience', 20)
        )
    else:
        history = trainer.train_pretrained(
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
