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
import glob
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from data.utils import convert_to_simpler_labels, topk_accuracy, pr_auc_score



# Add src to Python path for imports
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

# Local imports
from models import create_model, get_model_info
from contrastive_learn_add import ContrastiveModel, ProjectionHead, ClassificationHead
from gat_model import GATv2ClassificationHead
# from contrastive_trainer import ContrastiveTrainer
# from contrastive_classifier_trainer import ConClassTrainer
# from contrastive_gat_classifier_trainer import ConClassGraphTrainer
from data.utils import load_samples, create_training_transform, create_validation_transform
from data.orion_data_processing import load_cell_crops_from_orion
from data.data import CellCropsDataset
from train import get_multiclass_ct_name, load_config, create_data_loaders, calculate_class_weights
from data.graph_data import GraphDataConstructor
from contrastive_losses import MultiPosConLoss, SupConLoss


use_mask = True # False set to true if you want to include the mask info
cifar = False

model_dict = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'convnextv2_tiny': 768,
    'new_fused': 512,
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


def inside_eval(config, list_of_logits, list_of_labels, list_of_probs, num_classes):
    # list_of_logits = torch.tensor(list_of_logits).squeeze(1)
    # list_of_labels = torch.tensor(list_of_labels).squeeze(1)
    topk_accs = topk_accuracy(list_of_logits, list_of_labels, ks=[1, 3, 5])
    
    save_path = config.get('save_dir', './test_checkpoints')
    with open(f"{save_path}/topk_accs.json", 'w') as f:
        json.dump(topk_accs, f, indent=4)
        
    list_logits_np = list_of_logits.numpy()
    np.savez(f"{save_path}/list_of_logits.npz", list_logits_np)
        
    list_labels_np = list_of_labels.numpy()
    np.savez(f"{save_path}/list_of_labels.npz", list_labels_np)
    
    # if config.get("simpler_labels", False):
    #     all_orig_labels = all_labels.copy()
    all_labels = list_of_labels
    all_preds = list_of_probs.argmax(dim=1).numpy()
    all_probs = list_of_probs.numpy()
    
    accuracy = accuracy_score(all_labels, all_preds)

    if num_classes <= 2:
        average_method = 'binary'
    else:
        average_method = 'weighted'

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Overall metrics
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=average_method, zero_division=0
    )
    # precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
    #     all_labels, all_preds, average=average_method, zero_division=0, pos_label=0
    # )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # AUC
    try:
        if num_classes <= 2:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            # # Fix the multi aucs
            # if config.get("simpler_labels", False):
            #     all_labels = all_orig_labels
                
            all_probs = np.vstack(all_probs)
            
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovo', average='weighted')
            multi_aucs = roc_auc_score(all_labels, all_probs, multi_class='ovr', average=None)
            
            pr_auc = pr_auc_score(all_labels, all_probs, average='weighted')
            multi_pr_aucs = pr_auc_score(all_labels, all_probs, average=None)
    except ValueError:
        auc = 0.0  # Handle case where only one class is present
        if num_classes > 2:
            multi_aucs = []


    # with open('output_logits.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerows(list_of_logits)

    # with open('output_probs.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerows(list_of_probs)

    # with open('output_labels.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerows(list_of_labels)

    results = {
        'accuracy': accuracy,
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1.tolist(),
        'support_per_class': support.tolist(),
        'precision_avg': precision_avg,
        'recall_avg': recall_avg,
        'f1_avg': f1_avg,
        'auc': auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm.tolist(),
    }

    if num_classes > 2:
        if isinstance(multi_aucs, list):
            multi_aucs_list = multi_aucs
            multi_pr_aucs_list = multi_pr_aucs
        else:
            multi_aucs_list = multi_aucs.tolist()
            multi_pr_aucs_list = multi_pr_aucs.tolist()
        results.update({'multi_class_aucs': multi_aucs_list, 'multi_class_pr_aucs': multi_pr_aucs_list})
            
    return results


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
    use_xenium = config.get("xenium", False)
    
    if use_xenium:
        folds = config.get("xenium_fold", None)
        
        if folds[0] is not None:
            root_dir = config["root_dir"]
            for fold in folds:
                fold_path = f"{root_dir}/CellTypes/cells2labels/{fold}"

                fold_file_names = os.listdir(fold_path)
                fold_file_names = [f"{fold}/{f.split('.')[0]}" for f in fold_file_names if os.path.isfile(fold_path + "/" + f)]
            image_names = fold_file_names
        else:
            root_dir = config["root_dir"]
            file_name_path = f"{root_dir}/CellTypes/cells2labels"

            file_names = os.listdir(file_name_path)
            file_names = [f"{f.split('.')[0]}" for f in file_names if os.path.isfile(file_name_path + "/" + f)]
            image_names = file_names
    else:
        image_names = config['test_set'] 


    print("Loading testing data...")
    test_crops = load_samples(config, image_names, already_cropped=use_xenium, testing=use_xenium)
    # test_crops = load_samples(config, image_names, testing=True)
    print(f"Loaded {len(test_crops)} testing samples")

    # Create transforms
    test_transform = create_validation_transform(crop_size=config['crop_input_size'])
    
    # Create datasets
    test_dataset = CellCropsDataset(
        crops=test_crops,
        transform=test_transform,
        mask=use_mask
    )
    
    print_dataset_stats(test_dataset, "Testing")
    
    # Create data loaders
    use_graph = config.get('graph', False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'], #  1
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return test_loader


def create_orion_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
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
    # /taiga/illinois/vetmed/cb/kwang222/mz_jason/orion_all_without_largest/_meta/cell_labeling/cell_patches_64_match5um_area50_3000
    folders = glob.glob("CRC*", root_dir=cell_patches_path)
    perm_indices = np.random.permutation(len(folders))
    
    folders_perm = np.array(folders)
    folders_perm = folders_perm[perm_indices]
    
    # Then split into folds based on this.
    test_crc_samples = folders_perm[32:len(folders)]
    test_crc_samples_single = [test_crc_samples[4]] 

    # The data is numbered 00000
    mask_name = "cell_masks"
    img_patch_name = "image_patches"
    labels_name = "meta"

    # count
    print("Loading testing data...")
    test_crops = []
    for sample in test_crc_samples_single:
        filelist = glob.glob(f"{cell_patches_path}/{sample}/{labels_name}_*.csv")
        crops = load_cell_crops_from_orion(f"{cell_patches_path}/{sample}", mask_name, img_patch_name, labels_name, filelist)
        test_crops.extend(crops)
    print(f"Loaded {len(test_crops)} testing samples")

    # Create transforms
    test_transform = create_validation_transform(crop_size=config['crop_input_size'])
    
    # Create datasets
    test_dataset = CellCropsDataset(
        crops=test_crops,
        transform=test_transform,
        mask=use_mask,
        contrastive=False,
    )
    
    # Create data loaders
    use_graph = config.get('graph', False)


    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'], #  1
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return test_loader


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
    
    if not config.get("orion", False):
        test_loader = create_contrastive_data_loaders(config)
    else:
        test_loader = create_orion_data_loaders(config)
        
    # Get input channels from a sample
    if use_mask:
        input_channels = 5
    else:
        input_channels = 3 # sample_batch['image'][0].shape[1] #  5 #

    # if args.cifar:
    #     input_channels = 3
    print(f"Input channels: {input_channels}")
    
    # Create model
    # create_contrastive_model
    chosen_model = 'new_fused' # 'convnextv2_tiny' resnet18 resnet50 resnet34
    encoder_kwargs = {
        'in_channel': input_channels, # 2*
        # 'num_classes': config['num_classes'],
    }
    projection_head_kwargs = {
        'feature_dims': (model_dict[chosen_model], 128), # resnet18 if resnet34  2048 512 ConvNeXtV2: 768 256
        # 'activation': nn.ReLU(),
        'use_batch_norm': True, # True False
        'normalize_output': True
    }
    classification_head_kwargs = {
        # 'input_dim': 512,
        'num_classes': config['num_classes'],
        'dropout_rate': 0.5,
        'name': chosen_model, # resnet50 resnet18
    }
    model = create_contrastive_model(
        encoder_kwargs=encoder_kwargs,
        projection_head_kwargs=projection_head_kwargs,
        classification_head_kwargs=classification_head_kwargs,
        model_type='new_fused', # new_fused resnet
        model_name=chosen_model
    )

    if config.get('classifier', False):
        use_graph = config.get('graph', False)
        if not use_graph:
            classifier = ClassificationHead(**classification_head_kwargs)
        else:
            classifier = GATv2ClassificationHead(**classification_head_kwargs)

        if config["class_path"] is not None:
            state_dict = torch.load(config["class_path"], weights_only=False)
            classifier.load_state_dict(state_dict['model_state_dict'])
            print("Loaded classifier weights from checkpoint")

    # Print model information
    model_info = get_model_info(model)
    print(f"\nModel: {model_info['architecture']}")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    print(f"Model size: {model_info['model_size_mb']:.2f} MB")
    
    # Calculate class weights for balanced training
    if config.get('cifar', False) == False:
        class_weights = calculate_class_weights(test_loader, config['num_classes'], device)
    
    # Create loss function with class weights
    if not config.get('classifier', False):
        criterion = SupConLoss(temperature=0.15) # try default 0.07 #  temperature=0.07 25 1
    else:
        if config.get('cifar', False) == False:
            criterion = nn.CrossEntropyLoss(weight=class_weights) # 
        else:
            criterion = nn.CrossEntropyLoss()
    
    # Create optimizer and scheduler
    if not config.get('classifier', False):
        optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    else:
        optimizer, scheduler = create_optimizer_and_scheduler(classifier, config)
    
    # Create save directory
    save_dir = config.get('save_dir', './test_checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save configuration
    config_save_path = os.path.join(save_dir, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_save_path}")
    
    graph_evaluator = GraphDataConstructor(embedding_model=model, classifier=classifier, device=device)
    
    smoothed_probs, nn_idx, logits, labels_list, coords_list, metadata = graph_evaluator.construct_knn_smoothing(test_loader, k=5, alpha=0.8)
    
    results = inside_eval(config=config, list_of_logits=logits, list_of_labels=labels_list, list_of_probs=smoothed_probs, num_classes=config['num_classes'])
    
    # Print final results
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETED")

    # save results json
    with open(f"{save_dir}/evaluation_results_graph_analysis.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    return smoothed_probs, nn_idx, logits, labels_list, coords_list, metadata, results


main(config_path="/taiga/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/src/config_files/config_new_more_cts_testing.json")