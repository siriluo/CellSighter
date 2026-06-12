import os
import sys
import argparse
from pathlib import Path
from inspect import signature
import json
from typing import Dict, Tuple, Any

# from CellSighter.src.gat_model import GATv2ClassificationHead
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import numpy as np
import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import pandas as pd

# Suppose you have a trained ResNet50 with a projection head:
# model.backbone = ResNet50
# model.projection_head = MLP

# Requirements: numpy, matplotlib, scikit-learn, (optionally) torch
import random
from data.utils import load_samples, create_training_transform, create_validation_transform, create_test_transform
from data.orion_data_processing import load_cell_crops_from_orion
from data.data import CellCropsDataset
from models import get_model_info
from train import load_config
from contrastive_learn_add import ClassificationHead, ContrastiveModel
from contrastive_runner import create_contrastive_data_loaders
# from gat_results_debug import comprehensive_debug
from data.graph_data import GraphDataConstructor

# each label index and embedding index correspond to each other, so use that to extract 100 of each class randomly?
# Create a list of indices for each class
# class_indices = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
use_mask = True


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

    # The data is numbered 00000
    mask_name = "cell_masks"
    img_patch_name = "image_patches"
    labels_name = "meta"

    # count
    print("Loading testing data...")
    test_crops = []
    for sample in test_crc_samples:
        filelist = glob.glob(f"{cell_patches_path}/{sample}/{labels_name}_*.csv")
        crops = load_cell_crops_from_orion(f"{cell_patches_path}/{sample}", mask_name, img_patch_name, labels_name, filelist)
        test_crops.extend(crops)
    print(f"Loaded {len(test_crops)} testing samples")

    # Create transforms
    test_transform = create_validation_transform(crop_size=config['crop_input_size']) # create_test_transform
    
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


def set_model(model, checkpoint_path): # , classifier, criterion
    model_to_load = model
    # criterion = torch.nn.CrossEntropyLoss()

    # classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt['model_state_dict'] 

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model_to_load.encoder = torch.nn.DataParallel(model_to_load.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model_to_load = model_to_load.cuda()
        # classifier = classifier.cuda()
        # criterion = criterion.cuda()
        cudnn.benchmark = True
        print(f"Loading model from {checkpoint_path}")
        model_to_load.load_state_dict(state_dict)
    else:
        raise NotImplementedError('This code requires GPU')

    return model_to_load


def create_embeddings():

    model_dict = {
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'convnextv2_tiny': 768,
        'new_fused': 512,
    }

    config_path = "/taiga/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/src/config_files/config_new_more_cts_testing.json"

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
        input_channels = 3 

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
    if chosen_model == 'new_fused':
        encoder_kwargs['backbone'] = 'resnet50' # resnet50 dinov2_vitb14 uni2h
        encoder_kwargs['freeze_backbone'] = False # True False
    projection_head_kwargs = {
        'feature_dims': (model_dict[chosen_model], 128), # resnet18 if resnet34  2048 512 ConvNeXtV2: 768 256
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
        model_type='new_fused', # new_fused resnet
        model_name=chosen_model
    )

    use_graph = config.get('graph', False)
    if not use_graph:
        classifier = ClassificationHead(**classification_head_kwargs)
    # else:
    #     classifier = GATv2ClassificationHead(**classification_head_kwargs)

    # if config["class_path"] is not None:
    #     state_dict = torch.load(config["class_path"], weights_only=False)
    #     classifier.load_state_dict(state_dict['model_state_dict'])
    #     print("Loaded classifier weights from checkpoint")

    encoder_model = set_model(model, config["ckpt_path"]) # , classifier, criterion

    # Print model information
    model_info = get_model_info(model)
    print(f"\nModel: {model_info['architecture']}")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    print(f"Model size: {model_info['model_size_mb']:.2f} MB")

    graph_data = GraphDataConstructor(embedding_model=encoder_model, classifier=classifier, device=device)

    encoder_embeds = True
    embeddings, labels_list, metadata, node_indices = graph_data.extract_embeddings(dataloader=test_loader, use_encoder=encoder_embeds)
    
    return embeddings, labels_list, metadata, node_indices


def to_numpy(array):
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def stratified_sample_indices(labels, n_per_class=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    labels = np.asarray(labels)

    sampled_indices = []
    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        n_sample = min(n_per_class, len(class_indices))
        sampled_indices.extend(
            rng.choice(class_indices, size=n_sample, replace=False)
        )

    sampled_indices = np.asarray(sampled_indices)
    rng.shuffle(sampled_indices)
    return sampled_indices


def run_tsne(
    embeddings,
    n_pca=50,
    perplexity=50,
    max_iter=3000,
    metric="cosine",
    random_state=42,
):
    embeddings = to_numpy(embeddings).astype(float)

    embeddings = embeddings / (
        np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    )

    n_pca = min(n_pca, embeddings.shape[1], embeddings.shape[0] - 1)
    emb_pca = PCA(n_components=n_pca, random_state=random_state).fit_transform(
        embeddings
    )

    tsne_kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        metric=metric,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )

    if "max_iter" in signature(TSNE).parameters:
        tsne_kwargs["max_iter"] = max_iter
    else:
        tsne_kwargs["n_iter"] = max_iter

    return TSNE(**tsne_kwargs).fit_transform(emb_pca)


def plot_model_tsne_panels(
    model_embeddings: dict,
    labels,
    class_names=None,
    n_per_class=1000,
    perplexity=50,
    max_iter=3000,
    output_path="model_embedding_tsne",
    random_state=42,
):
    labels = to_numpy(labels).astype(int)
    sampled_indices = stratified_sample_indices(
        labels,
        n_per_class=n_per_class,
        random_state=random_state,
    )
    sampled_labels = labels[sampled_indices]

    class_ids = np.unique(sampled_labels)
    if class_names is None:
        class_names = {class_id: f"Class {class_id}" for class_id in class_ids}
    elif not isinstance(class_names, dict):
        class_names = {i: name for i, name in enumerate(class_names)}

    colors = [
        "#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2",
        "#B07AA1", "#9C755F", "#BAB0AC", "#86BCB6", "#A0CBE8",
    ]
    color_map = {
        class_id: colors[i % len(colors)]
        for i, class_id in enumerate(class_ids)
    }

    projections = {}
    for model_name, embeddings in model_embeddings.items():
        sampled_embeddings = to_numpy(embeddings)[sampled_indices]
        projections[model_name] = run_tsne(
            sampled_embeddings,
            perplexity=perplexity,
            max_iter=max_iter,
            random_state=random_state,
        )

    with mpl.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 7,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
    }):
        n_models = len(projections)
        fig, axes = plt.subplots(
            1,
            n_models,
            figsize=(42 * n_models / 25.4, 42 / 25.4),
            constrained_layout=True,
        )

        if n_models == 1:
            axes = [axes]

        for ax, (model_name, proj) in zip(axes, projections.items()):
            for class_id in class_ids:
                mask = sampled_labels == class_id
                ax.scatter(
                    proj[mask, 0],
                    proj[mask, 1],
                    s=1.5,
                    alpha=0.65,
                    linewidths=0,
                    color=color_map[class_id],
                    label=class_names[class_id],
                )

            ax.set_title(model_name, pad=3)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            for spine in ax.spines.values():
                spine.set_visible(False)

        handles, legend_labels = axes[-1].get_legend_handles_labels()
        fig.legend(
            handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            markerscale=3,
            fontsize=6,
        )

        output_path = Path(output_path)
        fig.savefig(f"{output_path}.svg", bbox_inches="tight")
        fig.savefig(f"{output_path}.pdf", bbox_inches="tight")
        fig.savefig(f"{output_path}.png", dpi=600, bbox_inches="tight")

    return fig, projections, sampled_indices


if __name__ == "__main__":
    embeddings, labels_list, metadata, node_indices = create_embeddings()
    
    class_names=[
        "CD4+ T",
        "CD8+ T",
        "Treg",
        "B cell",
        "Mono/Macro",
        "Stromal",
        "Smooth Muscle",
        "Tumor",
        "Vasculature",
        "Granulocyte"
    ]
    
    fig, projections, sampled_indices = plot_model_tsne_panels(
    model_embeddings={
        "ResNet encoder": embeddings,
    },
    labels=labels_list,
    class_names=class_names,
    n_per_class=2000,
    perplexity=100,
    max_iter=3000,
    output_path="model_embedding_tsne",
)

