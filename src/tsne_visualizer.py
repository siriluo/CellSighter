import os
import sys
import argparse
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Any
import torch.backends.cudnn as cudnn
import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA

import pandas as pd

# Suppose you have a trained ResNet50 with a projection head:
# model.backbone = ResNet50
# model.projection_head = MLP

# Requirements: numpy, matplotlib, scikit-learn, (optionally) torch
import random
from models import create_model, get_model_info
from trainer import Trainer
from data.utils import load_samples, create_training_transform, create_validation_transform
from data.data import CellCropsDataset
from data.graph_data import *
from train import create_data_loaders, load_config, calculate_class_weights, create_optimizer_and_scheduler
from contrastive_learn_add import ContrastiveModel
from contrastive_runner import create_contrastive_data_loaders
from gat_results_debug import comprehensive_debug


use_mask = True

model_dict = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'convnextv2_tiny': 768,
}

with open("/projects/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/src/config_new_more_cts_con_classifier_graph_tester.json", 'r') as f:
    config = json.load(f)

if use_mask:
    input_channels = 5
else:
    input_channels = 3

# ---- Extract backbone embeddings (2048-d) ----
chosen_model = 'convnextv2_tiny' # 'convnextv2_tiny' resnet18
encoder_kwargs = {
    'in_channel': input_channels, # 2*
    # 'num_classes': config['num_classes'],
}
projection_head_kwargs = {
    'feature_dims': (model_dict[chosen_model], 128), # resnet18 if resnet34   2048 512 ConvNeXtV2: 768 256
    # 'activation': nn.ReLU(),
    'use_batch_norm': False,
    'normalize_output': True
}
classification_head_kwargs = {
    # 'input_dim': 512,
    'num_classes': config['num_classes'],
    'dropout_rate': 0.7,
    'name': chosen_model, # resnet50 resnet18
}
model = ContrastiveModel(
    base_model='convnext', # resnet convnext
    encoder_kwargs=encoder_kwargs,
    projection_head_kwargs=projection_head_kwargs,
    classification_head_kwargs=classification_head_kwargs,
    norm_proj_head_input=False,
)

checkpoint_path = config["ckpt_path"]
ckpt = torch.load(checkpoint_path, map_location='cpu')
state_dict = ckpt['model_state_dict'] 

if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        model.encoder = torch.nn.DataParallel(model.encoder)
    else:
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
    model = model.cuda()
    cudnn.benchmark = True

    model.load_state_dict(state_dict)
else:
    raise NotImplementedError('This code requires GPU')

graph_data = GraphDataConstructor(embedding_model=model)

train_loader, val_loader = create_contrastive_data_loaders(config)

encoder_embeds = True
embeddings, labels_list, metadata, node_indices = graph_data.extract_embeddings(dataloader=train_loader, use_encoder=encoder_embeds)

# each label index and embedding index correspond to each other, so use that to extract 100 of each class randomly?
# Create a list of indices for each class
# class_indices = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
class_indices = [[] for _ in range(10)]
for i, label in enumerate(labels_list):
    class_indices[label].append(i)

# Sample 100 indices from each class
sampled_indices = []
for indices in class_indices:
    sampled_indices.extend(random.sample(indices, 1000))

# Filter embeddings and labels based on sampled indices
sampled_embeddings = embeddings[sampled_indices]
sampled_labels_list = labels_list[sampled_indices]


# ---------- Example inputs (replace with your real embeddings + labels) ----------
# embeddings: numpy array shape (N, D) — output from your model (before classifier)
# labels: integer labels shape (N,)
# Optionally normalize if your model did not already
# Here we simulate:

# Note: try and keep an even number of embeddings for each group

# (If your embeddings are torch tensors, convert: embeddings = emb.detach().cpu().numpy())
# L2-normalize embeddings if using cosine similarity downstream:
sampled_embeddings = sampled_embeddings.numpy()
sampled_embeddings = sampled_embeddings / (np.linalg.norm(sampled_embeddings, axis=1, keepdims=True) + 1e-10)
# embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

# labels_list = labels_list.numpy()
sampled_labels_list = sampled_labels_list.numpy()

# ---------- PCA reduction (recommended) ----------
pca = PCA(n_components=50, random_state=0)
emb_pca = pca.fit_transform(sampled_embeddings)  # shape (N, 50)

# ---------- t-SNE ----------
lr = 500
rstate = 42
perplex = 50
iters = 15000
tsne = TSNE(
    n_components=2,
    perplexity=perplex,          # try 5, 30, 50
    # learning_rate=500,      # try 100, 200, 500
    n_iter=iters,              # increase if not converged, tried 1000
    metric='cosine',        # 'cosine' often suits contrastive embeddings; 'euclidean' also common
    random_state=rstate,
    init='pca'              # better initialization
)
proj = tsne.fit_transform(emb_pca)  # shape (N, 2)

# ---------- Plot ----------
plt.figure(figsize=(8, 6))
palette = plt.get_cmap('tab10')
for c in np.unique(sampled_labels_list):
    mask = sampled_labels_list == c
    plt.scatter(proj[mask, 0], proj[mask, 1], s=6, label=str(c), alpha=0.8)
plt.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('t-SNE of contrastive embeddings (colored by class)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.tight_layout()
plt.savefig(f'tsne_con_embeds_6_convnext_plx{perplex}_encoder_{encoder_embeds}_iters_{iters}.png', dpi=300)
plt.show()



