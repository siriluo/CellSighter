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


use_mask = False

with open("/projects/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/src/config_new_more_cts_con_classifier_graph.json", 'r') as f:
    config = json.load(f)

if use_mask:
    input_channels = 5
else:
    input_channels = 3

# ---- Extract backbone embeddings (2048-d) ----
encoder_kwargs = {
    'in_channel': input_channels, # 2*
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
    'num_classes': 10,
    'dropout_rate': 0.7,
    'name': 'resnet50',
}
model = ContrastiveModel(
    base_model='resnet50',
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

embeddings, labels_list, metadata, node_indices = graph_data.extract_embeddings(dataloader=train_loader)



# ---------- Example inputs (replace with your real embeddings + labels) ----------
# embeddings: numpy array shape (N, D) — output from your model (before classifier)
# labels: integer labels shape (N,)
# Optionally normalize if your model did not already
# Here we simulate:

# Note: try and keep an even number of embeddings for each group

# (If your embeddings are torch tensors, convert: embeddings = emb.detach().cpu().numpy())
# L2-normalize embeddings if using cosine similarity downstream:
embeddings = embeddings.numpy()
embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

labels_list = labels_list.numpy()

# ---------- PCA reduction (recommended) ----------
pca = PCA(n_components=50, random_state=0)
emb_pca = pca.fit_transform(embeddings)  # shape (N, 50)

# ---------- t-SNE ----------
tsne = TSNE(
    n_components=2,
    perplexity=30,          # try 5, 30, 50
    learning_rate=200,      # try 100, 200, 500
    n_iter=1000,
    metric='cosine',        # 'cosine' often suits contrastive embeddings; 'euclidean' also common
    random_state=42,
    init='pca'              # better initialization
)
proj = tsne.fit_transform(emb_pca)  # shape (N, 2)

# ---------- Plot ----------
plt.figure(figsize=(8, 6))
palette = plt.get_cmap('tab10')
for c in np.unique(labels_list):
    mask = labels_list == c
    plt.scatter(proj[mask, 0], proj[mask, 1], s=6, label=str(c), alpha=0.8)
plt.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('t-SNE of contrastive embeddings (colored by class)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.tight_layout()
plt.savefig('tsne_contrastive_embeddings.png', dpi=300)
plt.show()



