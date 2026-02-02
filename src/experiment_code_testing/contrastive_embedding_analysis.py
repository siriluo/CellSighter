import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
# from sklearn.neighbors 
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
from collections import Counter
import json
from pathlib import Path
import sys

src_dir = Path("/projects/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/src")
sys.path.insert(0, str(src_dir))

from data.utils import load_samples, create_training_transform, create_validation_transform
from data.data import CellCropsDataset
from contrastive_runner import create_contrastive_model
from train import get_multiclass_ct_name, load_config, create_data_loaders, calculate_class_weights
from models import create_model, get_model_info, CellEncoderResNet
from knn_algorithm import knn_count


use_mask = True

model_dict = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'convnextv2_tiny': 768,
}

def create_contrastive_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    Create test data loader.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        test_loader
    """
    
    print("Loading testing data...")
    test_crops = load_samples(config, config['test_set'])
    print(f"Loaded {len(test_crops)} validation samples")
    
    # Create transforms
    test_transform = create_validation_transform(crop_size=config['crop_input_size'])

    # no_op_transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Lambda(lambda x: x)
    # ])
    
    # Create datasets
    test_dataset = CellCropsDataset(
        crops=test_crops,
        transform=test_transform,
        mask=use_mask
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'], #  1
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return test_loader


def set_model(model, checkpoint_path):
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
        # cudnn.benchmark = True

        model_to_load.load_state_dict(state_dict)
    else:
        raise NotImplementedError('This code requires GPU')

    return model_to_load
    

def load_two_models(best_nc_model_path, config_c):
    """
    Load in the two models for comparison.
    
    Returns:
        loaded_contrast_model, loaded_nc_model
    """

    # First the contrastive model
    chosen_model = 'resnet18' # 'convnextv2_tiny' resnet18
    encoder_kwargs = {
        'in_channel': 5, # 2*
        # 'num_classes': config['num_classes'],
    }
    projection_head_kwargs = {
        'feature_dims': (model_dict[chosen_model], 128), # resnet18 if resnet34   2048 512 ConvNeXtV2: 768 256
        # 'activation': nn.ReLU(),
        'use_batch_norm': False, # True False
        'normalize_output': True
    }
    classification_head_kwargs = {
        # 'input_dim': 512,
        'num_classes': config_c['num_classes'],
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

    loaded_contrast_model = set_model(model=model, checkpoint_path=config_c["ckpt_path"])
    loaded_contrast_model.eval()


    ##
    # Now for the non-contrastive model
    ##
    model_nc = create_model(
        model_type='resnet',
        contrastive=False,
        input_channels=5,
        num_classes=config_nc['num_classes'],
        dropout_rate=config_nc.get('dropout_rate', 0.3)
    )

    # Now load the non-contrastive model checkpoint
    checkpoint = torch.load(best_nc_model_path, map_location='cpu')
    model_nc.load_state_dict(checkpoint['model_state_dict'])

    loaded_nc_model = CellEncoderResNet(model_nc).cuda()
    loaded_nc_model.eval()
    
    return loaded_contrast_model, loaded_nc_model


def calc_cosine_sim(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    emb1_pool = F.adaptive_avg_pool2d(x1, (1, 1)) #.squeeze()
    emb2_pool = F.adaptive_avg_pool2d(x2, (1, 1)) #.squeeze()

    similarity = F.cosine_similarity(emb1_pool, emb2_pool, dim=1)

    return similarity

# This module performs contrastive embedding analysis on cell image data.
# Load in the testing dataset.

# Non-contrastive resnet
config_path_1 = "/projects/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/experiment_results/new_full_class_testing_results/results_ver5_mask_aug/config.json"
with open(config_path_1, 'r') as f1:
    config_nc = json.load(f1)
best_nc_model_path = "/projects/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/experiment_results/new_full_class_testing_results/results_ver5_mask_aug/checkpoint_epoch_55.pth"

# Contrastive resnet
config_path_2 = "/projects/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/experiment_results/new_full_class_testing_results/second_testing_set/results_supcon_mask_aug_resnet18_1/config.json"
with open(config_path_2, "r") as f2:
    config_c = json.load(f2)

test_loader_nc = create_contrastive_data_loaders(config=config_nc)
test_loader_c = create_contrastive_data_loaders(config=config_c)

# Look at this for loading the embeddings: https://medium.com/vector-database/how-to-get-the-right-vector-embeddings-83295ced7f35

# Now load the two models:
loaded_contrast_model, loaded_noncontrast_model = load_two_models(best_nc_model_path=best_nc_model_path, config_c=config_c)

print("--- Loaded Both Models ---")

# Now, randomly choose some samples from the test set and get their embeddings from both models.
num_samples_to_evaluate = 1000
all_indices = list(range(len(test_loader_c.dataset)))
chosen_indices = np.random.choice(all_indices, size=num_samples_to_evaluate, replace=False)
contrastive_embeddings = []
non_contrastive_embeddings = []
labels = []

# add the per class ratios as well.

with torch.no_grad():
    for idx in tqdm(all_indices):
        sample = test_loader_c.dataset[idx]
        image = sample['image'].unsqueeze(0)  # Add batch dimension
        # print(image.shape)
        # break
        label = sample['label']
        # sample_nc = test_loader_nc.dataset[idx]
        # image_nc = sample['image'].unsqueeze(0)
        labels.append(label)

        m = sample.get('mask', None).unsqueeze(0)
        if m is not None:
            image = torch.cat([image, m], dim=1)

        if torch.cuda.is_available():
            image = image.cuda()

        # Get contrastive embedding
        contrastive_output = loaded_contrast_model.encoder(image)
        contrastive_embeddings.append(contrastive_output.cpu()) # .numpy().squeeze()

        # Get non-contrastive embedding
        non_contrastive_output = loaded_noncontrast_model(image)
        non_contrastive_embeddings.append(non_contrastive_output.cpu()) # .numpy().squeeze()

print("--- processed random samples ---")

# Convert lists to numpy arrays
# contrastive_embeddings = np.array(contrastive_embeddings)
# non_contrastive_embeddings = np.array(non_contrastive_embeddings)
labels = np.array(labels)

# Now we have embeddings from both models for the chosen samples.
# Next steps could include calculating pairwise distances, visualizing embeddings, etc.
# Example: Calculate pairwise distances within each class for both models

# Do it for each cell type, also do it for each category (same cells in pair vs different cells in pair)
# Get the average distances for all these metrics.
# Form pairs of indices.

row_idx, col_idx = np.triu_indices(len(non_contrastive_embeddings), k=1)

# Choose a random sample of pairs? or just use all pairs.
num_pairs = len(row_idx)
print(f"pairs: {num_pairs}")

total_nc_dists = []
total_c_dists = []
corresponding_cts = []
same_class_results = []
diff_class_results = []

class_sep_dists_nc_same = {
    "0": [],
    "1": [],
    "2": [],
    "3": [],
    "4": [],
    "5": [],
    "6": [],
    "7": [],
    "8": [],
    "9": [],
}
class_sep_dists_nc_diff = {
    "0": [],
    "1": [],
    "2": [],
    "3": [],
    "4": [],
    "5": [],
    "6": [],
    "7": [],
    "8": [],
    "9": [],
}

class_sep_dists_c_same = {
    "0": [],
    "1": [],
    "2": [],
    "3": [],
    "4": [],
    "5": [],
    "6": [],
    "7": [],
    "8": [],
    "9": [],
}
class_sep_dists_c_diff = {
    "0": [],
    "1": [],
    "2": [],
    "3": [],
    "4": [],
    "5": [],
    "6": [],
    "7": [],
    "8": [],
    "9": [],
}

# print(row_idx[:100], col_idx[:100])

for idx in np.arange(num_pairs):
    i, j = (row_idx[idx], col_idx[idx])

    cell1, cell2 = (labels[i], labels[j])

    emb_nc1, emb_nc2 = (non_contrastive_embeddings[i], non_contrastive_embeddings[j])
    nc_cos_dist = F.cosine_similarity(emb_nc1, emb_nc2)
    total_nc_dists.append(nc_cos_dist.numpy().squeeze())

    emb_c1, emb_c2 = (contrastive_embeddings[i], contrastive_embeddings[j])
    c_cos_dist = F.cosine_similarity(emb_c1, emb_c2)
    total_c_dists.append(c_cos_dist.numpy().squeeze())

    corresponding_cts.append((cell1, cell2))

    if cell1 == cell2:
        same_class_results.append([nc_cos_dist, c_cos_dist])
        class_sep_dists_nc_same[str(cell1)].append(nc_cos_dist.item())
        class_sep_dists_c_same[str(cell1)].append(c_cos_dist.item())
    else:
        diff_class_results.append([nc_cos_dist, c_cos_dist])
        class_sep_dists_nc_diff[str(cell1)].append(nc_cos_dist.item())
        class_sep_dists_nc_diff[str(cell2)].append(nc_cos_dist.item())
        class_sep_dists_c_diff[str(cell1)].append(c_cos_dist.item())
        class_sep_dists_c_diff[str(cell2)].append(c_cos_dist.item())

# look at the ratio change before and after the contrastive learning
print("--- Calculated Distances ---")

nc_ratio = (np.mean([x[0] for x in same_class_results]) / np.mean([x[0] for x in diff_class_results]))

c_ratio = (np.mean([x[1] for x in same_class_results]) / np.mean([x[1] for x in diff_class_results]))

ratios = {
    "nc_ratio": float(nc_ratio),
    "c_ratio": float(c_ratio)
}

# Save the dictionary
save_path = "/projects/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/src/experiment_code_testing"
# with open(f"{save_path}/class_cosine_dists.json", 'w') as f:
#     json.dump(class_sep_dists, f, indent=4) # indent makes the file human-readable

with open(f"{save_path}/ratio_cosine.json", 'w') as f:
    json.dump(ratios, f, indent=4) # indent makes the file human-readable


# Now process the individual classes.
classes = class_sep_dists_nc_same.keys()

for cell_type in classes:
    nc_same = class_sep_dists_nc_same[cell_type]
    nc_diff = class_sep_dists_nc_diff[cell_type]
    c_same = class_sep_dists_c_same[cell_type]
    c_diff = class_sep_dists_c_diff[cell_type]

    nc_ratio_ct = np.mean(nc_same) / np.mean(nc_diff) if len(nc_diff) > 0 else float('inf')
    c_ratio_ct = np.mean(c_same) / np.mean(c_diff) if len(c_diff) > 0 else float('inf')

    ratio_cts = {
        "nc_ratio_ct": float(nc_ratio_ct),
        "c_ratio_ct": float(c_ratio_ct)
    }

    with open(f"{save_path}/ratio_cosine_ct_{cell_type}.json", 'w') as f:
        json.dump(ratio_cts, f, indent=4) 


# input1 = torch.randn(1, 128)
# input2 = torch.randn(100, 128)
# output = F.cosine_similarity(input1, input2)
# print(output)

# After getting the distances, now process them? Treat the first cell in the pair as the "starting point" and the 2nd cell
# as the comparison
# saved_df = pd.DataFrame({
#     "total_nc_dists": total_nc_dists,
#     "total_c_dists": total_c_dists,
#     "corresponding_cts": corresponding_cts,
# })

# saved_df.to_csv(f"{save_path}/nc_c_cosine_dists.csv")

# class_sep_nc_df = pd.DataFrame(class_sep_dists_nc)
# class_sep_nc_df.to_csv(f"{save_path}/class_sep_nc_cosine_dists.csv")

# class_sep_c_df = pd.DataFrame(class_sep_dists_c)
# class_sep_c_df.to_csv(f"{save_path}/class_sep_c_cosine_dists.csv")

# same_class_results = np.array(same_class_results)
# diff_class_results = np.array(diff_class_results)
    
# ratio_calc_df = pd.DataFrame({
#     "nc_same_dists": same_class_results[:, 0],
#     "nc_diff_dists": diff_class_results[:, 0],
#     "c_same_dists": same_class_results[:, 1],
#     "c_diff_dists": diff_class_results[:, 1],
# })

# ratio_calc_df.to_csv(f"{save_path}/nc_c_cosine_dists_ratio_calc.csv")


# First compute distances between two cells for both of the models and compare their distances. Use cosine similarity
# Do this for all the embeddings? Or just randomly select pairs instead.

# test_list = []
# for i in np.arange(10):
#     test_list.append(torch.randn(1, 128))

# contrastive_embeddings = [emb.numpy().squeeze() for emb in contrastive_embeddings]
# contrastive_embeddings = np.array(contrastive_embeddings)

# for i in np.arange(len(contrastive_embeddings)):
#     # i, j = (row_idx[idx], col_idx[idx])

#     # cell1, cell2 = (labels[i], labels[j])
    
#     test_result = knn_count(data=ncontrastive_embeddings, labels=labels, test_idx=i, k=10)