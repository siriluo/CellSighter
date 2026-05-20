#!/usr/bin/env python3
"""Evaluate a contrastive encoder + classifier on PanNuke and Orion test data."""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from contrastive_learn_add import ClassificationHead, ContrastiveModel
from data.data import CellCropsDataset
from data.orion_data_processing import load_cell_crops_from_orion
from data.utils import create_validation_transform, load_samples


MODEL_DIMS = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "convnextv2_tiny": 768,
    "new_fused": 512,
}

ORION_LABEL_TO_ID = {
    "CD4_T": 0,
    "CD8_T": 1,
    "Treg": 2,
    "B_cell": 3,
    "Mono_Macro": 4,
    "Stromal": 5,
    "Smooth_Muscle": 6,
    "Tumor": 7,
    "Vasculature": 8,
    "Granulocyte": 9,
}

ORION_10_TO_BROAD = {
    0: 1,  # CD4+ T -> Immune
    1: 1,  # CD8+ T -> Immune
    2: 1,  # Treg -> Immune
    3: 1,  # B cells -> Immune
    4: 1,  # Monocytes / Macrophages -> Immune
    5: 2,  # Stromal Cells -> Connective/Stromal
    6: 2,  # Smooth Muscle -> Connective/Stromal
    7: 0,  # Tumor Cells -> Tumor/Epithelial
    8: 2,  # Vasculature -> Connective/Stromal
    9: 1,  # Granulocytes -> Immune
}

PANNUKE_TO_BROAD = {
    0: 0,  # Neoplastic
    1: 1,  # Inflammatory
    2: 2,  # Connective
    4: 0,  # Epithelial
}

BROAD_CLASS_NAMES = {
    0: "Tumor/Epithelial",
    1: "Immune/Inflammatory",
    2: "Connective/Stromal",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a CellSighter encoder/classifier on PanNuke fold 3 and Orion held-out samples."
    )
    parser.add_argument("--encoder-ckpt", required=True, help="Path to contrastive encoder checkpoint.")
    parser.add_argument("--classifier-ckpt", required=True, help="Path to classifier checkpoint.")
    parser.add_argument(
        "--pannuke-root",
        default="/taiga/illinois/vetmed/cb/kwang222/mz_jason/indepedent_test/PanNuke/cellsighter_processing_stuff_2",
        help="PanNuke CellSighter root directory containing CellTypes/.",
    )
    parser.add_argument("--pannuke-fold", type=int, default=3, help="PanNuke fold number to evaluate.")
    parser.add_argument("--pannuke-image-fraction", type=float, default=None, help="Optional PanNuke image fraction.")
    parser.add_argument("--pannuke-max-images", type=int, default=None, help="Optional max PanNuke images.")
    parser.add_argument(
        "--orion-root",
        default="/taiga/illinois/vetmed/cb/kwang222/mz_jason/orion_all_without_largest/_meta/cell_labeling/cell_patches_64_match5um_area50_3000",
        help="Orion root directory containing CRC*/ folders.",
    )
    parser.add_argument(
        "--orion-samples",
        nargs="*",
        default=None,
        help="Explicit Orion CRC sample IDs to evaluate. If omitted, uses the configured held-out fold.",
    )
    parser.add_argument("--orion-val-fold", type=int, default=0, help="Held-out Orion fold index if samples omitted.")
    parser.add_argument("--orion-sample-fraction", type=float, default=None, help="Optional Orion cell sample fraction.")
    parser.add_argument("--orion-max-samples", type=int, default=None, help="Optional Orion max cells per CRC sample.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--crop-input-size", type=int, default=64)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--backbone", default="resnet50", choices=["resnet50", "resnet18", "dinov2_vitb14", "uni2h"])
    parser.add_argument("--model-name", default="new_fused", choices=["new_fused", "resnet18", "resnet50"])
    parser.add_argument(
        "--eval-label-space",
        default="three",
        choices=["three", "binary", "native"],
        help="Label space used for reported metrics.",
    )
    parser.add_argument("--orion-positive-label", type=int, default=7, help="Orion tumor class id.")
    parser.add_argument("--pannuke-positive-label", type=int, default=1, help="PanNuke tumor/neoplastic label id.")
    parser.add_argument("--output-dir", default="./evaluation_pannuke_orion")
    return parser.parse_args()


def create_model(num_classes, model_name, backbone, device):
    encoder_kwargs = {"in_channel": 5}
    if model_name == "new_fused":
        encoder_kwargs["backbone"] = backbone
        encoder_kwargs["freeze_backbone"] = False

    projection_head_kwargs = {
        "feature_dims": (MODEL_DIMS[model_name], 128),
        "use_batch_norm": True,
        "normalize_output": True,
    }
    classification_head_kwargs = {
        "num_classes": num_classes,
        "dropout_rate": 0.2,
        "name": model_name,
    }

    model = ContrastiveModel(
        base_model="new_fused" if model_name == "new_fused" else "resnet",
        encoder_kwargs=encoder_kwargs,
        projection_head_kwargs=projection_head_kwargs,
        classification_head_kwargs=classification_head_kwargs,
        norm_proj_head_input=False,
        model_name=model_name,
        pretrained=True,
    ).to(device)
    classifier = ClassificationHead(**classification_head_kwargs).to(device)
    return model, classifier


def clean_state_dict(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def load_checkpoints(model, classifier, encoder_ckpt, classifier_ckpt, device):
    encoder_state = torch.load(encoder_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(clean_state_dict(encoder_state["model_state_dict"]))

    classifier_state = torch.load(classifier_ckpt, map_location=device, weights_only=False)
    classifier.load_state_dict(clean_state_dict(classifier_state["model_state_dict"]))
    model.eval()
    classifier.eval()


def build_pannuke_ids(root, fold):
    cells_dir = Path(root) / "CellTypes" / "cells"
    ids = [
        path.stem
        for path in cells_dir.glob(f"pannuke_f{fold}_*.npz")
    ]
    return sorted(ids, key=lambda x: int(x.split("_")[-1]))


def sample_ids(ids, fraction=None, max_count=None, seed=42):
    if fraction is None and max_count is None:
        return ids

    if fraction is not None:
        count = max(1, int(round(len(ids) * fraction)))
    else:
        count = int(max_count)

    count = min(len(ids), count)
    if count >= len(ids):
        return ids

    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(ids), size=count, replace=False).tolist())
    return [ids[i] for i in indices]


def build_pannuke_loader(args):
    image_ids = build_pannuke_ids(args.pannuke_root, args.pannuke_fold)
    if not image_ids:
        raise ValueError(f"No PanNuke files found for fold {args.pannuke_fold} under {args.pannuke_root}")
    image_ids = sample_ids(image_ids, args.pannuke_image_fraction, args.pannuke_max_images, args.seed)
    print(f"Evaluating {len(image_ids)} PanNuke fold {args.pannuke_fold} images")

    config = {
        "root_dir": args.pannuke_root,
        "to_pad": False,
        "crop_size": args.crop_size,
    }
    crops = load_samples(config, image_ids, already_cropped=False, testing=True)
    transform = create_validation_transform(crop_size=args.crop_input_size)
    dataset = CellCropsDataset(crops=crops, transform=transform, mask=True, contrastive=False)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def discover_orion_samples(root, val_fold, seed):
    folders = glob.glob("CRC*", root_dir=root)
    folders = np.array(folders)
    folders = folders[np.random.RandomState(seed).permutation(len(folders))]
    train_val_samples = folders[:32]
    splits = np.split(train_val_samples, 4)
    return splits[val_fold].tolist()


def create_orion_data_loaders():
    """
    Create training and validation data loaders.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # In this case, we can get the image names by looping through the files instead for our situation: 
    cell_patches_path = "/taiga/illinois/vetmed/cb/kwang222/mz_jason/orion_all_without_largest/_meta/cell_labeling/cell_patches_64_match5um_area50_3000"
    
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
    test_transform = create_validation_transform(crop_size=64)
    
    # Create datasets
    test_dataset = CellCropsDataset(
        crops=test_crops,
        transform=test_transform,
        mask=True,
        contrastive=False,
    )
    
    # Create data loaders
    # use_graph = config.get('graph', False)


    test_loader = DataLoader(
        test_dataset,
        batch_size=512, #  1
        shuffle=False,
        num_workers=8,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return test_loader


def prepare_batch(batch, device):
    images = batch["image"]
    masks = batch.get("mask")
    if masks is not None:
        images = torch.cat([images, masks], dim=1)
    return images.to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)


def map_binary(labels, positive_label):
    return (labels == positive_label).long()


def map_labels(labels, label_map):
    mapped = labels.clone()
    for source_label, target_label in label_map.items():
        mapped[labels == source_label] = target_label
    return mapped


def aggregate_probs(probs, output_map, eval_num_classes):
    if output_map is None:
        return probs

    aggregated = probs.new_zeros((probs.shape[0], eval_num_classes))
    for source_label, target_label in output_map.items():
        if source_label < probs.shape[1]:
            aggregated[:, target_label] += probs[:, source_label]
    return aggregated


def evaluate_loader(
        model,
        classifier,
        loader,
        device,
        eval_num_classes,
        binary_positive_label=None,
        label_map=None,
        output_map=None):
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            images, labels = prepare_batch(batch, device)
            features = model.encoder(images)
            logits = classifier(features)
            probs = torch.softmax(logits, dim=1)

            if binary_positive_label is not None:
                labels = map_binary(labels, binary_positive_label)
            elif label_map is not None:
                labels = map_labels(labels, label_map)

            eval_probs = aggregate_probs(probs, output_map, eval_num_classes)
            preds = eval_probs.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(eval_probs.cpu().numpy().tolist())

    labels_np = np.asarray(all_labels)
    preds_np = np.asarray(all_preds)
    probs_np = np.asarray(all_probs)
    average = "binary" if eval_num_classes == 2 else "weighted"
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_np, preds_np, average=None, zero_division=0
    )
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        labels_np, preds_np, average=average, zero_division=0
    )

    try:
        if eval_num_classes == 2:
            auc = roc_auc_score(labels_np, probs_np[:, 1])
        else:
            auc = roc_auc_score(labels_np, probs_np, multi_class="ovr", average="weighted")
    except ValueError:
        auc = 0.0

    return {
        "num_samples": int(labels_np.shape[0]),
        "class_names": [BROAD_CLASS_NAMES.get(i, str(i)) for i in range(eval_num_classes)],
        "accuracy": float(accuracy_score(labels_np, preds_np)),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.tolist(),
        "precision_avg": float(precision_avg),
        "recall_avg": float(recall_avg),
        "f1_avg": float(f1_avg),
        "auc": float(auc),
        "confusion_matrix": confusion_matrix(labels_np, preds_np).tolist(),
    }


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, classifier = create_model(args.num_classes, args.model_name, args.backbone, device)
    load_checkpoints(model, classifier, args.encoder_ckpt, args.classifier_ckpt, device)

    pannuke_loader = build_pannuke_loader(args)
    orion_loader = build_orion_loader(args)

    pannuke_positive = None
    orion_positive = None
    pannuke_label_map = None
    orion_label_map = None
    output_map = None
    eval_num_classes = args.num_classes

    if args.eval_label_space == "binary":
        eval_num_classes = 2
        pannuke_positive = args.pannuke_positive_label
        orion_positive = args.orion_positive_label
    elif args.eval_label_space == "three":
        eval_num_classes = 3
        pannuke_label_map = PANNUKE_TO_BROAD
        orion_label_map = ORION_10_TO_BROAD
        if args.num_classes == 10:
            output_map = ORION_10_TO_BROAD

    results = {
        "pannuke_fold": evaluate_loader(
            model,
            classifier,
            pannuke_loader,
            device,
            eval_num_classes,
            binary_positive_label=pannuke_positive,
            label_map=pannuke_label_map,
            output_map=output_map,
        ),
        "orion_test": evaluate_loader(
            model,
            classifier,
            orion_loader,
            device,
            eval_num_classes,
            binary_positive_label=orion_positive,
            label_map=orion_label_map,
            output_map=output_map,
        ),
    }

    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
    
# python src/eval_testing_scripts/evaluate_pannuke_orion.py \
#   --encoder-ckpt /taiga/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/contrastive_checkpoints_orion_to_pannuke_dann_sample10_baseline_test1/best_model.pth \
#   --classifier-ckpt /taiga/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/contrastive_classifier_checkpoints_orion_ADA_test1/best_model.pth \
#   --num-classes 10 \
#   --eval-label-space three \
#   --pannuke-fold 3 \
#   --output-dir ./eval_three_class