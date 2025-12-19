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
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score


def comprehensive_debug(model, train_loader, val_loader, criterion, device):
    """Comprehensive debugging for mysterious high validation loss."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE VALIDATION LOSS DEBUG")
    print("="*80)
    
    # Get sample batches
    train_batch = next(iter(train_loader)).to(device)
    val_batch = next(iter(val_loader)).to(device)
    
    # 1. Check batch structure
    print("\n### BATCH STRUCTURE ###")
    print(f"Training batch:")
    print(f"  batch_size: {train_batch.batch_size}")
    print(f"  num_nodes: {train_batch.num_nodes}")
    print(f"  num_edges: {train_batch.edge_index.shape[1]}")
    print(f"  y shape: {train_batch.y.shape}")
    
    print(f"\nValidation batch:")
    print(f"  batch_size: {val_batch.batch_size}")
    print(f"  num_nodes: {val_batch.num_nodes}")
    print(f"  num_edges: {val_batch.edge_index.shape[1]}")
    print(f"  y shape: {val_batch.y.shape}")
    
    # 2. Check model forward pass
    print("\n### MODEL FORWARD PASS ###")
    model.eval()
    
    with torch.no_grad():
        train_output = model(train_batch.x, train_batch.edge_index) # [:train_batch.batch_size]
        val_output = model(val_batch.x, val_batch.edge_index) # [:val_batch.batch_size]
        
        print(f"Training output shape: {train_output.shape}")
        print(f"Validation output shape: {val_output.shape}")
        
        print(f"\nTraining output stats:")
        print(f"  Range: [{train_output.min():.2f}, {train_output.max():.2f}]")
        print(f"  Mean: {train_output.mean():.2f}, Std: {train_output.std():.2f}")
        
        print(f"\nValidation output stats:")
        print(f"  Range: [{val_output.min():.2f}, {val_output.max():.2f}]")
        print(f"  Mean: {val_output.mean():.2f}, Std: {val_output.std():.2f}")
    
    # 3. Check loss computation
    print("\n### LOSS COMPUTATION ###")
    
    with torch.no_grad():
        # Training loss
        train_out_sliced = train_output[:train_batch.batch_size]
        train_labels = train_batch.y[:train_batch.batch_size].type(torch.LongTensor).to(device=device)
        train_loss = criterion(train_out_sliced, train_labels)
        
        print(f"Training:")
        print(f"  Output slice: {train_out_sliced.shape}")
        print(f"  Labels: {train_labels.shape}")
        print(f"  Loss: {train_loss.item():.4f}")
        
        # Validation loss
        val_out_sliced = val_output[:val_batch.batch_size]
        val_labels = val_batch.y[:val_batch.batch_size].type(torch.LongTensor).to(device=device)
        val_loss = criterion(val_out_sliced, val_labels)
        
        print(f"\nValidation:")
        print(f"  Output slice: {val_out_sliced.shape}")
        print(f"  Labels: {val_labels.shape}")
        print(f"  Loss: {val_loss.item():.4f}")
        
        if val_loss.item() > 100:
            print(f"\n⚠️⚠️⚠️ VALIDATION LOSS IS ABNORMALLY HIGH! ⚠️⚠️⚠️")
    
    # 4. Check predictions
    print("\n### PREDICTIONS ###")
    
    with torch.no_grad():
        train_probs = torch.softmax(train_out_sliced, dim=1)
        val_probs = torch.softmax(val_out_sliced, dim=1)

        train_pred = train_probs.argmax(dim=-1)
        val_pred = val_probs.argmax(dim=-1)
        
        train_acc = accuracy_score(train_labels.cpu().numpy(), train_pred.cpu().numpy())
        val_acc = accuracy_score(val_labels.cpu().numpy(), val_pred.cpu().numpy())
        # train_acc = (train_pred == train_labels).float().mean()
        # val_acc = (val_pred == val_labels).float().mean()
        
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")
        
        print(f"\nTraining predictions: {train_pred.unique().tolist()}")
        print(f"Validation predictions: {val_pred.unique().tolist()}")
        
        if len(val_pred.unique()) == 1:
            print(f"⚠️ Validation predicting only class {val_pred[0].item()}!")
    
    # 5. Check if issue is in data or model
    print("\n### DATA VS MODEL ISSUE ###")
    
    # Try computing loss on training data with validation code path
    with torch.no_grad():
        # Use validation batch but compute like training
        val_out_test = model(train_batch.x, train_batch.edge_index)[:train_batch.batch_size]
        val_labels_test = train_batch.y[:train_batch.batch_size].type(torch.LongTensor).to(device=device)   
        test_loss = criterion(val_out_test, val_labels_test)
        
        print(f"Training data processed like validation: {test_loss.item():.4f}")
        
        if test_loss.item() < 10:
            print("→ Issue is likely with VALIDATION DATA")
        else:
            print("→ Issue is likely with MODEL or LOSS COMPUTATION")
    
    print("\n" + "="*80)

# Run it
# use contrastive_runner to run this
# comprehensive_debug(model, train_loader, val_loader, criterion, device)