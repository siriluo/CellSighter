import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from collections import defaultdict


class Evaluator:
        
    def __init__(self,
                model: nn.Module,
                data_loader: DataLoader,
                num_classes: int,
                device: str = 'cuda',
                save_dir: str = './eval_metrics',
                log_interval: int = 10):
        self.model = model.to(device)
        self.test_loader = data_loader
        self.num_classes = num_classes
        self.device = device
        self.save_dir = save_dir
        self.log_interval = log_interval
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_auc': [],
            'learning_rates': []
        }

        if self.num_classes > 2:
            self.history.update({'val_multi_auc': []})
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.best_epoch = 0


    def get_training_metrics(self, save_dir: str ='') -> Dict[str, Any]:
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                images = batch['image'] #.to(self.device)
                labels = batch['label'].to(self.device)
                
                m = batch.get('mask', None)
                if m is not None:
                    images = torch.cat([images, m], dim=1)
                images = images.to(self.device)
                
                # Convert labels to binary
                long_labels = labels.long() # (labels > 0)  

                # Forward pass
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)

                if self.num_classes > 2:
                    preds = probs.argmax(1)
                else:
                    _, preds = torch.max(outputs, 1)
                # _, preds = torch.max(outputs, 1)
                
                # Collect results
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(long_labels.cpu().numpy())
                if self.num_classes <= 2:
                    all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of tumor class
                else:
                    all_probs.append(probs.cpu().numpy())

        # outputs = self.model(images)
        # features = outputs.pooler_output.squeeze()
        # print(features.shape)

        if self.num_classes <= 2:
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
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # AUC
        try:
            if self.num_classes <= 2:
                auc = roc_auc_score(all_labels, all_probs)
            else:
                all_probs = np.vstack(all_probs)
                
                auc = roc_auc_score(all_labels, all_probs, multi_class='ovo', average='weighted')
                multi_aucs = roc_auc_score(all_labels, all_probs, multi_class='ovr', average=None)
        except ValueError:
            auc = 0.0  # Handle case where only one class is present
            if self.num_classes > 2:
                multi_aucs = []
        
        results = {
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support_per_class': support.tolist(),
            'precision_avg': precision_avg,
            'recall_avg': recall_avg,
            'f1_avg': f1_avg,
            'auc': auc,
            'confusion_matrix': cm.tolist(),
            'class_names': ['Non-tumor', 'Tumor']
        }

        if self.num_classes > 2:
            multi_aucs_list = multi_aucs.tolist()
            results.update({'multi_class_aucs': multi_aucs_list})

        eval_save_path = os.path.join(save_dir, 'training_metrics_results.json')
        with open(eval_save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation results saved to {eval_save_path}")

        return results
