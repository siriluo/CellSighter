# hobbit long pytorch supcontrastive trainer
import os
import sys
import argparse
import time
import math
import csv

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from util.utils import AverageMeter, warmup_learning_rate
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from matplotlib import pyplot as plt
from PIL import Image
from torch_geometric.loader import HGTLoader, NeighborLoader


# Local imports
from models import create_model, get_model_info
from trainer import Trainer
from data.utils import load_samples, create_training_transform, create_validation_transform
from data.data import CellCropsDataset
from data.graph_data import *
from train import create_data_loaders, load_config, calculate_class_weights, create_optimizer_and_scheduler
from contrastive_learn_add import ContrastiveModel
import json

class ConClassEvaluator:
    """
    Trainer class for supervised contrastive learning using cells.
    
    Handles training, validation, evaluation, and model checkpointing.
    """
    
    def __init__(self,
                 model: nn.Module,
                 encoder_ckpt_path: str,
                 classifier: nn.Module,
                 test_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 num_classes: int,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cuda',
                 save_dir: str = './eval_metrics_checkpoints',
                 log_interval: int = 10,
                 args=None,
                 config=None):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler (optional)
            device: Device to use for training ('cuda' or 'cpu')
            save_dir: Directory to save checkpoints
            log_interval: Frequency of logging (in batches)
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.args = args
        self.encoder_ckpt = encoder_ckpt_path
        self.classifier = classifier
        self.config = config

        self.encoder_model, self.classifier, self.criterion = self.set_model(self.model, self.encoder_ckpt, classifier=self.classifier, criterion=self.criterion)
        
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
        self.best_val_loss = 110.0

        self.pos_bin_label = 7

        # construct graph data
        # Switch to non-graph classifier for now
        self.use_graph = self.config.get('graph', False)
        if self.use_graph:
            distance_threshold = 80  # Example threshold value
            graph_data = GraphDataConstructor(self.encoder_model, self.device)

            embeddings_te, labels_te, metadata_te, node_indices_te = graph_data.extract_embeddings(self.test_loader)
            graph_data_te = graph_data.construct_graph(embeddings_te, metadata_te, labels_te, node_indices_te, dist=distance_threshold)

            kwargs = {'batch_size': 512, 'num_workers': 4} # , 'persistent_workers': True
            self.graph_test_loader = NeighborLoader(graph_data_te, num_neighbors=[5], shuffle=True, **kwargs)


    def get_multiclass_ct_name(self, label):

        if self.num_classes == 10:
            new_mapping = {
                0: "CD4+ T",
                1: "CD8+ T",
                2: "Treg",
                3: "B cells",
                4: "Monocytes / Macrophages",
                5: "Stromal Cells",
                6: "Smooth Muscle",
                7: "Tumor Cells",
                8: "Vasculature",
                9: "Granulocytes",
            }
        else:
            new_mapping = {
                0: "CD4+ T",
                1: "CD8+ T",
                2: "Treg",
                3: "B cells",
                4: "NK Cells",
                5: "Dendritic Cells",
                6: "Monocytes / Macrophages",
                7: "Stromal Cells",
                8: "Smooth Muscle",
                9: "Tumor Cells",
                10: "Vasculature",
                11: "Granulocytes",
            }


        class_name = new_mapping[label]

        return class_name


    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)


    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.history = checkpoint.get('history', {})
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def save_history(self):
        """Save training history to JSON file."""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        
        # Convert numpy types to Python types for JSON serialization
        json_history = {}
        for key, values in self.history.items():
            json_history[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v 
                               for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(json_history, f, indent=2)


    def set_model(self, model, checkpoint_path, classifier, criterion):
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
            classifier = classifier.cuda()
            criterion = criterion.cuda()
            cudnn.benchmark = True

            model_to_load.load_state_dict(state_dict)
        else:
            raise NotImplementedError('This code requires GPU')

        return model_to_load, classifier, criterion #


    def evaluate_detailed(self, test_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Perform detailed evaluation with confusion matrix and class-wise metrics.
        
        Args:
            test_loader: Test data loader (uses validation loader if None)
            
        Returns:
            Dictionary with detailed evaluation results
        """
        if test_loader is None:
            if self.use_graph:
                test_loader = self.graph_val_loader
            else:
                test_loader = self.test_loader

        self.encoder_model.eval()
        self.classifier.eval()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        list_of_logits = []
        list_of_probs = []
        list_of_labels = []

        bin_pos_label = self.pos_bin_label
        bin_ct_name = self.get_multiclass_ct_name(bin_pos_label)

        losses = AverageMeter()

        with torch.no_grad():
            for idx, batch in enumerate(test_loader):
                # batch = batch.to(self.device)

                # batch_size = batch.batch_size

                # labels = batch.y[:batch_size].type(torch.LongTensor).to(device=self.device)

                # # compute loss
                # output = self.classifier(batch.x, batch.edge_index)[:batch_size].to(device=self.device)
                # loss = self.criterion(output, labels)
                
                # if self.args.cifar == False:
                images = batch['image'] # .to(self.device) [2, B, C, H, W] [0]
                labels = batch['label'] #.to(self.device) [1]
                # else:
                #     images = batch[0]
                #     labels = batch[1]
                
                # if self.args.cifar == False:
                m = batch.get('mask', None)
                if m is not None:
                    images = torch.cat([images, m], dim=1)

                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                bsz = labels.shape[0]

                # compute loss
                with torch.no_grad():
                    features = self.encoder_model.encoder(images)
                output = self.classifier(features.detach())
                loss = self.criterion(output, labels)

                losses.update(loss.item(), bsz)
                
                # Get predictions and probabilities
                probs = torch.softmax(output, dim=1)

                if self.num_classes > 2:
                    preds = probs.argmax(1)
                else:
                    _, preds = torch.max(output, 1) # Need to convert to binary class labels?
                    preds_temp = preds.clone()
                    preds[preds_temp == bin_pos_label] = 0
                    preds[preds_temp != bin_pos_label] = 1
                    
                
                # Collect results
                # running_loss += loss.item() * batch_size
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                if self.num_classes <= 2:
                    all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of tumor class
                else:
                    all_probs.append(probs.cpu().numpy())
                
                # list_of_probs.append(probs.cpu().numpy().tolist())
                # list_of_logits.append(output.cpu().numpy().tolist())
                # list_of_labels.append(labels.cpu().numpy().tolist())
        
        # For the mapping, tumor cells are class 7 (index 7).
        

        # Calculate metrics
        avg_loss = losses.avg # running_loss / len(self.graph_val_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        if self.num_classes <= 2:
            average_method = 'binary'
        else:
            average_method = 'weighted'

        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )

        # one by one binary
        # pos_labels_list = np.arange(self.num_classes)
        # for pos_label in pos_labels_list:
        #     p_i, r_i, f1_i, supp_i = precision_recall_fscore_support(
        #         all_labels, all_preds, average='binary', pos_label=pos_label, zero_division=0
        #     )
        
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
            'confusion_matrix': cm.tolist(),
            'class_names': [f'Non-{bin_ct_name}', f'{bin_ct_name}'],
            'loss': avg_loss
        }

        if self.num_classes > 2:
            multi_aucs_list = multi_aucs.tolist()
            results.update({'multi_class_aucs': multi_aucs_list})
        
        return results




