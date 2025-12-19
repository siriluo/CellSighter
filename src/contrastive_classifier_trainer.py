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


# Local imports
from models import create_model, get_model_info
from trainer import Trainer
from data.utils import load_samples, create_training_transform, create_validation_transform
from data.data import CellCropsDataset
from train import create_data_loaders, load_config, calculate_class_weights, create_optimizer_and_scheduler
from contrastive_learn_add import ContrastiveModel
import json

class ConClassTrainer:
    """
    Trainer class for supervised contrastive learning using cells.
    
    Handles training, validation, evaluation, and model checkpointing.
    """
    
    def __init__(self,
                 model: nn.Module,
                 encoder_ckpt_path: str,
                 classifier: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 num_classes: int,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cuda',
                 save_dir: str = './contrastive_checkpoints',
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
        self.train_loader = train_loader
        self.val_loader = val_loader
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


    def train_epoch(self, train_loader, encoder_model, classifier, criterion, optimizer, epoch): # , opt
        """one epoch training"""
        encoder_model.eval()
        classifier.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()
        for idx, batch in enumerate(train_loader):
        # for idx, (images, labels) in enumerate(train_loader):
            data_time.update(time.time() - end)

            images = batch['image'] # .to(self.device) [2, B, C, H, W]
            labels = batch['label'] #.to(self.device)

            # Code to append mask dimensions
            # for images[0] and images[1], append m[0] and m[1] respectively along channel dimension
            m = batch.get('mask', None)
            if m is not None:
                images = torch.cat([images, m], dim=1)

            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # warm-up learning rate
            # if self.args is not None:
            warmup_learning_rate(self.args, epoch, idx, len(train_loader), optimizer)

            # compute loss
            with torch.no_grad():
                features = encoder_model.encoder(images)
            output = classifier(features.detach())
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)

            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            probs = torch.softmax(output, dim=1)

            # print info
            if (idx + 1) % self.log_interval == 0:
                print('Train: [{0}][{1}/{2}]\t'
                    'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))
                sys.stdout.flush()

        return losses.avg


    def validate(self) -> Dict[str, float]:
        """
        Validate the model on validation set.
        
        Returns:
            Dictionary with validation metrics
        """
        self.encoder_model.eval()
        self.classifier.eval()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        # all_probs_test = []

        losses = AverageMeter()
        
        with torch.no_grad():
            for idx, batch in enumerate(self.val_loader):
                images = batch['image'] #.to(self.device) [0] #
                labels = batch['label'] #.to(self.device) [1] #
                
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
                
                # Get predictions and probabilities
                probs = torch.softmax(output, dim=1)

                if self.num_classes > 2:
                    preds = probs.argmax(1)
                else:
                    _, preds = torch.max(output, 1)
                
                # Collect results
                # running_loss += loss.item() * images.size(0)
                losses.update(loss.item(), bsz)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                if self.num_classes <= 2:
                    all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of tumor class
                else:
                    all_probs.append(probs.cpu().numpy())
        
        # Calculate metrics
        avg_loss = losses.avg # running_loss / len(self.val_loader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)

        if self.num_classes <= 2:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='binary', zero_division=0
            )
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='weighted', zero_division=0
            )
            
        # Calculate AUC if we have both classes
        try:
            if self.num_classes <= 2:
                auc = roc_auc_score(all_labels, all_probs)
            else:
                all_probs = np.vstack(all_probs)

                auc = roc_auc_score(all_labels, all_probs, multi_class='ovo', average='weighted')
                multi_aucs = roc_auc_score(all_labels, all_probs, multi_class='ovr', average=None)
                
        except ValueError as e:
            auc = 0.0  # Handle case where only one class is present
            if self.num_classes > 2:
                multi_aucs = []
            # raise 
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }

        if self.num_classes > 2:
            multi_aucs_list = multi_aucs.tolist()
            metrics.update({'multi_aucs': multi_aucs_list})
        
        return metrics


    def train(self, num_epochs: int, early_stopping_patience: int = 10) -> Dict[str, List]:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Stop training if no improvement for this many epochs
            
        Returns:
            Training history dictionary
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")

        if torch.cuda.is_available():
            print(torch.cuda.memory_summary())
        else:
            print("CUDA is not available.")
        
        epochs_without_improvement = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(self.train_loader, self.encoder_model, self.classifier, self.criterion, self.optimizer, epoch + 1) # , train_acc 
            
            # Validation
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['learning_rates'].append(current_lr)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            # self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_auc'].append(val_metrics['auc'])

            if self.num_classes > 2:
                self.history['val_multi_auc'].append(val_metrics['multi_aucs'])
            
            # Print epoch results
            epoch_time = time.time() - epoch_start_time
            print(f'\nEpoch {epoch + 1}/{num_epochs} - {epoch_time:.2f}s')
            print(f'Train Loss: {train_loss:.4f}') # , Train Acc: {train_acc:.4f}
            print(f'Val Loss: {val_metrics["loss"]:.4f}') # , Val Acc: {val_metrics["accuracy"]:.4f}
            print(f'Val Precision: {val_metrics["precision"]:.4f}, Val Recall: {val_metrics["recall"]:.4f}')
            print(f'Val F1: {val_metrics["f1"]:.4f}, Val AUC: {val_metrics["auc"]:.4f}')
            
            # Check for best model
            if val_metrics['f1'] > self.best_val_f1: # val_metrics['loss'] < self.best_val_loss: # train_loss < self.best_train_loss: # 
                self.best_val_f1 = val_metrics['f1']
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_loss = val_metrics['loss']
                # self.best_train_loss = train_loss
                self.best_epoch = epoch + 1
                epochs_without_improvement = 0
                
                # Save best model
                self.save_checkpoint(epoch + 1, is_best=True)
                print(f'New best model saved! F1: {self.best_val_f1:.4f}')
            else:
                epochs_without_improvement += 1
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1, is_best=False)
            
            # # Early stopping
            # if epochs_without_improvement >= early_stopping_patience:
            #     print(f'\nEarly stopping triggered after {epochs_without_improvement} epochs without improvement')
            #     break
            
            print('-' * 60)
        
        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time:.2f}s')
        print(f'Best validation F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}')
        
        # Save final history
        self.save_history()
        
        return self.history
    

    def evaluate_detailed(self, test_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Perform detailed evaluation with confusion matrix and class-wise metrics.
        
        Args:
            test_loader: Test data loader (uses validation loader if None)
            
        Returns:
            Dictionary with detailed evaluation results
        """
        if test_loader is None:
            test_loader = self.val_loader

        self.encoder_model.eval()
        self.classifier.eval()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        losses = AverageMeter()
        
        list_of_logits = []
        list_of_probs = []
        list_of_labels = []

        bin_pos_label = self.pos_bin_label
        bin_ct_name = self.get_multiclass_ct_name(bin_pos_label)

        with torch.no_grad():
            for idx, batch in enumerate(self.val_loader):
                images = batch['image'] #.to(self.device) [0] #
                labels = batch['label'] #.to(self.device) [1] #
                
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
                
                # Get predictions and probabilities
                probs = torch.softmax(output, dim=1)

                if self.num_classes > 2:
                    preds = probs.argmax(1)
                else:
                    _, preds = torch.max(output, 1)
                
                # Collect results
                # running_loss += loss.item() * images.size(0)
                losses.update(loss.item(), bsz)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                if self.num_classes <= 2:
                    all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of tumor class
                else:
                    all_probs.append(probs.cpu().numpy())
                
                # list_of_probs.append(probs.cpu().numpy().tolist())
                # list_of_logits.append(output.cpu().numpy().tolist())
                # list_of_labels.append(labels.cpu().numpy().tolist())

        # Calculate metrics 
        accuracy = accuracy_score(all_labels, all_preds)
        avg_loss = losses.avg # running_loss / len(self.val_loader.dataset)

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
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """
        Plot training curves for loss and accuracy.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        # axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy')
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # F1 Score
        axes[1, 0].plot(epochs, self.history['val_f1'], 'g-', label='Validation F1')
        axes[1, 0].plot(epochs, self.history['val_precision'], 'orange', label='Validation Precision')
        axes[1, 0].plot(epochs, self.history['val_recall'], 'purple', label='Validation Recall')
        axes[1, 0].set_title('Validation Metrics')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # AUC
        axes[1, 1].plot(epochs, self.history['val_auc'], 'm-', label='Validation AUC')
        axes[1, 1].set_title('Validation AUC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()




