"""
trainer.py - Training and Evaluation Logic for Cell Classification

This module contains the Trainer class that handles:
- Training loop with validation
- Model evaluation and metrics
- Checkpointing and model saving
- Loss tracking and logging
"""

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


class Trainer:
    """
    Trainer class for binary cell classification.
    
    Handles training, validation, evaluation, and model checkpointing.
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cuda',
                 save_dir: str = './checkpoints',
                 log_interval: int = 10):
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
        self.scheduler = scheduler
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
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.best_epoch = 0
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train the model for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            images = batch['image'] # .to(self.device)
            labels = batch['label'].to(self.device)

            # Code to append mask dimensions
            m = batch.get('mask', None)
            if m is not None:
                images = torch.cat([images, m], dim=1)

            images = images.to(self.device)
            
            # Convert labels to binary (0: non-tumor, 1: tumor)
            binary_labels = labels.long() # (labels > 0)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)

            loss = self.criterion(outputs, binary_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == binary_labels.data)
            total_samples += images.size(0)
            
            # Log progress
            if batch_idx % self.log_interval == 0:
                current_loss = running_loss / total_samples
                current_acc = running_corrects.double() / total_samples
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}: '
                      f'Loss: {current_loss:.4f}, Acc: {current_acc:.4f}')
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return epoch_loss, epoch_acc.item()
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model on validation set.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'] #.to(self.device)
                labels = batch['label'].to(self.device)
                
                m = batch.get('mask', None)
                if m is not None:
                    images = torch.cat([images, m], dim=1)
                images = images.to(self.device)

                # Convert labels to binary
                binary_labels = (labels > 0).long()
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, binary_labels)
                
                # Get predictions and probabilities
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                # Collect results
                running_loss += loss.item() * images.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(binary_labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of tumor class
        
        # Calculate metrics
        avg_loss = running_loss / len(self.val_loader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        # Calculate AUC if we have both classes
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0  # Handle case where only one class is present
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
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
        
        epochs_without_improvement = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(epoch + 1)
            
            # Validation
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['learning_rates'].append(current_lr)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_auc'].append(val_metrics['auc'])
            
            # Print epoch results
            epoch_time = time.time() - epoch_start_time
            print(f'\nEpoch {epoch + 1}/{num_epochs} - {epoch_time:.2f}s')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_metrics["loss"]:.4f}, Val Acc: {val_metrics["accuracy"]:.4f}')
            print(f'Val Precision: {val_metrics["precision"]:.4f}, Val Recall: {val_metrics["recall"]:.4f}')
            print(f'Val F1: {val_metrics["f1"]:.4f}, Val AUC: {val_metrics["auc"]:.4f}')
            
            # Check for best model
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_val_acc = val_metrics['accuracy']
                self.best_epoch = epoch + 1
                epochs_without_improvement = 0
                
                # Save best model
                self.save_checkpoint(epoch + 1, is_best=True)
                print(f'New best model saved! F1: {self.best_val_f1:.4f}')
            else:
                epochs_without_improvement += 1
            
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1, is_best=False)
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f'\nEarly stopping triggered after {epochs_without_improvement} epochs without improvement')
                break
            
            print('-' * 60)
        
        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time:.2f}s')
        print(f'Best validation F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}')
        
        # Save final history
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
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
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy')
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
        
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'] #.to(self.device)
                labels = batch['label'].to(self.device)
                
                m = batch.get('mask', None)
                if m is not None:
                    images = torch.cat([images, m], dim=1)
                images = images.to(self.device)
                
                # Convert labels to binary
                binary_labels = (labels > 0).long()
                
                # Forward pass
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                # Collect results
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(binary_labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        
        # Overall metrics
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # AUC
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0
        
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
            'class_names': ['Non-tumor', 'Tumor']
        }
        
        return results