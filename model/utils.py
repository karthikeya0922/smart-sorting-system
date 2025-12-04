"""
Utility functions for model training and evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import json


def save_checkpoint(model, optimizer, epoch, val_acc, val_loss, filepath):
    """
    Save model checkpoint.
    
    Args:
        model (nn.Module): PyTorch model
        optimizer (torch.optim.Optimizer): Optimizer
        epoch (int): Current epoch
        val_acc (float): Validation accuracy
        val_loss (float): Validation loss
        filepath (str): Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
    }
    torch.save(checkpoint, filepath)
    print(f"✅ Checkpoint saved: {filepath}")


def load_checkpoint(model, filepath, optimizer=None, device='cpu'):
    """
    Load model checkpoint.
    
    Args:
        model (nn.Module): PyTorch model
        filepath (str): Path to checkpoint
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state
        device (str): Device to load model to
    
    Returns:
        int: Epoch number from checkpoint
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    val_acc = checkpoint.get('val_acc', 0.0)
    val_loss = checkpoint.get('val_loss', 0.0)
    
    print(f"✅ Checkpoint loaded: epoch {epoch}, val_acc: {val_acc:.2f}%, val_loss: {val_loss:.4f}")
    return epoch


def calculate_accuracy(outputs, labels):
    """
    Calculate accuracy from model outputs and labels.
    
    Args:
        outputs (torch.Tensor): Model outputs (logits)
        labels (torch.Tensor): Ground truth labels
    
    Returns:
        float: Accuracy (0-100)
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return 100.0 * correct / total


def plot_training_history(history, save_path='model/results/training_history.png'):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history (dict): Training history with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path (str): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training history plot saved: {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='model/results/confusion_matrix.png'):
    """
    Plot confusion matrix.
    
    Args:
        y_true (list): True labels
        y_pred (list): Predicted labels
        class_names (list): List of class names
        save_path (str): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Confusion matrix saved: {save_path}")
    
    return cm


def generate_classification_report(y_true, y_pred, class_names, save_path='model/results/classification_report.txt'):
    """
    Generate and save classification report.
    
    Args:
        y_true (list): True labels
        y_pred (list): Predicted labels
        class_names (list): List of class names
        save_path (str): Path to save the report
    """
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    # Create directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    
    print(f"✅ Classification report saved: {save_path}")
    print()
    print(report)
    
    return report


def save_training_history(history, save_path='model/results/training_history.json'):
    """
    Save training history to JSON file.
    
    Args:
        history (dict): Training history dictionary
        save_path (str): Path to save JSON file
    """
    # Create directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"✅ Training history saved: {save_path}")


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=7, min_delta=0.001, mode='min'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as improvement
            mode (str): 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        elif self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True


if __name__ == "__main__":
    print("Model Utilities Module")
    print("Available functions:")
    print("  - save_checkpoint()")
    print("  - load_checkpoint()")
    print("  - calculate_accuracy()")
    print("  - plot_training_history()")
    print("  - plot_confusion_matrix()")
    print("  - generate_classification_report()")
    print("  - EarlyStopping class")
    print("  - AverageMeter class")
