import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize

class MetricTracker:
    """Track epoch-wise loss and accuracy, and save them as figures and csv"""
    def __init__(self, save_dir="results"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.history = {
            'epoch': [], 'train_loss': [], 'val_loss': [], 
            'train_acc': [], 'val_acc': []
        }

    def update(self, epoch, train_loss, val_loss, train_acc, val_acc):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)

    def save_and_plot(self):
        # CSV
        df = pd.DataFrame(self.history)
        csv_path = os.path.join(self.save_dir, "training_history.csv")
        df.to_csv(csv_path, index=False)

        # Loss curve figure
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['epoch'], self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['epoch'], self.history['val_loss'], label='Val Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "loss_curve.png"))
        plt.close()

        # Accuracy Curve figure
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['epoch'], self.history['train_acc'], label='Train Acc')
        plt.plot(self.history['epoch'], self.history['val_acc'], label='Val Acc')
        plt.title('Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "accuracy_curve.png"))
        plt.close()


class ModelEvaluator:
    """Generate Report, Confusion Matrix, ROC, PR Curve based on the model's prediction"""
    def __init__(self, class_names, save_dir="results"):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def evaluate(self, y_true, y_pred, y_prob):
        """
        y_true: GT label (1D Array)
        y_pred: Predicted label (1D Array)
        y_prob: Predicted class probabilities (2D Array, N x C)
        """
        self._save_classification_report(y_true, y_pred)
        self._plot_confusion_matrix(y_true, y_pred)
        self._plot_roc_curve(y_true, y_prob)
        self._plot_pr_curve(y_true, y_prob)

    def _save_classification_report(self, y_true, y_pred):
        # Save Classification Report as .txt
        report = classification_report(y_true, y_pred, target_names=self.class_names, digits=4)
        with open(os.path.join(self.save_dir, "classification_report.txt"), "w") as f:
            f.write(report)

    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.save_dir, "confusion_matrix.png"))
        plt.close()

    def _plot_roc_curve(self, y_true, y_prob):
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        plt.figure(figsize=(8, 6))
        
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{self.class_names[i]} (AUC = {roc_auc:.4f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "roc_curve.png"))
        plt.close()

    def _plot_pr_curve(self, y_true, y_prob):
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        plt.figure(figsize=(8, 6))
        
        for i in range(self.num_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
            plt.plot(recall, precision, label=f'{self.class_names[i]}')

        plt.xlabel('Recall (Sensitivity)')
        plt.ylabel('Precision')
        plt.title('Precision-Recall (PR) Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "pr_curve.png"))
        plt.close()


class FeatureMapLogger:
    """Extract and visualize Intermediate Feature Map of specified layer"""
    def __init__(self, module, save_dir="results/feature_maps"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.feature_map = None
        self.hook = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        # Save 1st sample's feature map only (Channels, H, W)
        self.feature_map = output[0].detach().cpu()

    def plot_feature_map(self, epoch_or_name, num_channels=16):
        """Visualize saved feature map's first few channels"""
        if self.feature_map is None:
            print("Feature map is empty. Run a forward pass first.")
            return
            
        c, h, w = self.feature_map.shape
        plot_c = min(c, num_channels)
        
        fig, axes = plt.subplots(int(np.ceil(plot_c/4)), 4, figsize=(12, 12))
        fig.suptitle(f"Feature Maps at {epoch_or_name}", fontsize=16)
        
        for i, ax in enumerate(axes.flat):
            if i < plot_c:
                ax.imshow(self.feature_map[i].numpy(), cmap='viridis')
                ax.axis('off')
            else:
                ax.axis('off')
                
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"feature_map_{epoch_or_name}.png"))
        plt.close()

    def remove_hook(self):
        self.hook.remove()


def log_parameter_histograms(model, epoch, save_dir="results/histograms"):
    """Save model weights' histograms"""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(15, 10))
    
    # Only visualize Conv2d and Linear layers' weights
    params = [p.detach().cpu().numpy().flatten() for n, p in model.named_parameters() 
              if p.requires_grad and ('weight' in n) and ('conv' in n or 'fc' in n or 'classifier' in n)]
    
    if not params:
        return
        
    for i, p in enumerate(params[:6]):  # Draw first 6 weights
        plt.subplot(2, 3, i + 1)
        sns.histplot(p, bins=50, kde=True)
        plt.title(f'Layer {i+1} Weights')
        
    plt.suptitle(f'Parameter Distributions at Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"param_hist_epoch_{epoch}.png"))
    plt.close()
