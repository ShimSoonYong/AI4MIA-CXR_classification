import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report

from utils import MetricTracker, ModelEvaluator, FeatureMapLogger, log_parameter_histograms

class Trainer:
    """
    Unified class for train, validation, test, and logging.
    """
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device,
                 class_names: list, num_epochs: int = 20, save_dir: str = "results",
                 feature_layer: nn.Module = None):
        """
        Args:
            model (nn.Module): PyTorch model to train
            train_loader (DataLoader): Train data loader
            val_loader (DataLoader): Validation data loader
            criterion (nn.Module): Loss function
            optimizer (torch.optim.Optimizer): Optimizer
            device (torch.device): Hardware (CPU or CUDA)
            class_names (list): Class names of target
            num_epochs (int): Total training epochs
            save_dir (str): Result saving path (Default: "results")
            feature_layer (nn.Module, optional): A layer to extract and visualize a feature map
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.class_names = class_names
        self.num_epochs = num_epochs
        
        self.results_dir = save_dir
        self.checkpoints_dir = os.path.join(self.results_dir, "checkpoints")
        self.figures_dir = os.path.join(self.results_dir, "figures")
        
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        
        self.best_model_path = os.path.join(self.checkpoints_dir, "best_model.pth")
        
        self.metric_tracker = MetricTracker(save_dir=self.figures_dir)
        self.evaluator = ModelEvaluator(class_names=class_names, save_dir=self.figures_dir)
        
        self.feature_logger = None
        if feature_layer is not None:
            self.feature_logger = FeatureMapLogger(feature_layer, save_dir=self.figures_dir)

    def _train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            #loss = self.criterion(outputs, labels)
            loss = self.model.compute_loss(outputs, labels, self.criterion)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            #_, predicted = torch.max(outputs, 1)
            preds_out = self.model.get_predictions(outputs)
            _, predicted = torch.max(preds_out, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        return running_loss / total, correct / total

    def _validate_epoch(self, return_predictions=False):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating", leave=False)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if return_predictions:
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        if return_predictions:
            return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds), np.array(all_probs)
        return epoch_loss, epoch_acc

    def fit(self):
        """Run training and validation loop."""
        print(f"--- Traning start (Device: {self.device}, Epochs: {self.num_epochs}) ---")
        best_val_acc = 0.0
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            
            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._validate_epoch()
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            
            # Record metrics
            self.metric_tracker.update(epoch, train_loss, val_loss, train_acc, val_acc)
            
            # Parameter histograms
            log_parameter_histograms(self.model, epoch, save_dir=self.figures_dir)
            
            # Feature maps
            if self.feature_logger is not None:
                self.feature_logger.plot_feature_map(epoch_or_name=f"epoch_{epoch}")
            
            # Save best modelse
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"[*] Best model saved at epoch {epoch} with Val Acc: {best_val_acc:.4f}")
                
        print("--- Training finished ---")
        
        # 1. Learning curve plot
        self.metric_tracker.save_and_plot()
        
        # 2. Final metrics as json
        self._evaluate_final_model()


    def _evaluate_final_model(self):
        """Load best model and create metrics.json"""
        print("--- Final model evaluation ---")
        self.model.load_state_dict(torch.load(self.best_model_path))
        
        _, _, y_true, y_pred, y_prob = self._validate_epoch(return_predictions=True)
        
        self.evaluator.evaluate(y_true, y_pred, y_prob)
        
        report_dict = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        metrics_path = os.path.join(self.results_dir, "metrics.json")
        
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=4, ensure_ascii=False)
            
        print(f"[Complete] Final metrics saved at: {metrics_path}")
        print(f"[Complete] Checkpoint and visualization saved at '{self.results_dir}/'")
        
        if self.feature_logger is not None:
            self.feature_logger.remove_hook()

if __name__ == '__main__':
    print("Testing Trainer class...")
    import os
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    class DummyModel(nn.Module):
        def __init__(self, num_classes=3):
            super().__init__()
            self.features = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(16, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            return self.fc(x)
            
        def compute_loss(self, outputs, labels, criterion):
            return criterion(outputs, labels)

        def get_predictions(self, outputs):
            return outputs
    # 1. Dummy data (Batch 8, RGB, 224x224, 3 classes)
    num_samples = 32
    dummy_x = torch.randn(num_samples, 3, 224, 224)
    dummy_y = torch.randint(0, 3, (num_samples,))

    dataset = TensorDataset(dummy_x, dummy_y)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=8, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DummyModel(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    class_names = ['Cat', 'Dog', 'Bird']
    save_dir = "test_results"
    
    os.makedirs(save_dir, exist_ok=True)

    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        class_names=class_names,
        num_epochs=2,
        save_dir=save_dir,
        feature_layer=model.features
    )

    print("Starting training loop...")
    try:
        trainer.fit() 
        print("Trainer test completed successfully.")
    except Exception as e:
        print(f"Error during training: {e}")
