import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import numpy as np
from torchvision import models
from torch.optim import Adam
from torchmetrics.classification import Accuracy
from torchvision.models import ResNet50_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

class LitResNet(pl.LightningModule):
    def __init__(self, num_classes=10, lr=1e-3):  # 10 classes for CIFAR-10
        super().__init__()
        self.save_hyperparameters()
        
        weights = ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)
        
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        # Metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_acc_history = []
        self.train_loss_history = []
        self.val_acc_history = []
        
        self.test_preds = []
        self.test_targets = []
        self.test_probs = []
        
        self.logger_dir = None
        
        # Initialize current metrics
        self.current_train_acc = 0.0
        self.current_train_loss = 0.0
        
    def on_train_start(self):
        if self.logger and hasattr(self.logger, 'log_dir'):
            self.logger_dir = self.logger.log_dir 
        else:
            self.logger_dir = "logs/unknown_version"  
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        self.train_acc(logits, y)
        self.train_loss(loss)
        
        return loss

    def on_train_epoch_end(self):
        # Compute but DON'T log yet
        self.current_train_acc = self.train_acc.compute().item()
        self.current_train_loss = self.train_loss.compute().item()
        
        # Store in history
        self.train_acc_history.append(self.current_train_acc)
        self.train_loss_history.append(self.current_train_loss)
        
        # Reset
        self.train_acc.reset()
        self.train_loss.reset()
        

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        self.val_acc(logits, y)
        self.val_loss(loss)

    def on_validation_epoch_end(self):
        # Compute validation metrics
        val_acc = self.val_acc.compute().item()
        val_loss = self.val_loss.compute().item()
        
        # Store in history
        self.val_acc_history.append(val_acc)
        
        # NOW log everything together in ONE row
        self.log('train_acc', self.current_train_acc, prog_bar=True)
        self.log('train_loss', self.current_train_loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)
        self.log('val_loss', val_loss, prog_bar=True)
        
        # Print metrics for SLURM visibility
        print(f"\nEpoch {self.current_epoch}:")
        print(f"Train Acc: {self.current_train_acc:.4f}, Train Loss: {self.current_train_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}\n")
        
        # Reset
        self.val_acc.reset()
        self.val_loss.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        loss = self.criterion(logits, y)

        # Update metrics
        self.test_acc(logits, y)
        self.test_loss(loss)
        
        # Store predictions
        self.test_probs.append(probs.cpu())
        self.test_preds.append(preds.cpu())
        self.test_targets.append(y.cpu())
        
        return loss

    def on_test_epoch_end(self):
        # Log final metrics
        final_test_loss = self.test_loss.compute()
        final_test_acc = self.test_acc.compute()
        
        self.log("final_test_loss", final_test_loss, prog_bar=True)
        self.log("final_test_acc", final_test_acc, prog_bar=True)
        
        # Save test results
        test_results_path = os.path.join(self.logger_dir, "test_results.npz")
        np.savez(
            test_results_path,
            probs=torch.cat(self.test_probs).numpy(),
            preds=torch.cat(self.test_preds).numpy(),
            targets=torch.cat(self.test_targets).numpy()
        )
        
        # Reset metrics
        self.test_loss.reset()
        self.test_acc.reset()
        self.test_probs.clear()
        self.test_preds.clear()
        self.test_targets.clear()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)

        scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3),
            "monitor": "val_loss",
            "interval": "epoch",
            "reduce_on_plateau" : True,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_end(self):
        # Save training metrics
        train_metrics_path = os.path.join(self.logger_dir, "training_metrics.npz")

        np.savez(
            train_metrics_path,
            train_acc=np.array(self.train_acc_history),
            val_acc=np.array(self.val_acc_history)
        )
