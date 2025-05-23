import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torchvision import models
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import ResNet50_Weights
import os
import numpy as np

class LitRetinaNet(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pretrained ResNet50
        weights = ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)
        
        # Modify final layer for binary classification
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 1)  # Binary classification
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.val_acc = torchmetrics.Accuracy(task='binary')
        self.test_acc = torchmetrics.Accuracy(task='binary')
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()
        
        self.criterion = nn.BCEWithLogitsLoss()  # Binary cross entropy with logits
        
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
        x, y = batch['image'], batch['label'].float()
        logits = self(x).squeeze()
        loss = self.criterion(logits, y)
        
        # Update metrics
        self.train_acc(logits, y)
        self.train_loss(loss)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label'].float()
        logits = self(x).squeeze()
        loss = self.criterion(logits, y)
        
        # Update metrics
        self.val_acc(logits, y)
        self.val_loss(loss)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label'].float()
        logits = self(x).squeeze()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
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
            "reduce_on_plateau": True,
        }
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler} 