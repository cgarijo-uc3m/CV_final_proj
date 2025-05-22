import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from datasets import CIFAR10Dataset

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=32):
        """
        Args:
            data_path (str): Directory where CIFAR-10 files are stored
            batch_size (int): Batch size for the dataloaders
        """
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        
        # Define transforms
        self.train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        self.val_test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def setup(self, stage=None):
        # Create datasets
        self.train_ds = CIFAR10Dataset(
            self.data_path, 
            train=True, 
            transform=self.train_transforms
        )
        
        self.val_ds = CIFAR10Dataset(
            self.data_path, 
            train=True,  # We'll split the training data for validation
            transform=self.val_test_transforms
        )
        
        self.test_ds = CIFAR10Dataset(
            self.data_path, 
            train=False, 
            transform=self.val_test_transforms
        )
        
        # Split training data into train and validation
        train_size = int(0.9 * len(self.train_ds))
        val_size = len(self.train_ds) - train_size
        self.train_ds, self.val_ds = torch.utils.data.random_split(
            self.train_ds, 
            [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.batch_size, 
            num_workers=4
        )
