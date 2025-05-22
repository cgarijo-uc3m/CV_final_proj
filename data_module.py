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
    

class DataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_val_dataset,
            test_dataset,
            train_transforms=None,
            val_test_transforms=None,
            batch_size=32,
            num_workers=4,
            split_ratio=0.9
    ):
        """
        Args:
            train_val_dataset: Dataset instance for training/validation
            test_dataset: Dataset instance for testing
            train_transforms: Transforms to apply to training data
            val_test_transforms: Transforms to apply to validation/test data
            batch_size (int): Batch size for the dataloaders
            num_workers (int): Number of workers for dataloaders
            split_ratio (float): Ratio for train/val split
        """
        super().__init__()
        self.train_val_dataset = train_val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.num_workers = num_workers
        
        
        self.train_transforms = train_transforms
        self.val_test_transforms = val_test_transforms

    def setup(self, stage=None):
        # Create datasets
        self.test_ds = self.test_dataset
        
        # Split training data into train and validation
        train_size = int(self.split_ratio * len(self.train_val_dataset))
        val_size = len(self.train_val_dataset) - train_size
        self.train_ds, self.val_ds = torch.utils.data.random_split(
            self.train_val_dataset, 
            [train_size, val_size]
        )
        # A little bit shabby but we load transforms into datasets after the instances are created.
        self.train_ds.transform = self.train_transforms
        self.val_ds.transform = self.val_test_transforms
        self.test_ds.transform = self.val_test_transforms

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )

