import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms
from retina_dataset import (
    RetinopathyDataset,
    CropByEye,
    Rescale,
    RandomCrop,
    CenterCrop,
    ToTensor,
    Normalize
)

class RetinaDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32):
        """
        Args:
            data_dir (str): Directory where retina dataset files are stored
            batch_size (int): Batch size for the dataloaders
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        # Pixel means and stds expected by models in torchvision
        self.pixel_mean = [0.485, 0.456, 0.406]
        self.pixel_std = [0.229, 0.224, 0.225]
        
        # Define transforms
        self.train_transforms = transforms.Compose([
            CropByEye(0.10, 1),
            Rescale(224),
            RandomCrop(224),
            ToTensor(),
            Normalize(mean=self.pixel_mean, std=self.pixel_std)
        ])
        
        self.val_test_transforms = transforms.Compose([
            CropByEye(0.10, 1),
            Rescale(224),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=self.pixel_mean, std=self.pixel_std)
        ])

    def setup(self, stage=None):
        # Create datasets
        self.train_ds = RetinopathyDataset(
            csv_file=os.path.join(self.data_dir, 'train.csv'),
            root_dir=self.data_dir,  # This will be used to find the 'images' folder
            transform=self.train_transforms
        )
        
        self.val_ds = RetinopathyDataset(
            csv_file=os.path.join(self.data_dir, 'val.csv'),
            root_dir=self.data_dir,
            transform=self.val_test_transforms
        )
        
        self.test_ds = RetinopathyDataset(
            csv_file=os.path.join(self.data_dir, 'test.csv'),
            root_dir=self.data_dir,
            transform=self.val_test_transforms
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        ) 