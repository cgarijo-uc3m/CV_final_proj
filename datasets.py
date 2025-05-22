import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pickle

def unpickle(file):
    """Load a CIFAR-10 batch file."""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class CIFAR10Dataset(Dataset):
    def __init__(self, data_path, train=True, transform=None):
        """
        Args:
            data_path (str): Directory where CIFAR-10 files are stored
            train (bool): If True, loads training data, else loads test data
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_path = data_path
        self.transform = transform
        self.train = train

        self.data = []
        self.labels = []

        if self.train:
            for i in range(1, 6):
                batch_file = os.path.join(data_path, f'data_batch_{i}')
                batch_dict = unpickle(batch_file)
                self.data.append(batch_dict[b'data'])
                self.labels.extend(batch_dict[b'labels'])
        else:
            batch_file = os.path.join(data_path, 'test_batch')
            batch_dict = unpickle(batch_file)
            self.data.append(batch_dict[b'data'])
            self.labels.extend(batch_dict[b'labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]  # shape: (3, 32, 32)
        label = self.labels[index]

        # Convert to HWC format and then to PIL Image
        img = img.transpose(1, 2, 0)  # CHW -> HWC
        img = Image.fromarray(img.astype(np.uint8))  # to PIL

        if self.transform:
            img = self.transform(img)

        return img, label
