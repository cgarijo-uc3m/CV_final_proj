import os
import torch
import pandas as pd
import numpy as np
from skimage import io, transform, util, color
import cv2
from torch.utils.data import Dataset

class CropByEye(object):
    """Crop the image using around the eye."""
    def __init__(self, threshold, border):
        self.threshold = threshold
        assert isinstance(border, (int, tuple))
        if isinstance(border, int):
            self.border = (border, border)
        else:
            self.border = border

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        h, w = image.shape[:2]
        imgray = color.rgb2gray(image)
        # Compute the mask
        th, mask = cv2.threshold(imgray, self.threshold, 1, cv2.THRESH_BINARY)
        # Compute the coordinates of the bounding box that contains the mask
        sidx = np.nonzero(mask)
        # In case the mask is too small, implies malfunctioning
        if len(sidx[0]) < 20:
            return {'image': image, 'eye': eye, 'label': label}
        minx = np.maximum(sidx[1].min() - self.border[1], 0)
        maxx = np.minimum(sidx[1].max() + 1 + self.border[1], w)
        miny = np.maximum(sidx[0].min() - self.border[0], 0)
        maxy = np.minimum(sidx[0].max() + 1 + self.border[1], h)
        # Crop the image
        image = image[miny:maxy, minx:maxx, ...]
        return {'image': image, 'eye': eye, 'label': label}

class Rescale(object):
    """Re-scale image to a predefined size."""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        image = transform.resize(image, (new_h, new_w))
        return {'image': image, 'eye': eye, 'label': label}

class RandomCrop(object):
    """Randomly crop the image."""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        if h > new_h:
            top = np.random.randint(0, h - new_h)
        else:
            top = 0

        if w > new_w:
            left = np.random.randint(0, w - new_w)
        else:
            left = 0

        image = image[top: top + new_h, left: left + new_w]
        return {'image': image, 'eye': eye, 'label': label}

class CenterCrop(object):
    """Crop the central area of the image."""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        rem_h = h - new_h
        rem_w = w - new_w

        if h > new_h:
            top = int(rem_h/2)
        else:
            top = 0

        if w > new_w:
            left = int(rem_w/2)
        else:
            left = 0

        image = image[top: top + new_h, left: left + new_w]
        return {'image': image, 'eye': eye, 'label': label}

class ToTensor(object):
    """Convert ndarrays into pytorch tensors."""
    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        # Change axes from H x W x C to C x H x W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        label = torch.tensor(label, dtype=torch.long)
        return {'image': image, 'eye': eye, 'label': label}

class Normalize(object):
    """Normalize data by subtracting means and dividing by standard deviations."""
    def __init__(self, mean, std):
        assert len(mean) == len(std), 'Length of mean and std vectors is not the same'
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        c, h, w = image.shape
        assert c == len(self.mean), 'Length of mean and image is not the same'
        dtype = image.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=image.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=image.device)
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return {'image': image, 'eye': eye, 'label': label}

class RetinopathyDataset(Dataset):
    """Retinopathy dataset."""
    def __init__(self, csv_file, root_dir, transform=None, maxSize=0):
        """
        Args:
            csv_file (string): Path to csv file with annotations
            root_dir (string): Root directory where 'images' folder is located
            transform (callable, optional): Optional transform to be applied on a sample
            maxSize (int): If > 0, use only maxSize samples (for debugging)
        """
        self.dataset = pd.read_csv(csv_file, header=0, dtype={'id': str, 'eye': int, 'label': int})
        
        if maxSize > 0:
            idx = np.random.RandomState(seed=42).permutation(range(len(self.dataset)))
            reduced_dataset = self.dataset.iloc[idx[0:maxSize]]
            self.dataset = reduced_dataset.reset_index(drop=True)
            
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'images')
        self.transform = transform
        self.levels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
        self.classes = ['No DR', 'DR']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Read the image
        img_name = os.path.join(self.img_dir, self.dataset.id[idx] + '.jpg')
        image = io.imread(img_name)
        
        # If the image is from right eye, mirror it
        if self.dataset.eye[idx] == 1:
            image = image[:, ::-1, :]
            
        sample = {
            'image': image,
            'eye': self.dataset.eye[idx],
            'label': (self.dataset.label[idx] > 0).astype(dtype=np.int64)
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample 