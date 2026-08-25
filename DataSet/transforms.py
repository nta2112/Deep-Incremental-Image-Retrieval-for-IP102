"""
Custom transforms for image preprocessing.
"""
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import random


class CovertBGR(object):
    """Convert RGB image to BGR"""
    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        if img.ndim == 3 and img.shape[2] == 3:
            return img[:, :, ::-1]
        return img


class Resize(object):
    """Resize image to given size"""
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
    
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        return img.resize((self.size, self.size), self.interpolation)


class RandomResizedCrop(object):
    """Random resized crop"""
    def __init__(self, scale=(0.08, 1.0), size=224, interpolation=Image.BILINEAR):
        self.scale = scale
        self.size = size
        self.interpolation = interpolation
    
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        
        width, height = img.size
        area = width * height
        
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (np.log(3/4), np.log(4/3))
            aspect_ratio = np.exp(random.uniform(*log_ratio))
            
            w = int(round(np.sqrt(target_area * aspect_ratio)))
            h = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if w <= width and h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                img = img.crop((j, i, j + w, i + h))
                return img.resize((self.size, self.size), self.interpolation)
        
        # Fallback to center crop
        return CenterCrop(self.size)(img)


class RandomHorizontalFlip(object):
    """Random horizontal flip"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class CenterCrop(object):
    """Center crop"""
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)
    
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        return T.CenterCrop(self.size)(img)


class ToTensor(object):
    """Convert PIL Image or numpy array to tensor"""
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
        elif isinstance(img, Image.Image):
            img = T.ToTensor()(img)
        return img


class Normalize(object):
    """Normalize tensor with mean and std"""
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
    
    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


class Compose(object):
    """Compose multiple transforms"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


# Alias for torchvision compatibility
Compose = T.Compose
Resize = T.Resize
RandomResizedCrop = T.RandomResizedCrop
RandomHorizontalFlip = T.RandomHorizontalFlip
CenterCrop = T.CenterCrop
ToTensor = T.ToTensor
Normalize = T.Normalize