import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import random


class SuperResolutionDataset(Dataset):
    def __init__(self, LR_path, GT_path, in_memory=True, transform=None, grayscale=False):
        self.LR_path = LR_path
        self.GT_path = GT_path
        self.in_memory = in_memory
        self.transform = transform
        self.grayscale = grayscale

        self.LR_img_list = sorted(os.listdir(LR_path))
        self.GT_img_list = sorted(os.listdir(GT_path))

        if in_memory:
            self.LR_img_list = [self._load_image(os.path.join(LR_path, name)) for name in self.LR_img_list]
            self.GT_img_list = [self._load_image(os.path.join(GT_path, name)) for name in self.GT_img_list]

    def _load_image(self, path):
        img = Image.open(path)
        img = img.convert("L") if self.grayscale else img.convert("RGB")
        arr = np.array(img).astype(np.uint8)
        if self.grayscale:
            arr = arr[..., np.newaxis]
        return arr

    def __len__(self):
        return len(self.LR_img_list)

    def __getitem__(self, idx):
        if self.in_memory:
            LR = self.LR_img_list[idx].astype(np.float32)
            GT = self.GT_img_list[idx].astype(np.float32)
        else:
            LR = self._load_image(os.path.join(self.LR_path, self.LR_img_list[idx])).astype(np.float32)
            GT = self._load_image(os.path.join(self.GT_path, self.GT_img_list[idx])).astype(np.float32)

        LR = (LR / 127.5) - 1.0
        GT = (GT / 127.5) - 1.0

        sample = {'LR': LR, 'GT': GT}

        if self.transform:
            sample = self.transform(sample)

        for key in sample:
            sample[key] = sample[key].transpose(2, 0, 1).astype(np.float32)
            sample[key] = torch.from_numpy(sample[key])

        return sample

    
    
class InferenceDataset(Dataset):
    def __init__(self, LR_path, in_memory=True, grayscale=False):
        self.LR_path = LR_path
        self.in_memory = in_memory
        self.grayscale = grayscale

        self.LR_img_list = sorted(os.listdir(LR_path))

        if in_memory:
            self.LR_img_list = [self._load_image(os.path.join(LR_path, name)) for name in self.LR_img_list]

    def _load_image(self, path):
        img = Image.open(path)
        img = img.convert("L") if self.grayscale else img.convert("RGB")
        arr = np.array(img).astype(np.uint8)
        if self.grayscale:
            arr = arr[..., np.newaxis]
        return arr

    def __len__(self):
        return len(self.LR_img_list)

    def __getitem__(self, idx):
        if self.in_memory:
            LR = self.LR_img_list[idx]
        else:
            LR = self._load_image(os.path.join(self.LR_path, self.LR_img_list[idx]))

        LR = (LR / 127.5) - 1.0
        LR = LR.transpose(2, 0, 1).astype(np.float32)
        LR = torch.from_numpy(LR)

        return {'LR': LR}



class CropPatch:
    def __init__(self, scale, patch_size):
        self.scale = scale
        self.patch_size = patch_size

    def __call__(self, sample):
        LR_img, GT_img = sample['LR'], sample['GT']
        ih, iw = LR_img.shape[:2]

        ix = random.randrange(0, iw - self.patch_size + 1)
        iy = random.randrange(0, ih - self.patch_size + 1)
        tx = ix * self.scale
        ty = iy * self.scale

        LR_patch = LR_img[iy:iy + self.patch_size, ix:ix + self.patch_size]
        GT_patch = GT_img[ty:ty + self.scale * self.patch_size, tx:tx + self.scale * self.patch_size]

        return {'LR': LR_patch, 'GT': GT_patch}


class Augmentation:
    def __call__(self, sample):
        LR_img, GT_img = sample['LR'], sample['GT']
        if random.random() > 0.5:
            LR_img = np.fliplr(LR_img).copy()
            GT_img = np.fliplr(GT_img).copy()
        if random.random() > 0.5:
            LR_img = np.flipud(LR_img).copy()
            GT_img = np.flipud(GT_img).copy()
        if random.random() > 0.5:
            LR_img = np.transpose(LR_img, (1, 0, 2)).copy()
            GT_img = np.transpose(GT_img, (1, 0, 2)).copy()
        return {'LR': LR_img, 'GT': GT_img}


class ResizePair:
    def __init__(self, hr_size, scale, mode='bicubic'):
        self.hr_size = hr_size
        self.lr_size = hr_size // scale
        self.mode = mode

    def __call__(self, sample):
        GT_img = Image.fromarray((sample['GT'] * 127.5 + 127.5).astype(np.uint8))
        LR_img = Image.fromarray((sample['LR'] * 127.5 + 127.5).astype(np.uint8))

        GT_img = GT_img.resize((self.hr_size, self.hr_size), resample=Image.BICUBIC)
        LR_img = LR_img.resize((self.lr_size, self.lr_size), resample=Image.BICUBIC)

        return {
            'GT': (np.array(GT_img).astype(np.float32) / 127.5) - 1.0,
            'LR': (np.array(LR_img).astype(np.float32) / 127.5) - 1.0
        }
