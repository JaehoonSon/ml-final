"""
Pascal VOC 2012 reframed as binary segmentation: foreground (any object class) vs. background.
    0  = background
    1+ = foreground (collapsed from VOC classes 1-20)
    255 (boundary) -> background

Training augmentations match advML_lab2:
    1.1x resize -> random crop to SIZE
    horizontal flip (p=0.5)
    brightness jitter   (p=0.5, factor in [0.8, 1.2])
    contrast jitter     (p=0.5, factor in [0.8, 1.2])
    ImageNet normalize
Validation: deterministic resize + normalize, no augmentation.
"""

import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import functional as TF

ROOT = "data/voc"
SIZE = 224
NUM_CLASSES = 2

_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class VOCDataset(Dataset):
    def __init__(self, split: str):
        assert split in {"train", "val"}
        self.split = split
        self.dataset = VOCSegmentation(root=ROOT, year="2012", image_set=split, download=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        if self.split == "train":
            image, mask = self._augment(image, mask)
        else:
            image = image.resize((SIZE, SIZE), Image.BILINEAR)
            mask = mask.resize((SIZE, SIZE), Image.NEAREST)

        image = _NORMALIZE(TF.to_tensor(image))
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        mask = ((mask >= 1) & (mask <= 20)).long()
        return image, mask

    @staticmethod
    def _augment(image, mask):
        # upscale 1.1x then random crop to SIZE (joint on image + mask)
        up = int(SIZE * 1.1)
        image = image.resize((up, up), Image.BILINEAR)
        mask = mask.resize((up, up), Image.NEAREST)
        i = random.randint(0, up - SIZE)
        j = random.randint(0, up - SIZE)
        image = TF.crop(image, i, j, SIZE, SIZE)
        mask = TF.crop(mask, i, j, SIZE, SIZE)

        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() < 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
        if random.random() < 0.5:
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))

        return image, mask
