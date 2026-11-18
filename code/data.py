"""PyTorch dataset for the preprocessed LIDC-IDRI 2D slices (dataset/processed)."""
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DATA_ROOT = Path(__file__).resolve().parent.parent / "dataset" / "processed"
MEAN, STD = 0.5, 0.25  # simple fixed normalisation of the [0,1] lung-window image


class LIDCSlices(Dataset):
    """Returns (image 1xHxW float, mask 1xHxW float {0,1}, has_nodule int).

    crop:    if set, random crop of this size (training); otherwise the full 512x512 slice.
    augment: flips, 90-degree rotations, brightness/contrast jitter, gaussian noise.
    pos_crop_prob: for nodule slices, probability that the crop is centred near a nodule.
    """

    def __init__(self, split, root=DATA_ROOT, crop=None, augment=False, pos_crop_prob=0.8, in_ch=1):
        self.root = Path(root)
        self.in_ch = in_ch  # 1 = single slice (images/), 3 = 2.5D (images25d/: above, slice, below)
        meta = pd.read_csv(self.root / "metadata.csv")
        self.df = meta[meta.split == split].reset_index(drop=True)
        self.crop, self.augment, self.pos_crop_prob = crop, augment, pos_crop_prob

    def __len__(self):
        return len(self.df)

    def _random_crop(self, img, mask):
        H, W = img.shape[:2]
        c = self.crop
        ys, xs = np.nonzero(mask)
        if len(ys) and random.random() < self.pos_crop_prob:
            k = random.randrange(len(ys))
            cy = ys[k] + random.randint(-c // 3, c // 3)
            cx = xs[k] + random.randint(-c // 3, c // 3)
            y0 = int(np.clip(cy - c // 2, 0, H - c))
            x0 = int(np.clip(cx - c // 2, 0, W - c))
        else:
            y0, x0 = random.randint(0, H - c), random.randint(0, W - c)
        return img[y0:y0 + c, x0:x0 + c], mask[y0:y0 + c, x0:x0 + c]

    def __getitem__(self, i):
        row = self.df.iloc[i]
        if getattr(self, "in_ch", 1) == 3:
            img = cv2.cvtColor(cv2.imread(str(self.root / "images25d" / row.file), cv2.IMREAD_COLOR),
                               cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # H,W,3
        else:
            img = cv2.imread(str(self.root / "images" / row.file), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        mask = (cv2.imread(str(self.root / "masks" / row.file), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)

        if self.crop:
            img, mask = self._random_crop(img, mask)
        if self.augment:
            if random.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                img, mask = img[::-1], mask[::-1]
            k = random.randint(0, 3)
            img, mask = np.rot90(img, k), np.rot90(mask, k)
            if random.random() < 0.5:
                img = img * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05)
            if random.random() < 0.3:
                img = img + np.random.normal(0, 0.02, img.shape).astype(np.float32)
            if getattr(self, "in_ch", 1) == 3 and random.random() < 0.5:
                img = img[:, :, ::-1]  # swap above/below: the scan direction should not matter
            img = np.clip(img, 0, 1)

        img = (np.ascontiguousarray(img) - MEAN) / STD
        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1).contiguous() if img.ndim == 3 else img.unsqueeze(0)
        return (img,
                torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0),
                int(row.has_nodule))
