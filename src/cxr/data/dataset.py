"""PyTorch dataset, augmentations and Lightning data module."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from cxr.data.store import IMAGES_FILE, load_metadata
from cxr.labels import LABELS
from cxr.preprocess import IMAGENET_MEAN, IMAGENET_STD, to_model_input

_MEAN = torch.from_numpy(IMAGENET_MEAN)
_STD = torch.from_numpy(IMAGENET_STD)


def build_train_augment(img_size: int) -> v2.Compose:
    # No horizontal flip: the heart sits on the left, flipping creates anatomically wrong images
    # (and dextrocardia is itself a finding).
    return v2.Compose(
        [
            v2.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.9, 1.1), antialias=True),
            v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            v2.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    )


class CXRDataset(Dataset):
    def __init__(self, data_dir: str | Path, frame, img_size: int, train: bool, clahe_p: float = 0.25):
        self.images_path = str(Path(data_dir) / IMAGES_FILE)
        self.rows = frame["idx"].to_numpy()
        self.labels = frame[LABELS].to_numpy(np.float32)
        self.img_size = img_size
        self.train = train
        self.clahe_p = clahe_p
        self.augment = build_train_augment(img_size) if train else None
        self._images = None  # opened lazily so each DataLoader worker gets its own memmap

    def __len__(self) -> int:
        return len(self.rows)

    def _image(self, i: int) -> np.ndarray:
        if self._images is None:
            self._images = np.load(self.images_path, mmap_mode="r")
        return np.array(self._images[self.rows[i]])

    def __getitem__(self, i: int):
        img = self._image(i)
        if self.train:
            if random.random() < self.clahe_p:
                img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
            t = self.augment(torch.from_numpy(img)[None])
            x = (t.float().div(255.0).expand(3, -1, -1) - _MEAN) / _STD
        else:
            x = torch.from_numpy(to_model_input(img, self.img_size))
        return x, torch.from_numpy(self.labels[i])


class CXRDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, img_size: int = 224, batch_size: int = 64, num_workers: int = 2):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.meta = None

    def setup(self, stage: str | None = None) -> None:
        if self.meta is None:
            self.meta = load_metadata(self.data_dir)

    def frame(self, split: str):
        self.setup()
        return self.meta[self.meta["split"] == split].reset_index(drop=True)

    def pos_weight(self, cap: float = 10.0) -> list[float]:
        """neg/pos ratio per class, capped: rare classes (Hernia ~0.2%) would otherwise dominate."""
        y = self.frame("train")[LABELS].to_numpy()
        pos = y.sum(0).clip(min=1)
        return np.minimum((len(y) - pos) / pos, cap).round(3).tolist()

    def _loader(self, split: str, train: bool) -> DataLoader:
        ds = CXRDataset(self.data_dir, self.frame(split), self.img_size, train=train)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=train,
            drop_last=train,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._loader("train", train=True)

    def val_dataloader(self):
        return self._loader("val", train=False)

    def test_dataloader(self):
        return self._loader("test", train=False)
