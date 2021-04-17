import logging
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.utils
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image


class AVA(torch.utils.data.Dataset):
    def __init__(
        self,
        mode: str,
        image_dir: str = "/scratch/AVA/images",
        percentage_of_dataset: int = None,
        horizontal_flip: bool = True,
        normalize: bool = True,
    ):
        self.image_dir = image_dir
        self.normalize = normalize
        self.mode = mode
        self.horizontal_flip = horizontal_flip
        self.percentage_of_dataset = percentage_of_dataset

        self.files = pd.read_csv(f"/scratch/AVA/{mode}_labels.csv")
        if self.percentage_of_dataset is not None:
            self.files = self.files[: int(len(self.files) * (self.percentage_of_dataset / 100))]

        logging.info(f"found {len(self.files)} files")

        def pad_square(im: Image.Image, min_size: int = 224, fill_color=(0, 0, 0)) -> Image.Image:
            im = transforms.Resize(224)(im)
            x, y = im.size
            size = max(min_size, x, y)
            new_im = Image.new("RGB", (size, size), fill_color)
            new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
            return transforms.Resize(224)(new_im)

        self.pad_square = pad_square

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx):
        try:
            return self._actualgetitem(idx)
        except:
            return self[random.randint(0, len(self))]

    def _actualgetitem(self, idx: int):
        path = str(int(self.files.iloc[idx][0])) + ".jpg"
        img = Image.open(Path(self.image_dir) / path).convert("RGB")
        img = self.pad_square(img)
        if self.horizontal_flip:
            img = transforms.RandomHorizontalFlip()(img)
        img = transforms.ToTensor()(img)
        if self.normalize:
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        labels = list(self.files.iloc[idx][1:])
        labels = [i / sum(labels) for i in labels]
        return {"img": img, "path": path, "y_true": torch.Tensor(labels)}
