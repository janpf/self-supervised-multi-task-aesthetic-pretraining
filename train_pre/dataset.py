import logging
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.utils
import torch.utils.data
import torchvision.transforms as transforms
from imagenet_c import corrupt
from PIL import Image

from train_pre.utils import filename2path, rotatedRectWithMaxArea


class SSPexelsSmall(torch.utils.data.Dataset):
    def __init__(
        self,
        file_list_path: str,
        mapping,
        return_file_name: bool = False,
        normalize: bool = True,
        orig_dir: str = "/scratch/pexels/images_small",
        edited_dir: str = "/scratch/pexels/edited_images_small",
    ):
        self.file_list_path = file_list_path
        self.mapping = mapping

        self.return_file_name = return_file_name
        self.normalize = normalize
        self.orig_dir = orig_dir
        self.edited_dir = edited_dir

        with open(file_list_path) as f:
            file_list = f.readlines()

        file_list = [line.strip() for line in file_list]
        self.file_list = [filename2path(p) for p in file_list]

        def pad_square(im: Image.Image, min_size: int = 224, fill_color=(0, 0, 0)) -> Image.Image:
            im = transforms.Resize(224)(im)
            x, y = im.size
            size = max(min_size, x, y)
            new_im = Image.new("RGB", (size, size), fill_color)
            new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
            return transforms.Resize(224)(new_im)

        self.pad_square = pad_square

        def pixelate(x: Image.Image, severity=1) -> Image.Image:
            previous = x.size
            c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

            x = x.resize((int(previous[0] * c), int(previous[1] * c)), Image.BOX)
            x = x.resize(previous, Image.BOX)

            return x

        self.pixelate = pixelate

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx):
        try:
            return self._actualgetitem(idx)
        except:
            return self[random.randint(0, len(self))]

    def _actualgetitem(self, idx):
        data = dict()
        data["original"] = transforms.Resize(224)(
            Image.open(str(Path(self.orig_dir) / self.file_list[idx])).convert("RGB")
        )

        for style_change in self.mapping["styles_changes"]:
            parameter, change = style_change.split(";")
            change = float(change) if "." in change else int(change)
            data[style_change] = transforms.Resize(224)(
                Image.open(str(Path(self.edited_dir) / parameter / str(change) / self.file_list[idx])).convert("RGB")
            )

        for technical_change in self.mapping["technical_changes"]:
            parameter, change = technical_change.split(";")
            change = int(change)
            if parameter == "pixelate":
                data[technical_change] = self.pixelate(data["original"], severity=change)
            else:
                img = corrupt(np.array(data["original"]), severity=change, corruption_name=parameter)
                data[technical_change] = Image.fromarray(img)

        crop_original = transforms.Resize(336)(
            Image.open(str(Path(self.orig_dir) / self.file_list[idx])).convert("RGB")
        )
        for composition_change in self.mapping["composition_changes"]:
            parameter, change = composition_change.split(";")
            change = int(change)

            if "ratio" == parameter:
                img_size = (data["original"].size[1], data["original"].size[0])
                if change > 0:
                    img_resize = (img_size[0] * (1 + change * (1 / 5)), img_size[1])
                else:
                    img_resize = (img_size[0], img_size[1] * (1 + -change * (1 / 5)))
                img_resize = (round(img_resize[0]), round(img_resize[1]))
                img = transforms.Resize(img_resize)(data["original"])
                data[composition_change] = transforms.CenterCrop(img_size)(img)

            elif "rotate" == parameter:
                max_change = 10

                rotated = crop_original.rotate(change, Image.BICUBIC, True)

                w, h = rotatedRectWithMaxArea(crop_original.size[0], crop_original.size[1], max_change)

                img = transforms.CenterCrop((h, w))(rotated)
                img = transforms.Resize(224)(img)
                data[composition_change] = transforms.CenterCrop((data["original"].size[1], data["original"].size[0]))(
                    img
                )

                if not "rotate_original" in data.keys():
                    rotated = crop_original.rotate(0, Image.BICUBIC, True)
                    w, h = rotatedRectWithMaxArea(crop_original.size[0], crop_original.size[1], max_change)
                    img = transforms.CenterCrop((h, w))(rotated)
                    img = transforms.Resize(224)(img)
                    data["rotate_original"] = transforms.CenterCrop(
                        (data["original"].size[1], data["original"].size[0])
                    )(img)

            elif "crop" in parameter:
                crop_size = data["original"].size

                center_left = round(crop_original.size[0] / 2 - crop_size[0] / 2)
                center_right = round(crop_original.size[0] / 2 + crop_size[0] / 2)
                center_top = round(crop_original.size[1] / 2 - crop_size[1] / 2)
                center_bottom = round(crop_original.size[1] / 2 + crop_size[1] / 2)

                v_move = 0  # centered
                h_move = 0  # centered
                if parameter == "vcrop":
                    v_move = change
                elif parameter == "hcrop":
                    h_move = change
                elif parameter == "leftcornerscrop":
                    h_move = -abs(change)
                    v_move = change
                elif parameter == "rightcornerscrop":
                    h_move = abs(change)
                    v_move = change

                offset_left = round(center_left * (-h_move * (0.2)))
                offset_top = round(center_top * (v_move * (0.2)))

                center_left -= offset_left
                center_right -= offset_left
                center_top -= offset_top
                center_bottom -= offset_top

                data[composition_change] = crop_original.crop((center_left, center_top, center_right, center_bottom))

                if not "crop_original" in data.keys():
                    crop_size = data["original"].size

                    center_left = round(crop_original.size[0] / 2 - crop_size[0] / 2)
                    center_right = round(crop_original.size[0] / 2 + crop_size[0] / 2)
                    center_top = round(crop_original.size[1] / 2 - crop_size[1] / 2)
                    center_bottom = round(crop_original.size[1] / 2 + crop_size[1] / 2)

                    data["crop_original"] = crop_original.crop((center_left, center_top, center_right, center_bottom))

        for k in data.keys():
            data[k] = self.pad_square(data[k])
            data[k] = transforms.ToTensor()(data[k])
            if self.normalize:
                data[k] = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(data[k])

        if self.return_file_name:
            data["file_name"] = self.file_list[idx]
        return data


class FolderDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir: str, normalize: bool, accepted_extensions: List[str] = ["jpg", "bmp", "png"]):
        self.image_dir = image_dir
        self.normalize = normalize
        self.files = [
            str(val) for val in Path(image_dir).glob("**/*") if val.name.split(".")[-1].lower() in accepted_extensions
        ]
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

    def __getitem__(self, idx: int):
        try:
            return self._actualgetitem(idx)
        except:
            return self[random.randint(0, len(self))]

    def _actualgetitem(self, idx: int):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        img = self.pad_square(img)
        img = transforms.ToTensor()(img)
        if self.normalize:
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return {"img": img, "path": path, "idx": idx}


class AVA(FolderDataset):
    def __init__(self, image_dir: str = "/scratch/AVA/images", normalize: bool = True):
        super().__init__(image_dir=image_dir, normalize=normalize)


class TID2013(FolderDataset):
    def __init__(self, image_dir: str = "/scratch/tid2013", normalize: bool = True):
        super().__init__(image_dir=image_dir, normalize=normalize)
