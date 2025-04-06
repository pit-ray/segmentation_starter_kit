import os
import random
from PIL import Image
from glob import glob

import numpy as np
import torch
from torchvision import transforms
from torch.utils import data


class SegmentationDataset(data.Dataset):
    def __init__(
            self,
            root_dir: str,
            size_hw: tuple,
            class_values: list = [0, 255],
            image_dir_name: str = 'images',
            mask_dir_name: str = 'masks',
            img_ext: str = '.png',
            is_train: bool = True,
            random_hflip: bool = True,
            random_vflip: bool = True,
            random_colorjit: bool = True,
            random_crop: bool = False,
            random_crop_size_hw: tuple = None):
        super().__init__()
        # Collect the target file names.
        self.image_filenames, self.mask_filenames = self._glob_filename(
            root_dir, image_dir_name, mask_dir_name, img_ext)

        self.height, self.width = size_hw
        self.class_values = class_values
        self.is_train = is_train
        self.random_hflip = is_train and random_hflip
        self.random_vflip = is_train and random_vflip
        self.random_colorjit = is_train and random_colorjit
        self.random_crop = is_train and random_crop

        # Define data transforms
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(size=(self.height, self.width))
        self.color_jit = transforms.ColorJitter(
            brightness=(0.8, 1.2),
            contrast=(0.8, 1.2),
            saturation=(0.8, 1.2),
            hue=(-0.1, 0.1))

        if self.random_crop:
            if random_crop_size_hw is None:
                raise ValueError(
                    'You need to specify the crop size. ',
                    'format: random_crop_size_hw=(crop_height, crop_width).')
            self.random_crop_size_hw = random_crop_size_hw

    def _glob_filename(
            self,
            root_dir: str,
            image_dir_name: str,
            mask_dir_name: str,
            img_ext: str) -> tuple:
        image_filenames = []
        mask_filenames = []

        filenames = glob(os.path.join(root_dir, image_dir_name, '*' + img_ext))
        for filename in filenames:
            basename = os.path.splitext(os.path.basename(filename))[0]

            # Collect images and masks based on the names of image files.
            img_file = os.path.join(
                root_dir, image_dir_name, basename + img_ext)
            mask_file = os.path.join(
                root_dir, mask_dir_name, basename + '.png')

            if os.path.isfile(img_file) and os.path.isfile(mask_file):
                image_filenames.append(img_file)
                mask_filenames.append(mask_file)

        return image_filenames, mask_filenames

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx: int) -> dict:
        # Load the image and the mask
        with open(self.image_filenames[idx], 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
        img = self.to_tensor(img)

        with open(self.mask_filenames[idx], 'rb') as f:
            with Image.open(f) as mask_values:
                mask_values = mask_values.convert('L')
        mask_values = torch.from_numpy(np.array(mask_values, dtype=np.uint16))
        H, W = mask_values.shape[-2:]
        mask_values = mask_values.view(1, H, W)
        mask = torch.zeros_like(mask_values, dtype=torch.int64)
        for class_id, value in enumerate(self.class_values):
            mask[mask_values == value] = class_id

        # Resize these images
        img = self.resize(img)
        mask = self.resize(mask)

        if self.random_colorjit and random.random() < 0.5:
            img = self.color_jit(img)

        if self.random_hflip and random.random() < 0.5:
            img = transforms.functional.hflip(img)
            mask = transforms.functional.hflip(mask)

        if self.random_vflip and random.random() < 0.5:
            img = transforms.functional.vflip(img)
            mask = transforms.functional.vflip(mask)

        if self.random_crop and random.random() < 0.5:
            i, j, h, w = transforms.RandomCrop.get_params(
                img, self.random_crop_size_hw)
            img = transforms.functional.crop(img, i, j, h, w)
            mask = transforms.functional.crop(mask, i, j, h, w)

        return {
            'image': img,
            'mask': mask
        }
