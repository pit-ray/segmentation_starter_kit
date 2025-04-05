import os
from PIL import Image
from glob import glob

import torch
from torchvision import transforms
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(
            self,
            root_dir: str,
            img_height: int,
            img_width: int,
            image_dir_name: str = 'images',
            mask_dir_name: str = 'masks',
            img_ext: str = '.png'):
        super().__init__()

        self.image_filenames, self.mask_filenames = self._glob_filename(
            root_dir, image_dir_name, mask_dir_name, img_ext)

        self.to_tensor = transforms.ToTensor()

        self.resize = transforms.Resize(size=(img_height, img_width))

        self.height = img_height
        self.width = img_width

        self.color_jit = transforms.ColorJitter(
            brightness=(0.8, 1.2),
            contrast=(0.8, 1.2),
            saturation=(0.8, 1.2),
            hue=(-0.1, 0.1))

    def _glob_filename(
            self,
            root_dir,
            image_dir_name,
            mask_dir_name,
            img_ext):
        image_filenames = []
        mask_filenames = []

        filenames = glob(os.path.join(root_dir, image_dir_name, '*' + img_ext))
        for filename in filenames:
            basename = os.path.splitext(os.path.basename(filename))[0]
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
        with open(self.image_filenames[idx], 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
        img = self.to_tensor(img)

        with open(self.mask_filenames[idx], 'rb') as f:
            with Image.open(f) as mask:
                mask = mask.convert('L')
        mask = (self.to_tensor(mask) > 0).to(torch.float32)

        # Resize
        img = self.resize(img)
        mask = self.resize(mask)

        if torch.randn(1) > 0:
            img = transforms.functional.hflip(img)
            mask = transforms.functional.hflip(mask)

        if torch.randn(1) > 0:
            img = transforms.functional.vflip(img)
            mask = transforms.functional.vflip(mask)

        if torch.randn(1) > 0:
            img = self.color_jit(img)

        return {
            'image': img,
            'mask': mask
        }
