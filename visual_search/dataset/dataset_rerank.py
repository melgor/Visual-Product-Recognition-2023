import os
import typing as t

import torch
import torch.utils.data as data
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Compose
import torchvision.transforms as T


class Product10KDatasetReRank(data.Dataset):
    def __init__(self, image_paths, root: str, transforms: Compose):
        self.root = root
        self.image_paths = image_paths
        self.transforms = transforms
        self.normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, int]:
        impath = self.image_paths[index]
        if len(impath) == 2:
            impath, (x, y, w, h) = impath
            full_imname = os.path.join(self.root, impath)
            img = read_image(full_imname, mode=ImageReadMode.RGB)
            img = img[:, y:y + h, x:x + w]
        else:
            full_imname = os.path.join(self.root, impath)
            img = read_image(full_imname, mode=ImageReadMode.RGB)

        img = self.transforms(img)
        img = img.float() / 255.
        img = self.normalize(img)
        return img

    def __len__(self) -> int:
        return len(self.image_paths)