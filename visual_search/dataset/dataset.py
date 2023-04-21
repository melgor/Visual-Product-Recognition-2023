import os
import typing as t

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torch.utils.data import ConcatDataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Compose
import torchvision.transforms as T


class Product10KDataset(data.Dataset):
    def __init__(self, annotation_file: str, root: str, transforms: Compose):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)[["name", "class"]]
        self.targets = self.imlist["class"].values.tolist()
        self.transforms = transforms
        self.normalize = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.add_to_target = 0
        self.value_counts = self.imlist.groupby("class").apply(lambda x: x.shape[0]).values

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, int]:
        impath, target = self.imlist.iloc[index]
        full_imname = os.path.join(self.root, impath)
        img = read_image(full_imname, mode=ImageReadMode.RGB)
        img = self.transforms(img)

        img = img.float() / 255.
        img = self.normalize(img)
        return img, int(target + self.add_to_target)

    def __len__(self) -> int:
        return len(self.imlist)

    @property
    def num_classes(self) -> int:
        return self.imlist["class"].max()


class SOPDataset(data.Dataset):
    def __init__(self, annotation_file: str, root: str, transforms: Compose):
        self.root = root
        self.imlist = pd.read_csv(annotation_file, sep=" ")[["class_id", "path"]]
        self.transforms = transforms
        self.normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.add_to_target = 0

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, int]:
        target, impath = self.imlist.iloc[index]
        full_imname = os.path.join(self.root, impath)
        img = read_image(full_imname, mode=ImageReadMode.RGB)
        img = self.transforms(img)
        img = img.float() / 255.
        img = self.normalize(img)
        return img, int(target + self.add_to_target)

    def __len__(self) -> int:
        return self.imlist.shape[0]

    @property
    def num_classes(self) -> int:
        return self.imlist["class_id"].max()


class CTSDataset(data.Dataset):
    """Consumer-to-Shop dataset"""
    def __init__(self, annotation_file: str, root: str, transforms: Compose, add_to_target: int = 0):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)[["image_name", "label_enc"]]
        self.transforms = transforms
        self.add_to_target = add_to_target
        self.normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, int]:
        impath, target = self.imlist.iloc[index]
        full_imname = os.path.join(self.root, impath)
        img = read_image(full_imname, mode=ImageReadMode.RGB)
        img = self.transforms(img)
        img = img.float() / 255.
        img = self.normalize(img)
        return img, int(target + self.add_to_target)

    def __len__(self) -> int:
        return len(self.imlist)

    @property
    def num_classes(self) -> int:
        return self.imlist["label_enc"].max()


class AliExpress(data.Dataset):
    """AliExpress dataset"""
    def __init__(self, annotation_file: str, root: str, transforms: Compose, add_to_target: int = 0):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)[["path", "label_enc"]]
        self.transforms = transforms
        self.add_to_target = add_to_target
        self.normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, int]:
        impath, target = self.imlist.iloc[index]
        full_imname = os.path.join(self.root, impath)
        img = read_image(full_imname, mode=ImageReadMode.RGB)
        img = self.transforms(img)
        img = img.float() / 255.
        img = self.normalize(img)
        return img, int(target + self.add_to_target)

    def __len__(self) -> int:
        return len(self.imlist)

    @property
    def num_classes(self) -> int:
        return self.imlist["label_enc"].max()


class AmazonReview(data.Dataset):
    """AmazonReview dataset"""
    def __init__(self, annotation_file: str, root: str, transforms: Compose, add_to_target: int = 0):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)[["image_path", "label"]]
        self.targets = [elem + add_to_target for elem in self.imlist["label"].tolist()]
        self.transforms = transforms
        self.add_to_target = add_to_target
        self.normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.value_counts = self.imlist.groupby("label").apply(lambda x: x.shape[0]).values

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, int]:
        impath, target = self.imlist.iloc[index]
        full_imname = os.path.join(self.root, impath)
        img = read_image(full_imname, mode=ImageReadMode.RGB)
        img = self.transforms(img)
        img = img.float() / 255.
        img = self.normalize(img)
        return img, int(target + self.add_to_target)

    def __len__(self) -> int:
        return len(self.imlist)

    @property
    def num_classes(self) -> int:
        return self.imlist["label_enc"].max()


class SubmissionDatasetForTraining(data.Dataset):
    def __init__(self, root: str, annotation_file: str, transforms: Compose, with_bbox: bool = False):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.with_bbox = with_bbox
        # self.normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.normalize = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, int]:
        full_imname = os.path.join(self.root, self.imlist['img_path'][index])
        target = self.imlist['target'][index]
        img = read_image(full_imname, mode=ImageReadMode.RGB)

        if self.with_bbox:
            x, y, w, h = self.imlist.loc[index, 'bbox_x':'bbox_h']
            img = img[:, y:y+h, x:x+w]

        img = self.transforms(img)
        img = img.float() / 255.
        img = self.normalize(img)
        return img, target

    def __len__(self) -> int:
        return len(self.imlist)


class ConcatDatasetWithTargetMove(ConcatDataset):
    def __init__(self, datasets: t.List[data.Dataset]):
        current_max_label = 0
        for dataset in datasets:
            dataset.add_to_target = current_max_label
            current_max_label += dataset.num_classes + 1
        print("Number of classes:", current_max_label)
        super().__init__(datasets)


class SubmissionDataset(data.Dataset):
    def __init__(self, root: str, annotation_file: str, transforms: Compose, with_bbox: bool = False):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.with_bbox = with_bbox
        # self.normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.normalize = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

    def remove_background(self, image: np.ndarray, bg_color: int = 255) -> np.ndarray:
        # assumes rgb image (w, h, c) # 0.304
        intensity_img = np.mean(image, axis=2)

        # identify indices of non-background rows and columns, then look for min/max indices
        non_bg_rows = np.nonzero(np.mean(intensity_img, axis=1) != bg_color)
        non_bg_cols = np.nonzero(np.mean(intensity_img, axis=0) != bg_color)
        r1, r2 = np.min(non_bg_rows), np.max(non_bg_rows)
        c1, c2 = np.min(non_bg_cols), np.max(non_bg_cols)

        # return cropped image
        return image[r1:r2 + 1, c1:c2 + 1, :]

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, int]:
        full_imname = os.path.join(self.root, self.imlist['img_path'][index])
        img = read_image(full_imname, mode=ImageReadMode.RGB)

        if self.with_bbox:
            x, y, w, h = self.imlist.loc[index, 'bbox_x':'bbox_h']
            img = img[:, y:y+h, x:x+w]
        else:
            img_np = img.permute(1, 2, 0).numpy()
            img_np = self.remove_background(img_np)
            img = torch.from_numpy(img_np).permute(2, 0, 1)


        img = self.transforms(img)
        img = img.float() / 255.
        img = self.normalize(img)
        product_id = self.imlist["product_id"][index]
        return img, product_id

    def __len__(self) -> int:
        return len(self.imlist)
