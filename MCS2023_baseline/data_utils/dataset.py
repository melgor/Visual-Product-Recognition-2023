import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torchvision.io import read_image, ImageReadMode


class Product10KDataset(data.Dataset):
    def __init__(self, root, annotation_file, transforms, is_inference=False,
                 with_bbox=False):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.is_inference = is_inference
        self.with_bbox = with_bbox

    def __getitem__(self, index):
        if self.is_inference:
            impath, _, _ = self.imlist.iloc[index]
        else:
            impath, target, _ = self.imlist.iloc[index]

        full_imname = os.path.join(self.root, impath)
        img = read_image(full_imname, mode=ImageReadMode.RGB)
        img = img/255.
        if self.with_bbox:
            x, y, w, h = self.table.loc[index, 'bbox_x':'bbox_h']
            img = img[y:y+h, x:x+w, :]

        img = self.transforms(img)

        if self.is_inference:
            return img
        else:
            return img, target

    def __len__(self):
        return len(self.imlist)


class SubmissionDataset(data.Dataset):
    def __init__(self, root, annotation_file, transforms, with_bbox=False):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.with_bbox = with_bbox

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

    # def remove_background_from_mask(self, image: np.ndarray, bg_color: int = 0):
    #     # identify indices of non-background rows and columns, then look for min/max indices
    #     non_bg_rows = np.nonzero(np.mean(image, axis=1) != bg_color)
    #     non_bg_cols = np.nonzero(np.mean(image, axis=0) != bg_color)
    #     r1, r2 = np.min(non_bg_rows), np.max(non_bg_rows)
    #     c1, c2 = np.min(non_bg_cols), np.max(non_bg_cols)
    #
    #     # return cropped image
    #     return r1, c1, r2 + 1, c2 + 1
    #
    # def remove_background_percentile(self, image: np.ndarray, percentile: int = 90) -> np.ndarray:
    #     # assumes rgb image (w, h, c)
    #     intensity_img = np.mean(image, axis=2)
    #     bg_color = np.percentile(intensity_img, percentile)
    #
    #     # identify indices of non-background rows and columns, then look for min/max indices
    #     non_bg_rows = np.nonzero(np.mean(intensity_img, axis=1) < bg_color)
    #     non_bg_cols = np.nonzero(np.mean(intensity_img, axis=0) < bg_color)
    #     r1, r2 = np.min(non_bg_rows), np.max(non_bg_rows)
    #     c1, c2 = np.min(non_bg_cols), np.max(non_bg_cols)
    #
    #     # return cropped image
    #     return image[r1:r2 + 1, c1:c2 + 1, :]

    def __getitem__(self, index: int):
        full_imname = os.path.join(self.root, self.imlist['img_path'][index])
        img = read_image(full_imname, mode=ImageReadMode.RGB)

        if self.with_bbox:
            x, y, w, h = self.imlist.loc[index, 'bbox_x':'bbox_h']
            img = img[:, y:y+h, x:x+w]
        else:
            img_np = img.permute(1, 2, 0).numpy()
            img_np = self.remove_background(img_np)
            # img_np = self.remove_background_percentile(img_np, 80)
            img = torch.from_numpy(img_np).permute(2, 0, 1)

        img = img / 255.
        img = self.transforms(img)

        return img

    def __len__(self):
        return len(self.imlist)


class SubmissionDatasetForBB(data.Dataset):
    """
    Dataset used to get Bounding-Boxes for gallery set
    """
    def __init__(self, root, annotation_file, transforms, with_bbox=False):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.with_bbox = with_bbox

    def __getitem__(self, index):
        full_imname = os.path.join(self.root, self.imlist['img_path'][index])
        img = read_image(full_imname, mode=ImageReadMode.RGB)
        img = img / 255.
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.imlist)


class SubmissionDatasetGalleryWithBB(data.Dataset):
    """
    Gallery dataset with BB
    """
    def __init__(self, root, annotation_file, transforms, boxes):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.boxes = boxes

    def crop_image(self, img: torch.Tensor, index: int):
        y1, y2, x1, x2 = self.boxes[index]
        _, h, w = img.size()
        y1, y2 = int(y1 * h), int(y2 * h)
        x1, x2 = int(x1 * w), int(x2 * w)
        return img[:, y1:y2, x1:x2]

    def __getitem__(self, index):
        full_imname = os.path.join(self.root, self.imlist['img_path'][index])
        img = read_image(full_imname, mode=ImageReadMode.RGB)
        img = self.crop_image(img, index)
        img = img / 255.
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.imlist)

