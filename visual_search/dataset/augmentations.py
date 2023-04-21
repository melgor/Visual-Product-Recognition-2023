import typing as t

import cv2
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_image_transforms(input_size: int) -> t.Tuple[T.Compose, T.Compose]:
    train_transforms = T.Compose(
            [
                 T.Resize(input_size, antialias=True),
                 T.AutoAugment(),
                 T.Resize((input_size, input_size), antialias=True)
            ]
        )

    valid_transform = T.Compose(
        [
         T.Resize((input_size, input_size), antialias=True),
         ]
    )
    return train_transforms, valid_transform


def transforms_cutout(image_size: int):
    train_transforms = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.5),
            A.Cutout(max_h_size=int(image_size * 0.4), max_w_size=int(image_size * 0.4), num_holes=1, p=0.5),
            ToTensorV2(),
        ])
    return train_transforms


def transforms_happy_whale(image_size):
    aug8p3 = A.OneOf([
            A.Sharpen(p=0.3),
            A.ToGray(p=0.3),
            A.CLAHE(p=0.3),
        ], p=0.5)

    train_transforms = A.Compose([
            A.ShiftScaleRotate(rotate_limit=15, scale_limit=0.1, border_mode=cv2.BORDER_REFLECT, p=0.5),
            A.Resize(image_size, image_size),
            aug8p3,
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ToTensorV2(),
        ])
    return train_transforms
