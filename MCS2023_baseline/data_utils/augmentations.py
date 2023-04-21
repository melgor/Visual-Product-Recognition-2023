import numpy as np
import torchvision as tv
import torch.nn.functional as F
from torchvision.transforms.functional import rotate


def get_train_aug(config):
    if config.dataset.augmentations == 'default':
        train_augs = tv.transforms.Compose([
            tv.transforms.RandomResizedCrop((config.dataset.input_size,
                                             config.dataset.input_size)),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])
    else:
        raise Exception("Unknonw type of augs: {}".format(
            config.dataset.augmentations
        ))
    return train_augs


class SquarePad:
    def __call__(self, image):
        c, h, w = image.size()
        max_wh = np.max([h, w])
        hp = int((max_wh - h) / 2)
        vp = int((max_wh - w) / 2)
        padding = (vp, vp, hp, hp)
        return F.pad(image, padding, 'constant', 255)


class Rotate:
    def __call__(self, image):
        c, h, w = image.size()
        if w > h:
            return image
        return rotate(image, -90, expand=True)


def get_val_aug_query(input_size: int):
    val_augs = tv.transforms.Compose([
        # Rotate(),
        # SquarePad(),
        tv.transforms.Resize((input_size,
                              input_size), antialias=True),
        tv.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        # tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                         std=[0.229, 0.224, 0.225])
    ])

    return val_augs


def get_val_aug_gallery(input_size: int):
    val_augs = tv.transforms.Compose([
        tv.transforms.Resize((input_size, input_size), antialias=True),
        tv.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    return val_augs


def get_val_bb_gallery(input_size: int):
    val_augs = tv.transforms.Compose([
        tv.transforms.Resize((input_size, input_size), antialias=True),
        tv.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    return val_augs
