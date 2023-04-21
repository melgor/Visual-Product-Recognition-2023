import os
import typing as t
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

from visual_search.dataset.dataset import Product10KDataset, SubmissionDataset, AmazonReview
from visual_search.model.xbm import ClassBalancedBatchSampler


class XBMDataModule(pl.LightningDataModule):
    def __init__(self, data_config,
                 train_data_transform: t.Optional[transforms.Compose] = None,
                 valid_data_transform: t.Optional[transforms.Compose] = None):
        super().__init__()
        self.train_path_dataset = os.path.join(data_config["product10k"]["train_valid_root"], data_config["product10k"]["train_list"])
        self.train_image_root = os.path.join(data_config["product10k"]["train_valid_root"], data_config["product10k"]["train_prefix"])
        self.test_path_dataset = os.path.join(data_config["product10k"]["train_valid_root"], data_config["product10k"]["val_list"])
        self.test_image_root = os.path.join(data_config["product10k"]["train_valid_root"], data_config["product10k"]["val_prefix"])

        self.eval_root = data_config["dev_set"]["eval_root"]
        self.gallery = os.path.join(data_config["dev_set"]["eval_root"], data_config["dev_set"]["gallery"])
        self.queries = os.path.join(data_config["dev_set"]["eval_root"], data_config["dev_set"]["queries"])


        self.train_path_dataset_amazon = os.path.join(data_config["amazon_review"]["train_valid_root"], data_config["amazon_review"]["train_list"])
        self.train_image_root_amazon = os.path.join(data_config["amazon_review"]["train_valid_root"], data_config["amazon_review"]["train_prefix"])
        self.batch_size = data_config["batch_size"]
        self.train_data_transform = train_data_transform
        self.valid_data_transform = valid_data_transform
        self.num_workers = data_config["num_workers"]

        # self.dims is returned when you call dm.size()
        self.dims = (1, 4)
        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_dataset_10k = Product10KDataset(self.train_path_dataset, self.train_image_root, transforms=self.train_data_transform)
            test_dataset_10k = Product10KDataset(self.test_path_dataset, self.test_image_root, transforms=self.train_data_transform)
            train_dataset_review = AmazonReview(self.train_path_dataset_amazon, self.train_image_root_amazon,
                                                transforms=self.train_data_transform, add_to_target=10000)
            self.train_targets = torch.tensor(train_dataset_10k.targets + test_dataset_10k.targets + train_dataset_review.targets, dtype=torch.int)
            self.train_dataset = ConcatDataset([train_dataset_10k, test_dataset_10k, train_dataset_review])
            self.train_dataset.value_counts = train_dataset_10k.value_counts

            self.query_dataset = SubmissionDataset(root=self.eval_root, annotation_file=self.queries, transforms=self.valid_data_transform,
                                                   with_bbox=True)
            self.gallery_dataset = SubmissionDataset(root=self.eval_root, annotation_file=self.gallery, transforms=self.valid_data_transform)
            self.gallery_query_dataset = ConcatDataset([self.query_dataset, self.gallery_dataset])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
                            self.train_dataset,
                            batch_sampler=ClassBalancedBatchSampler(self.train_targets, self.batch_size),
                            num_workers=self.num_workers,
                         )

    def val_dataloader(self) -> t.List[DataLoader]:
        return [DataLoader(
                self.gallery_query_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False
            )]

    def test_dataloader(self) -> t.List[DataLoader]:
        return self.val_dataloader()
