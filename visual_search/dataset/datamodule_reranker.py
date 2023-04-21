import os
import random
import typing as t
from collections import defaultdict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from visual_search.dataset.dataset import Product10KDataset
from visual_search.dataset.dataset_rerank import Product10KDatasetReRank


class ClassBalancedSampler:
    def __init__(self, targets: t.List[int], batch_size: int, examples_per_class: int, epoch_size: t.Optional[int] = None):
        self.targets = targets
        self.batch_size = batch_size
        self.epoch_size = epoch_size

        self.reverse_index = self.build_reverse_index()
        self.classes = list(self.reverse_index.keys())
        self.num_classes = len(self.reverse_index.keys())
        self.examples_per_class = examples_per_class
        self.classes_in_batch = batch_size // examples_per_class
        self.start_idx: int = 0

    def build_reverse_index(self) -> t.Dict[int, t.List[int]]:
        reverse_index = defaultdict(list)
        for i, target in enumerate(self.targets):
            reverse_index[target].append(i)
        return reverse_index

    def __iter__(self) -> t.Iterable[t.Iterable[int]]:
        self.start_idx = 0
        for i in range(len(self)):
            yield self.sample_batch()

    def sample_batch(self) -> t.Iterable[int]:
        selected_indices: t.List[int] = []
        selected_classes = random.choices(self.classes, k=self.classes_in_batch)
        for class_nb in selected_classes:
            class_indices = self.reverse_index[class_nb]
            indices_from_class = random.choices(class_indices, k=self.examples_per_class)
            selected_indices.extend(indices_from_class)

        assert len(selected_indices) == self.batch_size
        return selected_indices

    def __len__(self) -> int:
        if self.epoch_size:
            return self.epoch_size
        return len(self.targets) // self.batch_size


class ClassBalancedSamplerUsingCentroids(ClassBalancedSampler):
    def __init__(self, targets: t.List[int], batch_size: int, examples_per_class: int, class_centroids: torch.Tensor,
                 epoch_size: t.Optional[int] = None):
        super().__init__(targets, batch_size, examples_per_class, epoch_size)
        self.class_centroids = torch.nn.functional.normalize(class_centroids, dim=1)
        self.class_similarity = self.class_centroids.mm(self.class_centroids.t())
        self.similar_classes = torch.argsort(-self.class_similarity, dim=1).numpy()

    def sample_batch(self) -> t.Iterable[int]:
        selected_indices: t.List[int] = []
        selected_class = random.choice(self.classes)
        selected_classes = self.similar_classes[selected_class][:self.classes_in_batch]
        for class_nb in selected_classes:
            class_indices = self.reverse_index[class_nb]
            indices_from_class = random.choices(class_indices, k=self.examples_per_class)
            selected_indices.extend(indices_from_class)

        assert len(selected_indices) == self.batch_size
        return selected_indices


class Product10kDataModuleReRanker(pl.LightningDataModule):
    def __init__(self, data_config,
                 train_data_transform: t.Optional[transforms.Compose] = None,
                 valid_data_transform: t.Optional[transforms.Compose] = None):
        super().__init__()
        self.train_path_dataset = os.path.join(data_config.product10k.train_valid_root, data_config.product10k.train_list)
        self.train_image_root = os.path.join(data_config.product10k.train_valid_root, data_config.product10k.train_prefix)
        self.class_centroids = torch.load(os.path.join(data_config.product10k.train_valid_root, data_config.product10k.class_centroids))

        self.eval_root = data_config.dev_set.eval_root
        self.gallery = os.path.join(data_config.dev_set.eval_root, data_config.dev_set.gallery)
        self.queries = os.path.join(data_config.dev_set.eval_root, data_config.dev_set.queries)

        self.batch_size = data_config.batch_size
        self.train_data_transform = train_data_transform
        self.valid_data_transform = valid_data_transform
        self.num_workers = data_config.num_workers

        # self.dims is returned when you call dm.size()
        self.dims = (1, 4)
        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = Product10KDataset(self.train_path_dataset, self.train_image_root, transforms=self.train_data_transform)

            # combine data
            gallery_csv = pd.read_csv(os.path.join(self.eval_root, 'gallery.csv'))
            queries_csv = pd.read_csv(os.path.join(self.eval_root, 'queries.csv'))
            current_ranks = np.load(os.path.join(self.eval_root, 'sorted_distances.npy'))
            image_paths = []
            for idx in range(queries_csv.shape[0]):
                q_elem = queries_csv.iloc[idx]
                image_paths.append([q_elem.img_path, [q_elem.bbox_x, q_elem.bbox_y, q_elem.bbox_w, q_elem.bbox_h]])
                image_paths.extend(gallery_csv.iloc[current_ranks[idx][:50]].img_path)

            self.eval_dataset = Product10KDatasetReRank(image_paths, self.eval_root, self.valid_data_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_sampler=ClassBalancedSamplerUsingCentroids(self.train_dataset.targets, self.batch_size, examples_per_class=2,
                                                             class_centroids=self.class_centroids),
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.eval_dataset,
            batch_size=51,
            shuffle=False,
            num_workers=4,
            drop_last=False
           )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()
