import math

import numpy as np
import torch
from torch import nn
from typing import Tuple, List
import random
from collections import defaultdict
from torch.utils.data.sampler import Sampler


class XBM:
    def __init__(self, queue_size: int, emb_dim: int):
        self.K = queue_size
        self.feats = torch.zeros(self.K, emb_dim).cuda()
        self.targets = torch.zeros(queue_size, dtype=torch.long).cuda()
        self.ptr = 0

    @property
    def is_full(self) -> bool:
        return self.targets[-1].item() != 0

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.is_full:
            return self.feats, self.targets
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr]

    def enqueue_dequeue(self, feats: torch.Tensor, targets: torch.Tensor):
        q_size = len(targets)
        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size


class ClassBalancedBatchSampler(Sampler):
    def __init__(self, target_vector: torch.Tensor, batch_size: int) -> None:
        super().__init__(None)
        self._target_vector = target_vector
        self._batch_size = batch_size

        self._class_index = self._build_class_index()
        self._classes = list(self._class_index.keys())
        self._classes_probability = self.calculate_probability_per_class()
        self._num_classes = len(self._classes)

        self._elem_per_class = 2
        self._classes_per_class = int(self.batch_size / self._elem_per_class)
        self._iter_number = 0
        self._iters_in_epoch = self.calculate_iters_per_epoch()

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def calculate_iters_per_epoch(self) -> int:
        nb_valid_examples = sum(len(self._class_index[class_nb]) for class_nb in self._classes)
        return math.floor(nb_valid_examples / self._batch_size)

    def calculate_probability_per_class(self) -> List[float]:
        prob_per_class = [len(self._class_index[class_nb]) for class_nb in self._classes]
        nb_valid_examples = sum(prob_per_class)
        prob_per_class = [prob/nb_valid_examples for prob in prob_per_class]
        return prob_per_class

    def __iter__(self):
        self._iter_number = 0
        return self

    def __next__(self):
        if self._iter_number < self._iters_in_epoch:
            self._iter_number += 1
            selected_labels = np.random.choice(self._classes, self._classes_per_class, p=self._classes_probability)
            element_indices = []
            for class_idx in selected_labels:
                # random element from the class
                random_elements = random.choices(self._class_index[class_idx], k=self._elem_per_class)
                element_indices.extend(random_elements)

            return element_indices
        raise StopIteration

    def _build_class_index(self):
        class_index: dict[int, list[int]] = defaultdict(list)
        for idx, elem in enumerate(self._target_vector):
            class_index[elem.item()].append(idx)
        return class_index

    def __len__(self) -> int:
        return self._iters_in_epoch


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.margin = 0.5

    def forward(self, inputs_col: torch.Tensor, targets_col: torch.Tensor,
                inputs_row: torch.Tensor, target_row: torch.Tensor):

        n = inputs_col.size(0)
        # Compute similarity matrix
        sim_mat = torch.matmul(inputs_col, inputs_row.t())
        epsilon = 1e-5
        loss = list()

        neg_count = list()
        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], targets_col[i] == target_row)
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1 - epsilon)
            neg_pair_ = torch.masked_select(sim_mat[i], targets_col[i] != target_row)

            neg_pair = torch.masked_select(neg_pair_, neg_pair_ > self.margin)

            pos_loss = torch.sum(-pos_pair_ + 1)
            if len(neg_pair) > 0:
                neg_loss = torch.sum(neg_pair)
                neg_count.append(len(neg_pair))
            else:
                neg_loss = 0

            loss.append(pos_loss + neg_loss)

        loss = sum(loss) / n
        return loss
