import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from torch.optim.lr_scheduler import MultiplicativeLR
import timm
import typing as t

from visual_search.utils.mean_average_precision import calculate_map
from visual_search.validation.f1_score import F1Score


def ll(x: int) -> float:
    return 0.92


class RerankModel(pl.LightningModule):
    def __init__(self, train_config: t.Dict[str, t.Any], eval_config: t.Dict[str, t.Any], channel_last: bool = False):
        super().__init__()
        self.train_config = train_config
        self.model = timm.create_model(train_config["arch"], num_classes=1, pretrained=True, drop_rate=0.0, drop_path_rate=0.1)
        self.loss_function = torch.nn.BCEWithLogitsLoss()

        if channel_last:
            self.model = self.model.to(memory_format=torch.channels_last)
            self.memory_format = torch.channels_last
        else:
            self.memory_format = torch.contiguous_format

        gallery_csv = pd.read_csv(os.path.join(eval_config["eval_root"], 'gallery.csv'))
        queries_csv = pd.read_csv(os.path.join(eval_config["eval_root"], 'queries.csv'))
        self.gallery_labels = gallery_csv['product_id'].values
        self.query_labels = queries_csv['product_id'].values
        self.current_ranks = np.load(os.path.join(eval_config["eval_root"], 'sorted_distances.npy'))
        self.rerank_elements = 50

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions = self.model.forward_features(x.to(memory_format=self.memory_format))
        predictions = self.model.forward_head(predictions, pre_logits=False)
        return predictions

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.train_config["learning_rate"], weight_decay=self.train_config["weight_decay"])
        base_lr_scheduler = MultiplicativeLR(optimizer, lr_lambda=ll)
        return [optimizer], [base_lr_scheduler]

    def prepare_input(self, images: torch.Tensor, targets: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        image_pair_0, image_pair_1 = images[::2], images[1::2]
        # merge images on width
        positive_image_concat = torch.concat([image_pair_0, image_pair_1], dim=3)
        # concat other way around so negative
        negative_image_concat = torch.concat([image_pair_0, torch.flip(image_pair_1, dims=(0,))], dim=3)
        all_images = torch.concat([positive_image_concat, negative_image_concat], dim=0)
        targets = torch.zeros((all_images.size(0), 1), device=targets.device)
        targets[:positive_image_concat.size(0)] = 1
        return all_images, targets
        # return images, torch.zeros((images.size(0), 1), device=targets.device)

    def training_step(self, train_batch: t.Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = train_batch
        images, targets = self.prepare_input(x, y)
        predictions = self(images)
        loss = self.loss_function(predictions, targets)
        self.log('train/loss', loss,
                 batch_size=x.size(0),
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=True)

        probabilities = torch.sigmoid(predictions)
        predictions = probabilities > 0.5
        f1_score, true_count, precision, recall = F1Score.calc_f1_count_for_label(predictions, targets, 1, return_precision_recall=True)
        self.log(
            "train/f1_score",
            f1_score,
            batch_size=x.size(0),
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )
        self.log(
            "train/precision",
            precision,
            batch_size=x.size(0),
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )
        self.log(
            "train/recall",
            recall,
            batch_size=x.size(0),
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )
        return loss

    def validation_step(self, val_batch: torch.Tensor, batch_idx: int) -> np.ndarray:
        query = val_batch[0:1].repeat(self.rerank_elements, 1, 1, 1)
        gallery = val_batch[1:]
        image_merged = torch.cat([query, gallery], dim=3).cuda()
        scores = self(image_merged).cpu().numpy()
        return scores

    def validation_epoch_end(self, outputs) -> None:
        scores_for_query_c = np.stack(outputs)[:, :, 0]
        if scores_for_query_c.shape[0] < 1000:
            return
        sorted_distances_reranked = np.argsort(-scores_for_query_c, axis=1)
        reranked = np.take_along_axis(self.current_ranks[:, :self.rerank_elements], sorted_distances_reranked, axis=1)
        all_ranks = np.concatenate((reranked, self.current_ranks[:, self.rerank_elements:]), axis=1)
        mAP_rerank = calculate_map(all_ranks, self.query_labels, self.gallery_labels)
        self.log(
            "valid_mAP",
            mAP_rerank,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )



