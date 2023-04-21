import typing as t

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from transformers import get_cosine_schedule_with_warmup

from visual_search.model.model_xbm import ModelToUseXBM
from visual_search.model.xbm import XBM, ContrastiveLoss
from visual_search.utils.mean_average_precision import calculate_map
from visual_search.utils.ranking_evaulation import db_augmentation, calculate_sim_matrix
from torch.nn import functional as F

def ll(x: int) -> float:
    return 0.92


class ImageClassificationXBM(pl.LightningModule):
    def __init__(self, num_classes: int, train_config: t.Dict[str, t.Any], num_train_steps: int, class_value_counts: np.ndarray, channel_last: bool = False):
        super().__init__()
        self.max_predictions = 1000
        self.num_classes = num_classes
        self.train_config = train_config
        self.num_train_steps = num_train_steps
        if channel_last:
            self.memory_format = torch.channels_last
        else:
            self.memory_format = torch.contiguous_format

        self.model = ModelToUseXBM(train_config["arch"], pretrain_name=train_config["pretrain_name"], num_classes=num_classes, memory_format=self.memory_format,
                                embedding_dim=train_config["embedding_dim"], lr_head=train_config["learning_rate_head"],
                                wd_head=train_config["weight_decay_head"], backbone_lrs=train_config["learning_rates_backbone"],
                                backbone_wds=train_config["weight_decay_backbone"])
        self.model = self.model.to(memory_format=torch.channels_last)
        self.model = torch.compile(self.model)
        self.xbm = XBM(32000, train_config["embedding_dim"])
        self.criterion = ContrastiveLoss()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions = self.model(x.to(memory_format=self.memory_format))
        return F.normalize(predictions)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        predictions = self.model(x.to(memory_format=self.memory_format))
        return predictions

    def configure_optimizers(self):
        if self.train_config["freeze_backbone"] and not self.train_config["freeze_head"]:
            parameters = self.model.get_parameters_head()
            self.model.freeze_backbone(True)
        elif not self.train_config["freeze_backbone"] and self.train_config["freeze_head"]:
            parameters = self.model.get_parameters_backbone()
            self.model.freeze_head(True)
        elif not self.train_config["freeze_backbone"] and not self.train_config["freeze_head"]:
            parameters = self.model.get_parameters()
        else:
            raise ValueError("Entire model freezed!")

        if self.train_config["optimizer"] == "Adam":
            optimizer = torch.optim.AdamW(parameters) #, betas=(0.9, 0.98))
        elif self.train_config["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(parameters, lr=self.train_config["learning_rate_head"], momentum=self.train_config["momentum"],
                                        weight_decay=self.train_config["weight_decay_head"])
        else:
            raise ValueError

        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=self.num_train_steps * self.train_config["n_epoch"],
                                                    num_warmup_steps=self.train_config["warmup_steps"])
        # scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.train_config["warmup_steps"])
        return [optimizer], [{
                                'scheduler': scheduler,
                                'interval': 'step',
                                'frequency': 1
                                                }]

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.model.parameters(), self.model_momentum_encoder.parameters()):
            param_k.data = param_k.data * self.em + param_q.data * (1.0 - self.em)

    def training_step(self, train_batch: t.Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = train_batch
        predictions = self(x)
        with torch.no_grad():
            self.xbm.enqueue_dequeue(predictions.detach(), y.detach())

        loss = self.criterion(predictions, y, predictions, y)
        xbm_feats, xbm_targets = self.xbm.get()
        xbm_feats = xbm_feats.to(predictions.device)
        xbm_targets = xbm_targets.to(predictions.device)
        xbm_loss = self.criterion(predictions, y, xbm_feats, xbm_targets)
        loss = loss + 1.0 * xbm_loss
        self.log('train/loss', loss,
                 batch_size=x.size(0),
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=True)
        return loss

    def validation_step_from_retrieval(self, val_batch: t.Tuple[torch.Tensor, torch.Tensor]) -> t.Tuple[torch.Tensor, torch.Tensor]:
        x, y = val_batch
        predictions = self.forward_features(x)
        return predictions.cpu(), y.cpu()

    def validation_step(self, val_batch: t.Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> \
            t.Tuple[torch.Tensor, torch.Tensor]:
        return self.validation_step_from_retrieval(val_batch)

    def validate_simple(self, query_embeddings: np.ndarray, gallery_embeddings: np.ndarray,
                        query_labels: np.ndarray, gallery_labels: np.ndarray) -> float:
        distances = pairwise_distances(query_embeddings, gallery_embeddings)
        sorted_distances = np.argsort(distances, axis=1)[:, :self.max_predictions]
        mAP = calculate_map(sorted_distances, query_labels, gallery_labels)
        return mAP

    def validate_complicated(self, query_embeddings: np.ndarray, gallery_embeddings: np.ndarray,
                        query_labels: np.ndarray, gallery_labels: np.ndarray) -> float:
        """
        Use tricks to boost performance:
        - subtract mean and std value of embeddings
        - use DB augmentation with swaped input, so gallery is mixed with query and query with query
        """
        reference_vecs, query_vecs = db_augmentation(gallery_embeddings, query_embeddings, top_k=5)
        similarities = calculate_sim_matrix(query_vecs, reference_vecs)
        sorted_similarities = np.argsort(-similarities, axis=1)[:, :self.max_predictions]
        mAP = calculate_map(sorted_similarities, query_labels, gallery_labels)
        return mAP

    def validation_epoch_end(self, outputs: t.Tuple[torch.Tensor, torch.Tensor, int]):
        # validate retrieval
        y_embeddings, targets = zip(*outputs)
        y_embeddings = torch.cat(y_embeddings, dim=0).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()

        if y_embeddings.shape[0] < 1000:
            return

        y_embeddings = normalize(y_embeddings)
        query_embeddings = y_embeddings[:1935]
        gallery_embeddings = y_embeddings[1935:]
        query_labels = targets[:1935]
        gallery_labels = targets[1935:]

        mAP = self.validate_simple(query_embeddings, gallery_embeddings, query_labels, gallery_labels)
        self.log(
            "valid_mAP",
            mAP,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

        mAP_tricks = self.validate_complicated(query_embeddings, gallery_embeddings, query_labels, gallery_labels)
        self.log(
            "valid_mAP_tricks",
            mAP_tricks,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )
