from typing import Optional, Dict

import open_clip
import torch
from torch import nn
from torch.nn import functional as F
from visual_search.model.train_utils import ArcMarginProduct_subcenter


class HeadXBM(nn.Module):
    def __init__(self, emb_size: int, n_classes: int):
        super(HeadXBM, self).__init__()
        self.dropout = nn.Dropout(.2)

    def forward(self, x: torch.Tensor):
        return F.normalize(x), F.normalize(x)


class ModelToUseXBM(nn.Module):
    def __init__(self, arch_name: str, num_classes: int, memory_format, embedding_dim: int, lr_head: float, wd_head: float,
                 backbone_lrs: Dict[int, float], backbone_wds: Dict[int, float], pretrain_name: Optional[str] = None):
        super().__init__()
        if pretrain_name == "":
            pretrain_name = None

        vit_model, model_transforms, preprocess = open_clip.create_model_and_transforms(arch_name, pretrain_name)
        self.model = vit_model.visual
        self.head = HeadXBM(embedding_dim, num_classes)

        self.memory_format = memory_format
        self.lr_head = lr_head
        self.wd_head = wd_head
        self.backbone_lrs = {int(float(key[1:])): value for key, value in backbone_lrs.items()}
        self.backbone_wds = {int(float(key[1:])): value for key, value in backbone_wds.items()}

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.model(x.to(memory_format=self.memory_format))
        return F.normalize(embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.model(x.to(memory_format=self.memory_format))
        predictions, embeddings = self.head(embeddings)
        return F.normalize(embeddings)

    def freeze_backbone(self, freeze: bool = True):
        self.model.trunk.requires_grad_(not freeze)

    def freeze_head(self, freeze: bool = False):
        self.head.requires_grad_(not freeze)

    def unfreeze_all(self):
        self.freeze_head(False)
        self.freeze_backbone(False)

    def get_parameters(self):
        param_dicts = self.get_parameters_backbone()
        param_dicts_head = self.get_parameters_head()
        param_dicts.extend(param_dicts_head)
        return param_dicts

    def get_parameters_head(self):
        param_dicts = [{'params': self.head.parameters(),
                        'lr': self.lr_head,
                        'weight_decay': self.wd_head}]
        return param_dicts

    def get_parameters_backbone(self):
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(self.model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        mul = 0.01
        param_dicts = [{"params": gain_or_bias_params, "weight_decay": 0., 'lr': self.lr_head * mul},
                       {"params": rest_params, "weight_decay": self.wd_head, 'lr': self.lr_head * mul}]
        return param_dicts


    def get_parameters_backbone_trunk(self):
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(self.model.trunk.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        mul = 0.01
        param_dicts = [{"params": gain_or_bias_params, "weight_decay": 0., 'lr': self.lr_head * mul},
                       {"params": rest_params, "weight_decay": self.wd_head, 'lr': self.lr_head * mul}]
        return param_dicts

    def get_parameters_backbone_head(self):
        param_dicts = [{'params': self.model.head.parameters(),
                        'lr': self.lr_head,
                        'weight_decay': self.wd_head}]
        return param_dicts
