import torch
import torch.nn.functional as F
from torch import nn

from visual_search.model.train_utils import ArcFaceLossAdaptiveMargin


class NormSoftmaxLayer(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=16.0, m=0.0):
        super(NormSoftmaxLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input: torch.Tensor, label: torch.Tensor):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # phi = cosine - self.m
        # # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), device='cuda')
        # # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        # output *= self.s
        output = cosine * self.s
        return output
