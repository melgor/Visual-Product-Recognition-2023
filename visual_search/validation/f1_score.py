from typing import Tuple, Union

import torch


class F1Score:
    """
    Class for f1 calculation in Pytorch.
    """

    def __init__(self, average: str = "weighted"):
        """
        Init.

        Args:
            average: averaging method
        """
        self.average = average
        if average not in [None, "micro", "macro", "weighted"]:
            raise ValueError("Wrong value of average parameter")

    @staticmethod
    def calc_f1_micro(predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Calculate f1 micro.

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels

        Returns:
            f1 score
        """
        true_positive = torch.eq(labels, predictions).sum().float()
        f1_score = torch.div(true_positive, len(labels))
        return f1_score.item()

    @staticmethod
    def calc_f1_count_for_label(
        predictions: torch.Tensor, labels: torch.Tensor, label_id: int, return_precision_recall: bool = False
    ) -> Union[Tuple[float, int], Tuple[float, int, float, float]]:
        """
        Calculate f1 and true count for the label

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels
            label_id: id of current label

        Returns:
            f1 score and true count for label
        """
        # label count
        true_count = int(torch.eq(labels, label_id).sum().item())

        # true positives: labels equal to prediction and to label_id
        true_positive = torch.logical_and(torch.eq(labels, predictions), torch.eq(labels, label_id)).sum().float()
        # precision for label
        precision = torch.div(true_positive, torch.eq(predictions, label_id).sum().float())
        # replace nan values with 0
        precision = torch.where(torch.isnan(precision), torch.zeros_like(precision).type_as(true_positive), precision)

        # recall for label
        recall = torch.div(true_positive, true_count)
        # f1
        f1 = 2 * precision * recall / (precision + recall)
        # replace nan values with 0
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1).type_as(true_positive), f1)
        if return_precision_recall:
            return f1.item(), true_count, precision.item(), recall.item()
        else:
            return f1.item(), true_count

    def __call__(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Calculate f1 score based on averaging method defined in init.

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels

        Returns:
            f1 score
        """

        # simpler calculation for micro
        if self.average == "micro":
            return self.calc_f1_micro(predictions, labels)

        f1_score = 0.0

        for label_id in range(0, len(labels.unique())):  # type: ignore
            f1, true_count = self.calc_f1_count_for_label(predictions, labels, label_id)

            if self.average == "weighted":
                f1_score += f1 * true_count
            elif self.average == "macro":
                f1_score += f1

        if self.average == "weighted":
            f1_score = torch.div(f1_score, len(labels)).item()

        elif self.average == "macro":
            f1_score = torch.div(f1_score, len(labels.unique())).item()

        return f1_score
