# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import mmcv
import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss
from lanedet.core.line.iou_calculators.iou_line_calculator import line_overlaps


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def line_iou_loss(pred, target, length=1e-3, aligned=True):
    """Line IoU loss.

    Computing the Line IoU loss between a set of predicted lines and target lines.
    The loss is calculated as negative log of IoU.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, x2, ..., x72),
            shape (n, 72).
        target (torch.Tensor): Corresponding gt lines, shape (n, 72).
        length: extended radius
        aligned: True for iou loss calculation, False for pair-wise ious in assign
    Return:
        torch.Tensor: Loss tensor.
    """
    loss = line_overlaps(pred, target, length, is_aligned=aligned)
    return loss


@LOSSES.register_module()
class LineIoULoss(nn.Module):
    """LineIoULoss.

    Computing the Line IoU loss between a set of predicted lines and target lines.

    Args:
        loss_weight (float): Weight of loss.

    """

    def __init__(self, loss_weight=1.0, length=1e-2):
        super(LineIoULoss, self).__init__()
        self.loss_weight = loss_weight
        self.length = length

    def forward(self,
                pred,
                target,
                weight,
                avg_factor=True,
                aligned=True
                ):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted bboxes of format (x1, x2, ..., x72),shape (n, 72).
            target (torch.Tensor): Corresponding gt lines, shape (n, 72).
            length: extended radius
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to True.
            aligned: True for iou loss calculation, False for pair-wise ious in assign
        """
        assert aligned, "pred and target should be the same shape in computing line iou loss"
        loss = self.loss_weight * line_iou_loss(
            pred,
            target,
            weight=weight,
            length=self.length,
            aligned=True,
            avg_factor=avg_factor
            )
        return loss

