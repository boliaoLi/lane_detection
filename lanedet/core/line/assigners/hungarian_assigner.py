# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import LINE_ASSIGNERS
from ..match_costs import build_match_cost
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@LINE_ASSIGNERS.register_module()
class HungarianAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        line_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation mode.
            Default type `LineIoU`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='LineL1Cost', weight=1.0)):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)

    def assign(self,
               line_pred,
               cls_pred,
               gt_lines,
               gt_labels,
               img_meta,
               gt_lines_ignore=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            line_pred (Tensor): Predicted boxes with normalized coordinates
                (x1,x2,x3,...,x72), which are all in range [0, 1]. Shape
                [num_query, 72].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_lines (Tensor): Ground truth boxes with unnormalized
                coordinates (x1,x2,x3,...,x72). Shape [num_gt, 72].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_lines_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_lines_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_lines = gt_lines.size(0), line_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = line_pred.new_full((num_lines, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = line_pred.new_full((num_lines, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_lines == 0:
            # No ground truth or lines, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
        img_h, img_w, _ = img_meta['img_shape']
        factor = img_w

        # 2. compute the weighted costs
        # classification and linecost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
        normalize_gt_lines = gt_lines / factor
        reg_cost = self.reg_cost(line_pred, normalize_gt_lines)
        # regression iou cost, defaultly LineIou is used in laneformer.
        lines = line_pred * factor
        # weighted sum of above three costs
        cost = cls_cost + reg_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            line_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            line_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds].long()
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
