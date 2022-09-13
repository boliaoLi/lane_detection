# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import force_fp32

from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from ..builder import HEADS, build_loss
from .detr_head import DETRHead
from .deformable_detr_head import DeformableDETRHead


@HEADS.register_module()
class LaneFormerHead(DETRHead):
    def __init__(self,
                 *args,
                 points_num=72,
                 loss_line=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='IoULoss', loss_weight=2.0),
                 with_box_refine=False,
                 transformer=None,
                 **kwargs):
        self.points_num = points_num
        super(LaneFormerHead, self).__init__(*args, with_box_refine=with_box_refine,
                                             transformer=transformer, **kwargs)
        self.loss_line = build_loss(loss_line)
        self.loss_iou = build_loss(loss_iou)

    # overwrite layer initialize
    def _init_layers(self):
        """Initialize layers of the transformer head."""
        self.input_proj = Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1)
        self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        self.reg_ffn = FFN(
            self.embed_dims,
            self.embed_dims,
            self.num_reg_fcs,
            self.act_cfg,
            dropout=0.0,
            add_residual=False)
        self.fc_reg = Linear(self.embed_dims, self.points_num)
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)

    def forward_single(self, x, img_metas):
        """"Forward function for a single feature level.

        Args:
            x (Tensor): Input feature from backbone's single stage, shape
                [bs, c, h, w].
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_line_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (x1,x2,...,x72).
                Shape [nb_dec, bs, num_query, points_num].
        """
        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        batch_size = x.size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        masks = x.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            masks[img_id, :img_h, :img_w] = 0

        x = self.input_proj(x)
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        # position encoding
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
        # outs_dec: [nb_dec, bs, num_query, embed_dim]
        outs_dec, _ = self.transformer(x, masks, self.query_embedding.weight,
                                       pos_embed)

        all_cls_scores = self.fc_cls(outs_dec)
        all_line_preds = self.fc_reg(self.activate(
            self.reg_ffn(outs_dec))).sigmoid()
        return all_cls_scores, all_line_preds

    @force_fp32(apply_to=('all_cls_scores_list', 'all_line_preds_list'))
    def loss(self,
             all_cls_scores_list,
             all_line_preds_list,
             gt_lines_list,
             gt_labels_list,
             img_metas,
             gt_lines_ignore=None):
        """"Loss function.

        Only outputs from the last feature level are used for computing
        losses by default.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_line_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor and shape
                [nb_dec, bs, num_query, points_num].
            gt_lines_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 72) in [x1,x2,...,x72] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_lines_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # NOTE defaultly only the outputs from the last feature scale is used.
        all_cls_scores = all_cls_scores_list[-1]
        all_line_preds = all_line_preds_list[-1]
        assert gt_lines_ignore is None, \
            'Only supports for gt_lines_ignore setting to None.'

        num_dec_layers = len(all_cls_scores)
        all_gt_lines_list = [gt_lines_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_lines_ignore_list = [
            gt_lines_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_line, losses_iou = multi_apply(
            self.loss_single, all_cls_scores, all_line_preds,
            all_gt_lines_list, all_gt_labels_list, img_metas_list,
            all_gt_lines_ignore_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_line'] = losses_line[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_line_i, loss_iou_i in zip(losses_cls[:-1],
                                                       losses_line[:-1],
                                                       losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_line'] = loss_line_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1
        return loss_dict

    def loss_single(self,
                    cls_scores,
                    line_preds,
                    gt_lines_list,
                    gt_labels_list,
                    img_metas,
                    gt_lines_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            line_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (x1,x2,...,x72) and
                shape [bs, num_query, points_num].
            gt_lines_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 72) in (x1,x2,...,x72) format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_lines_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        line_preds_list = [line_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, line_preds_list,
                                           gt_lines_list, gt_labels_list,
                                           img_metas, gt_lines_ignore_list)
        (labels_list, label_weights_list, line_targets_list, line_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        line_targets = torch.cat(line_targets_list, 0)
        line_weights = torch.cat(line_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, line_pred in zip(img_metas, line_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = line_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                line_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            line_preds, line_targets, line_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            line_preds, line_targets, line_weights, avg_factor=num_total_pos)
        return loss_cls, loss_line, loss_iou


    def get_targets(self,
                    cls_scores_list,
                    line_preds_list,
                    gt_lines_list,
                    gt_labels_list,
                    img_metas,
                    gt_lines_ignore_list):
        """"Compute regression and classification targets for a batch image.

                Outputs from a single decoder layer of a single feature level are used.

                Args:
                    cls_scores_list (list[Tensor]): Box score logits from a single
                        decoder layer for each image with shape [num_query,
                        cls_out_channels].
                    line_preds_list (list[Tensor]): Sigmoid outputs from a single
                        decoder layer for each image, shape [num_query, 72].
                    gt_lines_list (list[Tensor]): Ground truth bboxes for each image
                        with shape (num_gts, 72)
                    gt_labels_list (list[Tensor]): Ground truth class indices for each
                        image with shape (num_gts, ).
                    img_metas (list[dict]): List of image meta information.
                    gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                        boxes which can be ignored for each image. Default None.

                Returns:
                    tuple: a tuple containing the following targets.

                        - labels_list (list[Tensor]): Labels for all images.
                        - label_weights_list (list[Tensor]): Label weights for all \
                            images.
                        - line_targets_list (list[Tensor]): line targets for all \
                            images.
                        - line_weights_list (list[Tensor]): line weights for all \
                            images.
                        - num_total_pos (int): Number of positive samples in all \
                            images.
                        - num_total_neg (int): Number of negative samples in all \
                            images.
                """
        assert gt_lines_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_lines_ignore_list = [
            gt_lines_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, line_targets_list,
         line_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, line_preds_list,
            gt_lines_list, gt_labels_list, img_metas, gt_lines_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, line_targets_list,
                line_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self,
                           cls_score,
                           line_pred,
                           gt_lines,
                           gt_labels,
                           img_meta,
                           gt_lines_ignore=None):
        """"Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            line_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, shape [num_query, 72].
            gt_lines (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 72).
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            img_meta (dict): Meta information for one image.
            gt_lines_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - line_targets (Tensor): line targets of each image.
                - line_weights (Tensor): line weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_lines = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(line_pred, cls_score, gt_lines,
                                             gt_labels, img_meta,
                                             gt_lines_ignore)
        sampling_result = self.sampler.sample(assign_result, line_pred,
                                              gt_lines)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_lines.new_full((num_lines,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_lines.new_ones(num_lines)

        # bbox targets
        line_targets = torch.zeros_like(line_pred)
        line_weights = torch.zeros_like(line_pred)
        line_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size
        factor = line_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        pos_gt_lines_targets = sampling_result.pos_gt_lines / factor
        line_targets[pos_inds] = pos_gt_lines_targets
        return (labels, label_weights, line_targets, line_weights, pos_inds,
                neg_inds)
    
    