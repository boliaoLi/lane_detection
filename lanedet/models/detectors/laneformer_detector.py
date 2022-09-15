# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from lanedet.core import line2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class LaneFormerDetector(SingleStageDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict lines on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 line_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(LaneFormerDetector, self).__init__(backbone, neck, line_head, train_cfg,
                                                 test_cfg, pretrained, init_cfg)

    # over-write `forward_dummy` because:
    # the forward of bbox_head requires img_metas
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        warnings.warn('Warning! MultiheadAttention in DETR does not '
                      'support flops computation! Do not use the '
                      'results in your papers!')

        batch_size, _, height, width = img.shape
        dummy_img_metas = [
            dict(
                batch_input_shape=(height, width),
                img_shape=(height, width, 3)) for _ in range(batch_size)
        ]
        x = self.extract_feat(img)
        outs = self.line_head(x, dummy_img_metas)
        return outs

