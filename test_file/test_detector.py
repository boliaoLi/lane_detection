from lanedet.models.detectors.single_stage import SingleStageDetector
from lanedet.models.builder import build_detector
import numpy as np
import torch
import json


def read_gt():
    with open(r'D:\model\lane_detection\demo\demo_label.json', 'r') as file:
        str = file.read()
        data = json.loads(str)
    gt_lines = np.array(data['lines'])
    gt_labels = [0, 0, 1]
    return gt_lines, gt_labels


def parse_init():
    model = dict(
        type='LaneFormerDetector',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            style='pytorch'),
        line_head=dict(
            type='LaneFormerHead',
            num_classes=4,
            num_points=74,
            in_channels=2048,
            transformer=dict(
                type='Transformer',
                encoder=dict(
                    type='DetrTransformerEncoder',
                    num_layers=1,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1)
                        ],
                        feedforward_channels=2048,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
                decoder=dict(
                    type='DetrTransformerDecoder',
                    return_intermediate=True,
                    num_layers=6,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        feedforward_channels=2048,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                         'ffn', 'norm')),
                )),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True)),
        # training and testing settings
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='ClassificationCost', weight=1.),
                reg_cost=dict(type='LineL1Cost', weight=5.0),
                iou_cost=dict(type='LineIoUCost', weight=2.0))),
        test_cfg=dict(max_per_img=100))
    h = 540
    w = 960
    img_metas = [
        {'img_shape': (h, w, 3),
         'scale_factor': 1,
         'pad_shape': (h, w, 3),
         'batch_input_shape': (h, w)
    },
        {'img_shape': (h, w, 3),
         'scale_factor': 1,
         'pad_shape': (h, w, 3),
         'batch_input_shape': (h, w)
    }]
    return model, img_metas


if __name__ == '__main__':
    model, img_metas = parse_init()
    detector = build_detector(model)
    original_img = torch.rand(2, 3, 540, 960)
    gt_lines, gt_labels = read_gt()
    gt_lines = [torch.tensor(gt_lines, dtype=torch.float32), torch.tensor(gt_lines, dtype=torch.float32)]
    gt_labels = [torch.tensor(gt_labels), torch.tensor(gt_labels)]
    result = detector(original_img, img_metas, return_loss=True, gt_lines=gt_lines, gt_labels=gt_labels)
    print(result)
    print('done!')


