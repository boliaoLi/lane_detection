# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

import torch
from lanedet.models.dense_heads import LaneFormerHead
from mmcv import ConfigDict


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


def model_init():
    config = ConfigDict(
        dict(
            type='LaneFormerHead',
            num_points=72,
            num_classes=12,
            in_channels=200,
            transformer=dict(
                type='Transformer',
                encoder=dict(
                    type='DetrTransformerEncoder',
                    num_layers=6,
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
                        operation_order=('self_attn', 'norm', 'cross_attn',
                                         'norm', 'ffn', 'norm')),
                )),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_line=dict(type='L1Loss', loss_weight=5.0),
            loss_iou=dict(type='LineIoULoss', loss_weight=2.0)))
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3),
        'batch_input_shape': (s, s)
    },
        {
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3),
        'batch_input_shape': (s, s)
    }]
    self = LaneFormerHead(**config)
    self.init_weights()
    feat = [torch.rand(2, 200, 10, 10)]
    cls_scores, line_preds = self.forward(feat, img_metas)
    print(f'cls_scores:{cls_scores}, line_preds:{line_preds}')
    gt_lines = [torch.rand((4, 72)), torch.rand(5, 72)]
    gt_labels = [torch.LongTensor([1, 1, 0, 0]), torch.LongTensor([1, 1, 2, 0, 2])]
    gt_bboxes_ignore = None
    empty_gt_losses = self.loss(cls_scores, line_preds, gt_lines, gt_labels,
                                img_metas, gt_bboxes_ignore)
    print('empty_gt_losses:', empty_gt_losses)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    model_init()
    print_hi('PyCharm')

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
