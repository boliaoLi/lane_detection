
from lanedet.core import build_assigner


if __name__ == '__main__':

    assigner_cfg = dict(
        type='HungarianAssigner',
        cls_cost=dict(type='ClassificationCost', weight=1.),
        reg_cost=dict(type='LineL1Cost', weight=5.0),
        iou_cost=dict(type='LineIoUCost', weight=2.0))
    assigner = build_assigner(assigner_cfg)
    print("hello world")