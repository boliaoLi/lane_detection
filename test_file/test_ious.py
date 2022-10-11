import random

import numpy as np
from lanedet.core.evaluation.line_overlaps import line_overlaps
from lanedet.core.evaluation.mean_ap import tpfp_default
from lanedet.core.evaluation.recall import eval_recalls, plot_iou_recall


def test_iou(det_lines, gt_lines, length=10):
    ious = line_overlaps(det_lines, gt_lines, length)
    print(ious)


def test_TPFP(det_lines, gt_lines, length=10):
    tp, fp = tpfp_default(det_lines, gt_lines, length)
    print(tp, fp)


def test_recall(det_lines, gt_lines, length=10):
    recalls = eval_recalls([gt_lines], [det_lines], length, proposal_nums=6)
    print(recalls)


if __name__ == '__main__':
    gt_lines = []
    for i in range(1, 5):
        gt_line = np.linspace(100*i, 100*i+100, 72)
        gt_line = np.append(gt_line, [0, 1080])
        gt_lines.append(gt_line)
    gt_lines = np.stack(gt_lines)
    det_lines = []

    for i in range(1, 7):
        det_line = np.linspace(100*i+3*i, 100*i+100+3*i, 72)
        det_line = np.append(det_line, [0, 1080])
        det_lines.append(det_line)
    det_lines = np.stack(det_lines)
    test_iou(det_lines, gt_lines, length=10)
    test_TPFP(det_lines, gt_lines, length=10)
    test_recall(det_lines, gt_lines, length=10)