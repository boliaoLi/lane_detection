# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


def line_overlaps(lines1, lines2, length=20, is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.
    Args:
        lines1 (ndarray): shape (m, num_points) in <x1, x2, ..., xN, y1, y2>, N=num_points format or empty.
        lines2 (ndarray): shape (n, num_points) in <x1, x2, ..., xN, y1, y2>, N=num_points format or empty.
            If ``is_aligned`` is ``True``, then m and n must be equal.
        length:(int | float): extended radius.
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
    """

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert lines1.shape[:-2] == lines2.shape[:-2]
    num_lines1 = lines1.shape[-2]
    num_lines2 = lines2.shape[-2]
    px1 = lines1[:, :-2] - length
    px2 = lines1[:, :-2] + length
    tx1 = lines2[:, :-2] - length
    tx2 = lines2[:, :-2] + length

    if is_aligned:
        assert num_lines1 == num_lines2
        overlap = np.min(px2, tx2) - np.max(px1, tx1)
        union = np.max(px2, tx2) - np.min(px1, tx1)
        # overlap, union shape:[batch, m, num_points]
        eps = [eps]
        union = np.max(union, eps)
        line_iou = 1 - (overlap / union)
        return line_iou
    else:
        overlap = np.minimum(px2[..., :, None, :], tx2[..., None, :, :]) - np.maximum(px1[..., :, None, :], tx1[..., None, :, :])
        # 负数代表两个点不相交，因此将交集中小于0的地方设置为0
        overlap = np.maximum(overlap, 0)
        union = np.maximum(px2[..., :, None, :], tx2[..., None, :, :]) - np.minimum(px1[..., :, None, :], tx1[..., None, :, :])
        # overlap, union shape:[batch, m, n, num_points]
        union = np.maximum(union, eps)
        # line_iou = (1 - (overlap / union)).mean(axis=-1)
        line_iou = (overlap/union).mean(axis=-1)
        return line_iou
