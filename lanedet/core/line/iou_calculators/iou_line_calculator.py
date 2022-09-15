# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .builder import IOU_CALCULATORS


def cast_tensor_type(x, scale=1., dtype=None):
    if dtype == 'fp16':
        # scale is for preventing overflows
        x = (x / scale).half()
    return x


@IOU_CALCULATORS.register_module()
class LineOverlaps:
    """LineOverlaps Calculator."""

    def __init__(self, scale=1., num_points=72, dtype=None):
        self.scale = scale
        self.dtype = dtype
        self.num_points = num_points

    def __call__(self, lines1, lines2, length, is_aligned=False):
        """Calculate IoU between lines.

        Args:
            lines1 (Tensor): lines have shape (m, num_points).
            lines2 (Tensor): lines have shape (n, num_points), or be
                empty. If ``is_aligned `` is ``True``, then m and n must be
                equal.
            length:(int | float): extended radius
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """

        if self.dtype == 'fp16':
            # change tensor type to save cpu and cuda memory and keep speed
            lines1 = cast_tensor_type(lines1, self.scale, self.dtype)
            lines2 = cast_tensor_type(lines2, self.scale, self.dtype)
            overlaps = line_overlaps(lines1, lines2, is_aligned)
            if not overlaps.is_cuda and overlaps.dtype == torch.float16:
                # resume cpu float32
                overlaps = overlaps.float()
            return overlaps

        return line_overlaps(lines1, lines2, length, is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + f'(' \
            f'scale={self.scale}, dtype={self.dtype})'
        return repr_str


def line_overlaps(lines1, lines2, length=1e-5, is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.
    Args:
        lines1 (Tensor): shape (B, m, num_points) in <x1, x2, ..., xN>, N=num_points format or empty.
        lines2 (Tensor): shape (B, n, num_points) in <x1, x2, ..., xN>, N=num_points format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        length:(int | float): extended radius,default=1e-5
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
    num_lines1 = lines1.size(-2)
    num_lines2 = lines1.size(-2)
    px1 = lines1 - length
    px2 = lines1 + length
    tx1 = lines2 - length
    tx2 = lines2 + length

    if is_aligned:
        assert num_lines1 == num_lines2
        overlap = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)
        # overlap, union shape:[batch, m, num_points]
        eps = union.new_tensor([eps])
        union = torch.max(union, eps)
        line_iou = 1 - (overlap / union)
        return line_iou
    else:
        overlap = torch.min(px2[..., :, None, :], tx2[..., None, :, :]) - torch.max(px1[..., :, None, :], tx1[..., None, :, :])
        union = torch.max(px2[..., :, None, :], tx2[..., None, :, :]) - torch.min(px1[..., :, None, :], tx1[..., None, :, :])
        # overlap, union shape:[batch, m, n, num_points]
        eps = union.new_tensor([eps])
        union = torch.max(union, eps)
        line_iou = 1 - (overlap / union).sum(dim=-1)
        return line_iou