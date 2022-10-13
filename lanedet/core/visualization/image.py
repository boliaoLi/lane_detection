# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pycocotools.mask as mask_util
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

from ..mask.structures import bitmap_to_polygon
from ..utils import mask2ndarray
from .palette import get_palette, palette_val

__all__ = [
    'color_val_matplotlib', 'draw_lines', 'draw_labels',
    'imshow_det_lines', 'imshow_gt_det_lines'
]

EPS = 1e-2


def get_label_positions(lines):
    """
    compute the lines center position to draw labels on the image
    :param lines: shape[num_lines, 72]
    :return: positions: shape[num_lines, 2]
    """
    lines_x = lines[:, :-2]
    lines_y = lines[:, -2:]
    position_x = np.mean(lines_x, axis=1)
    position_y = np.mean(lines_y, axis=1)
    position = np.stack([position_x, position_y], axis=1)
    return position


def color_val_matplotlib(color):
    """Convert various input in BGR order to normalized RGB matplotlib color
    tuples.

    Args:
        color (:obj`Color` | str | tuple | int | ndarray): Color inputs.

    Returns:
        tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
    """
    color = mmcv.color_val(color)
    color = [color / 255 for color in color[::-1]]
    return tuple(color)


def _get_adaptive_scales(areas, min_area=800, max_area=30000):
    """Get adaptive scales according to areas.

    The scale range is [0.5, 1.0]. When the area is less than
    ``'min_area'``, the scale is 0.5 while the area is larger than
    ``'max_area'``, the scale is 1.0.

    Args:
        areas (ndarray): The areas of bboxes or masks with the
            shape of (n, ).
        min_area (int): Lower bound areas for adaptive scales.
            Default: 800.
        max_area (int): Upper bound areas for adaptive scales.
            Default: 30000.

    Returns:
        ndarray: The adaotive scales with the shape of (n, ).
    """
    scales = 0.5 + (areas - min_area) / (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales


def _get_bias_color(base, max_dist=30):
    """Get different colors for each masks.

    Get different colors for each masks by adding a bias
    color to the base category color.
    Args:
        base (ndarray): The base category color with the shape
            of (3, ).
        max_dist (int): The max distance of bias. Default: 30.

    Returns:
        ndarray: The new color for a mask with the shape of (3, ).
    """
    new_color = base + np.random.randint(
        low=-max_dist, high=max_dist + 1, size=3)
    return np.clip(new_color, 0, 255, new_color)


def draw_lines(ax, lines, color='g', alpha=0.8, thickness=2):
    """Draw bounding boxes on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        lines (ndarray): The input lines with the shape
            of (n, 72) in format(x1,x2,x3...,x70,y0,y1).
        color (list[tuple] | matplotlib.color): the colors for each
            bounding boxes.
        alpha (float): Transparency of bounding boxes. Default: 0.8.
        thickness (int): Thickness of lines. Default: 2.

    Returns:
        matplotlib.Axes: The result axes.
    """
    polylines = []
    for i, line in enumerate(lines):
        line_int_x = line[:-2].astype(np.int32)
        line_int_y = np.linspace(line[-2], line[-1], len(line)-2).astype(np.int32)
        polylines.append(Line2D(line_int_x, line_int_y))
    p = PatchCollection(
        polylines,
        facecolor='none',
        edgecolors=color,
        linewidths=thickness,
        alpha=alpha)
    ax.add_collection(p)

    return ax


def draw_labels(ax,
                labels,
                positions,
                scores=None,
                class_names=None,
                color='w',
                font_size=8,
                scales=None,
                horizontal_alignment='left'):
    """Draw labels on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        labels (ndarray): The labels with the shape of (n, ).
        positions (ndarray): The positions to draw each labels.
        scores (ndarray): The scores for each labels.
        class_names (list[str]): The class names.
        color (list[tuple] | matplotlib.color): The colors for labels.
        font_size (int): Font size of texts. Default: 8.
        scales (list[float]): Scales of texts. Default: None.
        horizontal_alignment (str): The horizontal alignment method of
            texts. Default: 'left'.

    Returns:
        matplotlib.Axes: The result axes.
    """
    for i, (pos, label) in enumerate(zip(positions, labels)):
        label_text = class_names[
            label] if class_names is not None else f'class {label}'
        if scores is not None:
            label_text += f'|{scores[i]:.02f}'
        text_color = color[i] if isinstance(color, list) else color

        font_size_mask = font_size if scales is None else font_size * scales[i]
        ax.text(
            pos[0],
            pos[1],
            f'{label_text}',
            bbox={
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            },
            color=text_color,
            fontsize=font_size_mask,
            verticalalignment='top',
            horizontalalignment=horizontal_alignment)

    return ax


def imshow_det_lines(img,
                     lines=None,
                     labels=None,
                     scores=None,
                     class_names=None,
                     score_thr=0,
                     line_color='random',
                     text_color='random',
                     thickness=2,
                     font_size=8,
                     win_name='',
                     show=True,
                     wait_time=0,
                     out_file=None):
    """Draw lines and class labels (with scores) on an image.

    Args:
        img (str | ndarray): The image to be displayed.
        lines (ndarray): lines, shaped (n, 74).
        scores(ndarray）：scores of lines.
        labels (ndarray): Labels of lines.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown. Default: 0.
        line_color (list[tuple] | tuple | str | None): Colors of bbox lines.
           If a single color is given, it will be applied to all classes.
           The tuple of color should be in RGB order. Default: 'green'.
        text_color (list[tuple] | tuple | str | None): Colors of texts.
           If a single color is given, it will be applied to all classes.
           The tuple of color should be in RGB order. Default: 'random'.
        thickness (int): Thickness of lines. Default: 2.
        font_size (int): Font size of texts. Default: 13.
        show (bool): Whether to show the image. Default: True.
        win_name (str): The window name. Default: ''.
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert lines is None or lines.ndim == 2, \
        f' lines ndim should be 2, but its ndim is {lines.ndim}.'
    assert scores is None or scores.ndim == 1, \
        f' scores ndim should be 1, but its ndim is {labels.ndim}.'
    assert scores is None or scores.shape[0] == lines.shape[0], \
        'scores.shape[0] should same as lines.shape[0].'
    assert labels is not None and labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert labels is not None and lines.shape[0] == labels.shape[0], \
        'labels.shape[0] should be same as lines.shape[0].'

    img = mmcv.imread(img).astype(np.uint8)
    score_in_line = True if scores is not None else False
    if score_thr > 0 and score_in_line:
        inds = scores > score_thr
        lines = lines[inds, :]    # lines.shape[num_line_thr, num_points+1]
        labels = labels[inds]

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    max_label = int(max(labels) if len(labels) > 0 else 0)
    text_palette = palette_val(get_palette(text_color, max_label + 1))
    text_colors = [text_palette[label] for label in labels]

    num_lines = 0
    if lines is not None:
        num_lines = lines.shape[0]
        line_palette = palette_val(get_palette(line_color, max_label + 1))
        colors = [line_palette[label] for label in labels[:num_lines]]
        horizontal_alignment = 'left'
        draw_lines(ax, lines, colors, alpha=0.8, thickness=thickness)
        positions = get_label_positions(lines).astype(np.int32) + thickness
        draw_labels(
            ax,
            labels[:num_lines],
            positions,
            scores=scores,
            class_names=class_names,
            color=text_colors,
            font_size=font_size,
            horizontal_alignment=horizontal_alignment)

    plt.imshow(img)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    plt.close()

    return img


def imshow_gt_det_lines(img,
                        annotation,
                        result,
                        class_names=None,
                        score_thr=0,
                        gt_line_color=(61, 102, 255),
                        gt_text_color=(200, 200, 200),
                        det_line_color=(241, 101, 72),
                        det_text_color=(200, 200, 200),
                        thickness=2,
                        font_size=13,
                        win_name='',
                        show=True,
                        wait_time=0,
                        out_file=None):
    """General visualization GT and result function.

    Args:
      img (str | ndarray): The image to be displayed.
      annotation (dict): Ground truth annotations where contain keys of
          'gt_lines' and 'gt_labels' or 'gt_masks'.
      result (tuple[list] | list | dict): The detection line result.
      class_names (list[str]): Names of each classes.
      score_thr (float): Minimum score of lines to be shown. Default: 0.
      gt_line_color (list[tuple] | tuple | str | None): Colors of gt_lines.
          If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (61, 102, 255).
      gt_text_color (list[tuple] | tuple | str | None): Colors of texts.
          If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (200, 200, 200).
      det_line_color (list[tuple] | tuple | str | None):Colors of bbox lines.
          If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (241, 101, 72).
      det_text_color (list[tuple] | tuple | str | None):Colors of texts.
          If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (200, 200, 200).
      thickness (int): Thickness of lines. Default: 2.
      font_size (int): Font size of texts. Default: 13.
      win_name (str): The window name. Default: ''.
      show (bool): Whether to show the image. Default: True.
      wait_time (float): Value of waitKey param. Default: 0.
      out_file (str, optional): The filename to write the image.
          Default: None.

    Returns:
        ndarray: The image with bboxes or masks drawn on it.
    """
    assert 'gt_lines' in annotation
    assert 'gt_labels' in annotation
    assert isinstance(result, (tuple, list, dict)), 'Expected ' \
        f'tuple or list or dict, but get {type(result)}'

    gt_lines = annotation['gt_lines']
    gt_labels = annotation['gt_labels']

    img = mmcv.imread(img)

    img = imshow_det_lines(
        img=img,
        lines=gt_lines,
        labels=gt_labels,
        class_names=class_names,
        line_color=gt_line_color,
        text_color=gt_text_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=False)

    if isinstance(result, dict):
        lines = result['pred_lines']
        scores = result['pred_scores'] if 'scores' in result else None
        assert 'pred_labels' in result, 'pred lines should have corresponding labels'
        labels = result['pred_labels']

    img = imshow_det_lines(
        img=img,
        lines=lines,
        scores=scores,
        labels=labels,
        class_names=class_names,
        score_thr=score_thr,
        line_color=det_line_color,
        text_color=det_text_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=show,
        wait_time=wait_time,
        out_file=out_file)
    return img
