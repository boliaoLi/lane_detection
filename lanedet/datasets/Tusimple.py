"""
Tusimple 数据集格式说明：
original img：
    img文件位于clips文件夹下，包含四个文件夹[0313-1, 0313-2, 0531, 0601],
    每一个文件夹代表一个道路场景，每个道路场景下包含N个图像序列文件夹，每一个图像序列包含20张图片。
annotation(车道线标签)：
    车道线标签由三个json文件，[0313, 0531, 0601]与original img对应
    json文件内容为长度为图片总数的列表list[dict]，列表中每一个元素为一个字典dict，其格式为
    {
        “lanes”: list[list], len(lanes)=图片中车道线的数量，len(lanes[0])=48
            eg:[[-2,-2,-2,632,625,617,609,....,299,-2,-2]]
        "h_samples": list[], len(h_samples)=48
            eg:[240,250,260,...,710]
        "raw_file":该标签对应的原始图片的相对路径
            eg:"clips/0313-1/6040/20.jpg"
    }
需要将tusimple转化为lanedet标准格式，格式如下：
    .. code-block:: none

        [
            {
                'filename': 'a.jpg',
                'width': 1280,
                'height': 720,
                'ann': {
                    'lines': <np.ndarray> (n, 74) in (x1, x2,..., y_start, y_end) order.
                    'labels': <np.ndarray> (n, ),
                    'bboxes_ignore': <np.ndarray> (k, 72), (optional field)
                    'labels_ignore': <np.ndarray> (k, 72) (optional field)
                }
            },
            ...
        ]

"""

import contextlib
import io
import itertools
import logging
import os
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from lanedet.core import eval_recalls
from .builder import DATASETS
from .custom import CustomDataset
import json


@DATASETS.register_module()
class TusimpleDataset(CustomDataset):

    CLASSES = ('crop_line', 'tusimple_line', 'coushude1', 'coushude2')
    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)]

    def load_annotations(self, ann_file):
        """
        :param ann_file: Annotation file path.
        :return: data_info
        """
        ann_list = os.listdir(ann_file)
        data_info = []
        for label in ann_list:
            file = open(os.path.join(ann_file, label), 'r', encoding='utf-8')
            for data in file.readlines():
                data_info.append(self.convert_data(json.loads(data)))
        return data_info

    def convert_data(self, data):
        """
        将tusimple数据格式转换为lane_detection通用格式
        :param data:
        {
        “lanes”：list[list],
        "h_samples":list,
        "raw_file":str
            }
        :return:
        """
        data_info = {}
        data_info['filename'] = data['raw_file']
        data_info['width'] = 1280
        data_info['height'] = 720
        h_samples = data['h_samples']
        lanes = data['lanes']
        ann = {}
        y_start = h_samples[0]
        y_end = h_samples[-1]
        line_new = []
        for lane in lanes:
            line = []
            for i in range(len(lane)):
                if lane[i] != -2:
                    line.append([lane[i], h_samples[i]])
                    if i != 0 and lane[i-1] == -2:
                        y_start = h_samples[i-1]
                    elif i != len(lane)-1 and lane[i+1] == -2:
                        y_end = h_samples[i+1]
                else:
                    continue
            line = np.array(line)
            if len(line):
                line_y = np.linspace(y_start, y_end, 72)
                line_x = np.interp(line_y, line[:, 1], line[:, 0])
                line = np.concatenate([line_x, [y_start, y_end]])
                line_new.append(line)
        ann['lines'] = np.array(line_new)
        ann['lines_x'] = ann['lines'][:, :-2]
        ann['lines_y'] = ann['lines'][:, -2:]
        ann['labels'] = np.full(ann['lines'].shape[0], self.CLASSES.index('tusimple_line'))
        data_info['ann'] = ann
        return data_info