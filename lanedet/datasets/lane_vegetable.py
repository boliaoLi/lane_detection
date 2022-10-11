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
class VegtableDataset(CustomDataset):

    CLASSES = ('crop_line', 'person', 'bicycle', 'car')
    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)]

    def load_annotation(self, file):
        with open(file, 'r', encoding='utf8') as fp:
            data_info = json.load(fp)
            num_class = len(self.CLASSES)
            class_names = {}
            for i in range(num_class+1):
                class_name = self.CLASSES[i-1]
                class_names[class_name] = i
            ann = data_info['ann']
            labels = ann['labels']
            labels_id = np.array([class_names[label] for label in labels], dtype=np.float32)
            lines_x = np.array(ann['lines_x'], dtype=np.float32)
            lines_y = np.array(ann['lines_y'], dtype=np.float32)
            lines = np.concatenate((lines_x, lines_y), axis=1)
            ann['labels'] = labels_id
            ann['lines_x'] = lines_x
            ann['lines_y'] = lines_y
            ann['lines'] = lines
            data_info['ann'] =ann
            return data_info

    def load_annotations(self, ann_file):
        ann_file_list = os.listdir(ann_file)
        data_info = []
        for file in ann_file_list:
            ann = os.path.join(ann_file, file)
            data_info.append(self.load_annotation(ann))
        return data_info



