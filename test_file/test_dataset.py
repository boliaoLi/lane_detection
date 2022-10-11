from lanedet.datasets.lane_vegetable import VegtableDataset
from mmcv import Config, DictAction
from lanedet.datasets import build_dataset, build_dataloader

if __name__ == '__main__':
    config_file = 'D:/model/lane_detection/configs/datasets/vegetabel.py'
    cfg = Config.fromfile(config_file)
    datasets = [build_dataset(cfg.data.train)]
    print('done')