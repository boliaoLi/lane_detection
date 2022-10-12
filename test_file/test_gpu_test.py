from lanedet.apis.test import single_gpu_test
from lanedet.models import build_detector
from lanedet.datasets import (build_dataset, build_dataloader,
                            replace_ImageToTensor)
from lanedet.utils import update_data_root
from mmcv import Config, DictAction
import copy


if __name__ == '__main__':
    config_file = 'D:/model/lane_detection/configs/detr/detr_vegetabel.py'
    cfg = Config.fromfile(config_file)
    # update data root according to MMDET_DATASETS
    update_data_root(cfg)
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    datasets = [build_dataset(cfg.data.val)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    val_dataloader_default_args = dict(
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=False,
        shuffle=False,
        persistent_workers=False)

    val_dataloader_args = {
        **val_dataloader_default_args,
        **cfg.data.get('val_dataloader', {})
    }
    if val_dataloader_args['samples_per_gpu'] > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.val.pipeline = replace_ImageToTensor(
            cfg.data.val.pipeline)
    val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

    val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
    single_gpu_test(model, val_dataloader)