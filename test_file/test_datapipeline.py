from mmcv import Config, DictAction
from lanedet.datasets import build_dataset,build_dataloader

if __name__ == '__main__':
    cfg_file = '/configs/_base_/datasets/vegetabel.py'
    cfg = Config.fromfile(cfg_file)
    datasets = [build_dataset(cfg.data.train)]
    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']
    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=1,
        dist=False,
        seed=None,
        runner_type=runner_type,
        persistent_workers=False)
    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in datasets]
    for data_loader in data_loaders:
        for i, data_batch in enumerate(data_loader):
            print(data_batch)

    print("芜湖！起飞！")