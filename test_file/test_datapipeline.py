from mmcv import Config, DictAction
from lanedet.datasets import build_dataset,build_dataloader

if __name__ == '__main__':
    config_file = r'D:\model\lane_detection\configs\_base_\datasets\Tusimple.py'
    cfg = Config.fromfile(config_file)
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
    j = 0
    for data_loader in data_loaders:
        for i, data_batch in enumerate(data_loader):
            j += 1
            print(j)
    print("芜湖！起飞！")