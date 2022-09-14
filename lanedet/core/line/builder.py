# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import Registry, build_from_cfg

LINE_ASSIGNERS = Registry('line_assigner')
LINE_SAMPLERS = Registry('line_sampler')
LINE_CODERS = Registry('line_coder')


def build_assigner(cfg, **default_args):
    """Builder of box assigner."""
    return build_from_cfg(cfg, LINE_ASSIGNERS, default_args)


def build_sampler(cfg, **default_args):
    """Builder of box sampler."""
    return build_from_cfg(cfg, LINE_SAMPLERS, default_args)


def build_bbox_coder(cfg, **default_args):
    """Builder of box coder."""
    return build_from_cfg(cfg, LINE_CODERS, default_args)
