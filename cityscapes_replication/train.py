#!/usr/bin/env python3
"""
Train mmsegmentation model using MMEngine Runner.

This is the standard way to train models in mmseg 1.x with mmengine.
The config file contains all training settings.

Usage:
    # Single GPU
    python train.py configs/segformer_mit-b5_cityscapes_1024x1024.py --work-dir ./work_dirs/segformer

    # Distributed training (4 GPUs)
    torchrun --nproc_per_node=4 train.py configs/segformer_mit-b5_cityscapes_1024x1024.py --work-dir ./work_dirs/segformer
"""

import argparse
import os
import os.path as osp

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--local_rank',
        '--local-rank',
        type=int,
        default=0,
        help='Local rank for distributed training')
    args = parser.parse_args()
    
    # For distributed training compatibility
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    return args


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    
    # Set work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    
    # Enable automatic-mixed-precision training
    if args.amp:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print('AMP training is already enabled in your config.')
        else:
            assert optim_wrapper == 'OptimWrapper', (
                'AMP training requires OptimWrapper, but got '
                f'{optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'
    
    # Resume training
    if args.resume:
        cfg.resume = True
    
    # Build the runner from config
    runner = Runner.from_cfg(cfg)
    
    # Start training
    runner.train()


if __name__ == '__main__':
    main()
