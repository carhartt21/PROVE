#!/usr/bin/env python3
"""
Critical test: Train model with FROZEN BACKBONE.
If mIoU drops significantly, backbone is learning.
If mIoU stays ~22%, backbone is irrelevant (pure decoder learning from labels).
"""

from pathlib import Path
import sys

sys.path.insert(0, '/home/mima2416/repositories/PROVE')

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.optim import DefaultOptimWrapperConstructor
import torch

def freeze_backbone_test():
    """Train with frozen backbone - if mIoU drops, backbone matters."""
    cfg_path = '/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/gen_stargan_v2/iddaw/pspnet_r50_ratio0p00/configs/training_config.py'
    output_dir = Path('/home/mima2416/repositories/PROVE/local_debug_batches/frozen_backbone_2000')
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(cfg_path)
    cfg.train_cfg.max_iters = 2000
    cfg.train_cfg.val_interval = 2000
    cfg.default_hooks['logger']['interval'] = 100
    cfg.work_dir = str(output_dir)

    cfg.mixed_dataloader.enabled = False
    cfg.generated_augmentation.enabled = False

    runner = Runner.from_cfg(cfg)
    
    # FREEZE THE BACKBONE
    print("\n" + "="*60)
    print("FREEZING BACKBONE")
    print("="*60)
    
    backbone = runner.model.backbone
    for param in backbone.parameters():
        param.requires_grad = False
    
    print(f"Backbone frozen. Trainable parameters:")
    total_params = sum(p.numel() for p in runner.model.parameters())
    trainable_params = sum(p.numel() for p in runner.model.parameters() if p.requires_grad)
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen: {total_params - trainable_params:,} ({100*(total_params-trainable_params)/total_params:.1f}%)")
    
    # Build new optimizer (only for trainable params)
    from mmengine.optim import OPTIM_WRAPPERS, OptimWrapperDict
    optim_cfg = cfg.optim_wrapper
    if isinstance(optim_cfg, dict):
        # Single optimizer wrapper
        optimizer_cfg = optim_cfg.optimizer
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, runner.model.parameters()),
            lr=optimizer_cfg.lr,
            momentum=optimizer_cfg.momentum,
            weight_decay=optimizer_cfg.weight_decay
        )
        runner.optim_wrapper.optimizer = optimizer
    # Actually, simpler: just let MMEngine handle it
    # The optimizer will only see gradients for trainable params anyway
    
    print(f"\nTraining with frozen backbone for 2000 iters...")
    print(f"{'='*60}\n")
    
    runner.train()


if __name__ == '__main__':
    freeze_backbone_test()
