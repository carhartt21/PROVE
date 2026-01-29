#!/usr/bin/env python3
"""Run 5 sanity tests for 2000 iterations and collect results."""

from pathlib import Path
import json
import re
import custom_transforms
import unified_datasets
import torch
import numpy as np

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.hooks import Hook
from mmseg.registry import DATASETS


def extract_mIoU_from_log(log_path):
    """Extract val/mIoU from training log."""
    with open(log_path) as f:
        content = f.read()
    # Look for pattern: val/mIoU: <number>
    match = re.search(r'val/mIoU: ([\d.]+)', content)
    if match:
        return float(match.group(1))
    return None


class NoiseInputHook(Hook):
    """Replace inputs and labels with noise/zeros during training."""
    def before_train_iter(self, runner):
        pass

    def after_train_step(self, runner):
        pass


def run_sanity_test(test_name, output_dir, cfg_path, setup_fn=None):
    """Run a single sanity test."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(cfg_path)
    cfg.train_cfg.max_iters = 2000
    cfg.train_cfg.val_interval = 2000
    cfg.default_hooks['logger']['interval'] = 100
    cfg.work_dir = str(output_dir)

    cfg.mixed_dataloader.enabled = False
    cfg.generated_augmentation.enabled = False

    runner = Runner.from_cfg(cfg)

    if setup_fn:
        setup_fn(runner)

    print(f"\n{'='*60}")
    print(f"Running: {test_name}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    runner.train()

    # Extract mIoU from log
    log_files = list(output_dir.glob('*/*.log'))
    if log_files:
        mIoU = extract_mIoU_from_log(log_files[0])
        return mIoU
    return None


def setup_noise(runner):
    """Setup noise hook for replacing inputs/labels."""
    class NoiseHook(Hook):
        def before_val(self, runner):
            # Store original dataset to restore after validation
            pass

        def after_val(self, runner):
            pass

    # Monkey patch the dataloader iteration
    orig_dataloader = runner.train_dataloader

    class NoiseDataLoader:
        def __init__(self, orig):
            self.orig = orig
            self.batch_size = orig.batch_size if hasattr(orig, 'batch_size') else 8
            self.dataset = orig.dataset if hasattr(orig, 'dataset') else None

        def __iter__(self):
            for batch in self.orig:
                # Replace inputs with noise
                if isinstance(batch, dict):
                    batch['inputs'] = torch.randn_like(batch['inputs'])
                    # Set labels to all zeros (ignore class)
                    for sample in batch['data_samples']:
                        if hasattr(sample, 'gt_sem_seg'):
                            sample.gt_sem_seg.data = torch.full_like(
                                sample.gt_sem_seg.data, fill_value=255, dtype=torch.uint8
                            )
                yield batch

        def __len__(self):
            return len(self.orig)

    runner._train_loop._dataloader = NoiseDataLoader(orig_dataloader)


def setup_real_only(runner):
    """Use real IDD-AW clear_day data."""
    # Already the default from config
    pass


def setup_stargan_v2(runner):
    """Use stargan_v2 generated images (100%)."""
    # Already configured in cfg, just use as-is
    
    dataset = runner.train_dataloader.dataset
    if hasattr(dataset, 'data_list'):
        # Load stargan_v2 generated image manifest
        import csv
        from pathlib import Path
        manifest_path = '/scratch/aaa_exchange/AWARE/GENERATED_IMAGES/stargan_v2/manifest.csv'
        gen_list = []
        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('dataset', '') == 'IDD-AW':
                    gen_list.append({
                        'img_path': row['generated_image_path'],
                        'seg_map_path': row['original_label_path'],
                        'reduce_zero_label': False,
                        'seg_fields': [],
                    })
        if gen_list:
            dataset.data_list = gen_list[:len(dataset.data_list)]


def setup_bdd10k_mismatch(runner):
    """Use BDD10k images with IDD-AW labels."""
    dataset = runner.train_dataloader.dataset
    if hasattr(dataset, 'data_list'):
        # Get BDD10k images
        bdd_cfg = dict(
            type='CityscapesDataset',
            data_root='/scratch/aaa_exchange/AWARE/FINAL_SPLITS',
            data_prefix=dict(
                img_path='train/images/BDD10k/clear_day',
                seg_map_path='train/labels/BDD10k/clear_day',
            ),
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
        )
        bdd_dataset = DATASETS.build(bdd_cfg)
        bdd_dataset.full_init()
        bdd_list = bdd_dataset.data_list

        # Get IDD-AW labels
        idd_list = dataset.data_list

        # Combine: BDD10k images + IDD-AW labels
        n = min(len(bdd_list), len(idd_list))
        mismatch_list = []
        for i in range(n):
            mismatch_list.append({
                'img_path': bdd_list[i]['img_path'],
                'seg_map_path': idd_list[i]['seg_map_path'],
                'reduce_zero_label': idd_list[i].get('reduce_zero_label', False),
                'seg_fields': [],
            })
        dataset.data_list = mismatch_list


def setup_outside15k_mismatch(runner):
    """Use OUTSIDE15k images with IDD-AW labels."""
    dataset = runner.train_dataloader.dataset
    if hasattr(dataset, 'data_list'):
        # Get OUTSIDE15k images
        outside_cfg = dict(
            type='CityscapesDataset',
            data_root='/scratch/aaa_exchange/AWARE/FINAL_SPLITS',
            data_prefix=dict(
                img_path='train/images/OUTSIDE15k/clear_day',
                seg_map_path='train/labels/OUTSIDE15k/clear_day',
            ),
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
        )
        outside_dataset = DATASETS.build(outside_cfg)
        outside_dataset.full_init()
        outside_list = outside_dataset.data_list

        # Get IDD-AW labels
        idd_list = dataset.data_list

        # Combine: OUTSIDE15k images + IDD-AW labels
        n = min(len(outside_list), len(idd_list))
        mismatch_list = []
        for i in range(n):
            mismatch_list.append({
                'img_path': outside_list[i]['img_path'],
                'seg_map_path': idd_list[i]['seg_map_path'],
                'reduce_zero_label': idd_list[i].get('reduce_zero_label', False),
                'seg_fields': [],
            })
        dataset.data_list = mismatch_list


# Configuration
cfg_path = '/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/gen_stargan_v2/iddaw/pspnet_r50_ratio0p00/configs/training_config.py'
base_output_dir = '/home/mima2416/repositories/PROVE/local_debug_batches'

tests = [
    ('noise_2000', setup_noise),
    ('stargan_v2_2000', setup_stargan_v2),
    ('real_only_2000', setup_real_only),
    ('bdd10k_mismatch_2000', setup_bdd10k_mismatch),
    ('outside15k_mismatch_2000', setup_outside15k_mismatch),
]

results = {}

for test_name, setup_fn in tests:
    output_dir = Path(base_output_dir) / test_name
    mIoU = run_sanity_test(test_name, output_dir, cfg_path, setup_fn)
    results[test_name] = mIoU
    print(f"\n{test_name}: val/mIoU = {mIoU}")

# Summary
print(f"\n{'='*60}")
print("SUMMARY (2000 iterations)")
print(f"{'='*60}")
for test_name, mIoU in results.items():
    print(f"{test_name:30s}: {mIoU:6.2f}")

# Save to JSON
results_file = Path(base_output_dir) / 'sanity_results_2000iters.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_file}")
