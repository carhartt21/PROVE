#!/usr/bin/env python3
"""
Comprehensive profiler for fine_grained_test.py to identify all bottlenecks.

This script profiles each component of the testing pipeline:
1. Image loading (cv2.imread)
2. Label loading (cv2.imread)
3. RGB label decoding (MapillaryVistas only)
4. Image preprocessing (normalization, tensor conversion)
5. Model inference
6. IoU metric computation
7. Batch collation and data transfer

Usage:
    bsub -q BatchGPU -R "rusage[mem=16G,ngpus_excl_p=1]" -n 4 \
        -J "profile_test" -o logs/profile_test_%J.out \
        'conda activate mmseg && python tools/profile_full_test.py'
"""

import os
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

# Add project root to path
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir.parent))

try:
    import cv2
    import torch
    from mmengine.config import Config
    from mmseg.registry import MODELS
    from mmengine.model.utils import revert_sync_batchnorm
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Please run: conda activate mmseg")
    sys.exit(1)

import custom_transforms
from fine_grained_test import (
    process_label_for_dataset,
    compute_iou_metrics,
    _get_mapillary_rgb_lut,
    detect_model_num_classes,
)


class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, name, times_dict):
        self.name = name
        self.times_dict = times_dict
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        elapsed = (time.perf_counter() - self.start) * 1000
        self.times_dict[self.name].append(elapsed)


def find_test_files(dataset_name, domain="clear_day", limit=50):
    """Find test images and labels for a dataset."""
    base_path = Path("${AWARE_DATA_ROOT}/FINAL_SPLITS/test")
    
    img_dir = base_path / "images" / dataset_name / domain
    label_dir = base_path / "labels" / dataset_name / domain
    
    if not img_dir.exists():
        print(f"ERROR: Image directory not found: {img_dir}")
        return []
    
    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))[:limit]
    
    pairs = []
    for img_path in img_files:
        label_path = label_dir / (img_path.stem + ".png")
        if label_path.exists():
            pairs.append((img_path, label_path))
    
    return pairs


def profile_dataset(dataset_name, model_path=None, config_path=None, num_images=50):
    """Profile the testing pipeline for a specific dataset."""
    
    print(f"\n{'='*60}")
    print(f"PROFILING: {dataset_name}")
    print(f"{'='*60}")
    
    # Determine number of classes
    num_classes = 66 if dataset_name == "MapillaryVistas" else 19
    
    # Find test files
    pairs = find_test_files(dataset_name, limit=num_images)
    if not pairs:
        print(f"No test files found for {dataset_name}")
        return None
    
    print(f"Found {len(pairs)} test image-label pairs")
    
    # Timing storage
    times = defaultdict(list)
    
    # Preprocessing constants
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    
    # Pre-build MapillaryVistas LUT if needed
    if dataset_name == "MapillaryVistas":
        print("Pre-building RGB LUT...")
        _ = _get_mapillary_rgb_lut()
    
    # Profile each component
    print(f"\nProfiling {len(pairs)} images...")
    
    for img_path, label_path in pairs:
        # 1. Image loading
        with Timer("img_load", times):
            img = cv2.imread(str(img_path))
        
        # 2. Color conversion
        with Timer("color_convert", times):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 3. Label loading
        with Timer("label_load", times):
            label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        
        # 4. Label processing (RGB decode for MapillaryVistas)
        with Timer("label_process", times):
            processed_label = process_label_for_dataset(label, dataset_name, num_classes)
        
        # 5. Image normalization
        with Timer("normalize", times):
            img_norm = (img_rgb.astype(np.float32) - mean) / std
        
        # 6. Tensor conversion
        with Timer("to_tensor", times):
            img_chw = img_norm.transpose(2, 0, 1)
            img_tensor = torch.from_numpy(img_chw).float()
        
        # 7. Simulate model inference (just tensor operations)
        with Timer("tensor_ops", times):
            # Simulate what happens during inference prep
            batch_tensor = img_tensor.unsqueeze(0)
            if torch.cuda.is_available():
                batch_tensor = batch_tensor.cuda()
        
        # 8. Simulate prediction (random output same shape as label)
        pred = np.random.randint(0, num_classes, processed_label.shape, dtype=np.uint8)
        
        # 9. IoU computation
        with Timer("iou_compute", times):
            compute_iou_metrics(pred, processed_label, num_classes)
    
    # If we have GPU and model, profile actual inference
    if model_path and config_path and torch.cuda.is_available():
        print("\nProfiling actual model inference...")
        try:
            cfg = Config.fromfile(config_path)
            model = MODELS.build(cfg.model)
            model = revert_sync_batchnorm(model)
            
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            model.load_state_dict(state_dict, strict=True)
            model = model.cuda().eval()
            
            # Profile actual inference
            for img_path, _ in pairs[:10]:  # Limit to 10 for inference
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = (img.astype(np.float32) - mean) / std
                img = img.transpose(2, 0, 1)
                img_tensor = torch.from_numpy(img).float().unsqueeze(0).cuda()
                
                img_meta = [{
                    'ori_shape': (512, 512),
                    'img_shape': (512, 512),
                    'pad_shape': (512, 512),
                    'scale_factor': (1.0, 1.0),
                }]
                
                # Warm up
                with torch.no_grad():
                    _ = model.inference(img_tensor, img_meta)
                torch.cuda.synchronize()
                
                # Actual timing
                with Timer("model_inference", times):
                    with torch.no_grad():
                        _ = model.inference(img_tensor, img_meta)
                    torch.cuda.synchronize()
                    
        except Exception as e:
            print(f"Could not profile model inference: {e}")
    
    # Print results
    print(f"\n{'-'*60}")
    print(f"RESULTS for {dataset_name} ({len(pairs)} images)")
    print(f"{'-'*60}")
    
    total_time = 0
    for name, values in sorted(times.items(), key=lambda x: -np.mean(x[1])):
        mean_ms = np.mean(values)
        std_ms = np.std(values)
        total_ms = np.sum(values)
        total_time += mean_ms
        print(f"  {name:<20}: {mean_ms:8.2f} ± {std_ms:5.2f} ms  (total: {total_ms/1000:.2f}s)")
    
    print(f"\n  {'TOTAL per image':<20}: {total_time:8.2f} ms")
    print(f"  {'Estimated for 4949 img':<20}: {total_time * 4949 / 1000 / 60:8.1f} min")
    
    return times


def main():
    parser = argparse.ArgumentParser(description="Profile fine_grained_test.py")
    parser.add_argument("--datasets", nargs="+", default=["BDD10k", "MapillaryVistas"],
                        help="Datasets to profile")
    parser.add_argument("--num-images", type=int, default=50,
                        help="Number of images to profile per dataset")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model checkpoint for inference profiling")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to model config for inference profiling")
    args = parser.parse_args()
    
    print("="*60)
    print("FINE-GRAINED TEST PROFILER")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("WARNING: No GPU available, inference profiling will be skipped")
    
    results = {}
    for dataset in args.datasets:
        results[dataset] = profile_dataset(
            dataset,
            model_path=args.model,
            config_path=args.config,
            num_images=args.num_images
        )
    
    # Compare datasets
    if len(results) > 1 and all(r is not None for r in results.values()):
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        
        # Get common timing categories
        all_keys = set()
        for times in results.values():
            all_keys.update(times.keys())
        
        print(f"\n{'Component':<20}", end="")
        for dataset in results.keys():
            print(f"{dataset:>15}", end="")
        print()
        print("-" * (20 + 15 * len(results)))
        
        for key in sorted(all_keys):
            print(f"{key:<20}", end="")
            for dataset, times in results.items():
                if key in times:
                    mean_ms = np.mean(times[key])
                    print(f"{mean_ms:>12.2f} ms", end="")
                else:
                    print(f"{'N/A':>15}", end="")
            print()
    
    print(f"\n{'='*60}")
    print("PROFILING COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
