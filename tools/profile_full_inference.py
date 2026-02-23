#!/usr/bin/env python
"""Profile full inference pipeline with proper GPU synchronization.

This measures real GPU inference time by synchronizing CUDA before/after.
Run this on a GPU node to get accurate measurements.
"""

import sys
import time
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
from typing import List, Tuple, Dict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def profile_inference(
    checkpoint_path: str,
    data_root: str,
    dataset: str = 'BDD10k',
    num_images: int = 50,
    batch_size: int = 4
) -> Dict:
    """Profile full inference with GPU timing."""
    from mmengine.config import Config
    from mmseg.apis import init_model
    
    # Load model
    print(f"\n{'='*60}")
    print(f"Loading model from: {checkpoint_path}")
    
    config_path = str(Path(checkpoint_path).parent / 'training_config.py')
    if not Path(config_path).exists():
        # Try to find config in parent directories
        for parent in Path(checkpoint_path).parents:
            cfg_path = parent / 'training_config.py'
            if cfg_path.exists():
                config_path = str(cfg_path)
                break
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model = init_model(config_path, checkpoint_path, device=device)
    model.eval()
    
    # Get model input size from config
    cfg = Config.fromfile(config_path)
    img_size = cfg.crop_size if hasattr(cfg, 'crop_size') else (512, 512)
    print(f"Input size: {img_size}")
    
    # Determine number of classes
    num_classes = model.decode_head.num_classes
    print(f"Number of classes: {num_classes}")
    
    # Find test images
    img_root = Path(data_root) / 'test/images' / dataset
    domain = 'clear_day'
    img_dir = img_root / domain
    if not img_dir.exists():
        domains = [d.name for d in img_root.iterdir() if d.is_dir()]
        if domains:
            domain = domains[0]
            img_dir = img_root / domain
    
    img_files = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))[:num_images]
    print(f"Found {len(img_files)} images from {domain}")
    
    # Warmup
    print("\nWarmup (10 batches)...")
    dummy_input = torch.randn(batch_size, 3, *img_size).to(device)
    dummy_metas = [{'ori_shape': img_size, 'img_shape': img_size, 
                    'pad_shape': img_size, 'scale_factor': (1.0, 1.0)} for _ in range(batch_size)]
    for _ in range(10):
        with torch.no_grad():
            _ = model.inference(dummy_input, dummy_metas)
        if device == 'cuda':
            torch.cuda.synchronize()
    
    # Normalization parameters
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    
    # Timing accumulators
    times = {
        'img_read': [],
        'img_preprocess': [],
        'to_gpu': [],
        'inference': [],
        'result_transfer': [],
        'postprocess': [],
        'total': []
    }
    
    # Process images in batches
    print(f"\nProfiling {len(img_files)} images in batches of {batch_size}...")
    
    num_batches = (len(img_files) + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(img_files))
        batch_files = img_files[start_idx:end_idx]
        actual_batch_size = len(batch_files)
        
        t_total_start = time.perf_counter()
        
        # 1. Image reading
        t0 = time.perf_counter()
        images = []
        for img_path in batch_files:
            img = cv2.imread(str(img_path))
            images.append(img)
        times['img_read'].append((time.perf_counter() - t0) * 1000)
        
        # 2. Preprocessing
        t0 = time.perf_counter()
        batch_tensors = []
        batch_metas = []
        for img in images:
            # Resize to model input size
            img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_LINEAR)
            # BGR to RGB
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            # Normalize
            img_float = img_rgb.astype(np.float32)
            img_norm = (img_float - mean) / std
            # HWC to CHW
            img_chw = img_norm.transpose(2, 0, 1)
            batch_tensors.append(torch.from_numpy(img_chw).float())
            batch_metas.append({
                'ori_shape': img.shape[:2],
                'img_shape': img_size,
                'pad_shape': img_size,
                'scale_factor': (1.0, 1.0)
            })
        times['img_preprocess'].append((time.perf_counter() - t0) * 1000)
        
        # 3. Transfer to GPU
        t0 = time.perf_counter()
        batch_input = torch.stack(batch_tensors).to(device)
        if device == 'cuda':
            torch.cuda.synchronize()
        times['to_gpu'].append((time.perf_counter() - t0) * 1000)
        
        # 4. Model inference (GPU)
        t0 = time.perf_counter()
        with torch.no_grad():
            results = model.inference(batch_input, batch_metas)
        if device == 'cuda':
            torch.cuda.synchronize()
        times['inference'].append((time.perf_counter() - t0) * 1000)
        
        # 5. Transfer results back (with GPU argmax optimization)
        t0 = time.perf_counter()
        pred_maps = []
        for i in range(actual_batch_size):
            if isinstance(results, torch.Tensor):
                result_tensor = results[i]
                # OPTIMIZATION: Do argmax on GPU before transfer (reduces 66MB → 1MB)
                if result_tensor.ndim == 3:
                    pred = result_tensor.argmax(dim=0).cpu().numpy()
                else:
                    pred = result_tensor.cpu().numpy()
            elif isinstance(results, list) and hasattr(results[i], 'pred_sem_seg'):
                result_tensor = results[i].pred_sem_seg.data.squeeze()
                if result_tensor.ndim == 3:
                    pred = result_tensor.argmax(dim=0).cpu().numpy()
                else:
                    pred = result_tensor.cpu().numpy()
            else:
                pred = results[i].cpu().numpy().squeeze() if isinstance(results[i], torch.Tensor) else results[i]
            pred_maps.append(pred)
        if device == 'cuda':
            torch.cuda.synchronize()
        times['result_transfer'].append((time.perf_counter() - t0) * 1000)
        
        # 6. Postprocessing (resize only, argmax already done on GPU)
        t0 = time.perf_counter()
        for i, pred in enumerate(pred_maps):
            # Resize back to original size
            ori_h, ori_w = images[i].shape[:2]
            if pred.shape != (ori_h, ori_w):
                pred = cv2.resize(pred.astype(np.uint8), (ori_w, ori_h), 
                                  interpolation=cv2.INTER_NEAREST)
        times['postprocess'].append((time.perf_counter() - t0) * 1000)
        
        times['total'].append((time.perf_counter() - t_total_start) * 1000)
    
    # Calculate statistics
    print(f"\n{'='*60}")
    print(f"TIMING RESULTS ({dataset}, batch_size={batch_size})")
    print(f"{'='*60}")
    
    results = {}
    for name, values in times.items():
        avg = np.mean(values)
        std_dev = np.std(values)
        per_img = avg / batch_size
        results[name] = {'avg_batch_ms': avg, 'std_batch_ms': std_dev, 'per_img_ms': per_img}
        print(f"{name:20s}: {avg:8.2f} ± {std_dev:5.2f} ms/batch  ({per_img:6.2f} ms/image)")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_per_img = results['total']['per_img_ms']
    inference_per_img = results['inference']['per_img_ms']
    io_per_img = results['img_read']['per_img_ms']
    print(f"Total time per image:     {total_per_img:.2f} ms")
    print(f"  - GPU inference:        {inference_per_img:.2f} ms ({inference_per_img/total_per_img*100:.1f}%)")
    print(f"  - Image I/O:            {io_per_img:.2f} ms ({io_per_img/total_per_img*100:.1f}%)")
    print(f"  - Other:                {total_per_img - inference_per_img - io_per_img:.2f} ms")
    print(f"\nThroughput:               {1000/total_per_img:.1f} images/sec")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Profile full inference pipeline')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--data-root', type=str, 
                        default='${AWARE_DATA_ROOT}/FINAL_SPLITS',
                        help='Path to data root')
    parser.add_argument('--dataset', type=str, default='BDD10k',
                        choices=['BDD10k', 'MapillaryVistas', 'IDD-AW', 'OUTSIDE15k'],
                        help='Dataset to profile')
    parser.add_argument('--num-images', type=int, default=50,
                        help='Number of images to profile')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    
    args = parser.parse_args()
    
    profile_inference(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        dataset=args.dataset,
        num_images=args.num_images,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
