#!/usr/bin/env python3
"""
Domain Adaptation Ablation Study: Cross-Dataset Domain Generalization

This script evaluates models trained on BDD10k, IDD-AW, or MapillaryVistas
on Cityscapes (clear_day) and ACDC (adverse weather: foggy, night, rainy, snowy).

Research Questions:
1. How well do models trained on one dataset generalize to European urban scenes?
2. How does performance vary across clear_day vs adverse weather conditions?
3. Which training dataset provides the best domain adaptation?

Usage:
    # Run single test (local, no LSF)
    python scripts/run_domain_adaptation_tests.py \
        --source-dataset BDD10k \
        --model pspnet_r50 \
        --strategy baseline
    
    # Run all tests for a strategy
    python scripts/run_domain_adaptation_tests.py --all --strategy baseline
    
    # Dry run to see what would be tested
    python scripts/run_domain_adaptation_tests.py --all --strategy baseline --dry-run

Output:
    Results saved to:
    /scratch/aaa_exchange/AWARE/WEIGHTS/{strategy}/{source_dataset}/{model}/domain_adaptation/{timestamp}/
        ├── results.json          # Per-domain metrics
        └── test_report.txt       # Summary report
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
import torch
import cv2
from tqdm import tqdm

# Add project root to path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Import custom transforms before MMSeg
import custom_transforms

from mmengine.config import Config
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

# Register all modules
register_all_modules(init_default_scope=True)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='mmseg')


# ============================================================================
# Configuration
# ============================================================================

DATA_ROOT = Path("/scratch/aaa_exchange/AWARE/FINAL_SPLITS")
WEIGHTS_ROOT = Path("/scratch/aaa_exchange/AWARE/WEIGHTS")

# Source datasets (models trained on these)
SOURCE_DATASETS = ['bdd10k', 'idd-aw', 'mapillaryvistas']

# Target domains for evaluation
TARGET_DOMAINS = {
    'Cityscapes': {
        'condition': 'clear_day',
        'cities': ['frankfurt', 'lindau', 'munster'],
    },
    'ACDC': {
        'conditions': ['foggy', 'night', 'rainy', 'snowy'],
    }
}

# Models to test
MODELS_TO_TEST = ['pspnet_r50', 'segformer_mit-b5', 'deeplabv3plus_r50']

# All available strategies
ALL_STRATEGIES = [
    # Baseline
    'baseline',
    # Standard augmentation strategies
    'std_autoaugment',
    'std_cutmix',
    'std_mixup',
    'std_randaugment',
    'photometric_distort',
    # Generative strategies - GAN-based
    'gen_cycleGAN',
    'gen_CUT',
    'gen_LANIT',
    'gen_stargan_v2',
    'gen_TSIT',
    'gen_SUSTechGAN',
    # Generative strategies - Diffusion-based
    'gen_cyclediffusion',
    'gen_flux_kontext',
    'gen_Img2Img',
    'gen_IP2P',
    'gen_step1x_new',
    'gen_step1x_v1p2',
    # Generative strategies - Other
    'gen_Attribute_Hallucination',
    'gen_CNetSeg',
    'gen_Qwen_Image_Edit',
    'gen_UniControl',
    'gen_VisualCloze',
    'gen_Weather_Effect_Generator',
    # Generative strategies - Classical augmentation
    'gen_albumentations_weather',
    'gen_augmenters',
    'gen_automold',
]

# Cityscapes labelID to trainID mapping
CITYSCAPES_LABELID_TO_TRAINID = {
    7: 0,   # road
    8: 1,   # sidewalk
    11: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    17: 5,  # pole
    19: 6,  # traffic light
    20: 7,  # traffic sign
    21: 8,  # vegetation
    22: 9,  # terrain
    23: 10, # sky
    24: 11, # person
    25: 12, # rider
    26: 13, # car
    27: 14, # truck
    28: 15, # bus
    31: 16, # train
    32: 17, # motorcycle
    33: 18, # bicycle
}

# Build lookup table for fast conversion
CITYSCAPES_LABEL_LUT = np.full(256, 255, dtype=np.uint8)
for label_id, train_id in CITYSCAPES_LABELID_TO_TRAINID.items():
    CITYSCAPES_LABEL_LUT[label_id] = train_id

# MapillaryVistas (66 classes) to Cityscapes (19 classes) mapping for predictions
# This is needed when testing MV-trained models on Cityscapes/ACDC
MAPILLARY_TO_CITYSCAPES = {
    13: 0,  # road
    15: 1,  # sidewalk
    17: 2,  # building
    6: 3,   # wall
    3: 4,   # fence
    45: 5,  # pole
    47: 6,  # traffic light
    48: 7,  # traffic sign
    30: 8,  # vegetation
    29: 9,  # terrain
    27: 10, # sky
    19: 11, # person
    20: 12, # rider
    55: 13, # car
    61: 14, # truck
    54: 15, # bus
    58: 16, # train
    57: 17, # motorcycle
    52: 18, # bicycle
}

# Build MV prediction mapping LUT
MAPILLARY_PRED_LUT = np.full(256, 255, dtype=np.uint8)
for mv_id, cs_id in MAPILLARY_TO_CITYSCAPES.items():
    MAPILLARY_PRED_LUT[mv_id] = cs_id

# Cityscapes class names
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle'
]


# ============================================================================
# Helper Functions
# ============================================================================

def detect_model_num_classes(cfg) -> int:
    """Detect number of output classes from model config."""
    if hasattr(cfg, 'model') and 'decode_head' in cfg.model:
        return cfg.model.decode_head.get('num_classes', 19)
    return 19


def get_test_images_and_labels(dataset: str, domain: str) -> List[Tuple[Path, Path]]:
    """Get list of (image_path, label_path) tuples for a dataset/domain."""
    image_dir = DATA_ROOT / "test" / "images" / dataset / domain
    label_dir = DATA_ROOT / "test" / "labels" / dataset / domain
    
    if not image_dir.exists():
        raise ValueError(f"Image directory not found: {image_dir}")
    if not label_dir.exists():
        raise ValueError(f"Label directory not found: {label_dir}")
    
    pairs = []
    for img_path in sorted(image_dir.glob("*.png")):
        # Construct label filename based on dataset
        if dataset == "Cityscapes":
            # frankfurt_000000_000294_leftImg8bit.png -> frankfurt_000000_000294_gtFine_labelIds.png
            label_name = img_path.name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
        elif dataset == "ACDC":
            # GOPR0122_frame_000001_rgb_anon.png -> GOPR0122_frame_000001_gt_labelIds.png
            label_name = img_path.name.replace("_rgb_anon.png", "_gt_labelIds.png")
        else:
            label_name = img_path.name
        
        label_path = label_dir / label_name
        if label_path.exists():
            pairs.append((img_path, label_path))
    
    return pairs


def convert_cityscapes_label(label: np.ndarray) -> np.ndarray:
    """Convert Cityscapes labelID to trainID."""
    if label.ndim == 3:
        label = label[:, :, 0]
    return CITYSCAPES_LABEL_LUT[label]


def map_predictions_to_cityscapes(pred: np.ndarray, model_num_classes: int) -> np.ndarray:
    """Map model predictions to Cityscapes trainIDs if needed."""
    if model_num_classes == 66:
        # MapillaryVistas model
        return MAPILLARY_PRED_LUT[pred.astype(np.uint8)]
    # BDD10k, IDD-AW models already predict Cityscapes trainIDs
    return pred


def compute_iou_metrics(pred: np.ndarray, label: np.ndarray, 
                        num_classes: int = 19, ignore_index: int = 255) -> Dict:
    """Compute IoU metrics for a batch of predictions."""
    # Filter out ignore pixels
    mask = label != ignore_index
    pred = pred[mask].astype(np.int64)
    label = label[mask].astype(np.int64)
    
    # Clip to valid range
    pred = np.clip(pred, 0, num_classes - 1)
    label = np.clip(label, 0, num_classes - 1)
    
    # Compute areas using bincount
    area_pred = np.bincount(pred, minlength=num_classes).astype(np.float64)
    area_label = np.bincount(label, minlength=num_classes).astype(np.float64)
    
    # Intersection: count where pred == label
    match_mask = pred == label
    matched = pred[match_mask]
    area_intersect = np.bincount(matched, minlength=num_classes).astype(np.float64)
    
    # Union
    area_union = area_pred + area_label - area_intersect
    
    return {
        'intersect': area_intersect,
        'union': area_union,
        'pred': area_pred,
        'label': area_label
    }


def aggregate_metrics(all_areas: List[Dict], num_classes: int = 19) -> Dict:
    """Aggregate per-image metrics into summary statistics."""
    total_intersect = np.zeros(num_classes, dtype=np.float64)
    total_union = np.zeros(num_classes, dtype=np.float64)
    total_label = np.zeros(num_classes, dtype=np.float64)
    
    for areas in all_areas:
        total_intersect += areas['intersect']
        total_union += areas['union']
        total_label += areas['label']
    
    # Per-class IoU
    iou_per_class = total_intersect / np.maximum(total_union, 1)
    
    # Accuracy
    total_correct = total_intersect.sum()
    total_pixels = total_label.sum()
    aAcc = (total_correct / max(total_pixels, 1)) * 100
    
    # mIoU (only over present classes)
    valid_mask = total_label > 0
    mIoU = np.nanmean(iou_per_class[valid_mask]) * 100 if valid_mask.any() else 0.0
    
    return {
        'mIoU': float(mIoU),
        'aAcc': float(aAcc),
        'per_class_iou': {CITYSCAPES_CLASSES[i]: float(iou_per_class[i] * 100) 
                         for i in range(num_classes) if valid_mask[i]}
    }


def load_model(config_path: Path, checkpoint_path: Path, device: str = 'cuda:0'):
    """Load a segmentation model from config and checkpoint."""
    cfg = Config.fromfile(str(config_path))
    model_num_classes = detect_model_num_classes(cfg)
    
    # Build model
    model = MODELS.build(cfg.model)
    model = revert_sync_batchnorm(model)
    
    # Load checkpoint
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    return model, cfg, model_num_classes


def run_inference_batch(model, images: List[np.ndarray], device: str = 'cuda:0') -> List[np.ndarray]:
    """Run inference on a batch of images."""
    # Get normalization params from data preprocessor
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    
    # Preprocess
    batch_tensors = []
    original_sizes = []
    batch_img_metas = []
    
    for img in images:
        h, w = img.shape[:2]
        original_sizes.append((h, w))
        
        # Create metadata
        batch_img_metas.append({
            'ori_shape': (h, w),
            'img_shape': (h, w),
            'pad_shape': (h, w),
            'scale_factor': (1.0, 1.0),
        })
        
        img_norm = (img.astype(np.float32) - mean) / std
        img_norm = img_norm.transpose(2, 0, 1)  # HWC -> CHW
        batch_tensors.append(torch.from_numpy(img_norm).float())
    
    # Stack and move to device
    batch = torch.stack(batch_tensors).to(device)
    
    # Inference using model.inference() which handles metadata properly
    with torch.no_grad():
        results = model.inference(batch, batch_img_metas)
    
    # Post-process
    predictions = []
    for i, (result, orig_size) in enumerate(zip(results, original_sizes)):
        # Extract prediction
        if isinstance(result, torch.Tensor):
            if result.ndim == 3:
                pred = result.argmax(dim=0).cpu().numpy()
            else:
                pred = result.cpu().numpy()
        elif hasattr(result, 'pred_sem_seg'):
            result_tensor = result.pred_sem_seg.data.squeeze()
            if result_tensor.ndim == 3:
                pred = result_tensor.argmax(dim=0).cpu().numpy()
            else:
                pred = result_tensor.cpu().numpy()
        else:
            pred = np.array(result).squeeze()
        
        pred = pred.astype(np.uint8)
        
        # Resize to original size if needed
        if pred.shape != orig_size:
            pred = cv2.resize(pred, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_NEAREST)
        predictions.append(pred)
    
    return predictions


def evaluate_on_domain(model, model_num_classes: int, dataset: str, domain: str,
                       batch_size: int = 4, device: str = 'cuda:0',
                       max_images: int = 0) -> Dict:
    """Evaluate model on a specific domain.
    
    Args:
        max_images: Maximum images to evaluate (0=all, for quick testing)
    """
    pairs = get_test_images_and_labels(dataset, domain)
    if not pairs:
        return {'error': f'No images found for {dataset}/{domain}'}
    
    # Limit images for quick testing
    if max_images > 0:
        pairs = pairs[:max_images]
    
    all_areas = []
    
    # Process in batches
    for i in tqdm(range(0, len(pairs), batch_size), desc=f"{dataset}/{domain}"):
        batch_pairs = pairs[i:i+batch_size]
        
        # Load images and labels
        images = []
        labels = []
        for img_path, label_path in batch_pairs:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            
            label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
            label = convert_cityscapes_label(label)
            labels.append(label)
        
        # Run inference
        predictions = run_inference_batch(model, images, device)
        
        # Map predictions if needed (for MV models)
        predictions = [map_predictions_to_cityscapes(p, model_num_classes) for p in predictions]
        
        # Compute metrics
        for pred, label in zip(predictions, labels):
            areas = compute_iou_metrics(pred, label)
            all_areas.append(areas)
    
    # Aggregate
    metrics = aggregate_metrics(all_areas)
    metrics['num_images'] = len(pairs)
    
    return metrics


# ============================================================================
# Main Test Runner
# ============================================================================

def evaluate_with_model(
    model,
    model_num_classes: int,
    weights_dir: Path,
    source_dataset: str,
    model_name: str,
    strategy: str,
    batch_size: int = 4,
    device: str = 'cuda:0',
    max_images: int = 0
) -> Dict:
    """Run domain adaptation evaluation with an already-loaded model.
    
    Args:
        model: Loaded PyTorch model (already on device)
        model_num_classes: Number of output classes
        weights_dir: Path to save results
        source_dataset: Source dataset name
        model_name: Model architecture name
        strategy: Training strategy name
        batch_size: Batch size for inference
        device: Device for inference
        max_images: Maximum images per domain (0=all, for quick testing)
    
    Returns:
        Dict with evaluation results
    """
    # Results structure
    results = {
        'source_dataset': source_dataset,
        'model': model_name,
        'strategy': strategy,
        'model_num_classes': model_num_classes,
        'timestamp': datetime.now().isoformat(),
        'domains': {}
    }
    
    # Test on Cityscapes (all cities combined as clear_day)
    print("\n  Testing on Cityscapes (clear_day)...")
    cityscapes_areas = []
    for city in TARGET_DOMAINS['Cityscapes']['cities']:
        pairs = get_test_images_and_labels('Cityscapes', city)
        if max_images > 0:
            pairs = pairs[:max(1, max_images // 3)]  # Split across 3 cities
        
        for i in tqdm(range(0, len(pairs), batch_size), desc=f"Cityscapes/{city}", leave=False):
            batch_pairs = pairs[i:i+batch_size]
            images = []
            labels = []
            for img_path, label_path in batch_pairs:
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
                label = convert_cityscapes_label(label)
                labels.append(label)
            
            predictions = run_inference_batch(model, images, device)
            predictions = [map_predictions_to_cityscapes(p, model_num_classes) for p in predictions]
            
            for pred, label in zip(predictions, labels):
                areas = compute_iou_metrics(pred, label)
                cityscapes_areas.append(areas)
    
    if cityscapes_areas:
        results['domains']['clear_day'] = aggregate_metrics(cityscapes_areas)
        print(f"    clear_day mIoU: {results['domains']['clear_day']['mIoU']:.2f}%")
    
    # Test on ACDC domains
    print("\n  Testing on ACDC (adverse weather)...")
    for condition in TARGET_DOMAINS['ACDC']['conditions']:
        print(f"    Testing {condition}...")
        domain_results = evaluate_on_domain(model, model_num_classes, 'ACDC', condition,
                                           batch_size, device, max_images)
        if 'error' not in domain_results:
            results['domains'][condition] = domain_results
            print(f"    {condition} mIoU: {domain_results['mIoU']:.2f}%")
    
    # Compute summary
    domain_miou = [r['mIoU'] for r in results['domains'].values() if 'mIoU' in r]
    results['summary'] = {
        'avg_mIoU': float(np.mean(domain_miou)) if domain_miou else 0.0,
        'clear_day_mIoU': results['domains'].get('clear_day', {}).get('mIoU', 0.0),
        'adverse_avg_mIoU': float(np.mean([
            results['domains'].get(d, {}).get('mIoU', 0.0) 
            for d in ['foggy', 'night', 'rainy', 'snowy']
        ])) if 'foggy' in results['domains'] else 0.0
    }
    
    # Save results
    output_dir = weights_dir / "domain_adaptation" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_file}")
    
    # Generate report
    report_file = output_dir / "test_report.txt"
    with open(report_file, 'w') as f:
        f.write(f"Domain Adaptation Test Report\n")
        f.write(f"{'='*50}\n")
        f.write(f"Source Dataset: {source_dataset}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Strategy: {strategy}\n")
        f.write(f"Timestamp: {results['timestamp']}\n\n")
        f.write(f"Per-Domain Results:\n")
        f.write(f"{'-'*50}\n")
        for domain, metrics in results['domains'].items():
            f.write(f"  {domain}: mIoU={metrics['mIoU']:.2f}%\n")
        f.write(f"\nSummary:\n")
        f.write(f"  Average mIoU: {results['summary']['avg_mIoU']:.2f}%\n")
        f.write(f"  Clear Day mIoU: {results['summary']['clear_day_mIoU']:.2f}%\n")
        f.write(f"  Adverse Avg mIoU: {results['summary']['adverse_avg_mIoU']:.2f}%\n")
    
    print(f"  Report saved to: {report_file}")
    
    return results


def run_domain_adaptation_test(
    source_dataset: str,
    model_name: str,
    strategy: str = 'baseline',
    batch_size: int = 4,
    device: str = 'cuda:0',
    dry_run: bool = False,
    max_images: int = 0,
    regenerate: bool = False,
    preloaded_model: tuple = None
) -> Optional[Dict]:
    """Run domain adaptation test for a single source model.
    
    Args:
        max_images: Maximum images per domain (0=all, for quick testing)
        regenerate: If True, re-run even if results exist
        preloaded_model: Optional tuple of (model, cfg, model_num_classes) to reuse
    """
    
    # Construct paths
    weights_dir = WEIGHTS_ROOT / strategy / source_dataset / model_name
    if not weights_dir.exists():
        weights_dir = WEIGHTS_ROOT / strategy / source_dataset.lower() / (model_name + '_ratio0p50')
        if not weights_dir.exists():
            print(f"  [SKIP] Weights directory not found: {weights_dir}")
            return None
    config_path = weights_dir / "training_config.py"
    checkpoint_path = weights_dir / "iter_80000.pth"
    
    print(f"\n{'='*60}")
    print(f"Source: {source_dataset} | Model: {model_name} | Strategy: {strategy}")
    print(f"{'='*60}")
    
    # Check if checkpoint exists
    if not checkpoint_path.exists():
        print(f"  [SKIP] Checkpoint not found: {checkpoint_path}")
        return None
    
    if not config_path.exists():
        print(f"  [SKIP] Config not found: {config_path}")
        return None
    
    # Check if results already exist
    domain_adapt_dir = weights_dir / "domain_adaptation"
    if domain_adapt_dir.exists() and not regenerate:
        # Find the most recent results
        result_dirs = sorted(domain_adapt_dir.glob("*/results.json"))
        if result_dirs:
            latest_result = result_dirs[-1]
            print(f"  [SKIP] Results already exist: {latest_result}")
            print(f"         Use --regenerate to re-run")
            # Load and return existing results
            with open(latest_result) as f:
                return json.load(f)
    
    print(f"  Config: {config_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    
    if dry_run:
        print("  [DRY RUN] Would test on Cityscapes + ACDC")
        return {'dry_run': True, 'strategy': strategy, 'source_dataset': source_dataset, 'model': model_name}
    
    # Load model (or use preloaded)
    if preloaded_model is not None:
        model, cfg, model_num_classes = preloaded_model
        print(f"  Using preloaded model (num_classes={model_num_classes})")
    else:
        print("  Loading model...")
        model, cfg, model_num_classes = load_model(config_path, checkpoint_path, device)
        print(f"  Model loaded (num_classes={model_num_classes})")
    
    # Run evaluation
    return evaluate_with_model(
        model=model,
        model_num_classes=model_num_classes,
        weights_dir=weights_dir,
        source_dataset=source_dataset,
        model_name=model_name,
        strategy=strategy,
        batch_size=batch_size,
        device=device,
        max_images=max_images
    )


def main():
    parser = argparse.ArgumentParser(description='Domain Adaptation Ablation Study')
    parser.add_argument('--source-dataset', choices=['bdd10k', 'idd-aw', 'mapillaryvistas'],
                       help='Source dataset to test')
    parser.add_argument('--model', choices=['pspnet_r50', 'segformer_mit-b5', 'deeplabv3plus_r50'],
                       help='Model architecture')
    parser.add_argument('--strategy', choices=ALL_STRATEGIES, default='baseline',
                       help='Training strategy (default: baseline). Available: ' + ', '.join(ALL_STRATEGIES[:5]) + '...')
    parser.add_argument('--all', action='store_true',
                       help='Run all source datasets and models for the specified strategy')
    parser.add_argument('--all-strategies', action='store_true',
                       help='Run all strategies (use with --all for full matrix)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for inference')
    parser.add_argument('--device', default='cuda:0',
                       help='Device for inference')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be tested without running')
    parser.add_argument('--quick-test', type=int, default=0,
                       help='Only test this many images per domain (0=all, for debugging)')
    parser.add_argument('--list-strategies', action='store_true',
                       help='List all available strategies and exit')
    parser.add_argument('--regenerate', action='store_true',
                       help='Re-run tests even if results already exist (default: skip existing)')
    
    args = parser.parse_args()
    
    # List strategies and exit
    if args.list_strategies:
        print("Available strategies:")
        for i, strategy in enumerate(ALL_STRATEGIES, 1):
            print(f"  {i:2d}. {strategy}")
        return
    
    # Determine which strategies to run
    strategies_to_run = ALL_STRATEGIES if args.all_strategies else [args.strategy]
    
    if args.all:
        # Run all combinations
        all_results = []
        for strategy in strategies_to_run:
            for source_ds in SOURCE_DATASETS:
                for model_name in MODELS_TO_TEST:
                    result = run_domain_adaptation_test(
                        source_dataset=source_ds,
                        model_name=model_name,
                        strategy=strategy,
                        batch_size=args.batch_size,
                        device=args.device,
                        dry_run=args.dry_run,
                        max_images=args.quick_test,
                        regenerate=args.regenerate
                    )
                    if result:
                        all_results.append(result)
        
        # Print summary
        if not args.dry_run and all_results:
            print("\n" + "="*90)
            print("SUMMARY: Domain Adaptation Results")
            print("="*90)
            print(f"{'Strategy':<30} {'Source':<15} {'Model':<20} {'Clear':<8} {'Foggy':<8} {'Night':<8} {'Rainy':<8} {'Snowy':<8} {'Avg':<8}")
            print("-"*90)
            for r in all_results:
                if 'domains' in r:
                    print(f"{r['strategy']:<30} {r['source_dataset']:<15} {r['model']:<20} "
                          f"{r['domains'].get('clear_day', {}).get('mIoU', 0):<8.1f} "
                          f"{r['domains'].get('foggy', {}).get('mIoU', 0):<8.1f} "
                          f"{r['domains'].get('night', {}).get('mIoU', 0):<8.1f} "
                          f"{r['domains'].get('rainy', {}).get('mIoU', 0):<8.1f} "
                          f"{r['domains'].get('snowy', {}).get('mIoU', 0):<8.1f} "
                          f"{r['summary']['avg_mIoU']:<8.1f}")
    else:
        if not args.source_dataset or not args.model:
            parser.error("--source-dataset and --model required unless --all is specified")
        
        run_domain_adaptation_test(
            source_dataset=args.source_dataset,
            model_name=args.model,
            strategy=args.strategy,
            batch_size=args.batch_size,
            device=args.device,
            dry_run=args.dry_run,
            max_images=args.quick_test,
            regenerate=args.regenerate
        )


if __name__ == '__main__':
    main()