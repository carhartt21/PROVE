#!/usr/bin/env python3
"""
Fine-grained testing for PROVE segmentation models.

This script provides detailed test results including:
- Per-domain (weather condition) metrics
- Per-class IoU breakdown
- Timestamped output folders
- Comprehensive JSON output with all metrics

Output Structure:
    {timestamp}/
    ├── results.json          # Complete metrics (overall, per-domain, per-class)
    └── test_report.txt       # Human-readable summary

Usage:
    python fine_grained_test.py --config /path/to/config.py --checkpoint /path/to/checkpoint.pth \
        --output-dir /path/to/output --dataset ACDC
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch
import cv2
from tqdm import tqdm

# Add project root to path
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

# Import custom transforms before MMSeg
import custom_transforms

from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.registry import DATASETS, MODELS
from mmseg.utils import register_all_modules
import mmseg.models  # Register all mmseg models including SegDataPreProcessor

# Register all modules
register_all_modules(init_default_scope=True)

# Suppress MMSegmentation deprecation warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='mmseg')


# Dataset domain configurations
# Updated to match actual folder structure in FINAL_SPLITS/test/images/
DATASET_DOMAINS = {
    'ACDC': ['foggy', 'night', 'rainy', 'snowy'],  # ACDC test split only has 4 domains
    'BDD10k': ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy'],
    'BDD100k': ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy'],
    'IDD-AW': ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy'],
    'MapillaryVistas': ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy'],
    'OUTSIDE15k': ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy'],
}

# Map lowercase dataset names to folder names (for case-insensitive lookup)
DATASET_FOLDER_MAP = {
    'acdc': 'ACDC',
    'bdd10k': 'BDD10k',
    'bdd100k': 'BDD100k',
    'cityscapes': 'Cityscapes',
    'idd-aw': 'IDD-AW',
    'mapillaryvistas': 'MapillaryVistas',
    'outside15k': 'OUTSIDE15k',
    # Also add proper case versions
    'ACDC': 'ACDC',
    'BDD10k': 'BDD10k',
    'BDD100k': 'BDD100k',
    'Cityscapes': 'Cityscapes',
    'IDD-AW': 'IDD-AW',
    'MapillaryVistas': 'MapillaryVistas',
    'OUTSIDE15k': 'OUTSIDE15k',
}

# Dataset label type configuration for proper label processing during testing
# Maps dataset folder names to their label type and class configuration
DATASET_LABEL_CONFIG = {
    'ACDC': {
        'label_type': 'cityscapes_labelid',  # Cityscapes labelIDs (0-33) -> trainIds
        'num_classes': 19,
        'classes': None,  # Use CITYSCAPES_CLASSES
    },
    'BDD10k': {
        'label_type': 'cityscapes_trainid',  # Already Cityscapes trainIDs (0-18, 255)
        'num_classes': 19,
        'classes': None,
    },
    'BDD100k': {
        'label_type': 'cityscapes_trainid',
        'num_classes': 19,
        'classes': None,
    },
    'Cityscapes': {
        'label_type': 'cityscapes_trainid',
        'num_classes': 19,
        'classes': None,
    },
    'IDD-AW': {
        'label_type': 'cityscapes_trainid',  # Already Cityscapes trainIDs (0-18, 255) - NOT labelIDs!
        'num_classes': 19,
        'classes': None,
    },
    'MapillaryVistas': {
        'label_type': 'mapillary_rgb',  # RGB color-encoded -> native Mapillary IDs (0-65)
        'num_classes': 66,
        'classes': [  # Native Mapillary class names
            'Bird', 'Ground Animal', 'Curb', 'Fence', 'Guard Rail', 'Barrier',
            'Wall', 'Bike Lane', 'Crosswalk - Plain', 'Curb Cut', 'Parking',
            'Pedestrian Area', 'Rail Track', 'Road', 'Service Lane', 'Sidewalk',
            'Bridge', 'Building', 'Tunnel', 'Person', 'Bicyclist', 'Motorcyclist',
            'Other Rider', 'Lane Marking - Crosswalk', 'Lane Marking - General',
            'Mountain', 'Sand', 'Sky', 'Snow', 'Terrain', 'Vegetation', 'Water',
            'Banner', 'Bench', 'Bike Rack', 'Billboard', 'Catch Basin', 'CCTV Camera',
            'Fire Hydrant', 'Junction Box', 'Mailbox', 'Manhole', 'Phone Booth',
            'Pothole', 'Street Light', 'Pole', 'Traffic Sign Frame', 'Utility Pole',
            'Traffic Light', 'Traffic Sign (Back)', 'Traffic Sign (Front)', 'Trash Can',
            'Bicycle', 'Boat', 'Bus', 'Car', 'Caravan', 'Motorcycle', 'On Rails',
            'Other Vehicle', 'Trailer', 'Truck', 'Wheeled Slow', 'Car Mount',
            'Ego Vehicle', 'Unlabeled',
        ],
    },
    'OUTSIDE15k': {
        'label_type': 'outside15k_native',  # Native OUTSIDE15k labels (0-23)
        'num_classes': 24,
        'classes': [  # Native OUTSIDE15k class names
            'unlabeled', 'animal', 'barrier', 'bicycle', 'boat', 'bridge',
            'building', 'grass', 'ground', 'mountain', 'object', 'person',
            'pole', 'road', 'sand', 'sidewalk', 'sign', 'sky', 'street light',
            'traffic light', 'tunnel', 'vegetation', 'vehicle', 'water',
        ],
    },
}

# Cityscapes class names (19 classes)
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]


def to_python_type(val):
    """Convert tensor values to Python types for JSON serialization."""
    if isinstance(val, torch.Tensor):
        return val.item() if val.numel() == 1 else val.tolist()
    elif isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, (np.floating, np.integer)):
        return val.item()
    return val


def get_dataset_config(dataset_name: str):
    """Get label configuration for a dataset.
    
    Returns:
        Tuple of (num_classes, class_names)
    """
    config = DATASET_LABEL_CONFIG.get(dataset_name, DATASET_LABEL_CONFIG['Cityscapes'])
    num_classes = config['num_classes']
    classes = config['classes'] if config['classes'] else CITYSCAPES_CLASSES
    return num_classes, classes


def process_label_for_dataset(gt_seg_map: np.ndarray, dataset_name: str) -> np.ndarray:
    """Process ground truth labels according to dataset type.
    
    Different datasets have different label formats that need conversion:
    - ACDC, IDD-AW: Cityscapes labelIDs (0-33) -> trainIds (0-18)
    - BDD10k, BDD100k: Already Cityscapes trainIds, no conversion needed
    - MapillaryVistas: RGB color-encoded -> native class IDs (0-65) - KEEPS NATIVE CLASSES
    - OUTSIDE15k: Native labels (0-23) - KEEPS NATIVE CLASSES
    
    Args:
        gt_seg_map: Raw label image (may be RGB or grayscale)
        dataset_name: Dataset folder name (e.g., 'MapillaryVistas', 'IDD-AW')
        
    Returns:
        Processed label image with appropriate class IDs
    """
    config = DATASET_LABEL_CONFIG.get(dataset_name, DATASET_LABEL_CONFIG['Cityscapes'])
    label_type = config['label_type']
    
    if label_type == 'mapillary_rgb':
        # MapillaryVistas: RGB color-encoded -> native Mapillary class IDs (0-65)
        # Keeps native classes for single-dataset evaluation
        if gt_seg_map.ndim == 3 and gt_seg_map.shape[-1] == 3:
            # Decode RGB to Mapillary native class IDs
            h, w = gt_seg_map.shape[:2]
            output = np.full((h, w), 255, dtype=np.uint8)
            
            # Pack RGB values for fast lookup
            r = gt_seg_map[:, :, 0].astype(np.int32)
            g = gt_seg_map[:, :, 1].astype(np.int32)
            b = gt_seg_map[:, :, 2].astype(np.int32)
            packed = r * 65536 + g * 256 + b
            
            # Lookup RGB -> class ID
            rgb_lookup = {}
            for rgb, class_id in custom_transforms.MAPILLARY_RGB_TO_ID.items():
                packed_rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
                rgb_lookup[packed_rgb] = class_id
            
            for packed_rgb, class_id in rgb_lookup.items():
                mask = packed == packed_rgb
                output[mask] = class_id
            
            gt_seg_map = output
        elif gt_seg_map.ndim == 3:
            # If 3-channel but not proper RGB, take first channel
            gt_seg_map = gt_seg_map[:, :, 0]
        # Note: Do NOT convert to Cityscapes trainIds - keep native Mapillary classes
        
    elif label_type == 'cityscapes_labelid':
        # ACDC, IDD-AW: Cityscapes labelIDs (0-33) -> trainIds (0-18)
        if gt_seg_map.ndim == 3:
            gt_seg_map = gt_seg_map[:, :, 0]
        
        lut = np.full(256, 255, dtype=np.uint8)
        for label_id, train_id in custom_transforms.CITYSCAPES_ID_TO_TRAINID.items():
            if 0 <= label_id < 256:
                lut[label_id] = train_id
        gt_seg_map = lut[gt_seg_map]
        
    elif label_type == 'outside15k_native':
        # OUTSIDE15k: native labels (0-23) - keep native classes
        if gt_seg_map.ndim == 3:
            gt_seg_map = gt_seg_map[:, :, 0]
        # Note: Do NOT convert to Cityscapes trainIds - keep native OUTSIDE15k classes
        
    else:  # cityscapes_trainid
        # BDD10k, BDD100k, Cityscapes: Already Cityscapes trainIds
        if gt_seg_map.ndim == 3:
            gt_seg_map = gt_seg_map[:, :, 0]
    
    return gt_seg_map


def compute_iou_metrics(
    pred: np.ndarray, 
    label: np.ndarray, 
    num_classes: int,
    ignore_index: int = 255
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute IoU metrics for a single prediction.
    
    Returns:
        area_intersect, area_union, area_pred, area_label
    """
    mask = label != ignore_index
    pred = pred[mask]
    label = label[mask]
    
    area_intersect = np.zeros(num_classes, dtype=np.float64)
    area_union = np.zeros(num_classes, dtype=np.float64)
    area_pred = np.zeros(num_classes, dtype=np.float64)
    area_label = np.zeros(num_classes, dtype=np.float64)
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        label_cls = (label == cls)
        
        area_intersect[cls] = np.sum(pred_cls & label_cls)
        area_union[cls] = np.sum(pred_cls | label_cls)
        area_pred[cls] = np.sum(pred_cls)
        area_label[cls] = np.sum(label_cls)
    
    return area_intersect, area_union, area_pred, area_label


def compute_metrics_from_areas(
    area_intersect: np.ndarray,
    area_union: np.ndarray,
    area_pred: np.ndarray,
    area_label: np.ndarray
) -> Dict[str, float]:
    """Compute summary metrics from aggregated areas."""
    # Per-class IoU
    iou = area_intersect / np.maximum(area_union, 1)
    
    # Per-class accuracy (recall)
    acc = area_intersect / np.maximum(area_label, 1)
    
    # Overall accuracy
    total_correct = area_intersect.sum()
    total_pixels = area_label.sum()
    aAcc = total_correct / max(total_pixels, 1) * 100
    
    # Mean IoU
    valid_mask = area_label > 0
    mIoU = np.nanmean(iou[valid_mask]) * 100 if valid_mask.any() else 0.0
    
    # Mean Accuracy
    mAcc = np.nanmean(acc[valid_mask]) * 100 if valid_mask.any() else 0.0
    
    # Frequency-weighted IoU
    if total_pixels > 0:
        freq = area_label / total_pixels
        fwIoU = (freq * iou).sum() * 100
    else:
        fwIoU = 0.0
    
    return {
        'aAcc': float(aAcc),
        'mIoU': float(mIoU),
        'mAcc': float(mAcc),
        'fwIoU': float(fwIoU)
    }


def get_per_class_metrics(
    area_intersect: np.ndarray,
    area_union: np.ndarray,
    area_pred: np.ndarray,
    area_label: np.ndarray,
    class_names: list = None
) -> Dict[str, Dict[str, float]]:
    """Get per-class breakdown of metrics."""
    if class_names is None:
        class_names = CITYSCAPES_CLASSES
    
    iou = area_intersect / np.maximum(area_union, 1)
    acc = area_intersect / np.maximum(area_label, 1)
    
    per_class = {}
    for i in range(len(area_intersect)):
        class_name = class_names[i] if i < len(class_names) else f'class_{i}'
        per_class[class_name] = {
            'IoU': float(iou[i] * 100),
            'Acc': float(acc[i] * 100),
            'area_intersect': float(area_intersect[i]),
            'area_union': float(area_union[i]),
            'area_pred': float(area_pred[i]),
            'area_label': float(area_label[i])
        }
    return per_class


def run_fine_grained_test(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    dataset_name: str,
    data_root: str,
    test_split: str = 'test'
) -> Dict[str, Any]:
    """Run fine-grained testing with per-domain and per-class results.
    
    Uses a single model instance and manually iterates over domains to avoid
    creating multiple Runners.
    """
    
    # Normalize dataset name to match folder structure in FINAL_SPLITS
    folder_name = DATASET_FOLDER_MAP.get(dataset_name, dataset_name)
    print(f"Dataset mapping: {dataset_name} -> {folder_name}")
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path(output_dir) / timestamp
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Fine-grained test output directory: {output_path}")
    
    # Load config
    cfg = Config.fromfile(config_path)
    
    # Build model
    print("Building model...")
    model = MODELS.build(cfg.model)
    model = revert_sync_batchnorm(model)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    
    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    
    # Get domains for this dataset
    domains = ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy']
    
    # Get dataset-specific configuration (num_classes, class_names)
    num_classes, class_names = get_dataset_config(folder_name)
    print(f"Using {num_classes} classes for {folder_name}")
    
    # Results storage
    all_results = {
        'config': {
            'config_path': config_path,
            'checkpoint_path': checkpoint_path,
            'dataset': dataset_name,
            'test_split': test_split,
            'timestamp': timestamp,
            'classes': class_names,
            'num_classes': num_classes,
        },
        'overall': {},
        'per_domain': {},
        'per_class': {}
    }
    
    # Overall aggregated metrics
    total_area_intersect = np.zeros(num_classes, dtype=np.float64)
    total_area_union = np.zeros(num_classes, dtype=np.float64)
    total_area_pred = np.zeros(num_classes, dtype=np.float64)
    total_area_label = np.zeros(num_classes, dtype=np.float64)
    
    # Test pipeline from config
    test_pipeline = cfg.get('test_pipeline', cfg.test_dataloader.dataset.pipeline)
    pipeline = Compose(test_pipeline)
    
    # Process each domain
    test_domains = domains if domains else ['all']
    
    for domain in test_domains:
        print(f"\n{'='*60}")
        print(f"Testing domain: {domain if domain != 'all' else 'all data'}")
        print('='*60)
        
        # Get image and label paths for this domain (use folder_name for correct case)
        if domain == 'all':
            img_dir = Path(data_root) / test_split / 'images' / folder_name
            label_dir = Path(data_root) / test_split / 'labels' / folder_name
        else:
            img_dir = Path(data_root) / test_split / 'images' / folder_name / domain
            label_dir = Path(data_root) / test_split / 'labels' / folder_name / domain
        
        if not img_dir.exists():
            print(f"  Skipping: folder not found at {img_dir}")
            continue
        
        # Get all images
        img_files = sorted(list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg')))
        if not img_files:
            print(f"  Skipping: no images found")
            continue
        
        print(f"  Found {len(img_files)} images")
        
        # Domain aggregated metrics
        domain_area_intersect = np.zeros(num_classes, dtype=np.float64)
        domain_area_union = np.zeros(num_classes, dtype=np.float64)
        domain_area_pred = np.zeros(num_classes, dtype=np.float64)
        domain_area_label = np.zeros(num_classes, dtype=np.float64)
        
        # Process images
        for img_path in tqdm(img_files, desc=f"  Processing {domain}"):
            # Find corresponding label
            label_path = label_dir / img_path.name
            if not label_path.exists():
                # Try different extension
                for ext in ['.png', '.jpg']:
                    label_path = label_dir / (img_path.stem + ext)
                    if label_path.exists():
                        break
            
            if not label_path.exists():
                continue
            
            # Prepare data dict for pipeline
            data_dict = {
                'img_path': str(img_path),
                'seg_map_path': str(label_path),
                'reduce_zero_label': False,
                'img_info': {'filename': img_path.name},
                'seg_fields': [],  # Required by LoadAnnotations
            }
            
            # Run through pipeline
            try:
                data = pipeline(data_dict)
            except Exception as e:
                # Fall back to direct loading
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                gt_seg_map = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
                
                # Process labels according to dataset type
                gt_seg_map = process_label_for_dataset(gt_seg_map, folder_name)
                
                # Simple preprocessing
                h, w = img.shape[:2]
                
                # Normalize
                mean = np.array([123.675, 116.28, 103.53])
                std = np.array([58.395, 57.12, 57.375])
                img = (img - mean) / std
                img = img.transpose(2, 0, 1)  # HWC to CHW
                inputs = torch.from_numpy(img).float().unsqueeze(0).to(device)
                
                # Run inference
                with torch.no_grad():
                    result = model.inference(inputs, None)
                    pred_seg_map = result[0].pred_sem_seg.data.cpu().numpy().squeeze()
                
                # Resize prediction back to original size if needed
                if pred_seg_map.shape != gt_seg_map.shape:
                    pred_seg_map = cv2.resize(pred_seg_map.astype(np.uint8), 
                                              (gt_seg_map.shape[1], gt_seg_map.shape[0]),
                                              interpolation=cv2.INTER_NEAREST)
                
                # Compute metrics
                area_intersect, area_union, area_pred, area_label = compute_iou_metrics(
                    pred_seg_map, gt_seg_map, num_classes
                )
                
                # Aggregate
                domain_area_intersect += area_intersect
                domain_area_union += area_union
                domain_area_pred += area_pred
                domain_area_label += area_label
                continue
            
            # Get input - need to handle as raw image data for preprocessing
            # Load image and preprocess manually
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gt_seg_map = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
            
            # Process labels according to dataset type (e.g., MapillaryVistas RGB -> trainIds)
            gt_seg_map = process_label_for_dataset(gt_seg_map, folder_name)
            
            # Preprocess image
            h, w = img.shape[:2]
            mean = np.array([123.675, 116.28, 103.53])
            std = np.array([58.395, 57.12, 57.375])
            img = (img.astype(np.float32) - mean) / std
            img = img.transpose(2, 0, 1)  # HWC to CHW
            inputs = torch.from_numpy(img).float().unsqueeze(0).to(device)
            
            # Get image shape for metadata
            img_shape = (h, w)
            batch_img_metas = [{
                'ori_shape': img_shape,
                'img_shape': img_shape,
                'pad_shape': img_shape,
                'scale_factor': (1.0, 1.0),
            }]
            
            # Run inference
            with torch.no_grad():
                result = model.inference(inputs, batch_img_metas)
                # Result may be tensor or list of SegDataSample
                if isinstance(result, torch.Tensor):
                    pred_seg_map = result.cpu().numpy().squeeze()
                elif isinstance(result, list) and hasattr(result[0], 'pred_sem_seg'):
                    pred_seg_map = result[0].pred_sem_seg.data.cpu().numpy().squeeze()
                else:
                    pred_seg_map = result[0].cpu().numpy().squeeze() if isinstance(result, list) else result.cpu().numpy().squeeze()
                
                # Take argmax if needed (logits case)
                if pred_seg_map.ndim == 3:
                    pred_seg_map = pred_seg_map.argmax(axis=0)
            
            # Compute metrics
            area_intersect, area_union, area_pred, area_label = compute_iou_metrics(
                pred_seg_map, gt_seg_map, num_classes
            )
            
            # Aggregate
            domain_area_intersect += area_intersect
            domain_area_union += area_union
            domain_area_pred += area_pred
            domain_area_label += area_label
        
        # Compute domain metrics
        domain_metrics = compute_metrics_from_areas(
            domain_area_intersect, domain_area_union,
            domain_area_pred, domain_area_label
        )
        domain_metrics['num_images'] = len(img_files)
        
        # Per-class for this domain
        domain_per_class = get_per_class_metrics(
            domain_area_intersect, domain_area_union,
            domain_area_pred, domain_area_label,
            class_names=class_names
        )
        
        # Store results
        if domain == 'all':
            all_results['overall'] = domain_metrics
            all_results['per_class'] = domain_per_class
        else:
            all_results['per_domain'][domain] = {
                'summary': domain_metrics,
                'per_class': domain_per_class
            }
        
        # Aggregate to overall
        total_area_intersect += domain_area_intersect
        total_area_union += domain_area_union
        total_area_pred += domain_area_pred
        total_area_label += domain_area_label
        
        # Print domain summary
        print(f"\n  Domain {domain} results:")
        for k, v in domain_metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.2f}")
            else:
                print(f"    {k}: {v}")
    
    # Compute overall metrics if we tested multiple domains
    if domains and len(all_results['per_domain']) > 0:
        overall_metrics = compute_metrics_from_areas(
            total_area_intersect, total_area_union,
            total_area_pred, total_area_label
        )
        overall_per_class = get_per_class_metrics(
            total_area_intersect, total_area_union,
            total_area_pred, total_area_label,
            class_names=class_names
        )
        
        # Count total images
        total_images = sum(
            d['summary']['num_images'] 
            for d in all_results['per_domain'].values() 
            if isinstance(d, dict) and 'summary' in d
        )
        overall_metrics['num_images'] = total_images
        
        all_results['overall'] = overall_metrics
        all_results['per_class'] = overall_per_class
    
    # Save results - Single unified JSON output
    # Ensure output directory exists (NFS filesystem can have race conditions)
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Create readable text report
    report_file = output_path / 'test_report.txt'
    with open(report_file, 'w') as f:
        f.write(f"PROVE Fine-Grained Test Report\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        
        f.write(f"\n{'='*60}\n")
        f.write(f"OVERALL METRICS\n")
        f.write(f"{'='*60}\n\n")
        
        for key, value in all_results.get('overall', {}).items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        
        if all_results['per_domain']:
            f.write(f"\n{'='*60}\n")
            f.write(f"PER-DOMAIN METRICS\n")
            f.write(f"{'='*60}\n")
            
            for domain, data in all_results['per_domain'].items():
                f.write(f"\n--- {domain} ---\n")
                metrics = data.get('summary', data) if isinstance(data, dict) else {}
                for key, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
        
        if all_results['per_class']:
            f.write(f"\n{'='*60}\n")
            f.write(f"PER-CLASS METRICS (Overall)\n")
            f.write(f"{'='*60}\n\n")
            
            f.write(f"{'Class':<20} {'IoU':>10} {'Acc':>10}\n")
            f.write(f"{'-'*40}\n")
            for class_name, metrics in all_results['per_class'].items():
                f.write(f"{class_name:<20} {metrics['IoU']:>10.2f} {metrics['Acc']:>10.2f}\n")
    
    print(f"Report saved to: {report_file}")
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS SUMMARY")
    print('='*60)
    for key, value in all_results.get('overall', {}).items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Fine-grained testing for PROVE models')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--dataset', required=True, help='Dataset name (ACDC, BDD10k, etc.)')
    parser.add_argument('--data-root', default='/scratch/aaa_exchange/AWARE/FINAL_SPLITS',
                       help='Data root directory')
    parser.add_argument('--test-split', default='test', choices=['val', 'test'],
                       help='Test split to use')
    
    args = parser.parse_args()
    
    run_fine_grained_test(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        data_root=args.data_root,
        test_split=args.test_split
    )


if __name__ == '__main__':
    main()
