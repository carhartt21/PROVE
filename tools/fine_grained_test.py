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
from concurrent.futures import ThreadPoolExecutor, Future
import queue
import threading

import numpy as np
import torch
import cv2
from tqdm import tqdm

# Add project root to path
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

# Import custom transforms and losses before MMSeg
from utils import custom_transforms
from utils import custom_losses  # Registers SegBoundaryLoss, BoundaryLossIgnoreWeight etc.

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
    'Cityscapes': ['frankfurt', 'lindau', 'munster'],  # City-based splits, not weather domains
    'IDD-AW': ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy'],
    'MapillaryVistas': ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy'],
    'OUTSIDE15k': ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy'],
}

# Map dataset names to folder names (for case-insensitive lookup)
# Includes multiple case variants to handle different input formats
DATASET_FOLDER_MAP = {
    # Lowercase versions
    'acdc': 'ACDC',
    'bdd10k': 'BDD10k',
    'bdd100k': 'BDD100k',
    'cityscapes': 'Cityscapes',
    'idd-aw': 'IDD-AW',
    'iddaw': 'IDD-AW',
    'mapillaryvistas': 'MapillaryVistas',
    'outside15k': 'OUTSIDE15k',
    # Proper case versions
    'ACDC': 'ACDC',
    'BDD10k': 'BDD10k',
    'BDD100k': 'BDD100k',
    'Cityscapes': 'Cityscapes',
    'IDD-AW': 'IDD-AW',
    'MapillaryVistas': 'MapillaryVistas',
    'OUTSIDE15k': 'OUTSIDE15k',
    # All-caps versions (commonly used in scripts)
    'BDD10K': 'BDD10k',
    'BDD100K': 'BDD100k',
    'CITYSCAPES': 'Cityscapes',
    'IDDAW': 'IDD-AW',
    'IDD_AW': 'IDD-AW',
    'MAPILLARYVISTAS': 'MapillaryVistas',
    'OUTSIDE15K': 'OUTSIDE15k',  # This was missing and caused test failures
}

# Cityscapes class names (19 classes)
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

# Mapillary Vistas class names (66 classes) - ordered by class ID 0-65
# Matches the class definitions in label_unification.py
MAPILLARY_CLASSES = [
    'Bird', 'Ground Animal', 'Curb', 'Fence', 'Guard Rail', 'Barrier', 'Wall',
    'Bike Lane', 'Crosswalk - Plain', 'Curb Cut', 'Parking', 'Pedestrian Area',
    'Rail Track', 'Road', 'Service Lane', 'Sidewalk', 'Bridge', 'Building', 'Tunnel',
    'Person', 'Bicyclist', 'Motorcyclist', 'Other Rider', 'Lane Marking - Crosswalk',
    'Lane Marking - General', 'Mountain', 'Sand', 'Sky', 'Snow', 'Terrain', 'Vegetation',
    'Water', 'Banner', 'Bench', 'Bike Rack', 'Billboard', 'Catch Basin', 'CCTV Camera',
    'Fire Hydrant', 'Junction Box', 'Mailbox', 'Manhole', 'Phone Booth', 'Pothole',
    'Street Light', 'Pole', 'Traffic Sign Frame', 'Utility Pole', 'Traffic Light',
    'Traffic Sign (Back)', 'Traffic Sign (Front)', 'Trash Can', 'Bicycle', 'Boat',
    'Bus', 'Car', 'Caravan', 'Motorcycle', 'On Rails', 'Other Vehicle', 'Trailer',
    'Truck', 'Wheeled Slow', 'Car Mount', 'Ego Vehicle', 'Unlabeled'
]

# OUTSIDE15k class names (24 classes) - ordered by class ID 0-23
OUTSIDE15K_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
    'construction', 'animal', 'water', 'other', 'ignore'
]

# Dataset label type configuration for proper label processing during testing
# Maps dataset folder names to their label type and class configuration
#
# CROSS-DOMAIN TESTING STRATEGY:
# - Models may be trained with native classes (66 for Mapillary, 24 for OUTSIDE15k)
# - Test datasets may use Cityscapes trainIDs (0-18)
# - For cross-domain: map MODEL PREDICTIONS to Cityscapes, keep GT as-is (if already Cityscapes)
# - For same-dataset: use native classes for both predictions and GT
#
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
        'label_type': 'cityscapes_labelid',  # FINAL_SPLITS has labelIDs (0-33), needs conversion to trainIds
        'num_classes': 19,
        'classes': None,
    },
    'IDD-AW': {
        'label_type': 'cityscapes_trainid',  # Already Cityscapes trainIDs (0-18, 255) - NOT labelIDs!
        'num_classes': 19,
        'classes': None,
    },
    'MapillaryVistas': {
        # For native MapillaryVistas models (66 classes), keep in native format
        # For cross-domain evaluation on Cityscapes test data, convert GT labels
        'label_type': 'mapillary_rgb_to_native',  # RGB color-encoded -> native IDs (0-65)
        'num_classes': 66,  # Native MapillaryVistas classes
        'classes': MAPILLARY_CLASSES,  # Use proper Mapillary class names for native evaluation
    },
    'OUTSIDE15k': {
        # For native OUTSIDE15k models (24 classes), keep in native format
        # For cross-domain evaluation on Cityscapes test data, convert predictions
        'label_type': 'native',  # Already native class IDs (0-23)
        'num_classes': 24,  # Native OUTSIDE15k classes
        'classes': OUTSIDE15K_CLASSES,  # Use proper OUTSIDE15k class names
    },
}

# Mapping from native predictions to Cityscapes trainIDs for cross-domain evaluation
# Used when model predicts native classes but test data uses Cityscapes labels
MAPILLARY_PRED_TO_CITYSCAPES = custom_transforms.MAPILLARY_TO_TRAINID
OUTSIDE15K_PRED_TO_CITYSCAPES = custom_transforms.OUTSIDE15K_TO_TRAINID

# Pre-built 24-bit lookup table for MapillaryVistas RGB decoding
# This provides O(1) per-pixel decoding instead of O(66) iteration
# Memory: 16MB for the LUT, but ~66x faster decoding
_MAPILLARY_RGB_LUT_24BIT = None

def _get_mapillary_rgb_lut():
    """Get or build the 24-bit RGB to class ID lookup table.
    
    Uses lazy initialization - builds LUT on first call, then caches it.
    The LUT maps packed RGB (R*65536 + G*256 + B) directly to class ID.
    
    Returns:
        np.ndarray: 16MB uint8 array mapping packed RGB to class ID (255 = ignore)
    """
    global _MAPILLARY_RGB_LUT_24BIT
    if _MAPILLARY_RGB_LUT_24BIT is None:
        # Build 24-bit direct lookup table (16MB)
        _MAPILLARY_RGB_LUT_24BIT = np.full(256**3, 255, dtype=np.uint8)
        for rgb, class_id in custom_transforms.MAPILLARY_RGB_TO_ID.items():
            packed_rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
            _MAPILLARY_RGB_LUT_24BIT[packed_rgb] = class_id
    return _MAPILLARY_RGB_LUT_24BIT


def detect_model_num_classes(cfg) -> int:
    """Detect number of output classes from model config."""
    # Try decode_head first
    if hasattr(cfg, 'model') and 'decode_head' in cfg.model:
        decode_head = cfg.model.decode_head
        # Handle case where decode_head is a list (e.g., OCRNet with cascade heads)
        if isinstance(decode_head, list):
            for head in decode_head:
                if 'num_classes' in head:
                    return head['num_classes']
        elif isinstance(decode_head, dict):
            return decode_head.get('num_classes', 19)
    # Fall back to top-level num_classes
    if hasattr(cfg, 'num_classes'):
        return cfg.num_classes
    return 19  # Default to Cityscapes


def map_predictions_to_cityscapes(pred_seg_map: np.ndarray, model_num_classes: int) -> np.ndarray:
    """Map model predictions from native classes to Cityscapes trainIDs.
    
    This is used for cross-domain evaluation where:
    - Model was trained on MapillaryVistas (66 classes) or OUTSIDE15k (24 classes)
    - Test data uses Cityscapes trainIDs (0-18)
    
    Args:
        pred_seg_map: Model predictions in native class space
        model_num_classes: Number of classes the model was trained with
    
    Returns:
        Predictions mapped to Cityscapes trainIDs (0-18, 255=ignore)
    """
    if model_num_classes == 66:
        # MapillaryVistas (66 classes) -> Cityscapes (19 classes)
        lut = np.full(256, 255, dtype=np.uint8)
        for native_id, train_id in MAPILLARY_PRED_TO_CITYSCAPES.items():
            if 0 <= native_id < 256:
                lut[native_id] = train_id
        return lut[pred_seg_map.astype(np.uint8)]
    
    elif model_num_classes == 24:
        # OUTSIDE15k (24 classes) -> Cityscapes (19 classes)
        lut = np.full(256, 255, dtype=np.uint8)
        for native_id, train_id in OUTSIDE15K_PRED_TO_CITYSCAPES.items():
            if 0 <= native_id < 256:
                lut[native_id] = train_id
        return lut[pred_seg_map.astype(np.uint8)]
    
    # No mapping needed (already Cityscapes)
    return pred_seg_map


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


def process_label_for_dataset(gt_seg_map: np.ndarray, dataset_name: str, 
                               model_num_classes: int = 19) -> np.ndarray:
    """Process ground truth labels according to dataset type.
    
    For CROSS-DOMAIN TESTING (model trained on one dataset, tested on another):
    - Test data is typically in Cityscapes trainID format (BDD10K, IDD-AW)
    - Model predictions are mapped to Cityscapes in a separate step
    - This function handles GT label format conversion only
    
    Label format handling:
    - ACDC: Cityscapes labelIDs (0-33) -> trainIds (0-18)
    - BDD10k, BDD100k, Cityscapes, IDD-AW: Already Cityscapes trainIds, no conversion
    - MapillaryVistas: RGB color-encoded -> native IDs (0-65) for native evaluation
    - OUTSIDE15k: Already native class IDs (0-23) for native evaluation
    
    REVERSE CROSS-DOMAIN (Cityscapes model on native test data):
    - When model_num_classes == 19 and test dataset is MapillaryVistas or OUTSIDE15k
    - Convert native GT labels to Cityscapes trainIDs for fair comparison
    
    Args:
        gt_seg_map: Raw label image (may be RGB or grayscale)
        dataset_name: Dataset folder name (e.g., 'MapillaryVistas', 'IDD-AW')
        model_num_classes: Number of classes the model was trained with (for determining evaluation mode)
        
    Returns:
        Processed label image (format depends on cross-domain vs same-dataset testing)
    """
    config = DATASET_LABEL_CONFIG.get(dataset_name, DATASET_LABEL_CONFIG['Cityscapes'])
    label_type = config['label_type']
    dataset_num_classes = config['num_classes']
    
    # Cross-domain testing scenarios:
    # 1. Native model (66/24 classes) on Cityscapes test data (19 classes)
    #    -> Predictions are mapped in map_predictions_to_cityscapes(), GT stays as-is
    # 2. Cityscapes model (19 classes) on native test data (66/24 classes) [REVERSE]
    #    -> GT must be converted from native to Cityscapes format
    is_cross_domain = model_num_classes in [66, 24] and dataset_num_classes == 19
    is_reverse_cross_domain = model_num_classes == 19 and dataset_num_classes in [66, 24]
    
    if label_type == 'mapillary_rgb_to_native':
        # MapillaryVistas: RGB color-encoded -> native IDs (0-65)
        if gt_seg_map.ndim == 3 and gt_seg_map.shape[-1] == 3:
            # OPTIMIZED: Use pre-built 24-bit LUT for O(1) per-pixel decoding
            # This is ~66x faster than iterating over each class color
            
            # IMPORTANT: cv2.imread loads as BGR, but our lookup uses RGB
            # Channel 0 in cv2 is B, Channel 2 is R
            r = gt_seg_map[:, :, 2].astype(np.int32)  # OpenCV channel 2 = R
            g = gt_seg_map[:, :, 1].astype(np.int32)  # OpenCV channel 1 = G  
            b = gt_seg_map[:, :, 0].astype(np.int32)  # OpenCV channel 0 = B
            packed = r * 65536 + g * 256 + b
            
            # Direct LUT lookup - single array indexing operation
            lut_24bit = _get_mapillary_rgb_lut()
            gt_seg_map = lut_24bit[packed]
        elif gt_seg_map.ndim == 3:
            gt_seg_map = gt_seg_map[:, :, 0]
        
        # If cross-domain testing (evaluating with Cityscapes GT), convert to trainIds
        # Otherwise keep native for same-dataset evaluation
        if is_cross_domain or is_reverse_cross_domain:
            lut = np.full(256, 255, dtype=np.uint8)
            for mapillary_id, train_id in custom_transforms.MAPILLARY_TO_TRAINID.items():
                if 0 <= mapillary_id < 256:
                    lut[mapillary_id] = train_id
            gt_seg_map = lut[gt_seg_map]
        
    elif label_type == 'native':
        # OUTSIDE15k: native class IDs (0-23)
        if gt_seg_map.ndim == 3:
            gt_seg_map = gt_seg_map[:, :, 0]
        
        # For reverse cross-domain (Cityscapes model on OUTSIDE15k),
        # convert GT to Cityscapes format
        if is_reverse_cross_domain:
            lut = np.full(256, 255, dtype=np.uint8)
            for native_id, train_id in custom_transforms.OUTSIDE15K_TO_TRAINID.items():
                if 0 <= native_id < 256:
                    lut[native_id] = train_id
            gt_seg_map = lut[gt_seg_map]
        
    elif label_type == 'cityscapes_labelid':
        # ACDC: Cityscapes labelIDs (0-33) -> trainIds (0-18)
        if gt_seg_map.ndim == 3:
            gt_seg_map = gt_seg_map[:, :, 0]
        
        lut = np.full(256, 255, dtype=np.uint8)
        for label_id, train_id in custom_transforms.CITYSCAPES_ID_TO_TRAINID.items():
            if 0 <= label_id < 256:
                lut[label_id] = train_id
        gt_seg_map = lut[gt_seg_map]
        
    elif label_type == 'outside15k_to_trainid':
        # OUTSIDE15k: native labels (0-23) -> Cityscapes trainIds (0-18)
        if gt_seg_map.ndim == 3:
            gt_seg_map = gt_seg_map[:, :, 0]
        
        # Convert OUTSIDE15k native IDs to Cityscapes trainIds
        lut = np.full(256, 255, dtype=np.uint8)
        for outside_id, train_id in custom_transforms.OUTSIDE15K_TO_TRAINID.items():
            if 0 <= outside_id < 256:
                lut[outside_id] = train_id
        gt_seg_map = lut[gt_seg_map]
        
    else:  # cityscapes_trainid
        # BDD10k, BDD100k, Cityscapes, IDD-AW: Already Cityscapes trainIds
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
    
    OPTIMIZED: Uses np.bincount for O(n) computation instead of O(num_classes * n).
    This provides ~36x speedup for 66 classes (MapillaryVistas).
    
    Returns:
        area_intersect, area_union, area_pred, area_label
    """
    # Filter out ignore_index pixels
    mask = label != ignore_index
    pred = pred[mask].astype(np.int64)
    label = label[mask].astype(np.int64)
    
    # Use bincount for vectorized counting - O(n) instead of O(num_classes * n)
    # Clip values to valid range to avoid bincount errors
    pred = np.clip(pred, 0, num_classes - 1)
    label = np.clip(label, 0, num_classes - 1)
    
    # Count pixels per class for predictions and labels
    area_pred = np.bincount(pred, minlength=num_classes).astype(np.float64)
    area_label = np.bincount(label, minlength=num_classes).astype(np.float64)
    
    # For intersection: count pixels where pred == label
    match_mask = pred == label
    matched_classes = pred[match_mask]
    area_intersect = np.bincount(matched_classes, minlength=num_classes).astype(np.float64)
    
    # Union = pred_count + label_count - intersection
    area_union = area_pred + area_label - area_intersect
    
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


def load_and_preprocess_image(img_path: Path, label_path: Path, 
                               folder_name: str, model_num_classes: int,
                               mean: np.ndarray, std: np.ndarray,
                               target_size: Tuple[int, int] = (512, 512)) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
    """Load and preprocess a single image and its label.
    
    Args:
        img_path: Path to input image
        label_path: Path to label image
        folder_name: Dataset folder name for label processing
        model_num_classes: Number of classes in model
        mean: Normalization mean
        std: Normalization std
        target_size: Target size (H, W) for resizing. Default: (512, 512)
    
    Returns:
        img_tensor: Preprocessed image tensor (CHW format)
        gt_seg_map: Ground truth segmentation map
        original_size: Original image size (H, W) before resizing
    """
    # Load image
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = (img.shape[0], img.shape[1])
    
    # Resize image to target size (512x512 by default)
    img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    
    # Load and process label
    gt_seg_map = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    gt_seg_map = process_label_for_dataset(gt_seg_map, folder_name, model_num_classes)
    # Resize label to target size using nearest neighbor (preserve class IDs)
    gt_seg_map = cv2.resize(gt_seg_map, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    
    # Normalize and convert to tensor
    img = (img.astype(np.float32) - mean) / std
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img_tensor = torch.from_numpy(img).float()
    
    return img_tensor, gt_seg_map, original_size


class PrefetchBatchLoader:
    """Batch loader with multi-threaded loading and pre-fetching.
    
    Loads batches in background threads while GPU processes current batch.
    This overlaps I/O with computation for better throughput.
    """
    
    def __init__(self, image_pairs: List[Tuple[Path, Path]], 
                 batch_size: int, folder_name: str, model_num_classes: int,
                 mean: np.ndarray, std: np.ndarray, num_workers: int = 4,
                 target_size: Tuple[int, int] = (512, 512)):
        """
        Args:
            image_pairs: List of (image_path, label_path) tuples
            batch_size: Number of images per batch
            folder_name: Dataset folder name for label processing
            model_num_classes: Number of classes in model
            mean: Normalization mean
            std: Normalization std
            num_workers: Number of parallel loading threads
            target_size: Target size (H, W) for resizing. Default: (512, 512)
        """
        self.image_pairs = image_pairs
        self.batch_size = batch_size
        self.folder_name = folder_name
        self.model_num_classes = model_num_classes
        self.mean = mean
        self.std = std
        self.num_workers = num_workers
        self.target_size = target_size
        
        self.num_batches = (len(image_pairs) + batch_size - 1) // batch_size
        self.current_batch_idx = 0
        
        # Pre-fetch queue (stores loaded batches ready for processing)
        self.prefetch_queue = queue.Queue(maxsize=2)  # Pre-fetch 2 batches ahead
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.loading_complete = threading.Event()
        
        # Start background pre-fetch thread
        self._prefetch_thread = threading.Thread(target=self._prefetch_batches, daemon=True)
        self._prefetch_thread.start()
    
    def _load_single_image(self, img_path: Path, label_path: Path):
        """Load a single image in a worker thread."""
        try:
            return load_and_preprocess_image(
                img_path, label_path, self.folder_name, 
                self.model_num_classes, self.mean, self.std,
                self.target_size
            )
        except Exception as e:
            return None
    
    def _load_batch(self, batch_idx: int):
        """Load a batch using multiple threads."""
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.image_pairs))
        batch_pairs = self.image_pairs[start_idx:end_idx]
        
        # Submit all images to thread pool
        futures = []
        for img_path, label_path in batch_pairs:
            future = self.executor.submit(self._load_single_image, img_path, label_path)
            futures.append(future)
        
        # Collect results
        batch_tensors = []
        batch_gt_maps = []
        batch_sizes = []
        
        for future in futures:
            result = future.result()
            if result is not None:
                img_tensor, gt_seg_map, original_size = result
                batch_tensors.append(img_tensor)
                batch_gt_maps.append(gt_seg_map)
                batch_sizes.append(original_size)
        
        return batch_tensors, batch_gt_maps, batch_sizes
    
    def _prefetch_batches(self):
        """Background thread that pre-fetches batches."""
        for batch_idx in range(self.num_batches):
            batch_data = self._load_batch(batch_idx)
            self.prefetch_queue.put((batch_idx, batch_data))
        self.loading_complete.set()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_batch_idx >= self.num_batches:
            self.shutdown()
            raise StopIteration
        
        # Get pre-fetched batch (blocks if not ready yet)
        batch_idx, batch_data = self.prefetch_queue.get()
        self.current_batch_idx += 1
        
        return batch_data
    
    def __len__(self):
        return self.num_batches
    
    def shutdown(self):
        """Clean up resources."""
        self.executor.shutdown(wait=False)


def process_batch(model, batch_tensors: List[torch.Tensor], 
                  batch_gt_maps: List[np.ndarray],
                  batch_sizes: List[Tuple[int, int]],
                  device: torch.device,
                  is_cross_domain: bool,
                  model_num_classes: int,
                  eval_num_classes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Process a batch of images through the model.
    
    Returns aggregated metrics for the batch.
    """
    # Stack tensors into batch
    batch_input = torch.stack(batch_tensors).to(device)
    batch_size = len(batch_tensors)
    
    # Create batch metadata
    batch_img_metas = []
    for h, w in batch_sizes:
        batch_img_metas.append({
            'ori_shape': (h, w),
            'img_shape': (h, w),
            'pad_shape': (h, w),
            'scale_factor': (1.0, 1.0),
        })
    
    # Batch inference
    with torch.no_grad():
        results = model.inference(batch_input, batch_img_metas)
    
    # Initialize batch metrics
    batch_area_intersect = np.zeros(eval_num_classes, dtype=np.float64)
    batch_area_union = np.zeros(eval_num_classes, dtype=np.float64)
    batch_area_pred = np.zeros(eval_num_classes, dtype=np.float64)
    batch_area_label = np.zeros(eval_num_classes, dtype=np.float64)
    
    # Process each result
    # OPTIMIZATION: Do argmax on GPU before transferring to CPU
    # For 66 classes: reduces transfer from 66MB to 1MB per batch
    for i in range(batch_size):
        # Extract prediction - apply argmax on GPU before transfer
        if isinstance(results, torch.Tensor):
            result_tensor = results[i]
            # If logits (3D), do argmax on GPU
            if result_tensor.ndim == 3:
                pred_seg_map = result_tensor.argmax(dim=0).cpu().numpy()
            else:
                pred_seg_map = result_tensor.cpu().numpy()
        elif isinstance(results, list) and hasattr(results[i], 'pred_sem_seg'):
            result_tensor = results[i].pred_sem_seg.data.squeeze()
            if result_tensor.ndim == 3:
                pred_seg_map = result_tensor.argmax(dim=0).cpu().numpy()
            else:
                pred_seg_map = result_tensor.cpu().numpy()
        else:
            result_tensor = results[i] if isinstance(results[i], torch.Tensor) else torch.tensor(results[i])
            if result_tensor.ndim == 3:
                pred_seg_map = result_tensor.argmax(dim=0).cpu().numpy()
            else:
                pred_seg_map = result_tensor.cpu().numpy().squeeze()
        
        gt_seg_map = batch_gt_maps[i]
        
        # Resize prediction back to original size if needed
        if pred_seg_map.shape != gt_seg_map.shape:
            pred_seg_map = cv2.resize(pred_seg_map.astype(np.uint8), 
                                      (gt_seg_map.shape[1], gt_seg_map.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
        
        # Map predictions to Cityscapes if cross-domain testing
        if is_cross_domain:
            pred_seg_map = map_predictions_to_cityscapes(pred_seg_map, model_num_classes)
        
        # Compute metrics
        area_intersect, area_union, area_pred, area_label = compute_iou_metrics(
            pred_seg_map, gt_seg_map, eval_num_classes
        )
        
        # Aggregate
        batch_area_intersect += area_intersect
        batch_area_union += area_union
        batch_area_pred += area_pred
        batch_area_label += area_label
    
    return batch_area_intersect, batch_area_union, batch_area_pred, batch_area_label


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
    test_split: str = 'test',
    batch_size: int = 4
) -> Dict[str, Any]:
    """Run fine-grained testing with per-domain and per-class results.
    
    Uses a single model instance and manually iterates over domains to avoid
    creating multiple Runners.
    
    Args:
        config_path: Path to model config file
        checkpoint_path: Path to model checkpoint
        output_dir: Output directory for results
        dataset_name: Name of dataset (ACDC, BDD10k, etc.)
        data_root: Root directory of test data
        test_split: Which split to use (test/val)
        batch_size: Number of images per batch for inference (default: 4)
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
    
    # Detect model's native number of classes
    model_num_classes = detect_model_num_classes(cfg)
    print(f"Model trained with {model_num_classes} classes")
    
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
    
    # Get domains for this dataset (use DATASET_DOMAINS if available, fallback to weather domains)
    domains = DATASET_DOMAINS.get(folder_name, ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy'])
    
    # Get dataset-specific configuration (num_classes for GT, class_names)
    gt_num_classes, class_names = get_dataset_config(folder_name)
    print(f"Test dataset '{folder_name}' has {gt_num_classes} classes")
    
    # Determine evaluation mode
    # For cross-domain testing scenarios:
    # 1. Native model (66/24) on Cityscapes test data (19): map predictions
    # 2. Cityscapes model (19) on native test data (66/24): convert GT labels [REVERSE]
    is_cross_domain = model_num_classes in [66, 24] and gt_num_classes == 19
    is_reverse_cross_domain = model_num_classes == 19 and gt_num_classes in [66, 24]
    
    if is_cross_domain:
        print(f"CROSS-DOMAIN TEST: Model ({model_num_classes} classes) -> Test data ({gt_num_classes} classes)")
        print(f"  Predictions will be mapped to Cityscapes classes for evaluation")
        eval_num_classes = 19  # Evaluate in Cityscapes space
        eval_class_names = CITYSCAPES_CLASSES
    elif is_reverse_cross_domain:
        print(f"REVERSE CROSS-DOMAIN TEST: Cityscapes model ({model_num_classes} classes) -> Native test data ({gt_num_classes} classes)")
        print(f"  GT labels will be converted to Cityscapes classes for evaluation")
        eval_num_classes = 19  # Evaluate in Cityscapes space
        eval_class_names = CITYSCAPES_CLASSES
    else:
        eval_num_classes = gt_num_classes
        eval_class_names = class_names
    
    # Results storage
    all_results = {
        'config': {
            'config_path': config_path,
            'checkpoint_path': checkpoint_path,
            'dataset': dataset_name,
            'test_split': test_split,
            'timestamp': timestamp,
            'classes': eval_class_names,
            'num_classes': eval_num_classes,
            'model_num_classes': model_num_classes,
            'is_cross_domain': is_cross_domain,
            'is_reverse_cross_domain': is_reverse_cross_domain,
        },
        'overall': {},
        'per_domain': {},
        'per_class': {}
    }
    
    # Overall aggregated metrics
    total_area_intersect = np.zeros(eval_num_classes, dtype=np.float64)
    total_area_union = np.zeros(eval_num_classes, dtype=np.float64)
    total_area_pred = np.zeros(eval_num_classes, dtype=np.float64)
    total_area_label = np.zeros(eval_num_classes, dtype=np.float64)
    
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
        domain_area_intersect = np.zeros(eval_num_classes, dtype=np.float64)
        domain_area_union = np.zeros(eval_num_classes, dtype=np.float64)
        domain_area_pred = np.zeros(eval_num_classes, dtype=np.float64)
        domain_area_label = np.zeros(eval_num_classes, dtype=np.float64)
        
        # Preprocessing constants
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        
        # Collect valid image-label pairs
        valid_pairs = []
        for img_path in img_files:
            label_path = label_dir / img_path.name
            if not label_path.exists():
                # Try stem + extension
                for ext in ['.png', '.jpg']:
                    label_path = label_dir / (img_path.stem + ext)
                    if label_path.exists():
                        break
            if not label_path.exists():
                # Cityscapes-specific naming: image has _leftImg8bit, label has _gtFine_labelIds
                if '_leftImg8bit' in img_path.name:
                    cs_label_name = img_path.name.replace('_leftImg8bit', '_gtFine_labelIds')
                    label_path = label_dir / cs_label_name
            if not label_path.exists():
                # ACDC-specific naming: image has _rgb_anon, label has _gt_labelIds
                if '_rgb_anon' in img_path.name:
                    acdc_label_name = img_path.name.replace('_rgb_anon', '_gt_labelIds')
                    label_path = label_dir / acdc_label_name
            if label_path.exists():
                valid_pairs.append((img_path, label_path))
        
        # Create prefetch batch loader for parallel I/O
        # All images are resized to 512x512 for consistent processing
        batch_loader = PrefetchBatchLoader(
            valid_pairs, batch_size, folder_name, model_num_classes,
            mean, std, num_workers=4, target_size=(512, 512)
        )
        
        # Process batches with prefetching
        for batch_tensors, batch_gt_maps, batch_sizes in tqdm(
            batch_loader, total=len(batch_loader),
            desc=f"  Processing {domain} (batches of {batch_size}, prefetch)"
        ):
            if not batch_tensors:
                continue
            
            # Process batch on GPU while next batch loads in background
            batch_intersect, batch_union, batch_pred, batch_label = process_batch(
                model, batch_tensors, batch_gt_maps, batch_sizes,
                device, is_cross_domain, model_num_classes, eval_num_classes
            )
            
            # Aggregate
            domain_area_intersect += batch_intersect
            domain_area_union += batch_union
            domain_area_pred += batch_pred
            domain_area_label += batch_label
        
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
            class_names=eval_class_names
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
            class_names=eval_class_names
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
    parser.add_argument('--data-root', default='${AWARE_DATA_ROOT}/FINAL_SPLITS',
                       help='Data root directory')
    parser.add_argument('--test-split', default='test', choices=['val', 'test'],
                       help='Test split to use')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for inference (default: 10). Larger = faster but uses more GPU memory.')
    
    args = parser.parse_args()
    
    run_fine_grained_test(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        data_root=args.data_root,
        test_split=args.test_split,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()