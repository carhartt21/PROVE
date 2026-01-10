#!/usr/bin/env python3
"""
Custom MMSegmentation transforms and metrics for PROVE training.

This module registers custom transforms and metrics needed for the PROVE dataset.
Import this module before building configs to ensure transforms are registered.
"""

import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS, METRICS
from mmseg.evaluation.metrics import IoUMetric


# Cityscapes label ID to trainId mapping
# From: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
CITYSCAPES_ID_TO_TRAINID = {
    -1: 255,  # unlabeled
    0: 255,   # unlabeled
    1: 255,   # ego vehicle
    2: 255,   # rectification border
    3: 255,   # out of roi
    4: 255,   # static
    5: 255,   # dynamic
    6: 255,   # ground
    7: 0,     # road
    8: 1,     # sidewalk
    9: 255,   # parking
    10: 255,  # rail track
    11: 2,    # building
    12: 3,    # wall
    13: 4,    # fence
    14: 255,  # guard rail
    15: 255,  # bridge
    16: 255,  # tunnel
    17: 5,    # pole
    18: 255,  # polegroup
    19: 6,    # traffic light
    20: 7,    # traffic sign
    21: 8,    # vegetation
    22: 9,    # terrain
    23: 10,   # sky
    24: 11,   # person
    25: 12,   # rider
    26: 13,   # car
    27: 14,   # truck
    28: 15,   # bus
    29: 255,  # caravan
    30: 255,  # trailer
    31: 16,   # train
    32: 17,   # motorcycle
    33: 18,   # bicycle
}


@TRANSFORMS.register_module(force=True)
class ReduceToSingleChannel(BaseTransform):
    """Convert 3-channel label images (where all channels are identical class IDs) to single channel.
    
    Some datasets store labels as RGB PNGs where all 3 channels contain the same class ID values.
    This transform extracts only the first channel for proper training.
    
    Required Keys:
        - gt_seg_map (np.ndarray): Segmentation ground truth with shape (H, W) or (H, W, 3)
        
    Modified Keys:
        - gt_seg_map (np.ndarray): Segmentation ground truth with shape (H, W)
    """
    
    def transform(self, results: dict) -> dict:
        """Take first channel of gt_seg_map if it has 3 channels."""
        if 'gt_seg_map' in results:
            seg_map = results['gt_seg_map']
            if seg_map.ndim == 3:
                if seg_map.shape[-1] == 3:
                    # Take first channel (all channels should be identical)
                    results['gt_seg_map'] = seg_map[:, :, 0]
                elif seg_map.shape[0] == 3:
                    # Channel-first format
                    results['gt_seg_map'] = seg_map[0, :, :]
        return results
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


@TRANSFORMS.register_module()
class CityscapesLabelIdToTrainId(BaseTransform):
    """Convert Cityscapes full label IDs (0-33) to trainIds (0-18, 255=ignore).
    
    Some datasets use Cityscapes full label IDs instead of trainIds. This transform
    converts them to the standard trainId format expected by segmentation models
    with 19 classes.
    
    The mapping follows the official Cityscapes label definitions.
    
    Required Keys:
        - gt_seg_map (np.ndarray): Segmentation ground truth with Cityscapes label IDs
        
    Modified Keys:
        - gt_seg_map (np.ndarray): Segmentation ground truth with trainIds (0-18, 255=ignore)
    """
    
    def __init__(self):
        # Create lookup table for fast mapping (values 0-255 to handle any input)
        self.lut = np.full(256, 255, dtype=np.uint8)
        for label_id, train_id in CITYSCAPES_ID_TO_TRAINID.items():
            if 0 <= label_id < 256:
                self.lut[label_id] = train_id
    
    def transform(self, results: dict) -> dict:
        """Convert label IDs to trainIds using lookup table."""
        if 'gt_seg_map' in results:
            seg_map = results['gt_seg_map']
            # Use lookup table for fast vectorized conversion
            results['gt_seg_map'] = self.lut[seg_map]
        return results
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


@METRICS.register_module(force=True)
class FWIoUMetric(IoUMetric):
    """IoU Metric with additional Frequency Weighted IoU (fwIoU) computation.
    
    Extends IoUMetric to include fwIoU = sum(freq_i * IoU_i) where
    freq_i = area_label_i / total_area.
    
    Args:
        iou_metrics: List of metrics to compute. Supports 'mIoU', 'mDice', 'mFscore', 'fwIoU'.
    """
    
    ALLOWED_METRICS = ['mIoU', 'mDice', 'mFscore', 'fwIoU']
    
    def __init__(self, iou_metrics=['mIoU'], **kwargs):
        # Validate and separate fwIoU
        for metric in iou_metrics:
            if metric not in self.ALLOWED_METRICS:
                raise KeyError(f"metric {metric} not in {self.ALLOWED_METRICS}")
        
        self.compute_fwiou = 'fwIoU' in iou_metrics
        parent_metrics = [m for m in iou_metrics if m != 'fwIoU']
        if not parent_metrics:
            parent_metrics = ['mIoU']
        
        super().__init__(iou_metrics=parent_metrics, **kwargs)
    
    def compute_metrics(self, results: list):
        """Compute metrics including fwIoU.
        
        Args:
            results: List of tuples (area_intersect, area_union, area_pred_label, area_label)
        """
        # Get parent metrics first
        metrics = super().compute_metrics(results)
        
        if self.compute_fwiou:
            # Results are list of tuples: [(intersect, union, pred, label), ...]
            # Convert to tuple of lists and sum
            results_tuple = tuple(zip(*results))
            assert len(results_tuple) == 4
            
            total_area_intersect = sum(results_tuple[0])
            total_area_union = sum(results_tuple[1])
            total_area_label = sum(results_tuple[3])  # Index 3 is area_label
            
            # Compute fwIoU: sum(freq_i * IoU_i)
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                iou = total_area_intersect / total_area_union
            
            total_area = total_area_label.sum()
            if total_area > 0:
                freq = total_area_label / total_area
                # Handle NaN values in IoU
                valid_mask = ~np.isnan(iou)
                iou = np.nan_to_num(iou, nan=0.0)
                freq = freq * valid_mask
                # Normalize frequencies
                freq_sum = freq.sum()
                if freq_sum > 0:
                    freq = freq / freq_sum
                fwiou = (freq * iou).sum()
            else:
                fwiou = 0.0
            
            metrics['fwIoU'] = np.round(fwiou * 100, 2)  # Percentage like other metrics
        
        return metrics


# OUTSIDE15k label ID to Cityscapes trainId mapping
# OUTSIDE15k has 24 classes (0-23) that need to be mapped to Cityscapes trainId (0-18)
OUTSIDE15K_TO_TRAINID = {
    0: 255,   # unlabeled -> ignore
    1: 255,   # animal -> ignore (no Cityscapes equivalent)
    2: 4,     # barrier -> fence
    3: 18,    # bicycle -> bicycle
    4: 255,   # boat -> ignore
    5: 255,   # bridge -> ignore
    6: 2,     # building -> building
    7: 8,     # grass -> vegetation
    8: 9,     # ground -> terrain
    9: 9,     # mountain -> terrain
    10: 255,  # object -> ignore
    11: 11,   # person -> person
    12: 5,    # pole -> pole
    13: 0,    # road -> road
    14: 9,    # sand -> terrain
    15: 1,    # sidewalk -> sidewalk
    16: 7,    # sign -> traffic sign
    17: 10,   # sky -> sky
    18: 5,    # street light -> pole
    19: 6,    # traffic light -> traffic light
    20: 2,    # tunnel -> building
    21: 8,    # vegetation -> vegetation
    22: 13,   # vehicle -> car
    23: 255,  # water -> ignore
}


@TRANSFORMS.register_module()
class Outside15kLabelTransform(BaseTransform):
    """Convert OUTSIDE15k label IDs (0-23) to Cityscapes trainIds (0-18, 255=ignore).
    
    OUTSIDE15k uses its own 24-class label format that needs to be mapped to
    the standard Cityscapes trainId format for unified training/evaluation.
    
    Required Keys:
        - gt_seg_map (np.ndarray): Segmentation ground truth with OUTSIDE15k label IDs
        
    Modified Keys:
        - gt_seg_map (np.ndarray): Segmentation ground truth with trainIds (0-18, 255=ignore)
    """
    
    def __init__(self):
        # Create lookup table for fast mapping (values 0-255 to handle any input)
        self.lut = np.full(256, 255, dtype=np.uint8)
        for label_id, train_id in OUTSIDE15K_TO_TRAINID.items():
            if 0 <= label_id < 256:
                self.lut[label_id] = train_id
    
    def transform(self, results: dict) -> dict:
        """Convert OUTSIDE15k label IDs to Cityscapes trainIds using lookup table."""
        if 'gt_seg_map' in results:
            seg_map = results['gt_seg_map']
            # Use lookup table for fast vectorized conversion
            results['gt_seg_map'] = self.lut[seg_map]
        return results
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


# Mapillary Vistas RGB to Class ID mapping
# Each tuple is (R, G, B) -> class_id (0-65)
MAPILLARY_RGB_TO_ID = {
    (165, 42, 42): 0,      # Bird
    (0, 192, 0): 1,        # Ground Animal
    (196, 196, 196): 2,    # Curb
    (190, 153, 153): 3,    # Fence
    (180, 165, 180): 4,    # Guard Rail
    (90, 120, 150): 5,     # Barrier
    (102, 102, 156): 6,    # Wall
    (128, 64, 255): 7,     # Bike Lane
    (140, 140, 200): 8,    # Crosswalk - Plain
    (170, 170, 170): 9,    # Curb Cut
    (250, 170, 160): 10,   # Parking
    (96, 96, 96): 11,      # Pedestrian Area
    (230, 150, 140): 12,   # Rail Track
    (128, 64, 128): 13,    # Road
    (110, 110, 110): 14,   # Service Lane
    (244, 35, 232): 15,    # Sidewalk
    (150, 100, 100): 16,   # Bridge
    (70, 70, 70): 17,      # Building
    (150, 120, 90): 18,    # Tunnel
    (220, 20, 60): 19,     # Person
    (255, 0, 0): 20,       # Bicyclist
    (255, 0, 100): 21,     # Motorcyclist
    (255, 0, 200): 22,     # Other Rider
    (200, 128, 128): 23,   # Lane Marking - Crosswalk
    (255, 255, 255): 24,   # Lane Marking - General
    (64, 170, 64): 25,     # Mountain
    (230, 160, 50): 26,    # Sand
    (70, 130, 180): 27,    # Sky
    (190, 255, 255): 28,   # Snow
    (152, 251, 152): 29,   # Terrain
    (107, 142, 35): 30,    # Vegetation
    (0, 170, 30): 31,      # Water
    (255, 255, 128): 32,   # Banner
    (250, 0, 30): 33,      # Bench
    (100, 140, 180): 34,   # Bike Rack
    (220, 220, 220): 35,   # Billboard
    (220, 128, 128): 36,   # Catch Basin
    (222, 40, 40): 37,     # CCTV Camera
    (100, 170, 30): 38,    # Fire Hydrant
    (40, 40, 40): 39,      # Junction Box
    (33, 33, 33): 40,      # Mailbox
    (100, 128, 160): 41,   # Manhole
    (142, 0, 0): 42,       # Phone Booth
    (70, 100, 150): 43,    # Pothole
    (210, 170, 100): 44,   # Street Light
    (153, 153, 153): 45,   # Pole
    (128, 128, 128): 46,   # Traffic Sign Frame
    (0, 0, 80): 47,        # Utility Pole
    (250, 170, 30): 48,    # Traffic Light
    (192, 192, 192): 49,   # Traffic Sign (Back)
    (220, 220, 0): 50,     # Traffic Sign (Front)
    (140, 140, 20): 51,    # Trash Can
    (119, 11, 32): 52,     # Bicycle
    (150, 0, 255): 53,     # Boat
    (0, 60, 100): 54,      # Bus
    (0, 0, 142): 55,       # Car
    (0, 0, 90): 56,        # Caravan
    (0, 0, 230): 57,       # Motorcycle
    (0, 80, 100): 58,      # On Rails
    (128, 64, 64): 59,     # Other Vehicle
    (0, 0, 110): 60,       # Trailer
    (0, 0, 70): 61,        # Truck
    (0, 0, 192): 62,       # Wheeled Slow
    (32, 32, 32): 63,      # Car Mount
    (120, 10, 10): 64,     # Ego Vehicle
    (0, 0, 0): 65,         # Unlabeled
}


@TRANSFORMS.register_module()
class MapillaryRGBToClassId(BaseTransform):
    """Convert MapillaryVistas RGB color-encoded labels to class IDs.
    
    MapillaryVistas labels are stored as RGB images where each unique (R,G,B)
    triplet represents a semantic class. This transform decodes them to class
    indices (0-65) suitable for training.
    
    Any unrecognized RGB values are mapped to 255 (ignore_index).
    
    Required Keys:
        - gt_seg_map (np.ndarray): RGB label with shape (H, W, 3)
        
    Modified Keys:
        - gt_seg_map (np.ndarray): Class indices with shape (H, W), values 0-65 or 255
    """
    
    def __init__(self, ignore_index: int = 255):
        self.ignore_index = ignore_index
        # Build optimized lookup using color encoding
        self._build_lookup()
    
    def _build_lookup(self):
        """Build lookup table for fast RGB to class ID conversion."""
        # Use a dictionary with packed RGB as key for fast lookup
        self.rgb_lookup = {}
        for rgb, class_id in MAPILLARY_RGB_TO_ID.items():
            # Pack RGB into single int for fast lookup
            packed = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
            self.rgb_lookup[packed] = class_id
    
    def transform(self, results: dict) -> dict:
        """Convert RGB labels to class IDs."""
        if 'gt_seg_map' in results:
            seg_map = results['gt_seg_map']
            
            if seg_map.ndim == 3 and seg_map.shape[-1] == 3:
                # RGB image - decode to class IDs
                h, w = seg_map.shape[:2]
                output = np.full((h, w), self.ignore_index, dtype=np.uint8)
                
                # Pack RGB values for fast lookup
                r = seg_map[:, :, 0].astype(np.int32)
                g = seg_map[:, :, 1].astype(np.int32)
                b = seg_map[:, :, 2].astype(np.int32)
                packed = r * 65536 + g * 256 + b
                
                # Vectorized lookup
                for packed_rgb, class_id in self.rgb_lookup.items():
                    mask = packed == packed_rgb
                    output[mask] = class_id
                
                results['gt_seg_map'] = output
            elif seg_map.ndim == 2:
                # Already class IDs - no conversion needed
                pass
        
        return results
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(ignore_index={self.ignore_index})'


@TRANSFORMS.register_module()
class MapillaryNativeLabelClamp(BaseTransform):
    """Clamp MapillaryVistas native labels to valid range (0-65) for single-dataset training.
    
    MapillaryVistas has 66 native classes (0-65), but some labels may contain invalid
    values like 255 for void/unlabeled regions. This transform ensures all labels are
    within the valid range by mapping invalid values (>=66) to the ignore index (255).
    
    This is essential because PyTorch's NLLLoss2d CUDA kernel has an assertion that
    checks `cur_target >= 0 && cur_target < n_classes` BEFORE the ignore_index is
    applied, which causes CUDA assertion failures if labels contain values >= num_classes.
    
    Required Keys:
        - gt_seg_map (np.ndarray): Segmentation ground truth with MapillaryVistas labels
        
    Modified Keys:
        - gt_seg_map (np.ndarray): Segmentation ground truth with valid labels (0-65, 255=ignore)
    """
    
    def __init__(self, num_classes: int = 66):
        self.num_classes = num_classes
    
    def transform(self, results: dict) -> dict:
        """Clamp invalid label values to ignore index."""
        if 'gt_seg_map' in results:
            seg_map = results['gt_seg_map']
            # Map any value >= num_classes to ignore (255), except 255 which stays 255
            # This handles cases where labels have values like 255 for void
            invalid_mask = (seg_map >= self.num_classes) & (seg_map != 255)
            if invalid_mask.any():
                seg_map = seg_map.copy()  # Don't modify original
                seg_map[invalid_mask] = 255
            results['gt_seg_map'] = seg_map
        return results
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_classes={self.num_classes})'


@TRANSFORMS.register_module()
class Outside15kNativeLabelClamp(BaseTransform):
    """Clamp OUTSIDE15k native labels to valid range (0-23) for single-dataset training.
    
    OUTSIDE15k has 24 native classes (0-23), but some labels may contain invalid
    values like 255 for void/unlabeled regions. This transform ensures all labels are
    within the valid range by mapping invalid values (>=24) to the ignore index (255).
    
    Required Keys:
        - gt_seg_map (np.ndarray): Segmentation ground truth with OUTSIDE15k labels
        
    Modified Keys:
        - gt_seg_map (np.ndarray): Segmentation ground truth with valid labels (0-23, 255=ignore)
    """
    
    def __init__(self, num_classes: int = 24):
        self.num_classes = num_classes
    
    def transform(self, results: dict) -> dict:
        """Clamp invalid label values to ignore index."""
        if 'gt_seg_map' in results:
            seg_map = results['gt_seg_map']
            # Map any value >= num_classes to ignore (255), except 255 which stays 255
            invalid_mask = (seg_map >= self.num_classes) & (seg_map != 255)
            if invalid_mask.any():
                seg_map = seg_map.copy()  # Don't modify original
                seg_map[invalid_mask] = 255
            results['gt_seg_map'] = seg_map
        return results
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_classes={self.num_classes})'


