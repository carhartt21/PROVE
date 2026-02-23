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


# Mapillary Vistas native class ID (0-65) to Cityscapes trainId (0-18, 255=ignore) mapping
# Based on label_unification.py MapillarytoCityscapes mapping
MAPILLARY_TO_TRAINID = {
    0: 255,   # Bird -> ignore
    1: 255,   # Ground Animal -> ignore
    2: 1,     # Curb -> sidewalk
    3: 4,     # Fence -> fence
    4: 4,     # Guard Rail -> fence
    5: 3,     # Barrier -> wall
    6: 3,     # Wall -> wall
    7: 0,     # Bike Lane -> road
    8: 1,     # Crosswalk Plain -> sidewalk
    9: 1,     # Curb Cut -> sidewalk
    10: 255,  # Parking -> ignore
    11: 1,    # Pedestrian Area -> sidewalk
    12: 255,  # Rail Track -> ignore
    13: 0,    # Road -> road
    14: 0,    # Service Lane -> road
    15: 1,    # Sidewalk -> sidewalk
    16: 2,    # Bridge -> building
    17: 2,    # Building -> building
    18: 2,    # Tunnel -> building
    19: 11,   # Person -> person
    20: 12,   # Bicyclist -> rider
    21: 12,   # Motorcyclist -> rider
    22: 12,   # Other Rider -> rider
    23: 0,    # Lane Marking Crosswalk -> road
    24: 0,    # Lane Marking General -> road
    25: 9,    # Mountain -> terrain
    26: 9,    # Sand -> terrain
    27: 10,   # Sky -> sky
    28: 255,  # Snow -> ignore
    29: 9,    # Terrain -> terrain
    30: 8,    # Vegetation -> vegetation
    31: 255,  # Water -> ignore
    32: 255,  # Banner -> ignore
    33: 255,  # Bench -> ignore
    34: 255,  # Bike Rack -> ignore
    35: 255,  # Billboard -> ignore
    36: 255,  # Catch Basin -> ignore
    37: 255,  # CCTV Camera -> ignore
    38: 255,  # Fire Hydrant -> ignore
    39: 255,  # Junction Box -> ignore
    40: 255,  # Mailbox -> ignore
    41: 255,  # Manhole -> ignore
    42: 255,  # Phone Booth -> ignore
    43: 255,  # Pothole -> ignore
    44: 5,    # Street Light -> pole
    45: 5,    # Pole -> pole
    46: 5,    # Traffic Sign Frame -> pole
    47: 5,    # Utility Pole -> pole
    48: 6,    # Traffic Light -> traffic light
    49: 7,    # Traffic Sign Back -> traffic sign
    50: 7,    # Traffic Sign Front -> traffic sign
    51: 255,  # Trash Can -> ignore
    52: 18,   # Bicycle -> bicycle
    53: 255,  # Boat -> ignore
    54: 15,   # Bus -> bus
    55: 13,   # Car -> car
    56: 255,  # Caravan -> ignore
    57: 17,   # Motorcycle -> motorcycle
    58: 16,   # On Rails -> train
    59: 255,  # Other Vehicle -> ignore
    60: 255,  # Trailer -> ignore
    61: 14,   # Truck -> truck
    62: 255,  # Wheeled Slow -> ignore
    63: 255,  # Car Mount -> ignore
    64: 255,  # Ego Vehicle -> ignore
    65: 255,  # Unlabeled -> ignore
}


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


@TRANSFORMS.register_module()
class MapillaryToTrainId(BaseTransform):
    """Convert MapillaryVistas native class IDs (0-65) to Cityscapes trainIds (0-18, 255=ignore).
    
    This transform maps Mapillary Vistas labels to the Cityscapes 19-class format
    used for unified training and evaluation. Must be applied AFTER MapillaryRGBToClassId
    which first converts RGB color-encoded labels to native class IDs.
    
    Required Keys:
        - gt_seg_map (np.ndarray): Segmentation ground truth with MapillaryVistas class IDs (0-65)
        
    Modified Keys:
        - gt_seg_map (np.ndarray): Segmentation ground truth with trainIds (0-18, 255=ignore)
    """
    
    def __init__(self):
        # Create lookup table for fast mapping (values 0-255 to handle any input)
        self.lut = np.full(256, 255, dtype=np.uint8)
        for mapillary_id, train_id in MAPILLARY_TO_TRAINID.items():
            if 0 <= mapillary_id < 256:
                self.lut[mapillary_id] = train_id
    
    def transform(self, results: dict) -> dict:
        """Convert MapillaryVistas class IDs to Cityscapes trainIds using lookup table."""
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


# IDD-AW label value mapping to Cityscapes trainId (0-18, 255=ignore)
# IDD-AW labels are mostly in Cityscapes trainId format (0-18) but have additional values:
#   - Value 19: appears to be IDD 'curb' class -> maps to 255 (ignore) per IDD39 csTrainId
#   - Value 20: appears to be IDD 'wall' class -> maps to 3 (wall) per IDD39 csTrainId
# This mapping corrects the out-of-range values while preserving valid trainIds
IDD_AW_TO_TRAINID = {
    # Valid Cityscapes trainIds (0-18) - pass through unchanged
    0: 0,     # road -> road
    1: 1,     # sidewalk -> sidewalk
    2: 2,     # building -> building
    3: 3,     # wall -> wall
    4: 4,     # fence -> fence
    5: 5,     # pole -> pole
    6: 6,     # traffic light -> traffic light
    7: 7,     # traffic sign -> traffic sign
    8: 8,     # vegetation -> vegetation
    9: 9,     # terrain -> terrain
    10: 10,   # sky -> sky
    11: 11,   # person -> person
    12: 12,   # rider -> rider
    13: 13,   # car -> car
    14: 14,   # truck -> truck
    15: 15,   # bus -> bus
    16: 16,   # train -> train
    17: 17,   # motorcycle -> motorcycle
    18: 18,   # bicycle -> bicycle
    # IDD-specific additional classes (19-20) - map based on IDD39 csTrainId
    # These are likely IDD class IDs that weren't converted to Cityscapes trainIds
    19: 255,  # IDD class 19 = curb -> Cityscapes ignore (255)
    20: 3,    # IDD class 20 = wall -> Cityscapes wall (trainId 3)
    # Ignore
    255: 255, # ignore -> ignore
}


@TRANSFORMS.register_module()
class IDDAWLabelTransform(BaseTransform):
    """Convert IDD-AW label values to Cityscapes trainIds (0-18, 255=ignore).
    
    IDD-AW labels are mostly in Cityscapes trainId format but contain additional
    values 19 and 20 that need to be mapped to valid Cityscapes classes:
    - 19 (IDD traffic light) -> 6 (Cityscapes traffic light)
    - 20 (IDD pole) -> 5 (Cityscapes pole)
    
    Required Keys:
        - gt_seg_map (np.ndarray): Segmentation ground truth with IDD-AW label values
        
    Modified Keys:
        - gt_seg_map (np.ndarray): Segmentation ground truth with trainIds (0-18, 255=ignore)
    """
    
    def __init__(self):
        # Create lookup table for fast mapping (values 0-255 to handle any input)
        # Default to 255 (ignore) for any unmapped values
        self.lut = np.full(256, 255, dtype=np.uint8)
        for label_val, train_id in IDD_AW_TO_TRAINID.items():
            if 0 <= label_val < 256:
                self.lut[label_val] = train_id
    
    def transform(self, results: dict) -> dict:
        """Convert IDD-AW label values to Cityscapes trainIds using lookup table."""
        if 'gt_seg_map' in results:
            seg_map = results['gt_seg_map']
            # Use lookup table for fast vectorized conversion
            results['gt_seg_map'] = self.lut[seg_map]
        return results
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


# Alias for backward compatibility with unified_training_config.py
@TRANSFORMS.register_module()
class IddawLabelClamp(IDDAWLabelTransform):
    """Alias for IDDAWLabelTransform for backward compatibility."""
    pass


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
    
    IMPORTANT: mmseg's LoadAnnotations (via mmcv.imfrombytes) returns images in
    BGR order regardless of the backend used. This transform handles both BGR
    input (from LoadAnnotations) and RGB input (from direct PIL loading).
    
    Any unrecognized RGB values are mapped to 255 (ignore_index).
    
    Required Keys:
        - gt_seg_map (np.ndarray): BGR label with shape (H, W, 3) from LoadAnnotations
        
    Modified Keys:
        - gt_seg_map (np.ndarray): Class indices with shape (H, W), values 0-65 or 255
    """
    
    def __init__(self, ignore_index: int = 255):
        self.ignore_index = ignore_index
        # Build optimized 24-bit LUT for O(1) per-pixel lookup
        self._build_lut()
    
    def _build_lut(self):
        """Build 24-bit lookup table for O(1) RGB to class ID conversion.
        
        Pre-allocates a 16MB table mapping all possible 24-bit RGB values
        directly to class IDs. This is ~66x faster than iterating over
        each class color during decode.
        """
        # 24-bit direct lookup table (16MB memory, O(1) decode)
        self.lut_24bit = np.full(256**3, self.ignore_index, dtype=np.uint8)
        for rgb, class_id in MAPILLARY_RGB_TO_ID.items():
            packed = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
            self.lut_24bit[packed] = class_id
    
    def transform(self, results: dict) -> dict:
        """Convert BGR labels to class IDs using optimized LUT.
        
        CRITICAL: mmseg's LoadAnnotations returns BGR images (via mmcv.imfrombytes
        which defaults to channel_order='bgr'). We must swap channels to RGB before
        looking up in our RGB-based LUT.
        """
        if 'gt_seg_map' in results:
            seg_map = results['gt_seg_map']
            
            if seg_map.ndim == 3 and seg_map.shape[-1] == 3:
                # BGR image from LoadAnnotations - swap to RGB for lookup
                # mmcv.imfrombytes defaults to BGR regardless of backend (pillow/cv2)
                # Channel 0 = B, Channel 1 = G, Channel 2 = R
                r = seg_map[:, :, 2].astype(np.int32)  # R is in channel 2
                g = seg_map[:, :, 1].astype(np.int32)  # G is in channel 1
                b = seg_map[:, :, 0].astype(np.int32)  # B is in channel 0
                packed = r * 65536 + g * 256 + b
                
                # Direct array indexing - O(1) per pixel
                results['gt_seg_map'] = self.lut_24bit[packed]
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


# ============================================================================
# Noise Ablation Transform
# ============================================================================

@TRANSFORMS.register_module()
class ReplaceWithNoise(BaseTransform):
    """Replace loaded image data with random noise of the same shape.

    Used in the noise ablation study to test whether models learn from
    actual image content or just memorize segmentation map layouts.

    Only applies to samples where ``_replace_with_noise`` is ``True``.
    Real images (without this flag) pass through unchanged.

    Args:
        noise_type (str): Type of noise. ``'uniform'`` samples pixel values
            uniformly from [0, 255]. ``'gaussian'`` samples from a clipped
            Gaussian distribution. Defaults to ``'uniform'``.
        mean (float): Mean for Gaussian noise. Defaults to ``128``.
        std (float): Standard deviation for Gaussian noise. Defaults to ``64``.

    Required Keys:
        - img (np.ndarray): Loaded image data (H, W, C).

    Modified Keys:
        - img (np.ndarray): Replaced with random noise of same shape and dtype.
    """

    def __init__(self, noise_type: str = 'uniform', mean: float = 128, std: float = 64):
        self.noise_type = noise_type
        self.mean = mean
        self.std = std

    def transform(self, results: dict) -> dict:
        if not results.get('_replace_with_noise', False):
            return results

        img = results['img']
        shape = img.shape  # (H, W, C)

        if self.noise_type == 'uniform':
            noise = np.random.randint(0, 256, shape, dtype=np.uint8)
        elif self.noise_type == 'gaussian':
            noise = np.random.normal(self.mean, self.std, shape)
            noise = np.clip(noise, 0, 255).astype(np.uint8)
        else:
            raise ValueError(f"Unknown noise_type: {self.noise_type}")

        results['img'] = noise
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'noise_type={self.noise_type!r}, '
                f'mean={self.mean}, std={self.std})')



