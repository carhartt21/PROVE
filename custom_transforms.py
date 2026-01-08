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


@TRANSFORMS.register_module()
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


@METRICS.register_module()
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


