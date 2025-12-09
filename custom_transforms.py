#!/usr/bin/env python3
"""
Custom MMSegmentation transforms for PROVE training.

This module registers custom transforms needed for the PROVE dataset.
Import this module before building configs to ensure transforms are registered.
"""

import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


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
