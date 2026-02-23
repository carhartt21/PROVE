#!/usr/bin/env python3
"""
Unified Dataset Classes for PROVE Pipeline
Custom MMSegmentation dataset classes for joint Cityscapes + Mapillary training

These dataset classes handle label transformation on-the-fly during training,
allowing seamless joint training of both datasets in a unified label space.
"""

import os
import os.path as osp
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from mmseg.datasets.basesegdataset import BaseSegDataset
    from mmseg.registry import DATASETS
    MMSEG_AVAILABLE = True
except ImportError:
    try:
        # Fallback for older mmseg versions
        from mmseg.datasets.builder import DATASETS
        from mmseg.datasets.basesegdataset import BaseSegDataset
        MMSEG_AVAILABLE = True
    except ImportError:
        MMSEG_AVAILABLE = False
        DATASETS = None

from label_unification import (
    LabelUnificationManager,
    MapillarytoCityscapes,
    MapillaryToUnified,
    CityscapesToUnified,
    CITYSCAPES_CLASSES,
    MAPILLARY_CLASSES,
    UNIFIED_CLASSES,
)


# =============================================================================
# CITYSCAPES CLASSES AND PALETTE
# =============================================================================

CITYSCAPES_CLASSES_NAMES = (
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle'
)

CITYSCAPES_PALETTE = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
]

# =============================================================================
# UNIFIED CLASSES AND PALETTE  
# =============================================================================

UNIFIED_CLASSES_NAMES = (
    'road', 'sidewalk', 'parking', 'rail track', 'bike lane',
    'building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel', 'barrier',
    'pole', 'traffic light', 'traffic sign', 'street light', 'utility pole', 'other object',
    'vegetation', 'terrain', 'sky', 'water', 'snow', 'mountain',
    'person', 'bicyclist', 'motorcyclist', 'other rider',
    'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
    'caravan', 'trailer', 'boat', 'other vehicle', 'wheeled slow', 'animal',
    'lane marking', 'crosswalk'
)

UNIFIED_PALETTE = [
    # Flat (0-4)
    [128, 64, 128], [244, 35, 232], [250, 170, 160], [230, 150, 140], [128, 64, 255],
    # Construction (5-11)
    [70, 70, 70], [102, 102, 156], [190, 153, 153], [180, 165, 180], [150, 100, 100],
    [150, 120, 90], [90, 120, 150],
    # Object (12-17)
    [153, 153, 153], [250, 170, 30], [220, 220, 0], [210, 170, 100], [0, 0, 80], [140, 140, 140],
    # Nature (18-23)
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [0, 170, 30], [190, 255, 255], [64, 170, 64],
    # Human (24-27)
    [220, 20, 60], [255, 0, 0], [255, 0, 100], [255, 0, 200],
    # Vehicle (28-39)
    [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32],
    [0, 0, 90], [0, 0, 110], [150, 0, 255], [128, 64, 64], [0, 0, 192], [165, 42, 42],
    # Marking (40-41)
    [255, 255, 255], [200, 128, 128],
]

# =============================================================================
# OUTSIDE15K CLASSES AND PALETTE (24 classes, 0-23)
# =============================================================================

OUTSIDE15K_CLASSES_NAMES = (
    'unlabeled', 'animal', 'barrier', 'bicycle', 'boat', 'bridge',
    'building', 'grass', 'ground', 'mountain', 'object', 'person',
    'pole', 'road', 'sand', 'sidewalk', 'sign', 'sky', 'street light',
    'traffic light', 'tunnel', 'vegetation', 'vehicle', 'water',
)

# Color palette for OUTSIDE15K (24 colors)
OUTSIDE15K_PALETTE = [
    [0, 0, 0],        # unlabeled
    [165, 42, 42],    # animal
    [90, 120, 150],   # barrier
    [119, 11, 32],    # bicycle
    [150, 0, 255],    # boat
    [150, 100, 100],  # bridge
    [70, 70, 70],     # building
    [152, 251, 152],  # grass
    [81, 0, 81],      # ground
    [64, 170, 64],    # mountain
    [140, 140, 140],  # object
    [220, 20, 60],    # person
    [153, 153, 153],  # pole
    [128, 64, 128],   # road
    [230, 160, 50],   # sand
    [244, 35, 232],   # sidewalk
    [220, 220, 0],    # sign
    [70, 130, 180],   # sky
    [210, 170, 100],  # street light
    [250, 170, 30],   # traffic light
    [150, 120, 90],   # tunnel
    [107, 142, 35],   # vegetation
    [0, 0, 142],      # vehicle
    [0, 170, 30],     # water
]


# =============================================================================
# MAPILLARY UNIFIED DATASET (Maps Mapillary to Cityscapes or Unified)
# =============================================================================

if MMSEG_AVAILABLE:
    @DATASETS.register_module()
    class MapillaryUnifiedDataset(BaseSegDataset):
        """
        Mapillary Vistas dataset with on-the-fly label transformation.
        
        This dataset loads Mapillary Vistas images and labels, then transforms
        the labels to either Cityscapes format or unified format for joint training.
        
        Args:
            target_space: Target label space ('cityscapes' or 'unified')
            data_root: Root directory of Mapillary Vistas dataset
            img_dir: Subdirectory containing images
            ann_dir: Subdirectory containing annotations
            **kwargs: Additional arguments passed to BaseSegDataset
        """
        
        METAINFO = dict(
            classes=CITYSCAPES_CLASSES_NAMES,
            palette=CITYSCAPES_PALETTE
        )
        
        def __init__(self,
                     target_space: str = 'cityscapes',
                     img_suffix: str = '.jpg',
                     seg_map_suffix: str = '.png',
                     **kwargs):
            self.target_space = target_space
            
            # Set appropriate classes and palette based on target space
            if target_space == 'unified':
                self.METAINFO = dict(
                    classes=UNIFIED_CLASSES_NAMES,
                    palette=UNIFIED_PALETTE
                )
                self.label_mapper = MapillaryToUnified()
            else:
                self.METAINFO = dict(
                    classes=CITYSCAPES_CLASSES_NAMES,
                    palette=CITYSCAPES_PALETTE
                )
                self.label_mapper = MapillarytoCityscapes()
            
            super().__init__(
                img_suffix=img_suffix,
                seg_map_suffix=seg_map_suffix,
                **kwargs
            )
        
        def get_gt_seg_map(self, idx: int) -> np.ndarray:
            """
            Get ground truth segmentation map with label transformation.
            
            Args:
                idx: Index of the sample
                
            Returns:
                Transformed segmentation map
            """
            seg_map = super().get_gt_seg_map(idx)
            
            # Transform Mapillary labels to target space
            transformed_seg_map = self.label_mapper.transform_label(seg_map)
            
            return transformed_seg_map


    @DATASETS.register_module()
    class UnifiedCityscapesDataset(BaseSegDataset):
        """
        Cityscapes dataset with on-the-fly label transformation to unified space.
        
        This dataset loads Cityscapes images and labels, then transforms
        the labels to the unified format for joint training.
        
        Args:
            data_root: Root directory of Cityscapes dataset
            img_dir: Subdirectory containing images
            ann_dir: Subdirectory containing annotations
            **kwargs: Additional arguments passed to BaseSegDataset
        """
        
        METAINFO = dict(
            classes=UNIFIED_CLASSES_NAMES,
            palette=UNIFIED_PALETTE
        )
        
        def __init__(self,
                     img_suffix: str = '_leftImg8bit.png',
                     seg_map_suffix: str = '_gtFine_labelTrainIds.png',
                     **kwargs):
            self.label_mapper = CityscapesToUnified()
            
            super().__init__(
                img_suffix=img_suffix,
                seg_map_suffix=seg_map_suffix,
                **kwargs
            )
        
        def get_gt_seg_map(self, idx: int) -> np.ndarray:
            """
            Get ground truth segmentation map with label transformation.
            
            Args:
                idx: Index of the sample
                
            Returns:
                Transformed segmentation map
            """
            seg_map = super().get_gt_seg_map(idx)
            
            # Transform Cityscapes labels to unified space
            transformed_seg_map = self.label_mapper.transform_label(seg_map)
            
            return transformed_seg_map


    @DATASETS.register_module()
    class JointCityscapesMapillaryDataset(BaseSegDataset):
        """
        Joint Cityscapes + Mapillary Vistas dataset for unified training.
        
        This dataset combines both Cityscapes and Mapillary Vistas datasets
        with automatic label unification. It handles different image sizes
        and provides a unified interface for training.
        
        Args:
            cityscapes_root: Root directory of Cityscapes dataset
            mapillary_root: Root directory of Mapillary Vistas dataset
            target_space: Target label space ('cityscapes' or 'unified')
            cityscapes_split: Cityscapes split ('train', 'val', 'test')
            mapillary_split: Mapillary split ('training', 'validation', 'testing')
            **kwargs: Additional arguments passed to BaseSegDataset
        """
        
        def __init__(self,
                     cityscapes_root: str,
                     mapillary_root: str,
                     target_space: str = 'cityscapes',
                     cityscapes_split: str = 'train',
                     mapillary_split: str = 'training',
                     **kwargs):
            
            self.cityscapes_root = cityscapes_root
            self.mapillary_root = mapillary_root
            self.target_space = target_space
            self.cityscapes_split = cityscapes_split
            self.mapillary_split = mapillary_split
            
            # Set classes and palette based on target space
            if target_space == 'unified':
                self.METAINFO = dict(
                    classes=UNIFIED_CLASSES_NAMES,
                    palette=UNIFIED_PALETTE
                )
                self.cs_mapper = CityscapesToUnified()
                self.mp_mapper = MapillaryToUnified()
            else:
                self.METAINFO = dict(
                    classes=CITYSCAPES_CLASSES_NAMES,
                    palette=CITYSCAPES_PALETTE
                )
                self.cs_mapper = None  # No transformation needed
                self.mp_mapper = MapillarytoCityscapes()
            
            # Initialize data lists
            self.cityscapes_samples = []
            self.mapillary_samples = []
            
            # Load file lists
            self._load_cityscapes_samples()
            self._load_mapillary_samples()
            
            # Combine samples
            self.data_list = self.cityscapes_samples + self.mapillary_samples
            
            super().__init__(**kwargs)
        
        def _load_cityscapes_samples(self):
            """Load Cityscapes sample file paths"""
            img_dir = osp.join(self.cityscapes_root, 'leftImg8bit', self.cityscapes_split)
            ann_dir = osp.join(self.cityscapes_root, 'gtFine', self.cityscapes_split)
            
            if not osp.exists(img_dir):
                return
            
            for city in os.listdir(img_dir):
                city_img_dir = osp.join(img_dir, city)
                city_ann_dir = osp.join(ann_dir, city)
                
                if not osp.isdir(city_img_dir):
                    continue
                
                for img_name in os.listdir(city_img_dir):
                    if img_name.endswith('_leftImg8bit.png'):
                        base_name = img_name.replace('_leftImg8bit.png', '')
                        ann_name = f"{base_name}_gtFine_labelTrainIds.png"
                        
                        sample = {
                            'img_path': osp.join(city_img_dir, img_name),
                            'seg_map_path': osp.join(city_ann_dir, ann_name),
                            'source': 'cityscapes',
                            'reduce_zero_label': False,
                        }
                        self.cityscapes_samples.append(sample)
        
        def _load_mapillary_samples(self):
            """Load Mapillary Vistas sample file paths"""
            img_dir = osp.join(self.mapillary_root, self.mapillary_split, 'images')
            ann_dir = osp.join(self.mapillary_root, self.mapillary_split, 'v1.2', 'labels')
            
            if not osp.exists(img_dir):
                # Try alternative structure
                ann_dir = osp.join(self.mapillary_root, self.mapillary_split, 'labels')
            
            if not osp.exists(img_dir):
                return
            
            for img_name in os.listdir(img_dir):
                if img_name.endswith(('.jpg', '.png')):
                    base_name = osp.splitext(img_name)[0]
                    ann_name = f"{base_name}.png"
                    
                    sample = {
                        'img_path': osp.join(img_dir, img_name),
                        'seg_map_path': osp.join(ann_dir, ann_name),
                        'source': 'mapillary',
                        'reduce_zero_label': False,
                    }
                    self.mapillary_samples.append(sample)
        
        def get_gt_seg_map(self, idx: int) -> np.ndarray:
            """
            Get ground truth segmentation map with appropriate label transformation.
            
            Args:
                idx: Index of the sample
                
            Returns:
                Transformed segmentation map
            """
            sample = self.data_list[idx]
            seg_map = super().get_gt_seg_map(idx)
            
            # Apply appropriate transformation based on source
            if sample['source'] == 'mapillary':
                seg_map = self.mp_mapper.transform_label(seg_map)
            elif sample['source'] == 'cityscapes' and self.cs_mapper is not None:
                seg_map = self.cs_mapper.transform_label(seg_map)
            
            return seg_map
        
        def __len__(self) -> int:
            return len(self.data_list)


    @DATASETS.register_module()
    class Outside15kDataset(BaseSegDataset):
        """
        OUTSIDE15k Dataset with native 24 classes.
        
        This dataset provides proper class metadata for OUTSIDE15k,
        which has 24 native classes (0-23). This is needed when training
        with native OUTSIDE15k labels to ensure correct metric evaluation.
        
        The OUTSIDE15k classes are:
        - unlabeled, animal, barrier, bicycle, boat, bridge,
        - building, grass, ground, mountain, object, person,
        - pole, road, sand, sidewalk, sign, sky, street light,
        - traffic light, tunnel, vegetation, vehicle, water
        
        Args:
            img_suffix: Image file suffix. Defaults to '.jpg'.
            seg_map_suffix: Segmentation map suffix. Defaults to '.png'.
            **kwargs: Additional arguments passed to BaseSegDataset
        """
        
        METAINFO = dict(
            classes=OUTSIDE15K_CLASSES_NAMES,
            palette=OUTSIDE15K_PALETTE
        )
        
        def __init__(self,
                     img_suffix: str = '.jpg',
                     seg_map_suffix: str = '.png',
                     **kwargs) -> None:
            super().__init__(
                img_suffix=img_suffix,
                seg_map_suffix=seg_map_suffix,
                **kwargs
            )


# =============================================================================
# DATA PIPELINE TRANSFORMS FOR LABEL TRANSFORMATION
# =============================================================================

if MMSEG_AVAILABLE:
    from mmseg.registry import TRANSFORMS
    from mmcv.transforms import BaseTransform
    
    @TRANSFORMS.register_module()
    class MapillaryLabelTransform(BaseTransform):
        """
        Transform Mapillary labels to target label space in data pipeline.
        
        This transform can be added to the data pipeline to transform
        Mapillary labels on-the-fly during data loading.
        
        Args:
            target_space: Target label space ('cityscapes' or 'unified')
        """
        
        def __init__(self, target_space: str = 'cityscapes'):
            self.target_space = target_space
            if target_space == 'unified':
                self.mapper = MapillaryToUnified()
            else:
                self.mapper = MapillarytoCityscapes()
        
        def transform(self, results: Dict) -> Dict:
            """Transform the label in results dict"""
            if 'gt_seg_map' in results:
                results['gt_seg_map'] = self.mapper.transform_label(results['gt_seg_map'])
            return results
        
        def __repr__(self) -> str:
            return f'{self.__class__.__name__}(target_space={self.target_space})'


    @TRANSFORMS.register_module()
    class CityscapesLabelTransform(BaseTransform):
        """
        Transform Cityscapes labels to unified label space in data pipeline.
        
        Args:
            target_space: Target label space (only 'unified' supported)
        """
        
        def __init__(self, target_space: str = 'unified'):
            self.target_space = target_space
            self.mapper = CityscapesToUnified()
        
        def transform(self, results: Dict) -> Dict:
            """Transform the label in results dict"""
            if 'gt_seg_map' in results:
                results['gt_seg_map'] = self.mapper.transform_label(results['gt_seg_map'])
            return results
        
        def __repr__(self) -> str:
            return f'{self.__class__.__name__}(target_space={self.target_space})'


# =============================================================================
# CONFIGURATION GENERATORS
# =============================================================================

def generate_joint_training_config(
    cityscapes_root: str,
    mapillary_root: str,
    target_space: str = 'cityscapes',
    crop_size: Tuple[int, int] = (512, 1024),
    batch_size: int = 2,
    workers: int = 4,
) -> Dict:
    """
    Generate a complete configuration for joint Cityscapes + Mapillary training.
    
    Args:
        cityscapes_root: Path to Cityscapes dataset
        mapillary_root: Path to Mapillary Vistas dataset
        target_space: Target label space ('cityscapes' or 'unified')
        crop_size: Crop size for training (height, width)
        batch_size: Batch size per GPU
        workers: Number of data loading workers per GPU
        
    Returns:
        Configuration dictionary compatible with MMSegmentation
    """
    
    if target_space == 'unified':
        classes = UNIFIED_CLASSES_NAMES
        palette = UNIFIED_PALETTE
        num_classes = len(UNIFIED_CLASSES_NAMES)
    else:
        classes = CITYSCAPES_CLASSES_NAMES
        palette = CITYSCAPES_PALETTE
        num_classes = len(CITYSCAPES_CLASSES_NAMES)
    
    # Image normalization config
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True
    )
    
    # Training pipeline
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(
            type='RandomResize',
            scale=(2048, 1024),
            ratio_range=(0.5, 2.0),
            keep_ratio=True
        ),
        dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
        dict(type='PackSegInputs'),
    ]
    
    # Add label transform for Mapillary
    mapillary_train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='MapillaryLabelTransform', target_space=target_space),
        dict(
            type='RandomResize',
            scale=(2048, 1024),
            ratio_range=(0.5, 2.0),
            keep_ratio=True
        ),
        dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
        dict(type='PackSegInputs'),
    ]
    
    # Cityscapes pipeline (with transform if using unified space)
    if target_space == 'unified':
        cityscapes_train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='CityscapesLabelTransform', target_space='unified'),
            dict(
                type='RandomResize',
                scale=(2048, 1024),
                ratio_range=(0.5, 2.0),
                keep_ratio=True
            ),
            dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type='PackSegInputs'),
        ]
    else:
        cityscapes_train_pipeline = train_pipeline
    
    # Validation pipeline
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
        dict(type='LoadAnnotations'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='PackSegInputs'),
    ]
    
    config = {
        # Dataset metadata
        'dataset_type': 'ConcatDataset',
        'num_classes': num_classes,
        'classes': classes,
        'palette': palette,
        
        # Data configuration
        'data': dict(
            samples_per_gpu=batch_size,
            workers_per_gpu=workers,
            
            train=dict(
                type='ConcatDataset',
                datasets=[
                    # Cityscapes training set
                    dict(
                        type='CityscapesDataset' if target_space == 'cityscapes' else 'UnifiedCityscapesDataset',
                        data_root=cityscapes_root,
                        img_dir='leftImg8bit/train',
                        ann_dir='gtFine/train',
                        pipeline=cityscapes_train_pipeline,
                    ),
                    # Mapillary training set
                    dict(
                        type='MapillaryUnifiedDataset',
                        data_root=mapillary_root,
                        img_dir='training/images',
                        ann_dir='training/v1.2/labels',
                        target_space=target_space,
                        pipeline=mapillary_train_pipeline,
                    ),
                ],
            ),
            
            val=dict(
                type='CityscapesDataset',
                data_root=cityscapes_root,
                img_dir='leftImg8bit/val',
                ann_dir='gtFine/val',
                pipeline=test_pipeline,
            ),
            
            test=dict(
                type='CityscapesDataset',
                data_root=cityscapes_root,
                img_dir='leftImg8bit/val',
                ann_dir='gtFine/val',
                pipeline=test_pipeline,
            ),
        ),
        
        # Training configuration
        'train_pipeline': train_pipeline,
        'test_pipeline': test_pipeline,
        'img_norm_cfg': img_norm_cfg,
    }
    
    return config


def generate_model_config(
    model_name: str = 'deeplabv3plus',
    backbone: str = 'r50',
    num_classes: int = 19,
) -> Dict:
    """
    Generate model configuration for segmentation.
    
    Args:
        model_name: Model architecture ('deeplabv3plus', 'pspnet', 'segformer', 'hrnet', 'segnext')
        backbone: Backbone network ('r50', 'r101', 'mit-b3', 'hr48', 'mscan-b')
        num_classes: Number of output classes
        
    Returns:
        Model configuration dictionary
    """
    
    if model_name == 'deeplabv3plus':
        model_config = dict(
            type='EncoderDecoder',
            pretrained='open-mmlab://resnet50_v1c',
            backbone=dict(
                type='ResNetV1c',
                depth=50 if backbone == 'r50' else 101,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                dilations=(1, 1, 2, 4),
                strides=(1, 2, 1, 1),
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True,
            ),
            decode_head=dict(
                type='DepthwiseSeparableASPPHead',
                in_channels=2048,
                in_index=3,
                channels=512,
                dilations=(1, 12, 24, 36),
                c1_in_channels=256,
                c1_channels=48,
                dropout_ratio=0.1,
                num_classes=num_classes,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0,
                    ignore_index=255,
                ),
            ),
            auxiliary_head=dict(
                type='FCNHead',
                in_channels=1024,
                in_index=2,
                channels=256,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=num_classes,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=0.4,
                    ignore_index=255,
                ),
            ),
            train_cfg=dict(),
            test_cfg=dict(mode='whole'),
        )
    else:
        # Default to PSPNet
        model_config = dict(
            type='EncoderDecoder',
            pretrained='open-mmlab://resnet50_v1c',
            backbone=dict(
                type='ResNetV1c',
                depth=50,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                dilations=(1, 1, 2, 4),
                strides=(1, 2, 1, 1),
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True,
            ),
            decode_head=dict(
                type='PSPHead',
                in_channels=2048,
                in_index=3,
                channels=512,
                pool_scales=(1, 2, 3, 6),
                dropout_ratio=0.1,
                num_classes=num_classes,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0,
                    ignore_index=255,
                ),
            ),
            auxiliary_head=dict(
                type='FCNHead',
                in_channels=1024,
                in_index=2,
                channels=256,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=num_classes,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=0.4,
                    ignore_index=255,
                ),
            ),
            train_cfg=dict(),
            test_cfg=dict(mode='whole'),
        )
    
    return model_config


# =============================================================================
# MAIN / EXAMPLE USAGE
# =============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("UNIFIED DATASET CLASSES FOR PROVE PIPELINE")
    print("=" * 80)
    
    print("\n--- Available Dataset Classes ---")
    print("1. MapillaryUnifiedDataset: Mapillary Vistas with label transformation")
    print("2. UnifiedCityscapesDataset: Cityscapes with unified label transformation")
    print("3. JointCityscapesMapillaryDataset: Combined dataset for joint training")
    
    print("\n--- Available Transforms ---")
    print("1. MapillaryLabelTransform: Transform Mapillary labels in pipeline")
    print("2. CityscapesLabelTransform: Transform Cityscapes labels to unified")
    
    print("\n--- Example Configuration Generation ---")
    
    # Generate example config
    config = generate_joint_training_config(
        cityscapes_root='./data/cityscapes',
        mapillary_root='./data/mapillary_vistas',
        target_space='cityscapes',
        crop_size=(512, 1024),
        batch_size=2,
    )
    
    print(f"\nGenerated config with {config['num_classes']} classes")
    print(f"Classes: {config['classes'][:5]}...")
    
    print("\n--- Label Space Options ---")
    print("1. 'cityscapes': Use standard Cityscapes 19-class format")
    print("   - Best for benchmarking on Cityscapes test set")
    print("   - Maps Mapillary 66 classes to Cityscapes 19 classes")
    print()
    print("2. 'unified': Use extended 42-class unified format")
    print("   - Best for training with maximum label granularity")
    print("   - Preserves more information from both datasets")
    print("   - Can map back to Cityscapes for evaluation")
