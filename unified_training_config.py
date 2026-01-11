#!/usr/bin/env python3
"""
PROVE Unified Training Configuration System

This module provides a centralized configuration system that eliminates
redundant config files by parameterizing:
- Base model (deeplabv3plus_r50, pspnet_r50, segformer_mit-b5, etc.)
- Dataset (ACDC, BDD10k, BDD100k, IDD-AW, MapillaryVistas, OUTSIDE15k)
- Augmentation strategy (baseline, photometric_distort, gen_<model>)
- Real-to-generated image ratio for mixed training

Usage:
    # Command line
    python unified_training_config.py --dataset ACDC --model deeplabv3plus_r50 \
        --strategy gen_cycleGAN --real-gen-ratio 0.5
    
    # As module
    from unified_training_config import UnifiedTrainingConfig
    config = UnifiedTrainingConfig()
    cfg = config.build(dataset='ACDC', model='deeplabv3plus_r50', 
                       strategy='gen_cycleGAN', real_gen_ratio=0.5)
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

# Suppress MMSegmentation deprecation warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='mmseg')

# Register custom transforms and metrics for handling 3-channel labels and fwIoU
try:
    import numpy as np
    from mmcv.transforms import BaseTransform
    from mmseg.registry import TRANSFORMS, METRICS
    from mmseg.evaluation.metrics import IoUMetric
    
    # Import custom transforms - they will be registered with force=True
    # This is the canonical source for all label transforms
    import custom_transforms
    
    # The following are registered by custom_transforms:
    # - ReduceToSingleChannel (force=True)
    # - CityscapesLabelIdToTrainId
    # - MapillaryRGBToClassId
    # - MapillaryNativeLabelClamp
    # - Outside15kNativeLabelClamp
    # - FWIoUMetric
            
except ImportError:
    # MMSeg not installed yet, skip registration (will be registered when imported during training)
    pass


# ============================================================================
# Constants
# ============================================================================

# Base paths - can be overridden via environment variables
DEFAULT_DATA_ROOT = os.environ.get('PROVE_DATA_ROOT', '/scratch/aaa_exchange/AWARE/FINAL_SPLITS')
DEFAULT_GEN_ROOT = os.environ.get('PROVE_GEN_ROOT', '/scratch/aaa_exchange/AWARE/GENERATED_IMAGES')
DEFAULT_WEIGHTS_ROOT = os.environ.get('PROVE_WEIGHTS_ROOT', '/scratch/aaa_exchange/AWARE/WEIGHTS')
DEFAULT_CONFIG_ROOT = os.environ.get('PROVE_CONFIG_ROOT', './multi_model_configs')

# Adverse weather conditions
ADVERSE_CONDITIONS = ['cloudy', 'dawn_dusk', 'fog', 'night', 'rainy', 'snowy']

# Available generative models (directory names in GENERATED_IMAGES)
# These are the actual directory names - some have hyphens
GENERATIVE_MODEL_DIRS = [
    'albumentations_weather', 'AOD-Net', 'Attribute_Hallucination', 'augmenters',
    'automold', 'CNetSeg', 'CUT', 'cyclediffusion', 'cycleGAN', 'EDICT',
    'flux2', 'flux_kontext', 'Img2Img', 'IP2P', 'LANIT', 'NST',
    'Qwen-Image-Edit', 'stargan_v2', 'step1x_new', 'step1x_v1p2', 'StyleID',
    'SUSTechGAN', 'TSIT', 'tunit', 'UniControl', 'VisualCloze',
    'Weather_Effect_Generator'
]

# Mapping from strategy name (with underscores) to directory name (with hyphens)
# Strategy names are used in shell scripts and need to be bash-compatible
def _strategy_name_to_dir(name: str) -> str:
    """Convert strategy-friendly name to actual directory name"""
    # Map underscores back to hyphens for specific directories
    hyphen_dirs = {'Qwen_Image_Edit': 'Qwen-Image-Edit'}
    return hyphen_dirs.get(name, name)

def _dir_to_strategy_name(dir_name: str) -> str:
    """Convert directory name to strategy-friendly name (bash-compatible)"""
    return dir_name.replace('-', '_')

# Strategy-friendly names (for use in bash scripts)
GENERATIVE_MODELS = [_dir_to_strategy_name(d) for d in GENERATIVE_MODEL_DIRS]

DATA_RESTORATION_MODELS = [
    'maxim', 'MPRNet', 'weatherformer', '2stageMultipleAdverseWeatherRemoval',
]


# ============================================================================
# Dataset Configurations
# ============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a specific dataset"""
    name: str
    task: str  # 'segmentation' or 'detection'
    format: str
    data_root: str = DEFAULT_DATA_ROOT
    ann_root: str = ''  # Optional: separate root for annotations (if different from data_root)
    train_img_dir: str = ''
    train_ann_dir: str = ''
    val_img_dir: str = ''
    val_ann_dir: str = ''
    test_img_dir: str = ''
    test_ann_dir: str = ''
    num_classes: int = 19
    classes: tuple = field(default_factory=tuple)
    img_suffix: str = '.png'  # Image file extension (labels always use .png)
    

# Cityscapes-style classes (used by most segmentation datasets)
CITYSCAPES_CLASSES = (
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle',
)

# OUTSIDE15k native classes (24 classes, 0-23)
# From tools/outside15k.json
OUTSIDE15K_CLASSES = (
    'unlabeled', 'animal', 'barrier', 'bicycle', 'boat', 'bridge',
    'building', 'grass', 'ground', 'mountain', 'object', 'person',
    'pole', 'road', 'sand', 'sidewalk', 'sign', 'sky', 'street light',
    'traffic light', 'tunnel', 'vegetation', 'vehicle', 'water',
)

# MapillaryVistas native classes (66 classes, 0-65)
# From label_unification.py MapillaryClass definitions
MAPILLARY_CLASSES = (
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
)

# Detection classes for BDD100k
BDD100K_DET_CLASSES = (
    'pedestrian', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle', 'traffic light', 'traffic sign',
)


# Updated DATASET_CONFIGS matching actual data structure at:
# /scratch/aaa_exchange/AWARE/FINAL_SPLITS/{train,test}/{images,labels}/{DATASET}/{condition}/
DATASET_CONFIGS = {
    'ACDC': DatasetConfig(
        name='ACDC',
        task='segmentation',
        format='cityscapes',
        train_img_dir='train/images/ACDC',
        train_ann_dir='train/labels/ACDC',
        val_img_dir='test/images/ACDC',
        val_ann_dir='test/labels/ACDC',
        test_img_dir='test/images/ACDC',
        test_ann_dir='test/labels/ACDC',
        num_classes=19,
        classes=CITYSCAPES_CLASSES,
        img_suffix='.png',  # ACDC images are PNG
    ),
    'BDD10k': DatasetConfig(
        name='BDD10k',
        task='segmentation',
        format='cityscapes',
        train_img_dir='train/images/BDD10k',
        train_ann_dir='train/labels/BDD10k',
        val_img_dir='test/images/BDD10k',
        val_ann_dir='test/labels/BDD10k',
        test_img_dir='test/images/BDD10k',
        test_ann_dir='test/labels/BDD10k',
        num_classes=19,
        classes=CITYSCAPES_CLASSES,
        img_suffix='.jpg',  # BDD10k images are JPEG
    ),
    'BDD100k': DatasetConfig(
        name='BDD100k',
        task='detection',
        format='bdd100k_json',
        train_img_dir='train/images/BDD100k',
        train_ann_dir='train/labels/BDD100k',
        val_img_dir='test/images/BDD100k',
        val_ann_dir='test/labels/BDD100k',
        test_img_dir='test/images/BDD100k',
        test_ann_dir='test/labels/BDD100k',
        num_classes=10,
        classes=BDD100K_DET_CLASSES,
        img_suffix='.jpg',  # BDD100k images are JPEG
    ),
    'IDD-AW': DatasetConfig(
        name='IDD-AW',
        task='segmentation',
        format='cityscapes',
        # IDD-AW labels are in FINAL_SPLITS/train/labels/IDD-AW (already in trainId format)
        train_img_dir='train/images/IDD-AW',
        train_ann_dir='train/labels/IDD-AW',
        val_img_dir='test/images/IDD-AW',
        val_ann_dir='test/labels/IDD-AW',
        test_img_dir='test/images/IDD-AW',
        test_ann_dir='test/labels/IDD-AW',
        num_classes=19,
        classes=CITYSCAPES_CLASSES,
        img_suffix='.png',  # IDD-AW images are PNG
    ),
    'MapillaryVistas': DatasetConfig(
        name='MapillaryVistas',
        task='segmentation',
        format='cityscapes',
        train_img_dir='train/images/MapillaryVistas',
        train_ann_dir='train/labels/MapillaryVistas',
        val_img_dir='test/images/MapillaryVistas',
        val_ann_dir='test/labels/MapillaryVistas',
        test_img_dir='test/images/MapillaryVistas',
        test_ann_dir='test/labels/MapillaryVistas',
        num_classes=19,  # Cityscapes 19 classes (for cross-dataset evaluation)
        classes=CITYSCAPES_CLASSES,  # Cityscapes class names
        img_suffix='.jpg',  # MapillaryVistas images are JPEG
    ),
    'OUTSIDE15k': DatasetConfig(
        name='OUTSIDE15k',
        task='segmentation',
        format='cityscapes',
        train_img_dir='train/images/OUTSIDE15k',
        train_ann_dir='train/labels/OUTSIDE15k',
        val_img_dir='test/images/OUTSIDE15k',
        val_ann_dir='test/labels/OUTSIDE15k',
        test_img_dir='test/images/OUTSIDE15k',
        test_ann_dir='test/labels/OUTSIDE15k',
        num_classes=19,  # Cityscapes 19 classes (for cross-dataset evaluation)
        classes=CITYSCAPES_CLASSES,  # Cityscapes class names
        img_suffix='.jpg',  # OUTSIDE15k images are JPEG
    ),
}


# ============================================================================
# Model Configurations
# ============================================================================

# Full model definitions for MMEngine/MMSeg
MODEL_DEFINITIONS = {
    'deeplabv3plus_r50': {
        'type': 'EncoderDecoder',
        'data_preprocessor': {
            'type': 'SegDataPreProcessor',
            'mean': [123.675, 116.28, 103.53],
            'std': [58.395, 57.12, 57.375],
            'bgr_to_rgb': True,
            'pad_val': 0,
            'seg_pad_val': 255,
            'size': (512, 512),
        },
        'backbone': {
            'type': 'ResNetV1c',
            'depth': 50,
            'num_stages': 4,
            'out_indices': (0, 1, 2, 3),
            'dilations': (1, 1, 2, 4),
            'strides': (1, 2, 1, 1),
            'norm_cfg': {'type': 'SyncBN', 'requires_grad': True},
            'norm_eval': False,
            'style': 'pytorch',
            'contract_dilation': True,
            'init_cfg': {'type': 'Pretrained', 'checkpoint': 'open-mmlab://resnet50_v1c'},
        },
        'decode_head': {
            'type': 'DepthwiseSeparableASPPHead',
            'in_channels': 2048,
            'in_index': 3,
            'channels': 512,
            'dilations': (1, 12, 24, 36),
            'c1_in_channels': 256,
            'c1_channels': 48,
            'dropout_ratio': 0.1,
            'num_classes': 19,
            'norm_cfg': {'type': 'SyncBN', 'requires_grad': True},
            'align_corners': False,
            'loss_decode': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0, 'avg_non_ignore': True},
        },
        'auxiliary_head': {
            'type': 'FCNHead',
            'in_channels': 1024,
            'in_index': 2,
            'channels': 256,
            'num_convs': 1,
            'concat_input': False,
            'dropout_ratio': 0.1,
            'num_classes': 19,
            'norm_cfg': {'type': 'SyncBN', 'requires_grad': True},
            'align_corners': False,
            'loss_decode': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 0.4, 'avg_non_ignore': True},
        },
        'train_cfg': {},
        'test_cfg': {'mode': 'whole'},
    },
    'pspnet_r50': {
        'type': 'EncoderDecoder',
        'data_preprocessor': {
            'type': 'SegDataPreProcessor',
            'mean': [123.675, 116.28, 103.53],
            'std': [58.395, 57.12, 57.375],
            'bgr_to_rgb': True,
            'pad_val': 0,
            'seg_pad_val': 255,
            'size': (512, 512),
        },
        'backbone': {
            'type': 'ResNetV1c',
            'depth': 50,
            'num_stages': 4,
            'out_indices': (0, 1, 2, 3),
            'dilations': (1, 1, 2, 4),
            'strides': (1, 2, 1, 1),
            'norm_cfg': {'type': 'SyncBN', 'requires_grad': True},
            'norm_eval': False,
            'style': 'pytorch',
            'contract_dilation': True,
            'init_cfg': {'type': 'Pretrained', 'checkpoint': 'open-mmlab://resnet50_v1c'},
        },
        'decode_head': {
            'type': 'PSPHead',
            'in_channels': 2048,
            'in_index': 3,
            'channels': 512,
            'pool_scales': (1, 2, 3, 6),
            'dropout_ratio': 0.1,
            'num_classes': 19,
            'norm_cfg': {'type': 'SyncBN', 'requires_grad': True},
            'align_corners': False,
            'loss_decode': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0, 'avg_non_ignore': True},
        },
        'auxiliary_head': {
            'type': 'FCNHead',
            'in_channels': 1024,
            'in_index': 2,
            'channels': 256,
            'num_convs': 1,
            'concat_input': False,
            'dropout_ratio': 0.1,
            'num_classes': 19,
            'norm_cfg': {'type': 'SyncBN', 'requires_grad': True},
            'align_corners': False,
            'loss_decode': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 0.4, 'avg_non_ignore': True},
        },
        'train_cfg': {},
        'test_cfg': {'mode': 'whole'},
    },
    'segformer_mit-b5': {
        'type': 'EncoderDecoder',
        'data_preprocessor': {
            'type': 'SegDataPreProcessor',
            'mean': [123.675, 116.28, 103.53],
            'std': [58.395, 57.12, 57.375],
            'bgr_to_rgb': True,
            'pad_val': 0,
            'seg_pad_val': 255,
            'size': (512, 512),
        },
        'backbone': {
            'type': 'MixVisionTransformer',
            'in_channels': 3,
            'embed_dims': 64,
            'num_stages': 4,
            'num_layers': [3, 6, 40, 3],
            'num_heads': [1, 2, 5, 8],
            'patch_sizes': [7, 3, 3, 3],
            'sr_ratios': [8, 4, 2, 1],
            'out_indices': (0, 1, 2, 3),
            'mlp_ratio': 4,
            'qkv_bias': True,
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
            'drop_path_rate': 0.1,
            'init_cfg': {
                'type': 'Pretrained',
                'checkpoint': 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth',
            },
        },
        'decode_head': {
            'type': 'SegformerHead',
            'in_channels': [64, 128, 320, 512],
            'in_index': [0, 1, 2, 3],
            'channels': 256,
            'dropout_ratio': 0.1,
            'num_classes': 19,
            'norm_cfg': {'type': 'SyncBN', 'requires_grad': True},
            'align_corners': False,
            'loss_decode': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0, 'avg_non_ignore': True},
        },
        'train_cfg': {},
        'test_cfg': {'mode': 'whole'},
    },
    'faster_rcnn_r50_fpn_1x': {
        'type': 'FasterRCNN',
        'data_preprocessor': {
            'type': 'DetDataPreprocessor',
            'mean': [123.675, 116.28, 103.53],
            'std': [58.395, 57.12, 57.375],
            'bgr_to_rgb': True,
            'pad_size_divisor': 32,
        },
        'backbone': {
            'type': 'ResNet',
            'depth': 50,
            'num_stages': 4,
            'out_indices': (0, 1, 2, 3),
            'frozen_stages': 1,
            'norm_cfg': {'type': 'BN', 'requires_grad': True},
            'norm_eval': True,
            'style': 'pytorch',
            'init_cfg': {'type': 'Pretrained', 'checkpoint': 'torchvision://resnet50'},
        },
        'neck': {
            'type': 'FPN',
            'in_channels': [256, 512, 1024, 2048],
            'out_channels': 256,
            'num_outs': 5,
        },
        'rpn_head': {
            'type': 'RPNHead',
            'in_channels': 256,
            'feat_channels': 256,
            'anchor_generator': {
                'type': 'AnchorGenerator',
                'scales': [8],
                'ratios': [0.5, 1.0, 2.0],
                'strides': [4, 8, 16, 32, 64],
            },
            'bbox_coder': {
                'type': 'DeltaXYWHBBoxCoder',
                'target_means': [0.0, 0.0, 0.0, 0.0],
                'target_stds': [1.0, 1.0, 1.0, 1.0],
            },
            'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 1.0},
            'loss_bbox': {'type': 'L1Loss', 'loss_weight': 1.0},
        },
        'roi_head': {
            'type': 'StandardRoIHead',
            'bbox_roi_extractor': {
                'type': 'SingleRoIExtractor',
                'roi_layer': {'type': 'RoIAlign', 'output_size': 7, 'sampling_ratio': 0},
                'out_channels': 256,
                'featmap_strides': [4, 8, 16, 32],
            },
            'bbox_head': {
                'type': 'Shared2FCBBoxHead',
                'in_channels': 256,
                'fc_out_channels': 1024,
                'roi_feat_size': 7,
                'num_classes': 10,
                'bbox_coder': {
                    'type': 'DeltaXYWHBBoxCoder',
                    'target_means': [0.0, 0.0, 0.0, 0.0],
                    'target_stds': [0.1, 0.1, 0.2, 0.2],
                },
                'reg_class_agnostic': False,
                'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0},
                'loss_bbox': {'type': 'L1Loss', 'loss_weight': 1.0},
            },
        },
        'train_cfg': {
            'rpn': {
                'assigner': {
                    'type': 'MaxIoUAssigner',
                    'pos_iou_thr': 0.7,
                    'neg_iou_thr': 0.3,
                    'min_pos_iou': 0.3,
                    'match_low_quality': True,
                    'ignore_iof_thr': -1,
                },
                'sampler': {
                    'type': 'RandomSampler',
                    'num': 256,
                    'pos_fraction': 0.5,
                    'neg_pos_ub': -1,
                    'add_gt_as_proposals': False,
                },
                'allowed_border': -1,
                'pos_weight': -1,
                'debug': False,
            },
            'rpn_proposal': {
                'nms_pre': 2000,
                'max_per_img': 1000,
                'nms': {'type': 'nms', 'iou_threshold': 0.7},
                'min_bbox_size': 0,
            },
            'rcnn': {
                'assigner': {
                    'type': 'MaxIoUAssigner',
                    'pos_iou_thr': 0.5,
                    'neg_iou_thr': 0.5,
                    'min_pos_iou': 0.5,
                    'match_low_quality': False,
                    'ignore_iof_thr': -1,
                },
                'sampler': {
                    'type': 'RandomSampler',
                    'num': 512,
                    'pos_fraction': 0.25,
                    'neg_pos_ub': -1,
                    'add_gt_as_proposals': True,
                },
                'pos_weight': -1,
                'debug': False,
            },
        },
        'test_cfg': {
            'rpn': {
                'nms_pre': 1000,
                'max_per_img': 1000,
                'nms': {'type': 'nms', 'iou_threshold': 0.7},
                'min_bbox_size': 0,
            },
            'rcnn': {
                'score_thr': 0.05,
                'nms': {'type': 'nms', 'iou_threshold': 0.5},
                'max_per_img': 100,
            },
        },
    },
    'yolox_l': {
        'type': 'YOLOX',
        'data_preprocessor': {
            'type': 'DetDataPreprocessor',
            'pad_size_divisor': 32,
            'batch_augments': [
                {'type': 'BatchSyncRandomResize', 'random_size_range': (480, 800), 'size_divisor': 32, 'interval': 10}
            ],
        },
        'backbone': {
            'type': 'CSPDarknet',
            'deepen_factor': 1.0,
            'widen_factor': 1.0,
            'out_indices': (2, 3, 4),
            'use_depthwise': False,
            'spp_kernal_sizes': (5, 9, 13),
            'norm_cfg': {'type': 'BN', 'momentum': 0.03, 'eps': 0.001},
            'act_cfg': {'type': 'SiLU', 'inplace': True},
        },
        'neck': {
            'type': 'YOLOXPAFPN',
            'in_channels': [256, 512, 1024],
            'out_channels': 256,
            'num_csp_blocks': 3,
            'use_depthwise': False,
            'upsample_cfg': {'scale_factor': 2, 'mode': 'nearest'},
            'norm_cfg': {'type': 'BN', 'momentum': 0.03, 'eps': 0.001},
            'act_cfg': {'type': 'SiLU', 'inplace': True},
        },
        'bbox_head': {
            'type': 'YOLOXHead',
            'num_classes': 10,
            'in_channels': 256,
            'feat_channels': 256,
            'stacked_convs': 2,
            'strides': (8, 16, 32),
            'use_depthwise': False,
            'norm_cfg': {'type': 'BN', 'momentum': 0.03, 'eps': 0.001},
            'act_cfg': {'type': 'SiLU', 'inplace': True},
            'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'reduction': 'sum', 'loss_weight': 1.0},
            'loss_bbox': {'type': 'IoULoss', 'mode': 'square', 'eps': 1e-16, 'reduction': 'sum', 'loss_weight': 5.0},
            'loss_obj': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'reduction': 'sum', 'loss_weight': 1.0},
            'loss_l1': {'type': 'L1Loss', 'reduction': 'sum', 'loss_weight': 1.0},
        },
        'train_cfg': {'assigner': {'type': 'SimOTAAssigner', 'center_radius': 2.5}},
        'test_cfg': {'score_thr': 0.01, 'nms': {'type': 'nms', 'iou_threshold': 0.65}},
    },
    'rtmdet_l': {
        'type': 'RTMDet',
        'data_preprocessor': {
            'type': 'DetDataPreprocessor',
            'mean': [103.53, 116.28, 123.675],
            'std': [57.375, 57.12, 58.395],
            'bgr_to_rgb': False,
            'batch_augments': None,
        },
        'backbone': {
            'type': 'CSPNeXt',
            'arch': 'P5',
            'expand_ratio': 0.5,
            'deepen_factor': 1.0,
            'widen_factor': 1.0,
            'channel_attention': True,
            'norm_cfg': {'type': 'SyncBN'},
            'act_cfg': {'type': 'SiLU', 'inplace': True},
        },
        'neck': {
            'type': 'CSPNeXtPAFPN',
            'in_channels': [256, 512, 1024],
            'out_channels': 256,
            'num_csp_blocks': 3,
            'expand_ratio': 0.5,
            'norm_cfg': {'type': 'SyncBN'},
            'act_cfg': {'type': 'SiLU', 'inplace': True},
        },
        'bbox_head': {
            'type': 'RTMDetSepBNHead',
            'num_classes': 10,
            'in_channels': 256,
            'stacked_convs': 2,
            'feat_channels': 256,
            'anchor_generator': {
                'type': 'MlvlPointGenerator',
                'offset': 0,
                'strides': [8, 16, 32],
            },
            'bbox_coder': {'type': 'DistancePointBBoxCoder'},
            'loss_cls': {'type': 'QualityFocalLoss', 'use_sigmoid': True, 'beta': 2.0, 'loss_weight': 1.0},
            'loss_bbox': {'type': 'GIoULoss', 'loss_weight': 2.0},
            'with_objectness': False,
            'exp_on_reg': True,
            'share_conv': True,
            'pred_kernel_size': 1,
            'norm_cfg': {'type': 'SyncBN'},
            'act_cfg': {'type': 'SiLU', 'inplace': True},
        },
        'train_cfg': {
            'assigner': {'type': 'DynamicSoftLabelAssigner', 'topk': 13},
            'allowed_border': -1,
            'pos_weight': -1,
            'debug': False,
        },
        'test_cfg': {
            'nms_pre': 30000,
            'min_bbox_size': 0,
            'score_thr': 0.001,
            'nms': {'type': 'nms', 'iou_threshold': 0.65},
            'max_per_img': 300,
        },
    },
}


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    name: str
    task: str  # 'segmentation' or 'detection'
    base_config: str  # Path to base model config (legacy, kept for reference)
    optimizer: str = 'SGD'
    lr: float = 0.01
    weight_decay: float = 0.0005


SEGMENTATION_MODELS = {
    'deeplabv3plus_r50': ModelConfig(
        name='deeplabv3plus_r50',
        task='segmentation',
        base_config='_base_/models/deeplabv3plus_r50-d8.py',
        optimizer='SGD',
        lr=0.01,
        weight_decay=0.0005,
    ),
    'pspnet_r50': ModelConfig(
        name='pspnet_r50',
        task='segmentation',
        base_config='_base_/models/pspnet_r50-d8.py',
        optimizer='SGD',
        lr=0.01,
        weight_decay=0.0005,
    ),
    'segformer_mit-b5': ModelConfig(
        name='segformer_mit-b5',
        task='segmentation',
        base_config='_base_/models/segformer_mit-b5.py',
        optimizer='AdamW',
        lr=0.00006,
        weight_decay=0.01,
    ),
}

DETECTION_MODELS = {
    'faster_rcnn_r50_fpn_1x': ModelConfig(
        name='faster_rcnn_r50_fpn_1x',
        task='detection',
        base_config='_base_/models/faster_rcnn_r50_fpn.py',
        optimizer='SGD',
        lr=0.02,
        weight_decay=0.0001,
    ),
    'yolox_l': ModelConfig(
        name='yolox_l',
        task='detection',
        base_config='_base_/models/yolox_l.py',
        optimizer='SGD',
        lr=0.01,
        weight_decay=0.0005,
    ),
    'rtmdet_l': ModelConfig(
        name='rtmdet_l',
        task='detection',
        base_config='_base_/models/rtmdet_l.py',
        optimizer='AdamW',
        lr=0.004,
        weight_decay=0.05,
    ),
}

ALL_MODELS = {**SEGMENTATION_MODELS, **DETECTION_MODELS}


# ============================================================================
# Training Configuration Templates
# ============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    max_iters: int = 80000
    batch_size: int = 2
    workers_per_gpu: int = 4
    checkpoint_interval: int = 10000
    eval_interval: int = 6666
    log_interval: int = 50
    warmup_iters: int = 500
    warmup_ratio: float = 0.001
    seed: int = 42
    deterministic: bool = True
    # Early stopping configuration
    early_stop: bool = True
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.001


TRAINING_CONFIGS = {
    'segmentation': TrainingConfig(
        max_iters=80000,
        batch_size=2,
        checkpoint_interval=10000,
        eval_interval=6666,
        early_stop=True,
        early_stop_patience=5,
        early_stop_min_delta=0.001,  # mIoU improvement threshold
    ),
    'detection': TrainingConfig(
        max_iters=40000,
        batch_size=2,
        checkpoint_interval=5000,
        eval_interval=3333,
        early_stop=True,
        early_stop_patience=5,
        early_stop_min_delta=0.001,  # mAP improvement threshold
    ),
}


# ============================================================================
# Augmentation Strategies
# ============================================================================

@dataclass
class AugmentationStrategy:
    """Configuration for an augmentation strategy"""
    name: str
    type: str  # 'none', 'transform', 'generated', 'standard'
    transforms: List[dict] = field(default_factory=list)
    generative_model: Optional[str] = None
    conditions: List[str] = field(default_factory=lambda: ADVERSE_CONDITIONS.copy())
    # Standard augmentation parameters
    standard_method: Optional[str] = None  # 'cutmix', 'mixup', 'autoaugment', 'randaugment'
    p_aug: float = 0.5  # Probability of applying augmentation
    
    def get_pipeline_transforms(self) -> List[dict]:
        """Get the transforms for this strategy"""
        return self.transforms.copy()


AUGMENTATION_STRATEGIES = {
    'baseline': AugmentationStrategy(
        name='baseline',
        type='none',
        transforms=[],
    ),
    'photometric_distort': AugmentationStrategy(
        name='photometric_distort',
        type='transform',
        transforms=[dict(type='PhotoMetricDistortion')],
    ),
}

# Add generative model strategies dynamically
# gen_model is the strategy-friendly name (bash-compatible, underscores)
# The actual directory lookup is done via _strategy_name_to_dir() when constructing paths
for gen_model in GENERATIVE_MODELS:
    AUGMENTATION_STRATEGIES[f'gen_{gen_model}'] = AugmentationStrategy(
        name=f'gen_{gen_model}',
        type='generated',
        transforms=[],
        generative_model=gen_model,  # Strategy-friendly name, converted to dir when needed
    )

# Add standard augmentation strategies (SOTA baselines)
# Reference: tools/standard_augmentations.py
STANDARD_AUGMENTATION_METHODS = ['cutmix', 'mixup', 'autoaugment', 'randaugment']

AUGMENTATION_STRATEGIES['std_cutmix'] = AugmentationStrategy(
    name='std_cutmix',
    type='standard',
    transforms=[],
    standard_method='cutmix',
    p_aug=0.5,
)

AUGMENTATION_STRATEGIES['std_mixup'] = AugmentationStrategy(
    name='std_mixup',
    type='standard',
    transforms=[],
    standard_method='mixup',
    p_aug=0.5,
)

AUGMENTATION_STRATEGIES['std_autoaugment'] = AugmentationStrategy(
    name='std_autoaugment',
    type='standard',
    transforms=[],
    standard_method='autoaugment',
    p_aug=0.5,
)

AUGMENTATION_STRATEGIES['std_randaugment'] = AugmentationStrategy(
    name='std_randaugment',
    type='standard',
    transforms=[],
    standard_method='randaugment',
    p_aug=0.5,
)

# ============================================================================
# Unified Training Configuration Builder
# ============================================================================

class UnifiedTrainingConfig:
    """
    Unified configuration builder for PROVE training pipeline.
    
    This class generates complete training configurations from high-level
    parameters, eliminating the need for separate config files for each
    dataset/model/strategy combination.
    
    Args:
        data_root: Root directory for training data. Default: from PROVE_DATA_ROOT env
        gen_root: Root directory for generated images. Default: from PROVE_GEN_ROOT env
        weights_root: Root directory for saving model weights. Default: from PROVE_WEIGHTS_ROOT env
        cache_dir: Directory for caching pretrained weights. When specified:
            - Sets TORCH_HOME environment variable to redirect PyTorch/MMEngine downloads
            - Updates relative pretrained paths to use cache_dir
            - Creates necessary subdirectories automatically
    
    Example:
        >>> config_builder = UnifiedTrainingConfig(cache_dir='/data/pretrained')
        >>> config = config_builder.build(dataset='ACDC', model='deeplabv3plus_r50')
        >>> # Pretrained weights will be stored in /data/pretrained/
    """
    
    def __init__(
        self,
        data_root: str = DEFAULT_DATA_ROOT,
        gen_root: str = DEFAULT_GEN_ROOT,
        weights_root: str = DEFAULT_WEIGHTS_ROOT,
        cache_dir: Optional[str] = None,
    ):
        self.data_root = data_root
        self.gen_root = gen_root
        self.weights_root = weights_root
        self.cache_dir = cache_dir
    
    def build(
        self,
        dataset: str,
        model: str,
        strategy: str = 'baseline',
        std_strategy: Optional[str] = None,
        real_gen_ratio: float = 1.0,
        custom_training_config: Optional[Dict] = None,
        custom_conditions: Optional[List[str]] = None,
        domain_filter: Optional[str] = None,
        use_native_classes: bool = False,
    ) -> Dict[str, Any]:
        """
        Build a complete training configuration.
        
        Args:
            dataset: Dataset name (e.g., 'ACDC', 'BDD10k')
            model: Model name (e.g., 'deeplabv3plus_r50')
            strategy: Main augmentation strategy (e.g., 'baseline', 'gen_cycleGAN', 'std_cutmix')
            std_strategy: Optional standard augmentation to combine with gen_* strategy
                         (e.g., 'std_cutmix', 'std_mixup'). When provided with a gen_* 
                         strategy, both augmentations will be applied.
            real_gen_ratio: Ratio of real images in mixed training (0.0-1.0)
                           1.0 = only real images, 0.0 = only generated images
                           0.5 = 50% real, 50% generated
            custom_training_config: Optional custom training parameters
            custom_conditions: Optional custom list of weather conditions
            domain_filter: Optional domain to filter training data (e.g., 'clear_day')
                          When specified, only images from this domain subdirectory
                          will be used for training.
            use_native_classes: If True, train with native class labels for MapillaryVistas
                               (66 classes) or OUTSIDE15k (24 classes) instead of converting
                               to Cityscapes 19 classes. Only applicable for these datasets.
            
        Returns:
            Complete configuration dictionary
            
        Raises:
            ValueError: If inputs are invalid or no generated images available for dataset
        """
        # Validate inputs
        if dataset not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset}. Available: {list(DATASET_CONFIGS.keys())}")
        if model not in ALL_MODELS:
            raise ValueError(f"Unknown model: {model}. Available: {list(ALL_MODELS.keys())}")
        if strategy not in AUGMENTATION_STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(AUGMENTATION_STRATEGIES.keys())}")
        
        # Validate std_strategy if provided
        if std_strategy is not None:
            if std_strategy not in AUGMENTATION_STRATEGIES:
                raise ValueError(f"Unknown std_strategy: {std_strategy}. Available: {list(AUGMENTATION_STRATEGIES.keys())}")
            std_aug = AUGMENTATION_STRATEGIES[std_strategy]
            if std_aug.type != 'standard':
                raise ValueError(f"std_strategy must be a standard augmentation (std_*), got: {std_strategy}")
        
        dataset_cfg = DATASET_CONFIGS[dataset]
        model_cfg = ALL_MODELS[model]
        aug_strategy = AUGMENTATION_STRATEGIES[strategy]
        training_cfg = TRAINING_CONFIGS.get(dataset_cfg.task, TRAINING_CONFIGS['segmentation'])
        
        # Validate generated images exist for this dataset if using generative strategy
        if aug_strategy.type == 'generated' and aug_strategy.generative_model:
            gen_model = aug_strategy.generative_model
            gen_model_dir = _strategy_name_to_dir(gen_model)
            manifest_path = os.path.join(self.gen_root, gen_model_dir, 'manifest.csv')
            
            if not os.path.exists(manifest_path):
                raise ValueError(
                    f"No generated images manifest found for strategy '{strategy}'.\n"
                    f"Expected manifest at: {manifest_path}\n"
                    f"Available generative models in {self.gen_root}:\n"
                    f"  {os.listdir(self.gen_root) if os.path.exists(self.gen_root) else 'Directory not found'}"
                )
            
            # Check if this dataset has any generated images in the manifest
            from generated_images_dataset import GeneratedImagesManifest
            manifest = GeneratedImagesManifest(manifest_path)
            dataset_count = manifest.get_dataset_count(dataset)
            
            if dataset_count == 0:
                available_datasets = manifest.get_available_datasets()
                raise ValueError(
                    f"No generated images found for dataset '{dataset}' in strategy '{strategy}'.\n"
                    f"Manifest path: {manifest_path}\n"
                    f"Available datasets in manifest:\n"
                    f"  {available_datasets if available_datasets else 'None'}\n"
                    f"Skipping training. Use 'baseline' strategy or choose a different dataset."
                )
            else:
                print(f"✓ Found {dataset_count} generated images for dataset '{dataset}' in strategy '{strategy}'")
        
        # Apply custom training config if provided
        if custom_training_config:
            for key, value in custom_training_config.items():
                if hasattr(training_cfg, key):
                    setattr(training_cfg, key, value)
        
        # Apply custom conditions if provided
        conditions = custom_conditions or aug_strategy.conditions
        
        # Build configuration
        config = self._build_base_config(dataset_cfg, model_cfg, training_cfg)
        
        # Add strategy info to metadata
        config['_prove_config']['strategy'] = strategy
        if std_strategy:
            config['_prove_config']['std_strategy'] = std_strategy
            config['_prove_config']['combined_strategy'] = f"{strategy}+{std_strategy}"
        
        config = self._add_dataset_config(config, dataset_cfg, domain_filter, use_native_classes)
        # Single-dataset training: use native class labels when specified (use_unified_labels=False)
        config = self._add_training_pipeline(config, dataset_cfg.task, aug_strategy, std_strategy, dataset, use_unified_labels=False, use_native_classes=use_native_classes)
        config = self._add_augmentation_config(config, aug_strategy, conditions, real_gen_ratio, std_strategy)
        config = self._add_mixed_dataloader_config(config, dataset_cfg, aug_strategy, real_gen_ratio, domain_filter)
        config = self._set_work_dir(config, dataset, model, strategy, real_gen_ratio, domain_filter, std_strategy)
        
        return config
    
    def build_multi_dataset(
        self,
        datasets: List[str],
        model: str,
        strategy: str = 'baseline',
        std_strategy: Optional[str] = None,
        real_gen_ratio: float = 1.0,
        weights: Optional[List[float]] = None,
        custom_training_config: Optional[Dict] = None,
        custom_conditions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Build a complete training configuration for multiple datasets (joint training).
        
        Args:
            datasets: List of dataset names (e.g., ['ACDC', 'MapillaryVistas'])
            model: Model name (e.g., 'deeplabv3plus_r50')
            strategy: Augmentation strategy (e.g., 'baseline', 'gen_cycleGAN')
            std_strategy: Optional standard augmentation to combine with gen_* strategy
                         (e.g., 'std_cutmix', 'std_mixup'). When provided with a gen_* 
                         strategy, both augmentations will be applied.
            real_gen_ratio: Ratio of real images in mixed training (0.0-1.0)
            weights: Optional sampling weights for each dataset (must sum to 1.0).
                    If None, uses balanced sampling (equal weight per dataset).
            custom_training_config: Optional custom training parameters
            custom_conditions: Optional custom list of weather conditions
            
        Returns:
            Complete configuration dictionary with ConcatDataset for joint training
        """
        import copy
        
        if len(datasets) < 2:
            raise ValueError(f"build_multi_dataset requires at least 2 datasets, got {len(datasets)}")
        
        # Validate inputs
        for ds_name in datasets:
            if ds_name not in DATASET_CONFIGS:
                raise ValueError(f"Unknown dataset: {ds_name}. Available: {list(DATASET_CONFIGS.keys())}")
        if model not in ALL_MODELS:
            raise ValueError(f"Unknown model: {model}. Available: {list(ALL_MODELS.keys())}")
        if strategy not in AUGMENTATION_STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(AUGMENTATION_STRATEGIES.keys())}")
        
        # Validate std_strategy if provided
        if std_strategy is not None:
            if std_strategy not in AUGMENTATION_STRATEGIES:
                raise ValueError(f"Unknown std_strategy: {std_strategy}. Available: {list(AUGMENTATION_STRATEGIES.keys())}")
            std_aug = AUGMENTATION_STRATEGIES[std_strategy]
            if std_aug.type != 'standard':
                raise ValueError(f"std_strategy must be a standard augmentation (std_*), got: {std_strategy}")
        
        # Validate weights if provided
        if weights is not None:
            if len(weights) != len(datasets):
                raise ValueError(f"Number of weights ({len(weights)}) must match number of datasets ({len(datasets)})")
            weight_sum = sum(weights)
            if not (0.99 <= weight_sum <= 1.01):
                raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")
        else:
            # Equal weights
            weights = [1.0 / len(datasets)] * len(datasets)
        
        # Use first dataset to get base configuration
        # All datasets must share the same task type and label space
        dataset_cfgs = [DATASET_CONFIGS[ds] for ds in datasets]
        first_cfg = dataset_cfgs[0]
        model_cfg = ALL_MODELS[model]
        aug_strategy = AUGMENTATION_STRATEGIES[strategy]
        training_cfg = TRAINING_CONFIGS.get(first_cfg.task, TRAINING_CONFIGS['segmentation'])
        
        # Validate all datasets have the same task
        for i, cfg in enumerate(dataset_cfgs):
            if cfg.task != first_cfg.task:
                raise ValueError(f"All datasets must have the same task. {datasets[0]} is '{first_cfg.task}' but {datasets[i]} is '{cfg.task}'")
        
        # Validate generated images exist for all datasets if using generative strategy
        if aug_strategy.type == 'generated' and aug_strategy.generative_model:
            gen_model = aug_strategy.generative_model
            gen_model_dir = _strategy_name_to_dir(gen_model)
            manifest_path = os.path.join(self.gen_root, gen_model_dir, 'manifest.csv')
            
            if not os.path.exists(manifest_path):
                raise ValueError(
                    f"No generated images manifest found for strategy '{strategy}'.\n"
                    f"Expected manifest at: {manifest_path}"
                )
            
            from generated_images_dataset import GeneratedImagesManifest
            manifest = GeneratedImagesManifest(manifest_path)
            
            for ds_name in datasets:
                dataset_count = manifest.get_dataset_count(ds_name)
                if dataset_count == 0:
                    available_datasets = manifest.get_available_datasets()
                    raise ValueError(
                        f"No generated images found for dataset '{ds_name}' in strategy '{strategy}'.\n"
                        f"Available datasets in manifest: {available_datasets}\n"
                        f"Skipping training. Use 'baseline' strategy or remove '{ds_name}' from datasets."
                    )
                print(f"✓ Found {dataset_count} generated images for dataset '{ds_name}' in strategy '{strategy}'")
        
        # Apply custom training config if provided
        if custom_training_config:
            training_cfg = copy.deepcopy(training_cfg)
            for key, value in custom_training_config.items():
                if hasattr(training_cfg, key):
                    setattr(training_cfg, key, value)
        
        conditions = custom_conditions or aug_strategy.conditions
        
        # Build base configuration using first dataset as template
        config = self._build_base_config(first_cfg, model_cfg, training_cfg)
        
        # Add strategy info to metadata
        config['_prove_config']['strategy'] = strategy
        if std_strategy:
            config['_prove_config']['std_strategy'] = std_strategy
            config['_prove_config']['combined_strategy'] = f"{strategy}+{std_strategy}"
        
        # Multi-dataset training: use unified labels mapped to Cityscapes 19-class (use_unified_labels=True)
        config = self._add_training_pipeline(config, first_cfg.task, aug_strategy, std_strategy, datasets[0], use_unified_labels=True)
        config = self._add_augmentation_config(config, aug_strategy, conditions, real_gen_ratio, std_strategy)
        
        # Build multi-dataset train dataloader with ConcatDataset
        config = self._add_multi_dataset_config(config, dataset_cfgs, datasets, weights, real_gen_ratio, aug_strategy)
        
        # Set work directory for multi-dataset training
        config = self._set_multi_dataset_work_dir(config, datasets, model, strategy, real_gen_ratio, std_strategy)
        
        # Update metadata
        config['_prove_config']['datasets'] = datasets
        config['_prove_config']['dataset'] = '+'.join(datasets)
        config['_prove_config']['multi_dataset'] = True
        config['_prove_config']['dataset_weights'] = weights
        
        return config
    
    def _add_multi_dataset_config(
        self,
        config: Dict[str, Any],
        dataset_cfgs: List[DatasetConfig],
        dataset_names: List[str],
        weights: List[float],
        real_gen_ratio: float,
        aug_strategy: AugmentationStrategy,
    ) -> Dict[str, Any]:
        """
        Add multi-dataset configuration using ConcatDataset with per-dataset pipelines.
        
        Automatically handles label unification by adding appropriate transforms:
        - MapillaryVistas: MapillaryLabelTransform (66 -> 19 class mapping)
        - ACDC, Cityscapes, etc.: Already use Cityscapes trainIDs (no transform needed)
        
        Args:
            config: Configuration dictionary to update
            dataset_cfgs: List of dataset configurations
            dataset_names: List of dataset names (for reference)
            weights: Sampling weights for each dataset
            real_gen_ratio: Ratio of real images (0.0-1.0)
        """
        
        data_root = self.data_root
        config['data_root'] = data_root
        
        # All segmentation datasets use Cityscapes format with same classes
        config['dataset_type'] = 'ConcatDataset'
        config['classes'] = dataset_cfgs[0].classes  # Use first dataset's classes
        
        batch_size = 2
        num_workers = 4
        
        # Datasets that need specific label transformations in UNIFIED mode:
        # Multi-dataset training ALWAYS uses unified labels (Cityscapes 19-class)
        # - MapillaryVistas: 66-class labels → 19-class Cityscapes trainIDs
        # - OUTSIDE15k: 24-class labels → 19-class Cityscapes trainIDs
        # - ACDC: Cityscapes label IDs (0-33) → Cityscapes trainIDs (0-18)
        # - IDD-AW: Uses fixed masks with correct Cityscapes trainIDs (masks regenerated from JSON polygons)
        # - BDD10k: Already uses Cityscapes trainIDs (no transform needed)
        MAPILLARY_DATASETS = {'MapillaryVistas', 'Mapillary'}
        OUTSIDE15K_DATASETS = {'OUTSIDE15k'}
        CITYSCAPES_LABEL_ID_DATASETS = {'ACDC'}  # Use Cityscapes label ID format (7=road, 8=sidewalk, etc.)
        IDDAW_DATASETS = {'IDD-AW'}  # IDD-AW uses fixed masks from JSON polygon annotations
        
        # Build individual dataset configs for ConcatDataset with per-dataset pipelines
        train_datasets = []
        for i, (cfg, name, weight) in enumerate(zip(dataset_cfgs, dataset_names, weights)):
            # Create per-dataset pipeline reference
            # Sanitize dataset name for valid Python variable (hyphens -> underscores)
            pipeline_name = f'train_pipeline_{name.lower().replace("-", "_")}'
            
            # Build custom pipeline based on dataset's label format
            # Check if baseline (no augmentation)
            is_baseline = aug_strategy.name == 'baseline'
            
            if name in MAPILLARY_DATASETS:
                # MapillaryVistas: needs MapillaryLabelTransform (66 → 19 classes)
                config[pipeline_name] = self._build_mapillary_training_pipeline(cfg, is_baseline)
                pipeline_ref = '{{' + pipeline_name + '}}'
            elif name in OUTSIDE15K_DATASETS:
                # OUTSIDE15k: needs Outside15kLabelTransform (24 → 19 classes)
                config[pipeline_name] = self._build_outside15k_training_pipeline(cfg, is_baseline)
                pipeline_ref = '{{' + pipeline_name + '}}'
            elif name in CITYSCAPES_LABEL_ID_DATASETS:
                # ACDC: uses Cityscapes label IDs (0-33), needs CityscapesLabelIdToTrainId
                config[pipeline_name] = self._build_cityscapes_training_pipeline(cfg, is_baseline)
                pipeline_ref = '{{' + pipeline_name + '}}'
            elif name in IDDAW_DATASETS:
                # IDD-AW: Now uses fixed masks with correct Cityscapes trainIDs (no transform needed)
                # Masks were regenerated from original JSON polygon annotations
                config[pipeline_name] = self._build_trainid_training_pipeline(cfg, is_baseline)
                pipeline_ref = '{{' + pipeline_name + '}}'
            else:
                # BDD10k: already has trainIDs, just needs ReduceToSingleChannel
                config[pipeline_name] = self._build_trainid_training_pipeline(cfg, is_baseline)
                pipeline_ref = '{{' + pipeline_name + '}}'
            
            # Determine annotation root (use ann_root if specified, otherwise data_root)
            ann_root = cfg.ann_root if cfg.ann_root else data_root
            
            ds_config = dict(
                type='CityscapesDataset',
                data_root=data_root,
                data_prefix=dict(
                    img_path=cfg.train_img_dir,
                    seg_map_path=os.path.join(ann_root, cfg.train_ann_dir) if cfg.ann_root else cfg.train_ann_dir,
                ),
                img_suffix=cfg.img_suffix,
                seg_map_suffix='.png',
                reduce_zero_label=False,
                pipeline=pipeline_ref,
            )
            train_datasets.append(ds_config)
        
        # ConcatDataset for training (combines all datasets)
        config['train_dataloader'] = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
            sampler=dict(type='InfiniteSampler', shuffle=True),
            dataset=dict(
                type='ConcatDataset',
                datasets=train_datasets,
            ),
        )
        
        # Store weights in config for potential weighted sampling
        config['multi_dataset_config'] = dict(
            enabled=True,
            datasets=dataset_names,
            weights=weights,
        )
        
        # Use first dataset for validation/test (can be changed as needed)
        first_cfg = dataset_cfgs[0]
        first_ann_root = first_cfg.ann_root if first_cfg.ann_root else data_root
        config['val_dataloader'] = dict(
            batch_size=1,
            num_workers=num_workers,
            persistent_workers=True,
            sampler=dict(type='DefaultSampler', shuffle=False),
            dataset=dict(
                type='CityscapesDataset',
                data_root=data_root,
                data_prefix=dict(
                    img_path=first_cfg.val_img_dir,
                    seg_map_path=os.path.join(first_ann_root, first_cfg.val_ann_dir) if first_cfg.ann_root else first_cfg.val_ann_dir,
                ),
                img_suffix=first_cfg.img_suffix,
                seg_map_suffix='.png',
                reduce_zero_label=False,
                pipeline='{{test_pipeline}}',
            ),
        )
        
        config['test_dataloader'] = config['val_dataloader'].copy()
        
        # Image normalization
        config['img_norm_cfg'] = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
        )
        
        return config
    
    def _build_mapillary_training_pipeline(self, dataset_cfg: DatasetConfig, is_baseline: bool = False) -> List[Dict]:
        """
        Build a training pipeline for Mapillary datasets with automatic label transformation.
        
        Includes MapillaryLabelTransform to convert 66 Mapillary classes to 19 Cityscapes trainIDs.
        This allows joint training with ACDC, Cityscapes, etc. using a unified label space.
        
        Args:
            dataset_cfg: Dataset configuration object
            is_baseline: If True, skip data augmentation transforms
            
        Returns:
            List of pipeline transforms
        """
        crop_size = (512, 512)
        
        # Pipeline with MapillaryLabelTransform added after LoadAnnotations
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            # Handle labels stored as 3-channel PNGs
            dict(type='ReduceToSingleChannel'),
            # Convert Mapillary labels (0-65) to Cityscapes trainIds (0-18, 255)
            # MapillaryLabelTransform is registered in unified_datasets.py
            dict(type='MapillaryLabelTransform', target_space='cityscapes'),
            dict(type='Resize', scale=(512, 512), keep_ratio=False),
        ]
        
        # Only add augmentations if not baseline
        if not is_baseline:
            pipeline.extend([
                dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(type='PhotoMetricDistortion'),
            ])
        
        pipeline.append(dict(type='PackSegInputs'))
        
        return pipeline
    
    def _build_cityscapes_training_pipeline(self, dataset_cfg: DatasetConfig, is_baseline: bool = False) -> List[Dict]:
        """
        Build a training pipeline for datasets using Cityscapes label ID format.
        
        Datasets like ACDC use the original Cityscapes label format where:
        - road = 7, sidewalk = 8, building = 11, etc.
        
        This pipeline includes CityscapesLabelIdToTrainId to convert these to trainIDs:
        - road = 0, sidewalk = 1, building = 2, etc.
        
        Args:
            dataset_cfg: Dataset configuration object
            is_baseline: If True, skip data augmentation transforms
            
        Returns:
            List of pipeline transforms
        """
        crop_size = (512, 512)
        
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            # Handle labels stored as 3-channel PNGs
            dict(type='ReduceToSingleChannel'),
            # Convert Cityscapes label IDs (0-33) to trainIDs (0-18)
            dict(type='CityscapesLabelIdToTrainId'),
            dict(type='Resize', scale=(512, 512), keep_ratio=False),
        ]
        
        # Only add augmentations if not baseline
        if not is_baseline:
            pipeline.extend([
                dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(type='PhotoMetricDistortion'),
            ])
        
        pipeline.append(dict(type='PackSegInputs'))
        
        return pipeline
    
    def _build_trainid_training_pipeline(self, dataset_cfg: DatasetConfig, is_baseline: bool = False) -> List[Dict]:
        """
        Build a training pipeline for datasets that already use trainID format.
        
        Datasets like BDD10k already have labels in Cityscapes trainID format:
        - road = 0, sidewalk = 1, building = 2, etc.
        
        These datasets only need ReduceToSingleChannel (for 3-channel PNGs) but NO
        label ID transformation.
        
        Args:
            dataset_cfg: Dataset configuration object
            is_baseline: If True, skip data augmentation transforms
            
        Returns:
            List of pipeline transforms
        """
        crop_size = (512, 512)
        
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            # Handle labels stored as 3-channel PNGs
            dict(type='ReduceToSingleChannel'),
            # NO CityscapesLabelIdToTrainId - labels already in trainID format
            dict(type='Resize', scale=(1024, 512), keep_ratio=True),
        ]
        
        # Only add augmentations if not baseline
        if not is_baseline:
            pipeline.extend([
                dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(type='PhotoMetricDistortion'),
            ])
        
        pipeline.append(dict(type='PackSegInputs'))
        
        return pipeline
    
    # NOTE: _build_iddaw_training_pipeline was removed. IDD-AW now uses fixed masks
    # with correct Cityscapes trainIDs, generated from original JSON polygon annotations.
    # The standard _build_trainid_training_pipeline is used instead.
    
    def _build_outside15k_training_pipeline(self, dataset_cfg: DatasetConfig, is_baseline: bool = False) -> List[Dict]:
        """
        Build a training pipeline for OUTSIDE15k dataset with automatic label transformation.
        
        Includes Outside15kLabelTransform to convert 24 OUTSIDE15k classes to 19 Cityscapes trainIDs.
        This allows joint training with other datasets using a unified label space.
        
        OUTSIDE15k native classes (24 classes, 0-23):
            unlabeled, animal, barrier, bicycle, boat, bridge, building, grass,
            ground, mountain, object, person, pole, road, sand, sidewalk, sign,
            sky, street light, traffic light, tunnel, vegetation, vehicle, water
        
        Args:
            dataset_cfg: Dataset configuration object
            is_baseline: If True, skip data augmentation transforms
            
        Returns:
            List of pipeline transforms
        """
        crop_size = (512, 512)
        
        # Pipeline with Outside15kLabelTransform added after LoadAnnotations
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            # Handle labels stored as 3-channel PNGs
            dict(type='ReduceToSingleChannel'),
            # Convert OUTSIDE15k labels (0-23) to Cityscapes trainIds (0-18, 255)
            # Outside15kLabelTransform is registered in custom_transforms.py
            dict(type='Outside15kLabelTransform'),
            dict(type='Resize', scale=(512, 512), keep_ratio=False),
        ]
        
        # Only add augmentations if not baseline
        if not is_baseline:
            pipeline.extend([
                dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(type='PhotoMetricDistortion'),
            ])
        
        pipeline.append(dict(type='PackSegInputs'))
        
        return pipeline
    
    def _set_multi_dataset_work_dir(
        self,
        config: Dict[str, Any],
        datasets: List[str],
        model: str,
        strategy: str,
        real_gen_ratio: float,
        std_strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Set the output work directory for multi-dataset training.
        
        Directory format: {weights_root}/{strategy}/multi_{ds1}+{ds2}/model
        If std_strategy is provided: {weights_root}/{strategy}+{std_strategy}/multi_{ds1}+{ds2}/model
        """
        
        if real_gen_ratio < 1.0:
            ratio_str = f'_ratio{real_gen_ratio:.2f}'.replace('.', 'p')
        else:
            ratio_str = ''
        
        # Create combined dataset name
        datasets_str = 'multi_' + '+'.join(ds.lower() for ds in datasets)
        
        # Build strategy name (with optional std_strategy suffix)
        if std_strategy:
            strategy_dir = f'{strategy}+{std_strategy}'
        else:
            strategy_dir = strategy
        
        config['work_dir'] = os.path.join(
            self.weights_root,
            strategy_dir,
            datasets_str,
            f'{model}{ratio_str}',
        )
        
        return config
    

    def _update_pretrained_paths(self, model_def: Dict[str, Any]) -> Dict[str, Any]:
        """Update pretrained model paths to use cache_dir if specified.
        
        This modifies init_cfg.checkpoint and pretrained paths to use the cache directory
        for storing downloaded pretrained weights.
        
        Args:
            model_def: Model definition dictionary
            
        Returns:
            Updated model definition with cache_dir-aware paths
        """
        if not self.cache_dir or not model_def:
            return model_def
        
        import os
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Update backbone init_cfg if present
        if 'backbone' in model_def and 'init_cfg' in model_def['backbone']:
            init_cfg = model_def['backbone']['init_cfg']
            if isinstance(init_cfg, dict) and 'checkpoint' in init_cfg:
                checkpoint = init_cfg['checkpoint']
                # Map open-mmlab and torchvision URLs to local cache paths
                if checkpoint.startswith('open-mmlab://'):
                    model_name = checkpoint.replace('open-mmlab://', '')
                    local_path = os.path.join(self.cache_dir, 'open-mmlab', f'{model_name}.pth')
                    # Only use local path if file exists, otherwise keep URL for download
                    if os.path.exists(local_path):
                        model_def['backbone']['init_cfg']['checkpoint'] = local_path
                    else:
                        # Set TORCH_HOME to cache_dir so MMEngine downloads to cache_dir
                        os.environ['TORCH_HOME'] = self.cache_dir
                elif checkpoint.startswith('torchvision://'):
                    model_name = checkpoint.replace('torchvision://', '')
                    local_path = os.path.join(self.cache_dir, 'torchvision', f'{model_name}.pth')
                    if os.path.exists(local_path):
                        model_def['backbone']['init_cfg']['checkpoint'] = local_path
                    else:
                        os.environ['TORCH_HOME'] = self.cache_dir
        
        # Update top-level pretrained path if present (e.g., for SegFormer)
        if 'pretrained' in model_def:
            pretrained = model_def['pretrained']
            if pretrained and not os.path.isabs(pretrained):
                # Convert relative path to cache_dir-relative path
                local_path = os.path.join(self.cache_dir, pretrained)
                if os.path.exists(local_path):
                    model_def['pretrained'] = local_path
                else:
                    # Create pretrain subdirectory in cache_dir
                    pretrain_dir = os.path.join(self.cache_dir, os.path.dirname(pretrained))
                    os.makedirs(pretrain_dir, exist_ok=True)
                    model_def['pretrained'] = os.path.join(self.cache_dir, pretrained)
        
        return model_def
    
    def _build_custom_hooks(
        self,
        dataset_cfg: 'DatasetConfig',
        training_cfg: TrainingConfig,
    ) -> List[Dict[str, Any]]:
        """Build custom hooks configuration, including early stopping.
        
        Args:
            dataset_cfg: Dataset configuration
            training_cfg: Training configuration
            
        Returns:
            List of custom hook configurations
        """
        hooks = []
        
        # Early stopping hook
        if training_cfg.early_stop:
            # Determine monitor metric based on task
            if dataset_cfg.task == 'segmentation':
                monitor = 'val/mIoU'  # Monitor validation mIoU for segmentation
                rule = 'greater'  # Higher is better
            else:
                monitor = 'coco/bbox_mAP'  # Monitor bbox mAP for detection
                rule = 'greater'
            
            hooks.append(dict(
                type='EarlyStoppingHook',
                monitor=monitor,
                rule=rule,
                patience=training_cfg.early_stop_patience,
                min_delta=training_cfg.early_stop_min_delta,
                strict=False,  # Don't crash if metric not found
                check_finite=True,  # Stop if NaN/Inf
            ))
        
        return hooks
    
    def _build_base_config(
        self,
        dataset_cfg: DatasetConfig,
        model_cfg: ModelConfig,
        training_cfg: TrainingConfig,
    ) -> Dict[str, Any]:
        """Build base configuration structure for MMEngine Runner"""
        
        metric = 'mIoU' if dataset_cfg.task == 'segmentation' else 'bbox'
        
        # Get model definition and update num_classes
        import copy
        model_def = copy.deepcopy(MODEL_DEFINITIONS.get(model_cfg.name, {}))
        if model_def:
            # Update num_classes in decode_head for segmentation models
            if 'decode_head' in model_def:
                model_def['decode_head']['num_classes'] = dataset_cfg.num_classes
            if 'auxiliary_head' in model_def:
                model_def['auxiliary_head']['num_classes'] = dataset_cfg.num_classes
            # Update num_classes for detection models
            if 'roi_head' in model_def and 'bbox_head' in model_def['roi_head']:
                model_def['roi_head']['bbox_head']['num_classes'] = len(BDD100K_DET_CLASSES)
            if 'bbox_head' in model_def and 'num_classes' in model_def['bbox_head']:
                model_def['bbox_head']['num_classes'] = len(BDD100K_DET_CLASSES)
            
            # Update pretrained paths with cache_dir
            model_def = self._update_pretrained_paths(model_def)
        
        # Build optimizer wrapper (new MMEngine format)
        optim_wrapper = dict(
            type='OptimWrapper',
            optimizer=dict(
                type=model_cfg.optimizer,
                lr=model_cfg.lr,
                weight_decay=model_cfg.weight_decay,
                **(dict(momentum=0.9) if model_cfg.optimizer == 'SGD' else dict(betas=(0.9, 0.999))),
            ),
        )
        
        # Build param scheduler (new MMEngine format)
        param_scheduler = [
            dict(
                type='LinearLR',
                start_factor=training_cfg.warmup_ratio,
                by_epoch=False,
                begin=0,
                end=training_cfg.warmup_iters,
            ),
            dict(
                type='PolyLR',
                eta_min=1e-6,
                power=0.9,
                begin=training_cfg.warmup_iters,
                end=training_cfg.max_iters,
                by_epoch=False,
            ),
        ]
        
        # Build train_cfg (new MMEngine format)
        train_cfg = dict(
            type='IterBasedTrainLoop',
            max_iters=training_cfg.max_iters,
            val_interval=training_cfg.eval_interval,
        )
        
        # Build val_cfg
        val_cfg = dict(type='ValLoop')
        
        # Build test_cfg
        test_cfg = dict(type='TestLoop')
        
        return {
            # Metadata
            '_prove_config': {
                'version': '2.0.0',
                'dataset': dataset_cfg.name,
                'model': model_cfg.name,
                'task': dataset_cfg.task,
            },
            
            # Model definition (inline, not inherited from _base_)
            'model': model_def,
            
            # MMEngine Runner configuration
            'train_cfg': train_cfg,
            'val_cfg': val_cfg,
            'test_cfg': test_cfg,
            
            # Optimizer and scheduler (new MMEngine format)
            'optim_wrapper': optim_wrapper,
            'param_scheduler': param_scheduler,
            
            # Checkpointing (new format)
            'default_hooks': dict(
                timer=dict(type='IterTimerHook'),
                logger=dict(type='LoggerHook', interval=training_cfg.log_interval),
                param_scheduler=dict(type='ParamSchedulerHook'),
                checkpoint=dict(
                    type='CheckpointHook',
                    interval=training_cfg.checkpoint_interval,
                    by_epoch=False,
                ),
                sampler_seed=dict(type='DistSamplerSeedHook'),
            ),
            
            # Early stopping hook (custom hooks)
            'custom_hooks': self._build_custom_hooks(dataset_cfg, training_cfg),
            
            # Evaluation (new format) - use FWIoUMetric for both mIoU and fwIoU
            'val_evaluator': dict(type='FWIoUMetric', iou_metrics=['mIoU', 'fwIoU'], prefix='val') if dataset_cfg.task == 'segmentation' else dict(type='CocoMetric', metric='bbox'),
            'test_evaluator': dict(type='FWIoUMetric', iou_metrics=['mIoU', 'fwIoU'], prefix='test') if dataset_cfg.task == 'segmentation' else dict(type='CocoMetric', metric='bbox'),
            
            # Logging
            'log_processor': dict(by_epoch=False),
            
            # Reproducibility - disable deterministic for CUDA compatibility
            'randomness': dict(seed=training_cfg.seed, deterministic=False),
            
            # Environment
            'env_cfg': dict(
                cudnn_benchmark=True,
                mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
                dist_cfg=dict(backend='nccl'),
            ),
            
            # Visualization
            'vis_backends': [dict(type='LocalVisBackend')],
            'visualizer': dict(
                type='SegLocalVisualizer',
                vis_backends=[dict(type='LocalVisBackend')],
                name='visualizer',
            ) if dataset_cfg.task == 'segmentation' else dict(
                type='DetLocalVisualizer',
                vis_backends=[dict(type='LocalVisBackend')],
                name='visualizer',
            ),
        }
    
    def _add_dataset_config(
        self,
        config: Dict[str, Any],
        dataset_cfg: DatasetConfig,
        domain_filter: Optional[str] = None,
        use_native_classes: bool = False,
    ) -> Dict[str, Any]:
        """
        Add dataset-specific configuration.
        
        Args:
            config: Configuration dictionary to update
            dataset_cfg: Dataset configuration object
            domain_filter: Optional domain to filter training data (e.g., 'clear_day')
            use_native_classes: If True, use native class count (66 for MapillaryVistas, 24 for OUTSIDE15k)
        """
        
        # Use base data root - dataset paths already include the dataset name
        data_root = self.data_root
        config['data_root'] = data_root
        config['dataset_type'] = 'CityscapesDataset' if dataset_cfg.format == 'cityscapes' else 'CocoDataset'
        
        # Handle native classes for MapillaryVistas and OUTSIDE15k
        if use_native_classes:
            if dataset_cfg.name in ['MapillaryVistas', 'Mapillary']:
                # Override to use 66 MapillaryVistas native classes
                native_num_classes = 66
                native_classes = MAPILLARY_CLASSES
                config['classes'] = native_classes
                # Update model num_classes
                if 'model' in config and 'decode_head' in config['model']:
                    config['model']['decode_head']['num_classes'] = native_num_classes
                if 'model' in config and 'auxiliary_head' in config['model']:
                    config['model']['auxiliary_head']['num_classes'] = native_num_classes
                config['_prove_config']['use_native_classes'] = True
                config['_prove_config']['native_num_classes'] = native_num_classes
                print(f"[INFO] Using native MapillaryVistas classes: {native_num_classes} classes")
            elif dataset_cfg.name == 'OUTSIDE15k':
                # Override to use 24 OUTSIDE15k native classes
                native_num_classes = 24
                native_classes = OUTSIDE15K_CLASSES
                config['classes'] = native_classes
                # Update model num_classes
                if 'model' in config and 'decode_head' in config['model']:
                    config['model']['decode_head']['num_classes'] = native_num_classes
                if 'model' in config and 'auxiliary_head' in config['model']:
                    config['model']['auxiliary_head']['num_classes'] = native_num_classes
                config['_prove_config']['use_native_classes'] = True
                config['_prove_config']['native_num_classes'] = native_num_classes
                print(f"[INFO] Using native OUTSIDE15k classes: {native_num_classes} classes")
            else:
                config['classes'] = dataset_cfg.classes
                print(f"[WARN] use_native_classes=True but dataset {dataset_cfg.name} doesn't have native classes")
        else:
            config['classes'] = dataset_cfg.classes
        
        # Determine annotation root (use ann_root if specified, otherwise data_root)
        ann_root = dataset_cfg.ann_root if dataset_cfg.ann_root else data_root
        
        # Apply domain filter to training directories if specified
        train_img_dir = dataset_cfg.train_img_dir
        train_ann_dir = dataset_cfg.train_ann_dir
        
        if domain_filter:
            # Append domain subdirectory to training paths
            # e.g., 'train/images/ACDC' -> 'train/images/ACDC/clear_day'
            train_img_dir = os.path.join(train_img_dir, domain_filter)
            train_ann_dir = os.path.join(train_ann_dir, domain_filter)
            config['domain_filter'] = domain_filter
            print(f"[INFO] Training data filtered to domain: {domain_filter}")
            print(f"[INFO] Using train_img_dir: {train_img_dir}")
            print(f"[INFO] Using train_ann_dir: {train_ann_dir}")
        
        # New MMEngine dataloader format
        batch_size = 2
        num_workers = 4
        
        if dataset_cfg.task == 'segmentation':
            # Train dataloader
            # Use CityscapesDataset with custom suffixes for our data format
            config['train_dataloader'] = dict(
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=True,
                sampler=dict(type='InfiniteSampler', shuffle=True),
                dataset=dict(
                    type='CityscapesDataset',
                    data_root=data_root,
                    data_prefix=dict(
                        img_path=train_img_dir,
                        seg_map_path=os.path.join(ann_root, train_ann_dir) if dataset_cfg.ann_root else train_ann_dir,
                    ),
                    # Custom suffixes for non-standard Cityscapes format
                    img_suffix=dataset_cfg.img_suffix,
                    seg_map_suffix='.png',
                    reduce_zero_label=False,  # Set here instead of LoadAnnotations
                    pipeline='{{train_pipeline}}',
                ),
            )
            
            # Val dataloader
            config['val_dataloader'] = dict(
                batch_size=1,
                num_workers=num_workers,
                persistent_workers=True,
                sampler=dict(type='DefaultSampler', shuffle=False),
                dataset=dict(
                    type='CityscapesDataset',
                    data_root=data_root,
                    data_prefix=dict(
                        img_path=dataset_cfg.val_img_dir,
                        seg_map_path=os.path.join(ann_root, dataset_cfg.val_ann_dir) if dataset_cfg.ann_root else dataset_cfg.val_ann_dir,
                    ),
                    # Custom suffixes for non-standard Cityscapes format
                    img_suffix=dataset_cfg.img_suffix,
                    seg_map_suffix='.png',
                    reduce_zero_label=False,  # Set here instead of LoadAnnotations
                    pipeline='{{test_pipeline}}',
                ),
            )
            
            # Test dataloader (same as val for now)
            config['test_dataloader'] = config['val_dataloader'].copy()
            
        else:  # detection
            # Train dataloader
            config['train_dataloader'] = dict(
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=True,
                sampler=dict(type='InfiniteSampler', shuffle=True),
                dataset=dict(
                    type='CocoDataset',
                    data_root=data_root,
                    ann_file=dataset_cfg.train_ann_dir,
                    data_prefix=dict(img=dataset_cfg.train_img_dir),
                    pipeline='{{train_pipeline}}',
                ),
            )
            
            # Val dataloader
            config['val_dataloader'] = dict(
                batch_size=1,
                num_workers=num_workers,
                persistent_workers=True,
                sampler=dict(type='DefaultSampler', shuffle=False),
                dataset=dict(
                    type='CocoDataset',
                    data_root=data_root,
                    ann_file=dataset_cfg.val_ann_dir,
                    data_prefix=dict(img=dataset_cfg.val_img_dir),
                    pipeline='{{test_pipeline}}',
                ),
            )
            
            # Test dataloader
            config['test_dataloader'] = config['val_dataloader'].copy()
        
        # Image normalization (for reference)
        config['img_norm_cfg'] = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
        )
        
        return config
    
    def _add_training_pipeline(
        self,
        config: Dict[str, Any],
        task: str,
        aug_strategy: AugmentationStrategy,
        std_strategy: Optional[str] = None,
        dataset: Optional[str] = None,
        use_unified_labels: bool = False,
        use_native_classes: bool = False,
    ) -> Dict[str, Any]:
        """Add training data pipeline
        
        Args:
            config: Configuration dictionary to update
            task: Task type ('segmentation' or 'detection')
            aug_strategy: Main augmentation strategy
            std_strategy: Optional standard augmentation to combine
            dataset: Dataset name to determine correct label transformation
            use_unified_labels: If True, map all labels to Cityscapes 19-class format
                               (for multi-dataset training or domain adaptation).
                               If False, use native class labels per dataset.
            use_native_classes: If True, keep MapillaryVistas (66 classes) or OUTSIDE15k (24 classes)
                               in their native format instead of converting to Cityscapes 19 classes.
                               This is for training native models that will be used with cross-domain
                               prediction mapping.
        """
        
        crop_size = (512, 512)
        is_baseline = aug_strategy.name == 'baseline'
        
        # Check if we're using standard augmentation (either as main or combined)
        using_std_aug = (aug_strategy.type == 'standard') or (std_strategy is not None)
        std_aug_method = None
        if aug_strategy.type == 'standard':
            std_aug_method = aug_strategy.standard_method
        elif std_strategy is not None:
            std_aug_method = AUGMENTATION_STRATEGIES[std_strategy].standard_method
        
        # Label transformation logic:
        # Single-dataset training with use_native_classes=True: Keep native class labels (no Cityscapes conversion)
        # Single-dataset training with use_native_classes=False: Convert to Cityscapes 19-class format
        # Multi-dataset training: Map all labels to Cityscapes 19-class format (unified labels)
        #
        # Dataset native formats:
        # - ACDC/Cityscapes: labelIds (0-33) - ALWAYS needs CityscapesLabelIdToTrainId transform
        # - BDD10k: Already Cityscapes trainIDs (0-18) - no transform needed
        # - IDD-AW: Extended Cityscapes trainIDs with values 19, 20 - needs IddawLabelClamp
        #           IDD values 19=traffic light (maps to trainId 6), 20=pole (maps to trainId 5)
        # - OUTSIDE15k: 24 native classes (0-23) - only transform to Cityscapes if NOT use_native_classes
        # - MapillaryVistas: 66 native classes (0-65) - only transform to Cityscapes if NOT use_native_classes
        CITYSCAPES_LABEL_ID_DATASETS = {'ACDC', 'Cityscapes'}  # Uses Cityscapes labelId format (0-33)
        OUTSIDE15K_DATASETS = {'OUTSIDE15k'}
        MAPILLARY_DATASETS = {'MapillaryVistas', 'Mapillary'}
        IDDAW_DATASETS = {'IDD-AW'}  # IDD-AW has extra classes 19=traffic light, 20=pole
        
        # ACDC/Cityscapes always need labelId->trainId conversion (format conversion)
        needs_label_id_transform = dataset in CITYSCAPES_LABEL_ID_DATASETS if dataset else True
        # OUTSIDE15k converts to Cityscapes trainIds ONLY IF not using native classes
        needs_outside15k_transform = (dataset in OUTSIDE15K_DATASETS and not use_native_classes) if dataset else False
        # MapillaryVistas converts to Cityscapes trainIds ONLY IF not using native classes
        needs_mapillary_transform = (dataset in MAPILLARY_DATASETS and not use_native_classes) if dataset else False
        # IDD-AW now uses fixed masks with correct trainIds (no transform needed)
        # MapillaryVistas uses RGB color-encoded labels that need decoding
        is_mapillary = dataset in MAPILLARY_DATASETS if dataset else False
        
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),  # reduce_zero_label set in dataset config
        ]
        
        # Handle label format conversion based on dataset
        if is_mapillary:
            # MapillaryVistas labels are RGB color-encoded - decode to class IDs (0-65)
            pipeline.append(dict(type='MapillaryRGBToClassId'))
            # If using native classes, clamp to valid range 0-65
            if use_native_classes:
                pipeline.append(dict(type='MapillaryNativeLabelClamp'))
        else:
            # Other datasets: labels stored as 3-channel PNGs with identical channels
            pipeline.append(dict(type='ReduceToSingleChannel'))
            # If using native classes for OUTSIDE15k, clamp to valid range 0-23
            if use_native_classes and dataset in OUTSIDE15K_DATASETS:
                pipeline.append(dict(type='Outside15kNativeLabelClamp'))
        
        # Apply dataset-specific label transformation to Cityscapes 19-class trainIds
        if needs_label_id_transform:
            # Convert Cityscapes full label IDs (0-33) to trainIds (0-18)
            # This is always needed for ACDC/Cityscapes
            pipeline.append(dict(type='CityscapesLabelIdToTrainId'))
        elif needs_outside15k_transform:
            # Convert OUTSIDE15k 24 classes (0-23) to Cityscapes 19 trainIDs (0-18)
            pipeline.append(dict(type='Outside15kLabelTransform'))
        elif needs_mapillary_transform:
            # Convert Mapillary native IDs (0-65) to Cityscapes 19 trainIDs (0-18)
            pipeline.append(dict(type='MapillaryToTrainId'))
        # NOTE: IDD-AW no longer needs IddawLabelClamp - masks have correct trainIds
        
        pipeline.append(dict(type='Resize', scale=(512, 512), keep_ratio=False))
        
        # Only add augmentations if not baseline
        if not is_baseline:
            pipeline.extend([
                dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(type='PhotoMetricDistortion'),
            ])
        
        # Add augmentation transforms from strategy (non-baseline strategies may add more)
        pipeline.extend(aug_strategy.get_pipeline_transforms())
        
        # Add standard augmentation transforms if using std_* strategy
        if using_std_aug and std_aug_method:
            # Standard augmentations are applied as batch-level transforms
            # They are configured in custom_hooks, not in the pipeline directly
            # But we can add a marker for them here
            config['_prove_config'] = config.get('_prove_config', {})
            config['_prove_config']['std_augmentation'] = std_aug_method
        
        # Add final packing
        if task == 'segmentation':
            pipeline.append(dict(type='PackSegInputs'))
        else:
            pipeline.append(dict(type='PackDetInputs'))
        
        config['train_pipeline'] = pipeline
        
        # Test pipeline (no augmentation)
        test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),  # reduce_zero_label set in dataset config
        ]
        
        # Handle label format conversion based on dataset (same as train pipeline)
        if is_mapillary:
            # MapillaryVistas labels are RGB color-encoded - decode to class IDs (0-65)
            test_pipeline.append(dict(type='MapillaryRGBToClassId'))
        else:
            # Other datasets: labels stored as 3-channel PNGs with identical channels
            test_pipeline.append(dict(type='ReduceToSingleChannel'))
        
        # Apply dataset-specific label transformation to Cityscapes 19-class trainIds
        if needs_label_id_transform:
            # Convert Cityscapes full label IDs (0-33) to trainIds (0-18)
            test_pipeline.append(dict(type='CityscapesLabelIdToTrainId'))
        elif needs_outside15k_transform:
            # Convert OUTSIDE15k 24 classes (0-23) to Cityscapes 19 trainIDs
            test_pipeline.append(dict(type='Outside15kLabelTransform'))
        elif needs_mapillary_transform:
            # Convert Mapillary native IDs (0-65) to Cityscapes 19 trainIDs
            test_pipeline.append(dict(type='MapillaryToTrainId'))
        # NOTE: IDD-AW no longer needs IddawLabelClamp - masks have correct trainIds
        
        # Resize to fixed 512x512 for testing (consistent with training)
        test_pipeline.append(dict(type='Resize', scale=(512, 512), keep_ratio=False))
        
        if task == 'segmentation':
            test_pipeline.append(dict(type='PackSegInputs'))
        else:
            test_pipeline.append(dict(type='PackDetInputs'))
        
        config['test_pipeline'] = test_pipeline
        
        # Add crop_size to config for use elsewhere
        config['crop_size'] = crop_size
        
        return config
    
    def _add_augmentation_config(
        self,
        config: Dict[str, Any],
        aug_strategy: AugmentationStrategy,
        conditions: List[str],
        real_gen_ratio: float,
        std_strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add augmentation-specific configuration
        
        Args:
            config: Configuration dictionary to update
            aug_strategy: Main augmentation strategy
            conditions: List of weather conditions
            real_gen_ratio: Ratio of real images
            std_strategy: Optional standard augmentation to combine with main strategy
        """
        
        if aug_strategy.type == 'generated' and aug_strategy.generative_model:
            gen_model = aug_strategy.generative_model
            # Convert strategy name to actual directory name (handles hyphens)
            gen_model_dir = _strategy_name_to_dir(gen_model)
            config['generated_augmentation'] = {
                'enabled': True,
                'generative_model': gen_model,
                'manifest_path': os.path.join(self.gen_root, gen_model_dir, 'manifest.csv'),
                'gen_root': os.path.join(self.gen_root, gen_model_dir),
                'conditions': conditions,
                'augmentation_multiplier': 1 + len(conditions),
                'real_gen_ratio': real_gen_ratio,
            }
            
            # Add standard augmentation if combined with gen_* strategy
            if std_strategy is not None:
                std_aug = AUGMENTATION_STRATEGIES[std_strategy]
                config['standard_augmentation'] = {
                    'enabled': True,
                    'method': std_aug.standard_method,
                    'p_aug': std_aug.p_aug,
                    'combined_with_gen': True,  # Flag indicating combined usage
                }
        
        elif aug_strategy.type == 'standard' and aug_strategy.standard_method:
            # Standard augmentation (CutMix, MixUp, AutoAugment, RandAugment)
            config['standard_augmentation'] = {
                'enabled': True,
                'method': aug_strategy.standard_method,
                'p_aug': aug_strategy.p_aug,
                'combined_with_gen': False,
            }
        
        # Handle std_strategy combined with non-generated strategies (e.g., baseline + std_cutmix)
        elif std_strategy is not None:
            std_aug = AUGMENTATION_STRATEGIES[std_strategy]
            config['standard_augmentation'] = {
                'enabled': True,
                'method': std_aug.standard_method,
                'p_aug': std_aug.p_aug,
                'combined_with_gen': False,  # Not combined with generated images
            }
        
        return config
    
    def _add_mixed_dataloader_config(
        self,
        config: Dict[str, Any],
        dataset_cfg: DatasetConfig,
        aug_strategy: AugmentationStrategy,
        real_gen_ratio: float,
        domain_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add configuration for mixed real/generated dataloader.
        
        This enables sampling from both real and generated images according
        to the specified ratio during training.
        
        Args:
            config: Configuration dictionary to update
            dataset_cfg: Dataset configuration object
            aug_strategy: Augmentation strategy object
            real_gen_ratio: Ratio of real images (0.0-1.0)
            domain_filter: Optional domain to filter training data (e.g., 'clear_day')
        """
        
        if aug_strategy.type != 'generated' or real_gen_ratio == 1.0:
            # No mixed dataloader needed
            config['mixed_dataloader'] = {'enabled': False}
            return config
        
        gen_model = aug_strategy.generative_model
        # Convert strategy name to actual directory name (handles hyphens)
        gen_model_dir = _strategy_name_to_dir(gen_model)
        
        # Apply domain filter to training directories if specified
        train_img_dir = dataset_cfg.train_img_dir
        train_ann_dir = dataset_cfg.train_ann_dir
        
        if domain_filter:
            train_img_dir = os.path.join(train_img_dir, domain_filter)
            train_ann_dir = os.path.join(train_ann_dir, domain_filter)
        
        # Get batch size from train_dataloader (new mmengine format)
        batch_size = config['train_dataloader']['batch_size']
        
        # Get dataset type from train_dataloader
        train_dataset = config['train_dataloader']['dataset']
        dataset_type = train_dataset.get('type', 'CityscapesDataset')
        
        config['mixed_dataloader'] = {
            'enabled': True,
            'real_gen_ratio': real_gen_ratio,
            'domain_filter': domain_filter,
            'real_dataset': {
                'type': dataset_type,
                'data_root': train_dataset.get('data_root', self.data_root),
                'img_dir': train_img_dir,
                'ann_dir': train_ann_dir,
            },
            'generated_dataset': {
                'type': 'GeneratedAugmentedDataset',
                'data_root': self.data_root,
                'generated_root': os.path.join(self.gen_root, gen_model_dir),
                'manifest_path': os.path.join(self.gen_root, gen_model_dir, 'manifest.csv'),
                'conditions': aug_strategy.conditions,
                'include_original': False,  # Only generated images
            },
            'sampling_strategy': 'ratio',  # 'ratio', 'alternating', 'batch_split'
            'batch_composition': {
                'total_batch_size': batch_size,
                'real_samples': int(batch_size * real_gen_ratio),
                'generated_samples': int(batch_size * (1 - real_gen_ratio)),
            },
        }
        
        return config
    
    def _set_work_dir(
        self,
        config: Dict[str, Any],
        dataset: str,
        model: str,
        strategy: str,
        real_gen_ratio: float,
        domain_filter: Optional[str] = None,
        std_strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Set the output work directory.
        
        Args:
            config: Configuration dictionary to update
            dataset: Dataset name
            model: Model name
            strategy: Augmentation strategy name
            real_gen_ratio: Ratio of real images (0.0-1.0)
            domain_filter: Optional domain filter (e.g., 'clear_day')
            std_strategy: Optional standard augmentation combined with strategy
        """
        
        # Include ratio in directory name if not 1.0
        if real_gen_ratio < 1.0:
            ratio_str = f'_ratio{real_gen_ratio:.2f}'.replace('.', 'p')
        else:
            ratio_str = ''
        
        # Include domain filter in dataset directory name if specified
        # Format: dataset_cd for clear_day, dataset for no filter
        domain_abbrev = {
            'clear_day': 'cd',
            'clear_night': 'cn',
            'rainy_day': 'rd',
            'rainy_night': 'rn',
            'fog': 'fg',
            'snow': 'sn',
        }
        if domain_filter:
            domain_str = f'_{domain_abbrev.get(domain_filter, domain_filter[:2])}'
        else:
            domain_str = ''
        
        # Include std_strategy in directory name if combined
        if std_strategy:
            std_str = f'+{std_strategy}'
        else:
            std_str = ''
        
        config['work_dir'] = os.path.join(
            self.weights_root,
            f'{strategy}{std_str}',
            f'{dataset.lower()}{domain_str}',  # Domain filter applied to dataset
            f'{model}{ratio_str}',  # Model no longer has domain suffix
        )
        
        return config
    
    def save_config(self, config: Dict[str, Any], filepath: str) -> str:
        """
        Save configuration to a Python file.
        
        Args:
            config: Configuration dictionary
            filepath: Output file path
            
        Returns:
            Saved file path
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            # Header
            f.write("# PROVE Unified Training Configuration\n")
            f.write(f"# Generated by unified_training_config.py\n")
            if '_prove_config' in config:
                meta = config['_prove_config']
                f.write(f"# Dataset: {meta.get('dataset', 'unknown')}\n")
                f.write(f"# Model: {meta.get('model', 'unknown')}\n")
                f.write(f"# Task: {meta.get('task', 'unknown')}\n")
            f.write("\n")
            
            # Add default_scope for mmsegmentation
            f.write("# MMSegmentation default scope\n")
            f.write("default_scope = 'mmseg'\n\n")
            
            # Define order of keys to ensure pipelines are defined before dataloaders
            priority_keys = ['train_pipeline', 'test_pipeline', 'val_pipeline']
            dataloader_keys = ['train_dataloader', 'val_dataloader', 'test_dataloader']
            
            # Write pipelines first
            for key in priority_keys:
                if key in config:
                    f.write(f"{key} = {repr(config[key])}\n")
            
            # Write other config items (except dataloaders and pipelines)
            for key, value in config.items():
                if key.startswith('_') or key in priority_keys or key in dataloader_keys:
                    continue
                value_str = repr(value)
                f.write(f"{key} = {value_str}\n")
            
            # Write dataloaders last with pipeline references
            for key in dataloader_keys:
                if key in config:
                    value_str = repr(config[key])
                    # Replace pipeline placeholders with actual variable references
                    value_str = value_str.replace("'{{train_pipeline}}'", "train_pipeline")
                    value_str = value_str.replace("'{{test_pipeline}}'", "test_pipeline")
                    value_str = value_str.replace("'{{val_pipeline}}'", "val_pipeline")
                    # Handle per-dataset pipelines (e.g., train_pipeline_mapillaryvistas)
                    import re
                    for pipeline_key in config.keys():
                        if pipeline_key.startswith('train_pipeline_'):
                            placeholder = "'{{" + pipeline_key + "}}'"
                            value_str = value_str.replace(placeholder, pipeline_key)
                    f.write(f"{key} = {value_str}\n")
        
        return filepath
    
    def get_available_options(self) -> Dict[str, List[str]]:
        """Get all available configuration options"""
        return {
            'datasets': list(DATASET_CONFIGS.keys()),
            'segmentation_models': list(SEGMENTATION_MODELS.keys()),
            'detection_models': list(DETECTION_MODELS.keys()),
            'strategies': list(AUGMENTATION_STRATEGIES.keys()),
            'conditions': ADVERSE_CONDITIONS,
        }


# ============================================================================
# Command Line Interface
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='PROVE Unified Training Configuration Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate config for baseline training
  python unified_training_config.py --dataset ACDC --model deeplabv3plus_r50 --strategy baseline

  # Generate config with cycleGAN augmentation
  python unified_training_config.py --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN

  # Generate config with 50% real, 50% generated images
  python unified_training_config.py --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN --real-gen-ratio 0.5

  # Train only on clear_day images
  python unified_training_config.py --dataset ACDC --model deeplabv3plus_r50 --domain-filter clear_day

  # List available options
  python unified_training_config.py --list

  # Generate all configs for a strategy
  python unified_training_config.py --generate-all --strategy gen_cycleGAN
        """
    )
    
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--strategy', type=str, default='baseline', help='Augmentation strategy')
    parser.add_argument('--real-gen-ratio', type=float, default=1.0,
                       help='Ratio of real images (0.0-1.0). 1.0=only real, 0.5=50%% each')
    parser.add_argument('--domain-filter', type=str, default=None,
                       help='Filter training data to specific domain (e.g., clear_day). '
                            'Only images from this domain subdirectory will be used.')
    parser.add_argument('--output', '-o', type=str, help='Output config file path')
    parser.add_argument('--output-dir', type=str, default='./multi_model_configs',
                       help='Output directory for generated configs')
    parser.add_argument('--list', action='store_true', help='List available options')
    parser.add_argument('--generate-all', action='store_true',
                       help='Generate configs for all dataset/model combinations')
    parser.add_argument('--print', action='store_true', help='Print config to stdout')
    parser.add_argument('--conditions', type=str, nargs='+',
                       help='Custom weather conditions to use')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    config_builder = UnifiedTrainingConfig()
    
    if args.list:
        options = config_builder.get_available_options()
        print("PROVE Unified Training Configuration - Available Options")
        print("=" * 60)
        print("\nDatasets:")
        for ds in options['datasets']:
            task = DATASET_CONFIGS[ds].task
            print(f"  - {ds} ({task})")
        print("\nSegmentation Models:")
        for m in options['segmentation_models']:
            print(f"  - {m}")
        print("\nDetection Models:")
        for m in options['detection_models']:
            print(f"  - {m}")
        print("\nAugmentation Strategies:")
        for s in options['strategies']:
            print(f"  - {s}")
        print("\nAdverse Conditions:")
        for c in options['conditions']:
            print(f"  - {c}")
        return
    
    if args.generate_all:
        if not args.strategy:
            print("Error: --strategy required with --generate-all")
            return
        
        print(f"Generating all configs for strategy: {args.strategy}")
        print("=" * 60)
        
        generated = 0
        for dataset_name, dataset_cfg in DATASET_CONFIGS.items():
            # Get models for this task
            if dataset_cfg.task == 'segmentation':
                models = SEGMENTATION_MODELS.keys()
            else:
                models = DETECTION_MODELS.keys()
            
            for model in models:
                try:
                    config = config_builder.build(
                        dataset=dataset_name,
                        model=model,
                        strategy=args.strategy,
                        real_gen_ratio=args.real_gen_ratio,
                        custom_conditions=args.conditions,
                        domain_filter=args.domain_filter,
                    )
                    
                    # Determine output path
                    output_path = Path(args.output_dir) / args.strategy / dataset_name.upper()
                    output_file = output_path / f'{dataset_name.lower()}_{model}_config.py'
                    
                    config_builder.save_config(config, str(output_file))
                    print(f"✓ {output_file}")
                    generated += 1
                    
                except Exception as e:
                    print(f"✗ {dataset_name}/{model}: {str(e)}")
        
        print(f"\nGenerated {generated} config files in {args.output_dir}")
        return
    
    # Generate single config
    if not args.dataset or not args.model:
        print("Error: --dataset and --model required (or use --list, --generate-all)")
        return
    
    config = config_builder.build(
        dataset=args.dataset,
        model=args.model,
        strategy=args.strategy,
        real_gen_ratio=args.real_gen_ratio,
        custom_conditions=args.conditions,
        domain_filter=args.domain_filter,
    )
    
    if args.print:
        import pprint
        pprint.pprint(config)
    
    if args.output:
        config_builder.save_config(config, args.output)
        print(f"Config saved to: {args.output}")
    elif not args.print:
        # Default output
        output_path = Path(args.output_dir) / args.strategy / args.dataset.upper()
        output_file = output_path / f'{args.dataset.lower()}_{args.model}_config.py'
        config_builder.save_config(config, str(output_file))
        print(f"Config saved to: {output_file}")


if __name__ == '__main__':
    main()
