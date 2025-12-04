# PROVE Configuration: Joint Cityscapes + Mapillary Vistas Training
# Using Unified label space (42 classes)
# Dataset paths: /scratch/aaa_exchange/AWARE/FINAL_SPLITS/

_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

# Dataset settings
cityscapes_root = '/scratch/aaa_exchange/AWARE/FINAL_SPLITS/train/cityscapes'
mapillary_root = '/scratch/aaa_exchange/AWARE/FINAL_SPLITS/train/mapillary_vistas'

# Label space configuration - Unified 42 classes
target_space = 'unified'
num_classes = 42

# Image normalization config
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

crop_size = (512, 512)

# Cityscapes training pipeline (with label transformation to unified)
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='CityscapesLabelTransform', target_space='unified'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='PackSegInputs'),
]

# Mapillary training pipeline (with label transformation to unified)
mapillary_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='MapillaryLabelTransform', target_space='unified'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='PackSegInputs'),
]

# Test pipeline (with unified label transformation)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='CityscapesLabelTransform', target_space='unified'),
    dict(type='PackSegInputs'),
]

# Joint training dataset configuration
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            # Cityscapes training set (transformed to unified)
            dict(
                type='UnifiedCityscapesDataset',
                data_root=cityscapes_root,
                data_prefix=dict(
                    img_path='leftImg8bit/train',
                    seg_map_path='gtFine/train'
                ),
                pipeline=cityscapes_train_pipeline
            ),
            # Mapillary Vistas training set (transformed to unified)
            dict(
                type='MapillaryUnifiedDataset',
                data_root=mapillary_root,
                data_prefix=dict(
                    img_path='training/images',
                    seg_map_path='training/v1.2/labels'
                ),
                target_space='unified',
                pipeline=mapillary_train_pipeline
            ),
        ]
    )
)

# Validation on Cityscapes (with unified labels)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='UnifiedCityscapesDataset',
        data_root=cityscapes_root,
        data_prefix=dict(
            img_path='leftImg8bit/val',
            seg_map_path='gtFine/val'
        ),
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

# Evaluator
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# Unified classes (42 classes)
classes = (
    # Flat (0-4)
    'road', 'sidewalk', 'parking', 'rail track', 'bike lane',
    # Construction (5-11)
    'building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel', 'barrier',
    # Object (12-17)
    'pole', 'traffic light', 'traffic sign', 'street light', 'utility pole', 'other object',
    # Nature (18-23)
    'vegetation', 'terrain', 'sky', 'water', 'snow', 'mountain',
    # Human (24-27)
    'person', 'bicyclist', 'motorcyclist', 'other rider',
    # Vehicle (28-39)
    'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
    'caravan', 'trailer', 'boat', 'other vehicle', 'wheeled slow', 'animal',
    # Marking (40-41)
    'lane marking', 'crosswalk'
)

# Palette for visualization (unified)
palette = [
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

# Model configuration - update num_classes
model = dict(
    decode_head=dict(num_classes=num_classes),
    auxiliary_head=dict(num_classes=num_classes)
)
