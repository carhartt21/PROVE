# PROVE Configuration: Joint Cityscapes + Mapillary Vistas Training
# Using Cityscapes label space (19 classes)
# Dataset paths: /scratch/aaa_exchange/AWARE/FINAL_SPLITS/

_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

# Dataset settings
cityscapes_root = '/scratch/aaa_exchange/AWARE/FINAL_SPLITS/train/cityscapes'
mapillary_root = '/scratch/aaa_exchange/AWARE/FINAL_SPLITS/train/mapillary_vistas'

# Label space configuration
target_space = 'cityscapes'
num_classes = 19

# Image normalization config
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

crop_size = (512, 512)

# Cityscapes training pipeline
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='PackSegInputs'),
]

# Mapillary training pipeline (with label transformation)
mapillary_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='MapillaryLabelTransform', target_space=target_space),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='PackSegInputs'),
]

# Test pipeline (Cityscapes validation)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
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
            # Cityscapes training set
            dict(
                type='CityscapesDataset',
                data_root=cityscapes_root,
                data_prefix=dict(
                    img_path='leftImg8bit/train',
                    seg_map_path='gtFine/train'
                ),
                pipeline=cityscapes_train_pipeline
            ),
            # Mapillary Vistas training set
            dict(
                type='MapillaryUnifiedDataset',
                data_root=mapillary_root,
                data_prefix=dict(
                    img_path='training/images',
                    seg_map_path='training/v1.2/labels'
                ),
                target_space=target_space,
                pipeline=mapillary_train_pipeline
            ),
        ]
    )
)

# Validation on Cityscapes
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CityscapesDataset',
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

# Classes (19 evaluation classes)
classes = (
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle'
)

# Palette for visualization
palette = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
]

# Model configuration - update num_classes
model = dict(
    decode_head=dict(num_classes=num_classes),
    auxiliary_head=dict(num_classes=num_classes)
)
