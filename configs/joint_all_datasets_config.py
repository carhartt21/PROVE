# PROVE Configuration: Joint Training on ALL AWARE Datasets
# Dataset path: /scratch/aaa_exchange/AWARE/FINAL_SPLITS/

_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_320k.py'
]

# Dataset settings
data_root = '/scratch/aaa_exchange/AWARE/FINAL_SPLITS'

# Image normalization config
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

crop_size = (512, 512)

# Common training pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='PackSegInputs'),
]

# Test pipeline
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
            # ACDC
            dict(
                type='CustomDataset',
                data_root=data_root,
                data_prefix=dict(
                    img_path='train/images/ACDC',
                    seg_map_path='train/labels/ACDC'
                ),
                img_suffix='.png',
                seg_map_suffix='.png',
                pipeline=train_pipeline
            ),
            # BDD100k
            dict(
                type='CustomDataset',
                data_root=data_root,
                data_prefix=dict(
                    img_path='train/images/BDD100k',
                    seg_map_path='train/labels/BDD100k'
                ),
                img_suffix='.jpg',
                seg_map_suffix='.png',
                pipeline=train_pipeline
            ),
            # BDD10k
            dict(
                type='CustomDataset',
                data_root=data_root,
                data_prefix=dict(
                    img_path='train/images/BDD10k',
                    seg_map_path='train/labels/BDD10k'
                ),
                img_suffix='.jpg',
                seg_map_suffix='.png',
                pipeline=train_pipeline
            ),
            # OUTSIDE15k
            dict(
                type='CustomDataset',
                data_root=data_root,
                data_prefix=dict(
                    img_path='train/images/OUTSIDE15k',
                    seg_map_path='train/labels/OUTSIDE15k'
                ),
                img_suffix='.png',
                seg_map_suffix='.png',
                pipeline=train_pipeline
            ),
            # IDD-AW
            dict(
                type='CustomDataset',
                data_root=data_root,
                data_prefix=dict(
                    img_path='train/images/IDD-AW',
                    seg_map_path='train/labels/IDD-AW'
                ),
                img_suffix='.png',
                seg_map_suffix='.png',
                pipeline=train_pipeline
            ),
            # MapillaryVistas
            dict(
                type='CustomDataset',
                data_root=data_root,
                data_prefix=dict(
                    img_path='train/images/MapillaryVistas',
                    seg_map_path='train/labels/MapillaryVistas'
                ),
                img_suffix='.jpg',
                seg_map_suffix='.png',
                pipeline=train_pipeline
            ),
        ]
    )
)

# Validation on ACDC (representative adverse weather dataset)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        data_prefix=dict(
            img_path='test/images/ACDC',
            seg_map_path='test/labels/ACDC'
        ),
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

# Evaluator
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# Classes (Cityscapes 19 classes)
classes = (
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle'
)
num_classes = 19

# Palette for visualization
palette = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
]
