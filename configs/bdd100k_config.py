# PROVE Configuration: BDD100k Object Detection Dataset
# Dataset path: /scratch/aaa_exchange/AWARE/FINAL_SPLITS/
# Format: BDD100k JSON with box2d annotations

_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_1x.py'
]

# Dataset settings - custom BDD100k dataset
dataset_type = 'CocoDataset'
data_root = '/scratch/aaa_exchange/AWARE/FINAL_SPLITS'

# BDD100k object detection classes
# Mapping: person->pedestrian, bike->bicycle, motor->motorcycle
classes = (
    'pedestrian', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign'
)
num_classes = 10

# Image normalization config
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

# Training pipeline for object detection
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='PackDetInputs')
]

# Test pipeline
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

# Dataset configuration
# Note: BDD100k JSON labels need to be converted to COCO format
# Labels are in: /scratch/aaa_exchange/AWARE/FINAL_SPLITS/{train,test}/labels/BDD100k/<weather>/*.json
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/labels/BDD100k/bdd100k_train_coco.json',  # Converted COCO format
        data_prefix=dict(img='train/images/BDD100k/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        metainfo=dict(classes=classes)
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test/labels/BDD100k/bdd100k_test_coco.json',  # Converted COCO format
        data_prefix=dict(img='test/images/BDD100k/'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=dict(classes=classes)
    )
)

test_dataloader = val_dataloader

# Evaluator - COCO-style mAP
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/test/labels/BDD100k/bdd100k_test_coco.json',
    metric='bbox',
    format_only=False
)
test_evaluator = val_evaluator

# Model configuration - update num_classes for Faster R-CNN
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes)
    )
)

# Palette for visualization
palette = [
    (220, 20, 60),    # pedestrian - red
    (255, 0, 0),      # rider - bright red
    (0, 0, 142),      # car - dark blue
    (0, 0, 70),       # truck - navy
    (0, 60, 100),     # bus - dark cyan
    (0, 80, 100),     # train - teal
    (0, 0, 230),      # motorcycle - blue
    (119, 11, 32),    # bicycle - dark red
    (250, 170, 30),   # traffic light - orange
    (220, 220, 0),    # traffic sign - yellow
]
