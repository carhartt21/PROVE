_base_ = [
    '../_base_/models/yolox_l.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]
checkpoint_config = dict(interval=5000)
classes = (
    'pedestrian',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle',
    'traffic light',
    'traffic sign',
)
data = dict(
    samples_per_gpu=2,
    test=dict(
        ann_file=
        '/scratch/aaa_exchange/AWARE/FINAL_SPLITS/labels/bdd100k_labels_images_val.json',
        classes=None,
        img_prefix='/scratch/aaa_exchange/AWARE/FINAL_SPLITS/images/100k/val/',
        type='CocoDataset'),
    train=dict(
        ann_file=
        '/scratch/aaa_exchange/AWARE/FINAL_SPLITS/labels/bdd100k_labels_images_train.json',
        classes=None,
        img_prefix=
        '/scratch/aaa_exchange/AWARE/FINAL_SPLITS/images/100k/train/',
        type='CocoDataset'),
    val=dict(
        ann_file=
        '/scratch/aaa_exchange/AWARE/FINAL_SPLITS/labels/bdd100k_labels_images_val.json',
        classes=None,
        img_prefix='/scratch/aaa_exchange/AWARE/FINAL_SPLITS/images/100k/val/',
        type='CocoDataset'),
    workers_per_gpu=4)
data_root = '/scratch/aaa_exchange/AWARE/FINAL_SPLITS'
dataset_type = 'CocoDataset'
deterministic = True
evaluation = dict(interval=3333, metric='bbox')
gpu_ids = [
    0,
]
img_norm_cfg = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
    interval=50)
lr_config = dict(
    policy='step',
    step=[
        8,
        11,
    ],
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001)
optimizer = dict(lr=0.02, momentum=0.9, type='SGD', weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
runner = dict(max_iters=40000, type='IterBasedRunner')
seed = 42
test_pipeline = []
train_pipeline = []
work_dir = './work_dirs/'
