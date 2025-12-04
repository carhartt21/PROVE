_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py',
]
checkpoint_config = dict(interval=5000)
classes = (
    'road',
    'sidewalk',
    'building',
    'wall',
    'fence',
    'pole',
    'traffic light',
    'traffic sign',
    'vegetation',
    'terrain',
    'sky',
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle',
)
data = dict(
    samples_per_gpu=2,
    test=dict(
        ann_dir='gtFine/test',
        data_root='/scratch/aaa_exchange/AWARE/FINAL_SPLITS',
        img_dir='leftImg8bit/test',
        type='CityscapesDataset'),
    train=dict(
        ann_dir='gtFine/train',
        data_root='/scratch/aaa_exchange/AWARE/FINAL_SPLITS',
        img_dir='leftImg8bit/train',
        type='CityscapesDataset'),
    val=dict(
        ann_dir='gtFine/val',
        data_root='/scratch/aaa_exchange/AWARE/FINAL_SPLITS',
        img_dir='leftImg8bit/val',
        type='CityscapesDataset'),
    workers_per_gpu=4)
data_root = '/scratch/aaa_exchange/AWARE/FINAL_SPLITS'
dataset_type = 'CityscapesDataset'
deterministic = True
evaluation = dict(interval=3333, metric='mIoU')
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
