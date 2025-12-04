# PROVE Augmentation Config: PhotoMetricDistort
# Dataset: IDD-AW
# Model: deeplabv3plus_r50

_base_ = ['../_base_/models/deeplabv3plus_r50.py']
runner = {'type': 'IterBasedRunner', 'max_iters': 40000}
checkpoint_config = {'interval': 5000}
evaluation = {'interval': 3333, 'metric': 'mIoU'}
log_config = {'interval': 50, 'hooks': [{'type': 'TextLoggerHook'}, {'type': 'TensorboardLoggerHook'}]}
data = {'samples_per_gpu': 2, 'workers_per_gpu': 4}
train_pipeline = [{'type': 'LoadImageFromFile'}, {'type': 'LoadAnnotations'}, {'type': 'Resize', 'scale': (512, 512), 'keep_ratio': True}, {'type': 'PhotoMetricDistortion'}, {'type': 'PackSegInputs'}]
work_dir = '/scratch/aaa_exchange/AWARE/WEIGHTS/photometric_distort/idd-aw/deeplabv3plus_r50'
seed = 42
deterministic = True
