# PROVE Augmentation Config: PhotoMetricDistort
# Dataset: BDD100k
# Model: faster_rcnn_r50_fpn_1x

_base_ = ['../_base_/models/faster_rcnn_r50_fpn_1x.py']
runner = {'type': 'IterBasedRunner', 'max_iters': 40000}
checkpoint_config = {'interval': 5000}
evaluation = {'interval': 3333, 'metric': 'bbox'}
log_config = {'interval': 50, 'hooks': [{'type': 'TextLoggerHook'}, {'type': 'TensorboardLoggerHook'}]}
data = {'samples_per_gpu': 2, 'workers_per_gpu': 4}
train_pipeline = [{'type': 'LoadImageFromFile'}, {'type': 'LoadAnnotations'}, {'type': 'Resize', 'scale': (512, 512), 'keep_ratio': True}, {'type': 'PhotoMetricDistortion'}, {'type': 'PackDetInputs'}]
work_dir = '/scratch/aaa_exchange/AWARE/WEIGHTS/photometric_distort/bdd100k/faster_rcnn_r50_fpn_1x'
seed = 42
deterministic = True
