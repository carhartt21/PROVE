# PROVE Augmentation Config: Generated Images (cycleGAN)
# Dataset: BDD10k
# Model: deeplabv3plus_r50
# Augmentation: 7x (original + 6 adverse conditions)

_base_ = ['../_base_/models/deeplabv3plus_r50.py']
runner = {'type': 'IterBasedRunner', 'max_iters': 40000}
checkpoint_config = {'interval': 5000}
evaluation = {'interval': 3333, 'metric': 'mIoU'}
log_config = {'interval': 50, 'hooks': [{'type': 'TextLoggerHook'}, {'type': 'TensorboardLoggerHook'}]}
data = {'samples_per_gpu': 2, 'workers_per_gpu': 4}
train_pipeline = [{'type': 'LoadImageFromFile'}, {'type': 'LoadAnnotations'}, {'type': 'Resize', 'scale': (512, 512), 'keep_ratio': True}, {'type': 'PackSegInputs'}]
generated_augmentation = {'enabled': True, 'generative_model': 'cycleGAN', 'manifest_path': '/scratch/aaa_exchange/AWARE/GENERATED_IMAGES/cycleGAN/manifest.csv', 'gen_root': '/scratch/aaa_exchange/AWARE/GENERATED_IMAGES/cycleGAN', 'conditions': ['cloudy', 'dawn_dusk', 'fog', 'night', 'rainy', 'snowy'], 'augmentation_multiplier': 7}
work_dir = '/scratch/aaa_exchange/AWARE/WEIGHTS/gen_cycleGAN/bdd10k/deeplabv3plus_r50'
seed = 42
deterministic = True
