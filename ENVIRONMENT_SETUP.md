# PROVE Environment Quick Reference

## Environment Setup

The PROVE environment has been created with the following compatible versions:

| Package | Version |
|---------|---------|
| Python | 3.10 |
| PyTorch | 2.1.2 |
| CUDA | 11.8 |
| MMCV | 2.1.0 |
| MMEngine | 0.10.7 |
| MMSegmentation | 1.2.2 |
| MMDetection | 3.3.0 |

## Activation

```bash
# Activate the environment
mamba activate prove

# Load project environment variables
cd ${HOME}/repositories/PROVE
source .env
```

## Quick Commands

```bash
# List all available options
bash train_unified.sh list

# Generate config only (no training)
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy baseline --config-only

# Train with baseline augmentation
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy baseline

# Train with generated images (cycleGAN)
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN

# Mixed training (50% real, 50% generated)
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN --real-gen-ratio 0.5

# Train with domain filter
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy baseline --domain-filter clear_day
```

## Files Created

- `environment.yml` - Mamba/Conda environment specification
- `setup_env.sh` - Automated setup script with verification
- `.env` - Project environment variables

## Troubleshooting

If you need to recreate the environment:
```bash
mamba env remove -n prove -y
mamba env create -f environment.yml
mamba activate prove
pip install openmim pycocotools cityscapesscripts wandb mmsegmentation==1.2.2 mmdet==3.3.0
```

## Data Paths

Configure in `.env`:
- `PROVE_DATA_ROOT` - Path to dataset root
- `PROVE_GEN_ROOT` - Path to generated images
- `PROVE_WEIGHTS_ROOT` - Path to model weights output
