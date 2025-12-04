# PROVE Training Scripts

Shell scripts for training semantic segmentation models on different datasets.

## Dataset Location

All scripts use datasets from:
```
/scratch/aaa_exchange/AWARE/FINAL_SPLITS/train/
├── cityscapes/
└── mapillary_vistas/
```

## Scripts Overview

### Local Training Scripts

| Script | Dataset | Label Space | Classes |
|--------|---------|-------------|---------|
| `train_cityscapes.sh` | Cityscapes | Cityscapes | 19 |
| `train_mapillary.sh` | Mapillary Vistas | Cityscapes | 19 |
| `train_mapillary_unified.sh` | Mapillary Vistas | Unified | 42 |
| `train_joint_cityscapes.sh` | CS + Mapillary | Cityscapes | 19 |
| `train_joint_unified.sh` | CS + Mapillary | Unified | 42 |
| `train_all.sh` | All above | - | - |

### LSF Scripts (HPC Cluster)

| Script | Dataset | Description |
|--------|---------|-------------|
| `lsf_train_cityscapes.sh` | Cityscapes | Single GPU, 24h |
| `lsf_train_mapillary.sh` | Mapillary | Single GPU, 48h |
| `lsf_train_joint.sh` | Joint | Multi-GPU, 72h |
| `lsf_submit_all.sh` | All | Submit all jobs |

## Usage

### Local Training

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Train on Cityscapes (single GPU)
./scripts/train_cityscapes.sh

# Train on Cityscapes (multi-GPU)
./scripts/train_cityscapes.sh --gpus 4

# Train with custom work directory
./scripts/train_cityscapes.sh --work-dir ./my_experiment/

# Resume training
./scripts/train_cityscapes.sh --resume

# Train all models sequentially
GPUS=2 ./scripts/train_all.sh
```

### LSF Cluster Training

```bash
# Submit single job
bsub < scripts/lsf_train_cityscapes.sh

# Submit joint training with specific label space
bsub < scripts/lsf_train_joint.sh

# Submit all training jobs
./scripts/lsf_submit_all.sh

# Monitor jobs
bjobs -u $USER

# Cancel job
bkill <job_id>
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GPUS` | 1 | Number of GPUs to use |
| `PORT` | 29500+ | Master port for distributed training |

## Customization

### Modify LSF Resources

Edit the LSF scripts to adjust:
- `#BSUB -n N` - Number of tasks
- `#BSUB -R "rusage[ngpus_excl_p=N]"` - Number of GPUs
- `#BSUB -R "rusage[mem=XMB]"` - Memory allocation
- `#BSUB -W HH:MM` - Time limit
- `#BSUB -q NAME` - Queue name

## Output

Training outputs are saved to:
```
/scratch/aaa_exchange/AWARE/WEIGHTS/<experiment_name>/
├── *.log           # Training logs
├── *.json          # Training metrics
├── iter_*.pth      # Checkpoint files
└── latest.pth      # Latest checkpoint
```

SLURM logs are saved to:
```
./logs/
├── <job_name>.out  # Standard output
└── <job_name>.err  # Standard error
```
