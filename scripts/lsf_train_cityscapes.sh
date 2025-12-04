#!/bin/bash
#BSUB -J prove_cityscapes
#BSUB -oo lsf_cityscapes_%J_gpu.log
#BSUB -eo lsf_cityscapes_%J_gpu.err
#BSUB -n 6
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=24G"
#BSUB -q BatchGPU
#BSUB -L /bin/bash

# PROVE LSF Training Script: Cityscapes Dataset
# Dataset: /scratch/aaa_exchange/AWARE/FINAL_SPLITS/train/cityscapes

set -e

# Load modules (adjust based on your cluster)
module purge
module load anaconda3
module load cuda/11.8

# Activate conda environment
source activate prove

# Change to project directory
cd $LS_SUBCWD

# Create logs directory
mkdir -p logs

# Configuration
CONFIG="configs/cityscapes_config.py"
WORK_DIR="/scratch/aaa_exchange/AWARE/WEIGHTS/cityscapes_deeplabv3plus_r50"

echo "============================================"
echo "LSF Job: ${LSB_JOBID}"
echo "Host: ${LSB_HOSTS}"
echo "GPUs: 1"
echo "============================================"

# Run training
python prove.py train \
    --config-path ${CONFIG} \
    --work-dir ${WORK_DIR}

echo "Training completed!"
