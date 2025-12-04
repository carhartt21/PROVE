#!/bin/bash
#BSUB -J prove_outside15k
#BSUB -oo lsf_outside15k_%J_gpu.log
#BSUB -eo lsf_outside15k_%J_gpu.err
#BSUB -n 6
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=24G"
#BSUB -q BatchGPU
#BSUB -L /bin/bash

# PROVE LSF Training Script: OUTSIDE15k Dataset
# Dataset: /scratch/aaa_exchange/AWARE/FINAL_SPLITS/

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
CONFIG="configs/outside15k_config.py"
WORK_DIR="/scratch/aaa_exchange/AWARE/WEIGHTS/outside15k_deeplabv3plus_r50"

echo "============================================"
echo "LSF Job: ${LSB_JOBID}"
echo "Host: ${LSB_HOSTS}"
echo "GPUs: 1"
echo "Dataset: OUTSIDE15k"
echo "============================================"

# Run training
python prove.py train \
    --config-path ${CONFIG} \
    --work-dir ${WORK_DIR}

echo "Training completed!"
