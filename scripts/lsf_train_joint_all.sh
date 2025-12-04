#!/bin/bash
#BSUB -J prove_joint_all
#BSUB -oo lsf_joint_all_%J_gpu.log
#BSUB -eo lsf_joint_all_%J_gpu.err
#BSUB -n 6
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=24G"
#BSUB -q BatchGPU
#BSUB -L /bin/bash

# PROVE LSF Training Script: Joint Training on ALL AWARE Datasets
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
CONFIG="configs/joint_all_datasets_config.py"
WORK_DIR="/scratch/aaa_exchange/AWARE/WEIGHTS/joint_all_deeplabv3plus_r50"

echo "============================================"
echo "LSF Job: ${LSB_JOBID}"
echo "Host: ${LSB_HOSTS}"
echo "GPUs: 1"
echo "Datasets: ACDC, BDD100k, BDD10k, OUTSIDE15k, IDD-AW, MapillaryVistas"
echo "============================================"

# Single GPU training
python prove.py train \
    --config-path ${CONFIG} \
    --work-dir ${WORK_DIR}

echo "Training completed!"
