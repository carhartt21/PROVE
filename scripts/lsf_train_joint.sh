#!/bin/bash
#BSUB -J prove_joint
#BSUB -oo lsf_joint_%J_gpu.log
#BSUB -eo lsf_joint_%J_gpu.err
#BSUB -n 6
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=24G"
#BSUB -q BatchGPU
#BSUB -L /bin/bash

# PROVE LSF Training Script: Joint Cityscapes + Mapillary
# Dataset: /scratch/aaa_exchange/AWARE/FINAL_SPLITS/train/

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

# Configuration - choose label space
LABEL_SPACE=${1:-"cityscapes"}  # cityscapes or unified

if [ "$LABEL_SPACE" == "unified" ]; then
    CONFIG="configs/joint_unified_config.py"
    WORK_DIR="/scratch/aaa_exchange/AWARE/WEIGHTS/joint_deeplabv3plus_r50_unified"
else
    CONFIG="configs/joint_cityscapes_mapillary_config.py"
    WORK_DIR="/scratch/aaa_exchange/AWARE/WEIGHTS/joint_deeplabv3plus_r50_cs"
fi

echo "============================================"
echo "LSF Job: ${LSB_JOBID}"
echo "Host: ${LSB_HOSTS}"
echo "GPUs: 1"
echo "Label Space: ${LABEL_SPACE}"
echo "============================================"

# Single GPU training
python prove.py train \
    --config-path ${CONFIG} \
    --work-dir ${WORK_DIR}

echo "Training completed!"
