#!/bin/bash
#BSUB -J prove_bdd100k
#BSUB -oo lsf_bdd100k_%J_gpu.log
#BSUB -eo lsf_bdd100k_%J_gpu.err
#BSUB -n 6
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=24G"
#BSUB -q BatchGPU
#BSUB -L /bin/bash

# PROVE LSF Training Script: BDD100k Object Detection Dataset
# Dataset: /scratch/aaa_exchange/AWARE/FINAL_SPLITS/
# Task: Object Detection (Faster R-CNN)

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
CONFIG="configs/bdd100k_config.py"
WORK_DIR="/scratch/aaa_exchange/AWARE/WEIGHTS/bdd100k_fasterrcnn_r50"
DATA_ROOT="/scratch/aaa_exchange/AWARE/FINAL_SPLITS"

# Check if COCO format labels exist, if not convert them
TRAIN_COCO="${DATA_ROOT}/train/labels/BDD100k/bdd100k_train_coco.json"
TEST_COCO="${DATA_ROOT}/test/labels/BDD100k/bdd100k_test_coco.json"

if [ ! -f "${TRAIN_COCO}" ]; then
    echo "Converting BDD100k train labels to COCO format..."
    python tools/convert_bdd100k_to_coco.py \
        --input-dir "${DATA_ROOT}/train/labels/BDD100k" \
        --image-dir "${DATA_ROOT}/train/images/BDD100k" \
        --output "${TRAIN_COCO}"
fi

if [ ! -f "${TEST_COCO}" ]; then
    echo "Converting BDD100k test labels to COCO format..."
    python tools/convert_bdd100k_to_coco.py \
        --input-dir "${DATA_ROOT}/test/labels/BDD100k" \
        --image-dir "${DATA_ROOT}/test/images/BDD100k" \
        --output "${TEST_COCO}"
fi

echo "============================================"
echo "LSF Job: ${LSB_JOBID}"
echo "Host: ${LSB_HOSTS}"
echo "GPUs: 1"
echo "Dataset: BDD100k (Object Detection)"
echo "============================================"

# Single GPU training
python prove.py train \
    --config-path ${CONFIG} \
    --work-dir ${WORK_DIR}

echo "Training completed!"
