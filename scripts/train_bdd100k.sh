#!/bin/bash
# PROVE Training Script: BDD100k Object Detection Dataset
# Dataset: /scratch/aaa_exchange/AWARE/FINAL_SPLITS/train/images/BDD100k
# Task: Object Detection (Faster R-CNN)

set -e

# Configuration
CONFIG="configs/bdd100k_config.py"
WORK_DIR="/scratch/aaa_exchange/AWARE/WEIGHTS/bdd100k_fasterrcnn_r50"
DATA_ROOT="/scratch/aaa_exchange/AWARE/FINAL_SPLITS"
GPUS=${GPUS:-1}
PORT=${PORT:-29501}

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

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --resume)
            RESUME="--resume"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "============================================"
echo "PROVE Training: BDD100k"
echo "============================================"
echo "Config: ${CONFIG}"
echo "Work Dir: ${WORK_DIR}"
echo "Data Root: ${DATA_ROOT}"
echo "GPUs: ${GPUS}"
echo "============================================"

# Create work directory
mkdir -p ${WORK_DIR}

# Single GPU training
if [ ${GPUS} -eq 1 ]; then
    python prove.py train \
        --config-path ${CONFIG} \
        --work-dir ${WORK_DIR} \
        ${RESUME}
else
    # Multi-GPU training with PyTorch distributed
    python -m torch.distributed.launch \
        --nproc_per_node=${GPUS} \
        --master_port=${PORT} \
        prove.py train \
        --config-path ${CONFIG} \
        --work-dir ${WORK_DIR} \
        --launcher pytorch \
        ${RESUME}
fi

echo "Training completed! Results saved to ${WORK_DIR}"
