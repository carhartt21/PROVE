#!/bin/bash
# PROVE Training Script: Mapillary Vistas Dataset (Cityscapes label space)
# Dataset: /scratch/aaa_exchange/AWARE/FINAL_SPLITS/train/mapillary_vistas

set -e

# Configuration
CONFIG="configs/mapillary_vistas_config.py"
WORK_DIR="/scratch/aaa_exchange/AWARE/WEIGHTS/mapillary_deeplabv3plus_r50_cs"
DATA_ROOT="/scratch/aaa_exchange/AWARE/FINAL_SPLITS/train/mapillary_vistas"
GPUS=${GPUS:-1}
PORT=${PORT:-29501}

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
echo "PROVE Training: Mapillary Vistas (CS labels)"
echo "============================================"
echo "Config: ${CONFIG}"
echo "Work Dir: ${WORK_DIR}"
echo "Data Root: ${DATA_ROOT}"
echo "GPUs: ${GPUS}"
echo "Label Space: Cityscapes (19 classes)"
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
