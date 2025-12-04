#!/bin/bash
# PROVE Training Script: Joint Cityscapes + Mapillary (Unified label space)
# Dataset: /scratch/aaa_exchange/AWARE/FINAL_SPLITS/train/

set -e

# Configuration
CONFIG="configs/joint_unified_config.py"
WORK_DIR="/scratch/aaa_exchange/AWARE/WEIGHTS/joint_deeplabv3plus_r50_unified"
DATA_ROOT="/scratch/aaa_exchange/AWARE/FINAL_SPLITS/train"
GPUS=${GPUS:-1}
PORT=${PORT:-29504}

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
echo "PROVE Training: Joint CS + Mapillary (Unified)"
echo "============================================"
echo "Config: ${CONFIG}"
echo "Work Dir: ${WORK_DIR}"
echo "Data Root: ${DATA_ROOT}"
echo "GPUs: ${GPUS}"
echo "Label Space: Unified (42 classes)"
echo "Datasets: Cityscapes + Mapillary Vistas"
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
