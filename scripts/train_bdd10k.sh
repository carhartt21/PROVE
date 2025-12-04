#!/bin/bash
# PROVE Training Script: BDD10k Dataset
# Dataset: /scratch/aaa_exchange/AWARE/FINAL_SPLITS/train/images/BDD10k

set -e

# Configuration
CONFIG="configs/bdd10k_config.py"
WORK_DIR="/scratch/aaa_exchange/AWARE/WEIGHTS/bdd10k_deeplabv3plus_r50"
DATA_ROOT="/scratch/aaa_exchange/AWARE/FINAL_SPLITS"
GPUS=${GPUS:-1}
PORT=${PORT:-29502}

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
echo "PROVE Training: BDD10k"
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
