#!/bin/bash
# PROVE Training Script: Train all models sequentially
# Dataset: /scratch/aaa_exchange/AWARE/FINAL_SPLITS/train/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS=${GPUS:-1}

echo "============================================"
echo "PROVE: Training All Models"
echo "============================================"
echo "GPUs: ${GPUS}"
echo "============================================"

# Train Cityscapes
echo ""
echo "[1/5] Training on Cityscapes..."
bash ${SCRIPT_DIR}/train_cityscapes.sh --gpus ${GPUS}

# Train Mapillary (Cityscapes labels)
echo ""
echo "[2/5] Training on Mapillary Vistas (Cityscapes labels)..."
bash ${SCRIPT_DIR}/train_mapillary.sh --gpus ${GPUS}

# Train Mapillary (Unified labels)
echo ""
echo "[3/5] Training on Mapillary Vistas (Unified labels)..."
bash ${SCRIPT_DIR}/train_mapillary_unified.sh --gpus ${GPUS}

# Train Joint (Cityscapes labels)
echo ""
echo "[4/5] Training Joint CS + Mapillary (Cityscapes labels)..."
bash ${SCRIPT_DIR}/train_joint_cityscapes.sh --gpus ${GPUS}

# Train Joint (Unified labels)
echo ""
echo "[5/5] Training Joint CS + Mapillary (Unified labels)..."
bash ${SCRIPT_DIR}/train_joint_unified.sh --gpus ${GPUS}

echo ""
echo "============================================"
echo "All training completed!"
echo "============================================"
