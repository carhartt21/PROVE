#!/bin/bash
# =============================================================================
# Retrain std_minimal and gen_cyclediffusion on MapillaryVistas with native classes
# 
# Issue: These strategies were trained with 19 Cityscapes classes instead of 
# 66 native MapillaryVistas classes, making metrics incomparable.
# 
# Solution: Retrain with --use-native-classes flag
# =============================================================================

set -e

PROVE_DIR="/home/mima2416/repositories/PROVE"
WEIGHTS_DIR="/scratch/aaa_exchange/AWARE/WEIGHTS"
LOG_DIR="${PROVE_DIR}/logs"
QUEUE="BatchGPU"
GPU_MEM="24G"
GPU_MODE="shared"
NUM_CPUS=10
WALL_TIME="24:00"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create log directory
mkdir -p "$LOG_DIR"

echo "==================================================================="
echo "Retraining with Native Classes - $TIMESTAMP"
echo "==================================================================="

# Step 1: Backup existing models (19-class versions)
echo ""
echo "Step 1: Backing up existing models..."

# std_minimal backup
SRC_DIR="${WEIGHTS_DIR}/std_minimal/mapillaryvistas_cd/deeplabv3plus_r50"
if [ -d "$SRC_DIR" ]; then
    BACKUP_DIR="${SRC_DIR}_19class_backup_${TIMESTAMP}"
    echo "  Backing up std_minimal: $SRC_DIR -> $BACKUP_DIR"
    mv "$SRC_DIR" "$BACKUP_DIR"
    echo "  Done."
fi

# gen_cyclediffusion backup
SRC_DIR="${WEIGHTS_DIR}/gen_cyclediffusion/mapillaryvistas_cd/deeplabv3plus_r50_ratio0p50"
if [ -d "$SRC_DIR" ]; then
    BACKUP_DIR="${SRC_DIR}_19class_backup_${TIMESTAMP}"
    echo "  Backing up gen_cyclediffusion: $SRC_DIR -> $BACKUP_DIR"
    mv "$SRC_DIR" "$BACKUP_DIR"
    echo "  Done."
fi

echo ""
echo "Step 2: Submitting retraining jobs..."

# Job 1: std_minimal on MapillaryVistas with native classes
JOB_NAME="retrain_std_minimal_mapillary_native"
TRAIN_CMD="cd ${PROVE_DIR} && source ~/.bashrc && conda activate prove && python unified_training.py --dataset MapillaryVistas --model deeplabv3plus_r50 --strategy std_minimal --domain-filter clear_day --use-native-classes"

echo ""
echo "  Submitting: ${JOB_NAME}"
echo "  Command: ${TRAIN_CMD}"

bsub -J "${JOB_NAME}" \
     -q "${QUEUE}" \
     -gpu "num=1:mode=${GPU_MODE}:gmem=${GPU_MEM}" \
     -n "${NUM_CPUS}" \
     -W "${WALL_TIME}" \
     -o "${LOG_DIR}/${JOB_NAME}_${TIMESTAMP}.log" \
     -e "${LOG_DIR}/${JOB_NAME}_${TIMESTAMP}.err" \
     "${TRAIN_CMD}"

# Job 2: gen_cyclediffusion on MapillaryVistas with native classes  
JOB_NAME="retrain_gen_cyclediffusion_mapillary_native"
TRAIN_CMD="cd ${PROVE_DIR} && source ~/.bashrc && conda activate prove && python unified_training.py --dataset MapillaryVistas --model deeplabv3plus_r50 --strategy gen_cyclediffusion --domain-filter clear_day --use-native-classes --real-gen-ratio 0.5"

echo ""
echo "  Submitting: ${JOB_NAME}"
echo "  Command: ${TRAIN_CMD}"

bsub -J "${JOB_NAME}" \
     -q "${QUEUE}" \
     -gpu "num=1:mode=${GPU_MODE}:gmem=${GPU_MEM}" \
     -n "${NUM_CPUS}" \
     -W "${WALL_TIME}" \
     -o "${LOG_DIR}/${JOB_NAME}_${TIMESTAMP}.log" \
     -e "${LOG_DIR}/${JOB_NAME}_${TIMESTAMP}.err" \
     "${TRAIN_CMD}"

echo ""
echo "==================================================================="
echo "Jobs submitted!"
echo ""
echo "Old models backed up with _19class_backup_${TIMESTAMP} suffix"
echo "New models will be trained with 66 native MapillaryVistas classes"
echo ""
echo "After training completes, run tests with:"
echo "  python scripts/auto_submit_tests.py"
echo "==================================================================="
