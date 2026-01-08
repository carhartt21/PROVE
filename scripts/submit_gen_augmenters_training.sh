#!/bin/bash
# ============================================================================
# PROVE: Submit Missing gen_augmenters Training Jobs
# ============================================================================
# This script submits training jobs for the missing gen_augmenters models.
#
# Usage:
#   ./scripts/submit_gen_augmenters_training.sh [--dry-run]
#
# ============================================================================

DRY_RUN=false
WEIGHTS_DIR="/scratch/aaa_exchange/AWARE/WEIGHTS"
DATA_ROOT="/scratch/aaa_exchange/AWARE/FINAL_SPLITS"
QUEUE="BatchGPU"
GPU_MEM="16G"
MAX_TIME="24:00"
STRATEGY="gen_augmenters"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Missing models - format: dataset|model|variant
MISSING_MODELS=(
    "BDD10k|pspnet_r50|_clear_day"
    "BDD10k|segformer_mit-b5|"
    "BDD10k|segformer_mit-b5|_clear_day"
    "IDD-AW|deeplabv3plus_r50|"
    "IDD-AW|deeplabv3plus_r50|_clear_day"
    "IDD-AW|pspnet_r50|"
    "IDD-AW|pspnet_r50|_clear_day"
    "IDD-AW|segformer_mit-b5|"
    "IDD-AW|segformer_mit-b5|_clear_day"
    "MapillaryVistas|deeplabv3plus_r50|"
    "MapillaryVistas|deeplabv3plus_r50|_clear_day"
    "MapillaryVistas|pspnet_r50|"
    "MapillaryVistas|pspnet_r50|_clear_day"
    "MapillaryVistas|segformer_mit-b5|"
    "MapillaryVistas|segformer_mit-b5|_clear_day"
)

echo "========================================================================"
echo "PROVE gen_augmenters Training Job Submission"
echo "========================================================================"
echo ""
echo "Strategy: $STRATEGY"
echo "Dry Run: $DRY_RUN"
echo "Queue: $QUEUE"
echo "GPU Memory: $GPU_MEM"
echo "Max Time: $MAX_TIME"
echo "Total missing models: ${#MISSING_MODELS[@]}"
echo ""

submitted=0
skipped=0

for job in "${MISSING_MODELS[@]}"; do
    IFS='|' read -r dataset model variant <<< "$job"
    
    full_model="${model}${variant}"
    
    echo "----------------------------------------"
    echo "Dataset: $dataset"
    echo "Model: $full_model"
    echo "Strategy: $STRATEGY"
    
    # Build job name
    JOB_NAME="prove_${dataset}_${full_model}_${STRATEGY}"
    
    # Build training command
    if [[ -n "$variant" ]]; then
        # Clear day variant
        TRAIN_CMD="mamba run -n prove python unified_training.py --dataset ${dataset} --model ${model} --strategy ${STRATEGY} --domain-filter clear_day --ratio 1.0 --work-dir ${WEIGHTS_DIR} --data-root ${DATA_ROOT} --no-early-stop"
    else
        # Full model
        TRAIN_CMD="mamba run -n prove python unified_training.py --dataset ${dataset} --model ${model} --strategy ${STRATEGY} --ratio 1.0 --work-dir ${WEIGHTS_DIR} --data-root ${DATA_ROOT} --no-early-stop"
    fi
    
    # Build LSF submission command
    SUBMIT_CMD="bsub -J ${JOB_NAME} -q ${QUEUE} -n 4 -gpu \"num=1:mode=exclusive_process:gmem=${GPU_MEM}\" -W ${MAX_TIME} -o logs/${JOB_NAME}_%J.log -e logs/${JOB_NAME}_%J.err \"${TRAIN_CMD}\""
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Would submit:"
        echo "  $TRAIN_CMD"
    else
        echo "Submitting job..."
        eval "$SUBMIT_CMD"
        ((submitted++)) || true
    fi
    echo ""
done

echo "========================================================================"
echo "Summary"
echo "========================================================================"
echo "  Submitted: $submitted"
echo "  Total missing: ${#MISSING_MODELS[@]}"
