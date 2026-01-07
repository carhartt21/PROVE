#!/bin/bash
# ============================================================================
# PROVE: Submit Incomplete Training Jobs
# ============================================================================
# This script resubmits training jobs that were interrupted and didn't complete.
# It resumes from the last checkpoint.
#
# Usage:
#   ./scripts/submit_incomplete_training.sh [--dry-run]
#
# ============================================================================

# Don't use set -e as arithmetic operations can return non-zero
# set -e

DRY_RUN=false
WEIGHTS_DIR="/scratch/aaa_exchange/AWARE/WEIGHTS"
DATA_ROOT="/scratch/aaa_exchange/AWARE/FINAL_SPLITS"
QUEUE="BatchGPU"
GPU_MEM="16G"
MAX_TIME="24:00"

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

# Define incomplete training jobs
# Format: strategy|dataset|model|variant|last_iter
INCOMPLETE_JOBS=(
    "gen_StyleID|bdd10k|pspnet_r50||60000"
    "gen_StyleID|idd-aw|deeplabv3plus_r50|_clear_day|50000"
    "photometric_distort|mapillaryvistas|deeplabv3plus_r50|_clear_day|30000"
)

echo "========================================================================"
echo "PROVE Incomplete Training Job Submission"
echo "========================================================================"
echo ""
echo "Dry Run: $DRY_RUN"
echo "Queue: $QUEUE"
echo "GPU Memory: $GPU_MEM"
echo "Max Time: $MAX_TIME"
echo ""

submitted=0
skipped=0

for job in "${INCOMPLETE_JOBS[@]}"; do
    IFS='|' read -r strategy dataset model variant last_iter <<< "$job"
    
    full_model="${model}${variant}"
    checkpoint_path="${WEIGHTS_DIR}/${strategy}/${dataset}/${full_model}/iter_${last_iter}.pth"
    
    # Check if checkpoint exists
    if [[ ! -f "$checkpoint_path" ]]; then
        echo "WARNING: Checkpoint not found: $checkpoint_path"
        echo "  Skipping..."
        ((skipped++))
        continue
    fi
    
    echo "----------------------------------------"
    echo "Strategy: $strategy"
    echo "Dataset: $dataset"  
    echo "Model: $full_model"
    echo "Resume from: iter_${last_iter}"
    echo "Checkpoint: $checkpoint_path"
    
    # Map lowercase dataset to proper name for unified_training.py
    case "$dataset" in
        "bdd10k") DATASET_NAME="BDD10k" ;;
        "idd-aw") DATASET_NAME="IDD-AW" ;;
        "mapillaryvistas") DATASET_NAME="MapillaryVistas" ;;
        *) DATASET_NAME="$dataset" ;;
    esac
    
    # Build job name
    JOB_NAME="prove_resume_${DATASET_NAME}_${full_model}_${strategy}"
    
    # Build training command
    if [[ -n "$variant" ]]; then
        # Clear day variant
        TRAIN_CMD="mamba run -n prove python unified_training.py --dataset ${DATASET_NAME} --model ${model} --strategy ${strategy} --domain-filter clear_day --resume-from ${checkpoint_path} --ratio 1.0 --work-dir ${WEIGHTS_DIR} --data-root ${DATA_ROOT} --no-early-stop"
    else
        # Full model
        TRAIN_CMD="mamba run -n prove python unified_training.py --dataset ${DATASET_NAME} --model ${model} --strategy ${strategy} --resume-from ${checkpoint_path} --ratio 1.0 --work-dir ${WEIGHTS_DIR} --data-root ${DATA_ROOT} --no-early-stop"
    fi
    
    # Build LSF submission command
    SUBMIT_CMD="bsub -J ${JOB_NAME} -q ${QUEUE} -n 4 -gpu \"num=1:mode=exclusive_process:gmodel=NVIDIAA100_PCIE_40GB:gmem=${GPU_MEM}\" -W ${MAX_TIME} -o logs/${JOB_NAME}_%J.log -e logs/${JOB_NAME}_%J.err \"${TRAIN_CMD}\""
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Would submit:"
        echo "  $TRAIN_CMD"
    else
        echo "Submitting job..."
        eval "$SUBMIT_CMD"
        ((submitted++))
    fi
    echo ""
done

echo "========================================================================"
echo "Summary"
echo "========================================================================"
echo "  Submitted: $submitted"
echo "  Skipped: $skipped"
