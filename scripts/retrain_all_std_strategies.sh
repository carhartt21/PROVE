#!/bin/bash
#
# COMPLETE Retrain of all std_* strategies with StandardAugmentationHook fix
#
# This script deletes ALL existing std_* checkpoints and retrains from scratch
# to properly apply CutMix, MixUp, AutoAugment, and RandAugment augmentations.
#
# WARNING: This will delete all existing std_* trained models!
#
# Total: 48 models (4 strategies × 4 datasets × 3 models)
# Estimated time: ~720 GPU hours (~15h per model)
#
# Usage:
#   ./scripts/retrain_all_std_strategies.sh --dry-run      # Preview (no deletion/submission)
#   ./scripts/retrain_all_std_strategies.sh --delete-only  # Delete checkpoints only
#   ./scripts/retrain_all_std_strategies.sh                # Delete and submit all jobs
#   ./scripts/retrain_all_std_strategies.sh --limit 10     # Submit first 10 jobs

set -e

DRY_RUN=false
DELETE_ONLY=false
LIMIT=0
SKIP_CONFIRM=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true ;;
        --delete-only) DELETE_ONLY=true ;;
        --limit) LIMIT="$2"; shift ;;
        --yes|-y) SKIP_CONFIRM=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Configuration
STRATEGIES=("std_cutmix" "std_mixup" "std_autoaugment" "std_randaugment")
DATASETS=("BDD10k" "IDD-AW" "MapillaryVistas" "OUTSIDE15k")
MODELS=("deeplabv3plus_r50" "pspnet_r50" "segformer_mit-b5")

WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS"
BACKUP_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS_STD_BACKUP_$(date +%Y%m%d)"
LOG_DIR="/home/mima2416/repositories/PROVE/logs"
PROVE_DIR="/home/mima2416/repositories/PROVE"

mkdir -p "$LOG_DIR"

echo "========================================================"
echo "COMPLETE Retraining of std_* Strategies"
echo "========================================================"
echo ""
echo "This will:"
echo "  1. Backup existing std_* models to: $BACKUP_ROOT"
echo "  2. Delete original std_* directories"
echo "  3. Retrain all 48 models with StandardAugmentationHook"
echo ""
echo "Strategies: ${STRATEGIES[*]}"
echo "Datasets:   ${DATASETS[*]}"
echo "Models:     ${MODELS[*]}"
echo "Total jobs: $((${#STRATEGIES[@]} * ${#DATASETS[@]} * ${#MODELS[@]}))"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - No changes will be made"
    echo ""
fi

# Confirmation prompt
if [ "$DRY_RUN" = false ] && [ "$SKIP_CONFIRM" = false ]; then
    read -p "Are you sure you want to proceed? (yes/no): " confirm
    if [[ "$confirm" != "yes" ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Function to convert dataset name to directory name
dataset_to_dir() {
    local ds="$1"
    case "$ds" in
        "BDD10k") echo "bdd10k" ;;
        "IDD-AW") echo "idd-aw" ;;
        "MapillaryVistas") echo "mapillaryvistas" ;;
        "OUTSIDE15k") echo "outside15k" ;;
        *) echo "$ds" | tr '[:upper:]' '[:lower:]' ;;
    esac
}

# Phase 1: Backup and delete existing checkpoints
echo "========================================================"
echo "Phase 1: Backup and Delete Existing Checkpoints"
echo "========================================================"

for strategy in "${STRATEGIES[@]}"; do
    strategy_dir="${WEIGHTS_ROOT}/${strategy}"
    backup_dir="${BACKUP_ROOT}/${strategy}"
    
    if [ -d "$strategy_dir" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] Would backup: $strategy_dir -> $backup_dir"
            echo "[DRY RUN] Would delete: $strategy_dir"
        else
            echo "Backing up: $strategy_dir -> $backup_dir"
            mkdir -p "$(dirname "$backup_dir")"
            mv "$strategy_dir" "$backup_dir"
            echo "Deleted: $strategy_dir"
        fi
    else
        echo "SKIP: $strategy_dir does not exist"
    fi
done

if [ "$DELETE_ONLY" = true ]; then
    echo ""
    echo "Delete-only mode. Exiting without submitting jobs."
    exit 0
fi

# Phase 2: Submit retraining jobs
echo ""
echo "========================================================"
echo "Phase 2: Submit Retraining Jobs"
echo "========================================================"

job_count=0
submitted=0

for strategy in "${STRATEGIES[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            ds_dir=$(dataset_to_dir "$dataset")
            
            # Job name (shortened)
            job_name="tr_${strategy}_${ds_dir}_${model}"
            job_name=$(echo "$job_name" | sed 's/deeplabv3plus_r50/dlv3/g' | sed 's/pspnet_r50/psp/g' | sed 's/segformer_mit-b5/segf/g' | cut -c1-35)
            
            # Log files
            out_log="${LOG_DIR}/retrain_${strategy}_${ds_dir}_${model}.out"
            err_log="${LOG_DIR}/retrain_${strategy}_${ds_dir}_${model}.err"
            
            job_count=$((job_count + 1))
            
            if [ "$DRY_RUN" = true ]; then
                echo "[$job_count] Would submit: ${strategy}/${ds_dir}/${model}"
                continue
            fi
            
            # Check limit
            if [ "$LIMIT" -gt 0 ] && [ "$submitted" -ge "$LIMIT" ]; then
                echo "Limit reached ($LIMIT jobs). Remaining jobs not submitted."
                break 3
            fi
            
            echo "[$job_count] Submitting: ${strategy}/${ds_dir}/${model}"
            
            # Submit job
            bsub -q BatchGPU \
                -gpu "num=1:mode=exclusive_process:gmem=24G" \
                -n 8 \
                -R "rusage[mem=8000]" \
                -W 24:00 \
                -J "$job_name" \
                -o "$out_log" \
                -e "$err_log" \
                "cd $PROVE_DIR && source ~/.bashrc && mamba activate prove && python unified_training.py \
                    --dataset $dataset \
                    --model $model \
                    --strategy $strategy \
                    --domain-filter clear_day"
            
            submitted=$((submitted + 1))
            
            # Small delay to avoid overwhelming scheduler
            sleep 0.3
        done
    done
done

echo ""
echo "========================================================"
echo "Summary"
echo "========================================================"
echo "Total jobs: $job_count"
if [ "$DRY_RUN" = true ]; then
    echo "Dry run - no changes made"
else
    echo "Jobs submitted: $submitted"
    if [ "$submitted" -lt "$job_count" ]; then
        echo "Jobs remaining: $((job_count - submitted))"
    fi
    echo ""
    echo "Backup location: $BACKUP_ROOT"
    echo "Monitor jobs with: bjobs -w | grep tr_std"
fi
