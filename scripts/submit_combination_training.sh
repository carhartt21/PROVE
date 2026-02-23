#!/bin/bash
# =============================================================================
# PROVE Augmentation Combination Training Script
# =============================================================================
# This script trains models using combinations of top generative (gen_*) and
# standard (std_*) augmentation strategies.
#
# Top Generative Strategies (from leaderboard/extended training analysis):
#   1. gen_cyclediffusion   - Best overall performance
#   2. gen_TSIT             - Fast convergence, good results
#   3. gen_cycleGAN         - Stable and reliable
#
# Top Standard Augmentation Strategies:
#   1. std_randaugment      - Best std strategy
#   2. std_mixup            - Good for generalization
#   3. std_cutmix           - Effective regularization
#
# Combinations to train (9 total):
#   gen_cyclediffusion + std_randaugment/std_mixup/std_cutmix
#   gen_TSIT + std_randaugment/std_mixup/std_cutmix
#   gen_cycleGAN + std_randaugment/std_mixup/std_cutmix
#
# Usage:
#   ./submit_combination_training.sh [OPTIONS]
#   ./submit_combination_training.sh --list        # Show all combinations
#   ./submit_combination_training.sh --dry-run     # Preview without submitting
#   ./submit_combination_training.sh --limit 5     # Submit only 5 jobs
# =============================================================================

set -e

# Configuration
PROVE_DIR="${HOME}/repositories/PROVE"
LOG_DIR="${PROVE_DIR}/logs"
QUEUE="BatchGPU"
GPU_MEM="24G"
GPU_MODE="shared"
NUM_CPUS=8
WALL_TIME="24:00"
DRY_RUN=false
LIST_ONLY=false
LIMIT=""

# Top strategies to combine
GEN_STRATEGIES=("gen_flux_kontext" "gen_step1x_new" "gen_cycleGAN")
STD_STRATEGIES=("std_randaugment" "std_mixup" "std_cutmix")

# Datasets (as requested: MapillaryVistas and IDD-AW)
DATASETS=("MapillaryVistas" "IDD-AW")

# Models (SegFormer and PSPNet - DeepLabV3+ excluded)
MODELS=("segformer_mit-b5" "pspnet_r50")

# Domain filter for Stage 1 training
DOMAIN_FILTER="clear_day"

# Real-to-generated ratio
RATIO="0.5"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --list) LIST_ONLY=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --queue) QUEUE="$2"; shift 2 ;;
        --gpu-mem) GPU_MEM="$2"; shift 2 ;;
        --ratio) RATIO="$2"; shift 2 ;;
        --all-domains) DOMAIN_FILTER=""; shift ;;  # Stage 2: all domains
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --list          List all combinations without submitting"
            echo "  --dry-run       Show bsub commands without executing"
            echo "  --limit N       Submit only N jobs"
            echo "  --queue QUEUE   LSF queue (default: BatchGPU)"
            echo "  --gpu-mem SIZE  GPU memory (default: 24G)"
            echo "  --ratio RATIO   Real-to-generated ratio (default: 0.5)"
            echo "  --all-domains   Train on all domains (Stage 2), default is clear_day only"
            echo ""
            echo "Combinations: ${#GEN_STRATEGIES[@]} gen × ${#STD_STRATEGIES[@]} std × ${#DATASETS[@]} datasets × ${#MODELS[@]} models"
            echo "Total jobs: $((${#GEN_STRATEGIES[@]} * ${#STD_STRATEGIES[@]} * ${#DATASETS[@]} * ${#MODELS[@]}))"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Create log directory
mkdir -p "$LOG_DIR"

# Count jobs
TOTAL_JOBS=$((${#GEN_STRATEGIES[@]} * ${#STD_STRATEGIES[@]} * ${#DATASETS[@]} * ${#MODELS[@]}))
JOB_COUNT=0
SUBMITTED=0

echo "=============================================="
echo "PROVE Augmentation Combination Training"
echo "=============================================="
echo ""
echo "Generative strategies: ${GEN_STRATEGIES[*]}"
echo "Standard strategies:   ${STD_STRATEGIES[*]}"
echo "Datasets:              ${DATASETS[*]}"
echo "Models:                ${MODELS[*]}"
echo "Domain filter:         ${DOMAIN_FILTER:-all_domains}"
echo "Ratio:                 ${RATIO}"
echo ""
echo "Total combinations: ${TOTAL_JOBS}"
echo "=============================================="
echo ""

if [ "$LIST_ONLY" = true ]; then
    echo "Job List:"
    echo "---------"
fi

for gen_strategy in "${GEN_STRATEGIES[@]}"; do
    for std_strategy in "${STD_STRATEGIES[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            for model in "${MODELS[@]}"; do
                JOB_COUNT=$((JOB_COUNT + 1))
                
                # Check limit
                if [ -n "$LIMIT" ] && [ $SUBMITTED -ge $LIMIT ]; then
                    continue
                fi
                
                # Build job name and combined strategy name
                COMBINED_STRATEGY="${gen_strategy}+${std_strategy}"
                
                # Dataset lowercase for paths
                dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
                
                if [ -n "$DOMAIN_FILTER" ]; then
                    JOB_NAME="combo_${gen_strategy#gen_}_${std_strategy#std_}_${dataset_lower}_${model%%_*}_cd"
                else
                    JOB_NAME="combo_${gen_strategy#gen_}_${std_strategy#std_}_${dataset_lower}_${model%%_*}_ad"
                fi
                
                if [ "$LIST_ONLY" = true ]; then
                    echo "${JOB_COUNT}. ${COMBINED_STRATEGY} / ${dataset} / ${model}"
                    continue
                fi
                
                # Build training command
                TRAIN_CMD="cd ${PROVE_DIR} && python unified_training.py"
                TRAIN_CMD="${TRAIN_CMD} --dataset ${dataset}"
                TRAIN_CMD="${TRAIN_CMD} --model ${model}"
                TRAIN_CMD="${TRAIN_CMD} --strategy ${gen_strategy}"
                TRAIN_CMD="${TRAIN_CMD} --std-strategy ${std_strategy}"
                TRAIN_CMD="${TRAIN_CMD} --real-gen-ratio ${RATIO}"
                
                if [ -n "$DOMAIN_FILTER" ]; then
                    TRAIN_CMD="${TRAIN_CMD} --domain-filter ${DOMAIN_FILTER}"
                fi
                
                # Full command with environment
                FULL_CMD="source ~/.bashrc && conda activate prove && ${TRAIN_CMD}"
                
                echo "[${JOB_COUNT}/${TOTAL_JOBS}] ${JOB_NAME}"
                echo "  Strategy: ${COMBINED_STRATEGY}"
                echo "  Dataset:  ${dataset}"
                echo "  Model:    ${model}"
                
                if [ "$DRY_RUN" = true ]; then
                    echo "  [DRY RUN] Would submit: bsub -J \"${JOB_NAME}\" ..."
                else
                    bsub -J "${JOB_NAME}" \
                         -q "${QUEUE}" \
                         -gpu "num=1:mode=${GPU_MODE}:gmem=${GPU_MEM}" \
                         -n "${NUM_CPUS}" \
                         -W "${WALL_TIME}" \
                         -o "${LOG_DIR}/${JOB_NAME}.log" \
                         -e "${LOG_DIR}/${JOB_NAME}.err" \
                         "${FULL_CMD}"
                    echo "  ✓ Submitted"
                fi
                echo ""
                
                SUBMITTED=$((SUBMITTED + 1))
            done
        done
    done
done

echo "=============================================="
if [ "$LIST_ONLY" = true ]; then
    echo "Listed ${TOTAL_JOBS} combinations"
elif [ "$DRY_RUN" = true ]; then
    echo "Dry run complete: ${SUBMITTED} jobs would be submitted"
else
    echo "Submitted ${SUBMITTED} jobs"
fi
echo "=============================================="
