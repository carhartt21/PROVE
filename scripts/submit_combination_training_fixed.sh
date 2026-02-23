#!/bin/bash
# =============================================================================
# PROVE Augmentation Combination Training Script (Fixed Version)
# =============================================================================
# This script trains models using combinations of generative (gen_*) and
# standard (std_*) augmentation strategies with the fixed StandardAugmentationHook.
#
# Configuration:
#   - Generative strategies: gen_step1x_new, gen_flux_kontext, gen_Qwen_Image_Edit,
#                            gen_stargan_v2, gen_Attribute_Hallucination
#   - Standard strategies: std_randaugment, std_mixup, std_cutmix, std_autoaugment
#   - Datasets: MapillaryVistas, IDD-AW
#   - Models: segformer_mit-b5, pspnet_r50
#
# Total combinations:
#   - Gen+Std: 5 × 4 = 20 combinations × 2 datasets × 2 models = 80 jobs
#   - Std+Std: C(4,2) = 6 combinations × 2 datasets × 2 models = 24 jobs
#   - Total: 104 jobs
#
# Usage:
#   ./submit_combination_training_fixed.sh [OPTIONS]
#   ./submit_combination_training_fixed.sh --list        # Show all combinations
#   ./submit_combination_training_fixed.sh --dry-run     # Preview without submitting
#   ./submit_combination_training_fixed.sh --limit 5     # Submit only 5 jobs
# =============================================================================

set -e

# Configuration
PROVE_DIR="/home/chge7185/repositories/PROVE"
WEIGHTS_DIR="${AWARE_DATA_ROOT}/WEIGHTS_COMBINATIONS_chge7185"
LOG_DIR="${PROVE_DIR}/logs"
QUEUE="BatchGPU"
GPU_MEM="24G"
GPU_MODE="shared"
NUM_CPUS=10
WALL_TIME="48:00"
DRY_RUN=false
LIST_ONLY=false
GEN_ONLY=false
STD_ONLY=false
LIMIT=""
SKIP_TEST=false

# Generative strategies
GEN_STRATEGIES=(
    "gen_step1x_new"
    "gen_flux_kontext"
    "gen_Qwen_Image_Edit"
    "gen_stargan_v2"
    "gen_Attribute_Hallucination"
)

# Standard strategies
STD_STRATEGIES=(
    "std_randaugment"
    "std_mixup"
    "std_cutmix"
    "std_autoaugment"
)

# Std+Std combinations
STD_COMBINATIONS=(
    "std_randaugment+std_mixup"
    "std_randaugment+std_cutmix"
    "std_randaugment+std_autoaugment"
    "std_mixup+std_cutmix"
    "std_mixup+std_autoaugment"
    "std_cutmix+std_autoaugment"
)

# Datasets
DATASETS=("MapillaryVistas" "IDD-AW")

# Models
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
        --gen-only) GEN_ONLY=true; shift ;;
        --std-only) STD_ONLY=true; shift ;;
        --no-test) SKIP_TEST=true; shift ;;
        --all-domains) DOMAIN_FILTER=""; shift ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --list          List all combinations without submitting"
            echo "  --dry-run       Show commands without executing"
            echo "  --limit N       Submit only N jobs"
            echo "  --queue QUEUE   LSF queue (default: BatchGPU)"
            echo "  --gpu-mem SIZE  GPU memory (default: 24G)"
            echo "  --ratio RATIO   Real-to-generated ratio (default: 0.5)"
            echo "  --gen-only      Only submit gen+std combinations"
            echo "  --std-only      Only submit std+std combinations"
            echo "  --no-test       Skip automatic testing after training"
            echo "  --all-domains   Train on all domains (Stage 2), default is clear_day"
            echo ""
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$WEIGHTS_DIR"

# Count totals
GEN_STD_TOTAL=$((${#GEN_STRATEGIES[@]} * ${#STD_STRATEGIES[@]} * ${#DATASETS[@]} * ${#MODELS[@]}))
STD_STD_TOTAL=$((${#STD_COMBINATIONS[@]} * ${#DATASETS[@]} * ${#MODELS[@]}))
TOTAL_JOBS=$((GEN_STD_TOTAL + STD_STD_TOTAL))

echo "=============================================="
echo "PROVE Combination Training (Fixed)"
echo "=============================================="
echo ""
echo "Generative strategies: ${GEN_STRATEGIES[*]}"
echo "Standard strategies:   ${STD_STRATEGIES[*]}"
echo "Std combinations:      ${STD_COMBINATIONS[*]}"
echo "Datasets:              ${DATASETS[*]}"
echo "Models:                ${MODELS[*]}"
echo "Domain filter:         ${DOMAIN_FILTER:-all_domains}"
echo "Real:Gen ratio:        ${RATIO}"
echo "Output directory:      ${WEIGHTS_DIR}"
echo ""
echo "Gen+Std jobs:          ${GEN_STD_TOTAL}"
echo "Std+Std jobs:          ${STD_STD_TOTAL}"
echo "Total jobs:            ${TOTAL_JOBS}"
echo "=============================================="
echo ""

JOB_COUNT=0
SUBMITTED=0
SKIPPED=0

# Function to get dataset dir name
get_dataset_dir() {
    local dataset=$1
    echo "${dataset,,}" | sed 's/mapillaryvistas/mapillaryvistas/' | sed 's/idd-aw/idd-aw/'
}

# Submit gen+std combinations
if [ "$STD_ONLY" != true ]; then
    echo "=== Gen+Std Combinations ==="
    for gen_strategy in "${GEN_STRATEGIES[@]}"; do
        for std_strategy in "${STD_STRATEGIES[@]}"; do
            for dataset in "${DATASETS[@]}"; do
                for model in "${MODELS[@]}"; do
                    JOB_COUNT=$((JOB_COUNT + 1))
                    
                    # Check limit
                    if [ -n "$LIMIT" ] && [ $SUBMITTED -ge $LIMIT ]; then
                        continue
                    fi
                    
                    # Build paths
                    COMBINED_STRATEGY="${gen_strategy}+${std_strategy}"
                    dataset_dir=$(get_dataset_dir "$dataset")
                    ratio_str="ratio0p$(echo "$RATIO" | sed 's/0\.//')"
                    model_dir="${model}_${ratio_str}"
                    WORK_DIR="${WEIGHTS_DIR}/${COMBINED_STRATEGY}/${dataset_dir}/${model_dir}"
                    CHECKPOINT="${WORK_DIR}/iter_80000.pth"
                    
                    # Build job name
                    JOB_NAME="combo_${gen_strategy#gen_}_${std_strategy#std_}_${dataset_dir}_${model%%_*}"
                    
                    # Check if already completed
                    if [ -f "$CHECKPOINT" ]; then
                        echo "[SKIP] Already exists: ${COMBINED_STRATEGY}/${dataset_dir}/${model_dir}"
                        SKIPPED=$((SKIPPED + 1))
                        continue
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
                    TRAIN_CMD="${TRAIN_CMD} --work-dir ${WORK_DIR}"
                    
                    if [ -n "$DOMAIN_FILTER" ]; then
                        TRAIN_CMD="${TRAIN_CMD} --domain-filter ${DOMAIN_FILTER}"
                    fi
                    
                    # Build test command
                    TEST_CMD=""
                    if [ "$SKIP_TEST" != true ]; then
                        TEST_CMD=" && python fine_grained_test.py"
                        TEST_CMD="${TEST_CMD} --config ${WORK_DIR}/training_config.py"
                        TEST_CMD="${TEST_CMD} --checkpoint ${CHECKPOINT}"
                        TEST_CMD="${TEST_CMD} --dataset ${dataset}"
                        TEST_CMD="${TEST_CMD} --output-dir ${WORK_DIR}/test_results_detailed"
                    fi
                    
                    # Full command
                    FULL_CMD="source ~/.bashrc && mamba activate prove && ${TRAIN_CMD}${TEST_CMD}"
                    
                    echo "[${JOB_COUNT}] ${JOB_NAME}"
                    echo "  Strategy:  ${COMBINED_STRATEGY}"
                    echo "  Dataset:   ${dataset}"
                    echo "  Model:     ${model}"
                    echo "  Work dir:  ${WORK_DIR}"
                    
                    if [ "$DRY_RUN" = true ]; then
                        echo "  [DRY RUN] Would submit"
                    else
                        bsub -J "${JOB_NAME}" \
                             -q "${QUEUE}" \
                             -gpu "num=1:mode=${GPU_MODE}:gmem=${GPU_MEM}" \
                             -n "${NUM_CPUS}" \
                             -W "${WALL_TIME}" \
                             -R "span[hosts=1]" \
                             -oo "${LOG_DIR}/${JOB_NAME}_%J.out" \
                             -eo "${LOG_DIR}/${JOB_NAME}_%J.err" \
                             "${FULL_CMD}"
                        echo "  ✓ Submitted"
                    fi
                    echo ""
                    
                    SUBMITTED=$((SUBMITTED + 1))
                done
            done
        done
    done
fi

# Submit std+std combinations
if [ "$GEN_ONLY" != true ]; then
    echo ""
    echo "=== Std+Std Combinations ==="
    for std_combo in "${STD_COMBINATIONS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            for model in "${MODELS[@]}"; do
                JOB_COUNT=$((JOB_COUNT + 1))
                
                # Check limit
                if [ -n "$LIMIT" ] && [ $SUBMITTED -ge $LIMIT ]; then
                    continue
                fi
                
                # Parse combination
                std1="${std_combo%+*}"
                std2="${std_combo#*+}"
                
                # Build paths (no ratio suffix for std+std)
                dataset_dir=$(get_dataset_dir "$dataset")
                WORK_DIR="${WEIGHTS_DIR}/${std_combo}/${dataset_dir}/${model}"
                CHECKPOINT="${WORK_DIR}/iter_80000.pth"
                
                # Build job name
                JOB_NAME="combo_${std1#std_}_${std2#std_}_${dataset_dir}_${model%%_*}"
                
                # Check if already completed
                if [ -f "$CHECKPOINT" ]; then
                    echo "[SKIP] Already exists: ${std_combo}/${dataset_dir}/${model}"
                    SKIPPED=$((SKIPPED + 1))
                    continue
                fi
                
                if [ "$LIST_ONLY" = true ]; then
                    echo "${JOB_COUNT}. ${std_combo} / ${dataset} / ${model}"
                    continue
                fi
                
                # Build training command
                TRAIN_CMD="cd ${PROVE_DIR} && python unified_training.py"
                TRAIN_CMD="${TRAIN_CMD} --dataset ${dataset}"
                TRAIN_CMD="${TRAIN_CMD} --model ${model}"
                TRAIN_CMD="${TRAIN_CMD} --strategy ${std1}"
                TRAIN_CMD="${TRAIN_CMD} --std-strategy ${std2}"
                TRAIN_CMD="${TRAIN_CMD} --work-dir ${WORK_DIR}"
                
                if [ -n "$DOMAIN_FILTER" ]; then
                    TRAIN_CMD="${TRAIN_CMD} --domain-filter ${DOMAIN_FILTER}"
                fi
                
                # Build test command
                TEST_CMD=""
                if [ "$SKIP_TEST" != true ]; then
                    TEST_CMD=" && python fine_grained_test.py"
                    TEST_CMD="${TEST_CMD} --config ${WORK_DIR}/training_config.py"
                    TEST_CMD="${TEST_CMD} --checkpoint ${CHECKPOINT}"
                    TEST_CMD="${TEST_CMD} --dataset ${dataset}"
                    TEST_CMD="${TEST_CMD} --output-dir ${WORK_DIR}/test_results_detailed"
                fi
                
                # Full command
                FULL_CMD="source ~/.bashrc && mamba activate prove && ${TRAIN_CMD}${TEST_CMD}"
                
                echo "[${JOB_COUNT}] ${JOB_NAME}"
                echo "  Strategy:  ${std_combo}"
                echo "  Dataset:   ${dataset}"
                echo "  Model:     ${model}"
                echo "  Work dir:  ${WORK_DIR}"
                
                if [ "$DRY_RUN" = true ]; then
                    echo "  [DRY RUN] Would submit"
                else
                    bsub -J "${JOB_NAME}" \
                         -q "${QUEUE}" \
                         -gpu "num=1:mode=${GPU_MODE}:gmem=${GPU_MEM}" \
                         -n "${NUM_CPUS}" \
                         -W "${WALL_TIME}" \
                         -R "span[hosts=1]" \
                         -oo "${LOG_DIR}/${JOB_NAME}_%J.out" \
                         -eo "${LOG_DIR}/${JOB_NAME}_%J.err" \
                         "${FULL_CMD}"
                    echo "  ✓ Submitted"
                fi
                echo ""
                
                SUBMITTED=$((SUBMITTED + 1))
            done
        done
    done
fi

echo "=============================================="
if [ "$LIST_ONLY" = true ]; then
    echo "Listed ${TOTAL_JOBS} combinations"
elif [ "$DRY_RUN" = true ]; then
    echo "Dry run complete"
    echo "  Would submit: ${SUBMITTED} jobs"
    echo "  Skipped:      ${SKIPPED} (already complete)"
else
    echo "Submission complete"
    echo "  Submitted: ${SUBMITTED} jobs"
    echo "  Skipped:   ${SKIPPED} (already complete)"
fi
echo "=============================================="
