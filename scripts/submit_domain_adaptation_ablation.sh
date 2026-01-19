#!/bin/bash
# PROVE Domain Adaptation Ablation - Job Submission Script
#
# This script submits evaluation jobs for the domain adaptation ablation study.
# It evaluates models trained on BDD10k/IDD-AW/MapillaryVistas on:
#   - Cityscapes (clear_day condition)
#   - ACDC (adverse weather conditions: foggy, night, rainy, snowy)
#
# Key features:
# - Evaluates cross-dataset domain adaptation capability
# - Uses Cityscapes as the "clear_day" baseline condition
# - Evaluates on ACDC adverse weather domains
# - Reports per-domain (weather condition) metrics
#
# Data structure:
#   test/images/Cityscapes/{city}/*_leftImg8bit.png (clear_day)
#   test/images/ACDC/{domain}/*_rgb_anon.png (adverse)
#
# Usage:
#   ./submit_domain_adaptation_ablation.sh [options]
#
# Options:
#   --all               Submit jobs for all source dataset / model combinations
#   --source-dataset    Specific source dataset (BDD10k, IDD-AW, MapillaryVistas)
#   --model             Specific model (deeplabv3plus_r50, pspnet_r50, segformer_mit-b5)
#   --dry-run           Show commands without executing
#   --list              List all jobs that would be submitted
#   --queue <name>      LSF queue name (default: BatchGPU)
#   --gpu-mem <size>    GPU memory requirement (default: 16G)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# Disable exit on error for the main loop (we handle errors manually)
set +e

# ============================================================================
# Configuration
# ============================================================================

# Source datasets (models trained on these)
SOURCE_DATASETS=("BDD10k" "IDD-AW" "MapillaryVistas")

# Models to evaluate (base models)
MODELS=("pspnet_r50" "segformer_mit-b5")

# Model variants
# "" = full dataset training (all weather conditions) - uses WEIGHTS_STAGE_2
# "_clear_day" = clear_day only training - uses WEIGHTS
MODEL_VARIANTS=("" "_clear_day")

# Top 15 augmentation strategies to evaluate (in addition to baseline)
# These will be evaluated using checkpoints from WEIGHTS/{strategy}/{dataset}/{model}_ratio0p50/
TOP_STRATEGIES=(
    "gen_cyclediffusion"
    "gen_flux_kontext"
    "gen_step1x_new"
    "gen_step1x_v1p2"
    "gen_stargan_v2"
    "gen_cycleGAN"
    "gen_automold"
    "gen_albumentations_weather"
    "gen_TSIT"
    "gen_UniControl"
    "std_randaugment"
    "std_autoaugment"
    "std_cutmix"
    "std_mixup"
    "photometric_distort"
)

# Default LSF settings
DEFAULT_QUEUE="BatchGPU"
DEFAULT_GPU_MEM="16G"
DEFAULT_GPU_MODE="shared"
DEFAULT_NUM_CPUS=4

# Paths
# WEIGHTS = clear_day only models, WEIGHTS_STAGE_2 = full dataset models
WEIGHTS_ROOT="${PROVE_WEIGHTS_ROOT:-/scratch/aaa_exchange/AWARE/WEIGHTS}"
WEIGHTS_STAGE_2_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2"
DATA_ROOT="${PROVE_DATA_ROOT:-/scratch/aaa_exchange/AWARE/FINAL_SPLITS}"
OUTPUT_ROOT="${WEIGHTS_ROOT}/domain_adaptation_ablation"

# Python environment
PYTHON_ENV="${PROJECT_ROOT}/venv/bin/activate"
CONDA_ENV="prove"

# ============================================================================
# Helper Functions
# ============================================================================

print_usage() {
    echo "PROVE Domain Adaptation Ablation - Job Submission Script"
    echo "========================================================="
    echo ""
    echo "Evaluates cross-dataset domain adaptation:"
    echo "  Source: BDD10k, IDD-AW, MapillaryVistas (trained models)"
    echo "  Target: Cityscapes (clear_day) + ACDC (foggy, night, rainy, snowy)"
    echo ""
    echo "Model variants:"
    echo "  '' (default)  - Models trained on full source dataset"
    echo "  '_clear_day'  - Models trained on clear_day subset only (baseline condition)"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --all               Submit jobs for all source/model combinations (18 jobs: 9 full + 9 clear_day)"
    echo "  --all-clear-day     Submit jobs only for _clear_day variants (9 jobs)"
    echo "  --all-full          Submit jobs only for full dataset models (9 jobs, no _clear_day)"
    echo "  --source-dataset <name>"
    echo "                      Specific source dataset: ${SOURCE_DATASETS[*]}"
    echo "  --model <name>      Specific model: ${MODELS[*]}"
    echo "  --variant <v>       Model variant: '' or '_clear_day' (default: '')"
    echo "  --dry-run           Show commands without executing"
    echo "  --list              List all jobs that would be submitted"
    echo "  --queue <name>      LSF queue name (default: $DEFAULT_QUEUE)"
    echo "  --gpu-mem <size>    GPU memory requirement (default: $DEFAULT_GPU_MEM)"
    echo "  --gpu-mode <mode>   GPU mode: shared or exclusive_process (default: $DEFAULT_GPU_MODE)"
    echo "  --num-cpus <n>      Number of CPUs per job (default: $DEFAULT_NUM_CPUS)"
    echo "  --skip-existing     Skip jobs with existing results"
    echo ""
    echo "Examples:"
    echo "  # Submit all 18 evaluation jobs (full + clear_day)"
    echo "  $0 --all"
    echo ""
    echo "  # Submit only the 9 clear_day baseline jobs"
    echo "  $0 --all-clear-day"
    echo ""
    echo "  # Submit only the 9 full dataset jobs (no clear_day)"
    echo "  $0 --all-full"
    echo ""
    echo "  # Submit single job with clear_day variant"
    echo "  $0 --source-dataset BDD10k --model deeplabv3plus_r50 --variant _clear_day"
    echo ""
    echo "  # Dry run - show what would be submitted"
    echo "  $0 --all --dry-run"
    echo ""
    echo "  # List available checkpoints"
    echo "  $0 --list"
    echo ""
}

# Check if checkpoint exists
# For full dataset models (no variant): use WEIGHTS_STAGE_2
# For clear_day models (_clear_day variant): use WEIGHTS
check_checkpoint() {
    local source_dataset="$1"
    local model="$2"
    local variant="${3:-}"  # Optional variant parameter
    
    # Determine which weights directory to use
    local weights_dir
    if [ -n "$variant" ] && [ "$variant" = "_clear_day" ]; then
        # Clear_day variant - use WEIGHTS (models trained on clear_day only)
        weights_dir="${WEIGHTS_ROOT}"
        # For clear_day variant, the model name in the path doesn't have the suffix
        local path_model="${model}"
    else
        # Full dataset variant - use WEIGHTS_STAGE_2 (models trained on all weather)
        weights_dir="${WEIGHTS_STAGE_2_ROOT}"
        local path_model="${model}"
    fi
    
    local checkpoint_dir="${weights_dir}/baseline/${source_dataset,,}/${path_model}"
    
    # Try common checkpoint names
    if [ -f "${checkpoint_dir}/iter_80000.pth" ]; then
        echo "${checkpoint_dir}/iter_80000.pth"
        return 0
    elif [ -f "${checkpoint_dir}/latest.pth" ]; then
        echo "${checkpoint_dir}/latest.pth"
        return 0
    fi
    
    # Try to find any checkpoint
    local best_ckpt=$(find "$checkpoint_dir" -name "best_*.pth" 2>/dev/null | head -n 1)
    if [ -n "$best_ckpt" ]; then
        echo "$best_ckpt"
        return 0
    fi
    
    return 1
}

# Check if strategy checkpoint exists
# Generative strategies use: WEIGHTS/{strategy}/{dataset}/{model}_ratio0p50/iter_80000.pth
# Standard strategies use: WEIGHTS/{strategy}/{dataset}/{model}/iter_80000.pth
check_strategy_checkpoint() {
    local strategy="$1"
    local source_dataset="$2"
    local model="$3"
    
    # Try with _ratio0p50 suffix first (generative strategies)
    local checkpoint_dir="${WEIGHTS_ROOT}/${strategy}/${source_dataset,,}/${model}_ratio0p50"
    
    if [ -f "${checkpoint_dir}/iter_80000.pth" ]; then
        echo "${checkpoint_dir}/iter_80000.pth"
        return 0
    fi
    
    # Try without _ratio0p50 suffix (standard augmentation strategies)
    checkpoint_dir="${WEIGHTS_ROOT}/${strategy}/${source_dataset,,}/${model}"
    
    if [ -f "${checkpoint_dir}/iter_80000.pth" ]; then
        echo "${checkpoint_dir}/iter_80000.pth"
        return 0
    fi
    
    # Try latest.pth as fallback
    for dir in "${WEIGHTS_ROOT}/${strategy}/${source_dataset,,}/${model}_ratio0p50" "${WEIGHTS_ROOT}/${strategy}/${source_dataset,,}/${model}"; do
        if [ -f "${dir}/latest.pth" ]; then
            echo "${dir}/latest.pth"
            return 0
        fi
    done
    
    return 1
}

# Check if strategy results already exist
check_strategy_results_exist() {
    local strategy="$1"
    local source_dataset="$2"
    local model="$3"
    
    # Check both path patterns
    local result_file1="${OUTPUT_ROOT}/${strategy}/${source_dataset,,}/${model}_ratio0p50/domain_adaptation_evaluation.json"
    local result_file2="${OUTPUT_ROOT}/${strategy}/${source_dataset,,}/${model}/domain_adaptation_evaluation.json"
    [ -f "$result_file1" ] || [ -f "$result_file2" ]
}

# Check if results already exist
check_results_exist() {
    local source_dataset="$1"
    local model="$2"
    
    local result_file="${OUTPUT_ROOT}/${source_dataset,,}/${model}/domain_adaptation_evaluation.json"
    [ -f "$result_file" ]
}

# Submit a single evaluation job
submit_job() {
    local source_dataset="$1"
    local model="$2"
    local variant="$3"  # "" or "_clear_day"
    local queue="$4"
    local gpu_mem="$5"
    local gpu_mode="$6"
    local num_cpus="$7"
    local dry_run="$8"
    
    # Build full model name
    local full_model="${model}${variant}"
    
    # Check for checkpoint - pass variant for correct weights directory
    local checkpoint=$(check_checkpoint "$source_dataset" "$model" "$variant")
    if [ -z "$checkpoint" ]; then
        echo "  SKIP: No checkpoint found for ${source_dataset}/${full_model}"
        return 1
    fi
    
    # Job name (include variant in job name)
    local variant_suffix=""
    if [ -n "$variant" ]; then
        variant_suffix="${variant}"
    fi
    local jobname="da_${source_dataset,,}_${model}${variant_suffix}_to_acdc"
    
    # Log directory
    local log_dir="${PROJECT_ROOT}/logs/domain_adaptation"
    mkdir -p "$log_dir"
    
    # Build the evaluation command (include variant if set)
    local eval_cmd="python ${PROJECT_ROOT}/tools/evaluate_domain_adaptation.py \
        --source-dataset ${source_dataset} \
        --model ${model} \
        --variant '${variant}' \
        --checkpoint ${checkpoint}"
    
    # LSF submission command
    local submit_cmd="bsub -gpu \"num=1:mode=${gpu_mode}:gmem=${gpu_mem}\" \
        -q ${queue} \
        -R \"span[hosts=1]\" \
        -n ${num_cpus} \
        -oo \"${log_dir}/${jobname}_%J.log\" \
        -eo \"${log_dir}/${jobname}_%J.err\" \
        -L /bin/bash \
        -J \"${jobname}\" \
        \"source ~/.bashrc && mamba activate ${CONDA_ENV}; ${eval_cmd}\""
    
    if [ "$dry_run" = true ]; then
        echo "  [DRY-RUN] Would submit: $jobname"
        echo "    Checkpoint: $checkpoint"
        echo "    Full model: $full_model"
        echo "    Command: $eval_cmd"
        echo ""
    else
        echo "  Submitting: $jobname"
        echo "    Checkpoint: $checkpoint"
        eval "$submit_cmd"
    fi
    
    return 0
}

# Submit a strategy evaluation job
submit_strategy_job() {
    local strategy="$1"
    local source_dataset="$2"
    local model="$3"
    local queue="$4"
    local gpu_mem="$5"
    local gpu_mode="$6"
    local num_cpus="$7"
    local dry_run="$8"
    
    # Check for checkpoint
    local checkpoint=$(check_strategy_checkpoint "$strategy" "$source_dataset" "$model")
    if [ -z "$checkpoint" ]; then
        echo "  SKIP: No checkpoint found for ${strategy}/${source_dataset}/${model}_ratio0p50"
        return 1
    fi
    
    # Job name
    local jobname="da_${strategy}_${source_dataset,,}_${model}_to_acdc"
    
    # Log directory
    local log_dir="${PROJECT_ROOT}/logs/domain_adaptation"
    mkdir -p "$log_dir"
    
    # Build the evaluation command
    local eval_cmd="python ${PROJECT_ROOT}/tools/evaluate_domain_adaptation.py \
        --source-dataset ${source_dataset} \
        --model ${model} \
        --strategy ${strategy} \
        --checkpoint ${checkpoint}"
    
    # LSF submission command
    local submit_cmd="bsub -gpu \"num=1:mode=${gpu_mode}:gmem=${gpu_mem}\" \
        -q ${queue} \
        -R \"span[hosts=1]\" \
        -n ${num_cpus} \
        -oo \"${log_dir}/${jobname}_%J.log\" \
        -eo \"${log_dir}/${jobname}_%J.err\" \
        -L /bin/bash \
        -J \"${jobname}\" \
        \"source ~/.bashrc && mamba activate ${CONDA_ENV}; ${eval_cmd}\""
    
    if [ "$dry_run" = true ]; then
        echo "  [DRY-RUN] Would submit: $jobname"
        echo "    Strategy: $strategy"
        echo "    Checkpoint: $checkpoint"
        echo "    Command: $eval_cmd"
        echo ""
    else
        echo "  Submitting: $jobname"
        echo "    Checkpoint: $checkpoint"
        eval "$submit_cmd"
    fi
    
    return 0
}

# List all available configurations
list_configurations() {
    echo "Available Domain Adaptation Ablation Configurations"
    echo "===================================================="
    echo ""
    echo "Note: Full dataset models use WEIGHTS_STAGE_2, clear_day models use WEIGHTS"
    echo ""
    echo "=== BASELINE MODELS ==="
    echo "Source Dataset / Model (Variant) / Checkpoint Status"
    echo "-----------------------------------------------------"
    
    local baseline_count=0
    for source_dataset in "${SOURCE_DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            for variant in "${MODEL_VARIANTS[@]}"; do
                local full_model="${model}${variant}"
                # Pass variant to check_checkpoint
                local checkpoint=$(check_checkpoint "$source_dataset" "$model" "$variant")
                local result_exists=""
                local variant_label=""
                
                if [ -n "$variant" ]; then
                    variant_label=" (${variant}) [WEIGHTS]"
                else
                    variant_label=" (full) [WEIGHTS_STAGE_2]"
                fi
                
                if check_results_exist "$source_dataset" "$full_model"; then
                    result_exists=" [RESULTS EXIST]"
                fi
                
                if [ -n "$checkpoint" ]; then
                    echo "  ✓ ${source_dataset} / ${model}${variant_label}${result_exists}"
                    echo "    Checkpoint: ${checkpoint}"
                    ((baseline_count++))
                else
                    echo "  ✗ ${source_dataset} / ${model}${variant_label} - NO CHECKPOINT"
                fi
            done
        done
    done
    
    echo ""
    echo "Baseline configurations available: $baseline_count / $((${#SOURCE_DATASETS[@]} * ${#MODELS[@]} * ${#MODEL_VARIANTS[@]}))"
    
    echo ""
    echo "=== TOP 15 AUGMENTATION STRATEGIES ==="
    echo "Strategy / Source Dataset / Model / Checkpoint Status"
    echo "------------------------------------------------------"
    
    local strategy_count=0
    for strategy in "${TOP_STRATEGIES[@]}"; do
        echo ""
        echo "[$strategy]"
        for source_dataset in "${SOURCE_DATASETS[@]}"; do
            for model in "${MODELS[@]}"; do
                local checkpoint=$(check_strategy_checkpoint "$strategy" "$source_dataset" "$model")
                local result_exists=""
                
                if check_strategy_results_exist "$strategy" "$source_dataset" "$model"; then
                    result_exists=" [RESULTS EXIST]"
                fi
                
                if [ -n "$checkpoint" ]; then
                    echo "  ✓ ${source_dataset} / ${model}${result_exists}"
                    ((strategy_count++))
                else
                    echo "  ✗ ${source_dataset} / ${model} - NO CHECKPOINT"
                fi
            done
        done
    done
    
    echo ""
    echo "Strategy configurations available: $strategy_count / $((${#TOP_STRATEGIES[@]} * ${#SOURCE_DATASETS[@]} * ${#MODELS[@]}))"
    echo ""
    echo "Total available: $((baseline_count + strategy_count))"
}

# ============================================================================
# Parse Command Line Arguments
# ============================================================================

ALL_MODE=false
ALL_CLEAR_DAY_MODE=false
ALL_FULL_MODE=false
ALL_STRATEGIES_MODE=false
DRY_RUN=false
LIST_MODE=false
SKIP_EXISTING=false
QUEUE="$DEFAULT_QUEUE"
GPU_MEM="$DEFAULT_GPU_MEM"
GPU_MODE="$DEFAULT_GPU_MODE"
NUM_CPUS="$DEFAULT_NUM_CPUS"
FILTER_SOURCE=""
FILTER_MODEL=""
FILTER_VARIANT=""
FILTER_STRATEGY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            ALL_MODE=true
            shift
            ;;
        --all-clear-day)
            ALL_CLEAR_DAY_MODE=true
            shift
            ;;
        --all-full)
            ALL_FULL_MODE=true
            shift
            ;;
        --all-strategies)
            ALL_STRATEGIES_MODE=true
            shift
            ;;
        --strategy)
            FILTER_STRATEGY="$2"
            shift 2
            ;;
        --source-dataset)
            FILTER_SOURCE="$2"
            shift 2
            ;;
        --model)
            FILTER_MODEL="$2"
            shift 2
            ;;
        --variant)
            FILTER_VARIANT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --list)
            LIST_MODE=true
            shift
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --queue)
            QUEUE="$2"
            shift 2
            ;;
        --gpu-mem)
            GPU_MEM="$2"
            shift 2
            ;;
        --gpu-mode)
            GPU_MODE="$2"
            shift 2
            ;;
        --num-cpus)
            NUM_CPUS="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# ============================================================================
# Main Execution
# ============================================================================

# List mode
if [ "$LIST_MODE" = true ]; then
    list_configurations
    exit 0
fi

# Determine which configurations to run
if [ "$ALL_MODE" = true ]; then
    SELECTED_SOURCES=("${SOURCE_DATASETS[@]}")
    SELECTED_MODELS=("${MODELS[@]}")
    SELECTED_VARIANTS=("${MODEL_VARIANTS[@]}")
    ALL_STRATEGIES_MODE=true  # Include strategies when --all is specified
elif [ "$ALL_CLEAR_DAY_MODE" = true ]; then
    SELECTED_SOURCES=("${SOURCE_DATASETS[@]}")
    SELECTED_MODELS=("${MODELS[@]}")
    SELECTED_VARIANTS=("_clear_day")
elif [ "$ALL_FULL_MODE" = true ]; then
    SELECTED_SOURCES=("${SOURCE_DATASETS[@]}")
    SELECTED_MODELS=("${MODELS[@]}")
    SELECTED_VARIANTS=("")
elif [ "$ALL_STRATEGIES_MODE" = true ]; then
    # Only strategies, no baseline
    SELECTED_SOURCES=()
    SELECTED_MODELS=()
    SELECTED_VARIANTS=()
elif [ -n "$FILTER_STRATEGY" ]; then
    # Single strategy filter
    SELECTED_SOURCES=()
    SELECTED_MODELS=()
    SELECTED_VARIANTS=()
elif [ -n "$FILTER_SOURCE" ] || [ -n "$FILTER_MODEL" ]; then
    if [ -n "$FILTER_SOURCE" ]; then
        SELECTED_SOURCES=("$FILTER_SOURCE")
    else
        SELECTED_SOURCES=("${SOURCE_DATASETS[@]}")
    fi
    
    if [ -n "$FILTER_MODEL" ]; then
        SELECTED_MODELS=("$FILTER_MODEL")
    else
        SELECTED_MODELS=("${MODELS[@]}")
    fi
    
    if [ -n "$FILTER_VARIANT" ]; then
        SELECTED_VARIANTS=("$FILTER_VARIANT")
    else
        SELECTED_VARIANTS=("")  # Default to full dataset only for single jobs
    fi
else
    print_usage
    echo "ERROR: Specify --all, --all-clear-day, --all-full, --all-strategies, --strategy, or --source-dataset/--model"
    exit 1
fi

# Count total jobs
total_jobs=$((${#SELECTED_SOURCES[@]} * ${#SELECTED_MODELS[@]} * ${#SELECTED_VARIANTS[@]}))

echo "========================================================================"
echo "PROVE Domain Adaptation Ablation - Job Submission"
echo "========================================================================"
echo ""
echo "Source Datasets: ${SELECTED_SOURCES[*]}"
echo "Models: ${SELECTED_MODELS[*]}"
echo "Variants: ${SELECTED_VARIANTS[*]:-(full dataset)}"
echo "Total Baseline Jobs: $total_jobs"
echo "Target: Cityscapes (clear_day) + ACDC (foggy, night, rainy, snowy)"
echo "Queue: $QUEUE"
echo "GPU Memory: $GPU_MEM"
echo "Dry Run: $DRY_RUN"
echo ""

# Submit baseline jobs
submitted_count=0
skipped_count=0

echo "=== BASELINE MODELS ==="
for source_dataset in "${SELECTED_SOURCES[@]}"; do
    for model in "${SELECTED_MODELS[@]}"; do
        for variant in "${SELECTED_VARIANTS[@]}"; do
            local_full_model="${model}${variant}"
            echo "Processing: ${source_dataset} / ${local_full_model}"
            
            # Check if results already exist
            if [ "$SKIP_EXISTING" = true ] && check_results_exist "$source_dataset" "$local_full_model"; then
                echo "  SKIP: Results already exist"
                ((skipped_count++))
                continue
            fi
            
            # Submit job
            if submit_job "$source_dataset" "$model" "$variant" "$QUEUE" "$GPU_MEM" "$GPU_MODE" "$NUM_CPUS" "$DRY_RUN"; then
                ((submitted_count++))
            else
                ((skipped_count++))
            fi
        done
    done
done

# Submit strategy jobs if requested
if [ "$ALL_STRATEGIES_MODE" = true ] || [ -n "$FILTER_STRATEGY" ]; then
    echo ""
    echo "=== AUGMENTATION STRATEGIES ==="
    
    # Determine which strategies to process
    if [ -n "$FILTER_STRATEGY" ]; then
        SELECTED_STRATEGIES=("$FILTER_STRATEGY")
    else
        SELECTED_STRATEGIES=("${TOP_STRATEGIES[@]}")
    fi
    
    for strategy in "${SELECTED_STRATEGIES[@]}"; do
        echo ""
        echo "[$strategy]"
        for source_dataset in "${SOURCE_DATASETS[@]}"; do
            for model in "${MODELS[@]}"; do
                echo "Processing: ${strategy} / ${source_dataset} / ${model}"
                
                # Check if results already exist
                if [ "$SKIP_EXISTING" = true ] && check_strategy_results_exist "$strategy" "$source_dataset" "$model"; then
                    echo "  SKIP: Results already exist"
                    ((skipped_count++))
                    continue
                fi
                
                # Submit job
                if submit_strategy_job "$strategy" "$source_dataset" "$model" "$QUEUE" "$GPU_MEM" "$GPU_MODE" "$NUM_CPUS" "$DRY_RUN"; then
                    ((submitted_count++))
                else
                    ((skipped_count++))
                fi
            done
        done
    done
fi

echo ""
echo "========================================================================"
echo "Summary"
echo "========================================================================"
echo "  Submitted: $submitted_count"
echo "  Skipped: $skipped_count"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "This was a DRY RUN. No jobs were actually submitted."
    echo "Remove --dry-run to submit jobs."
fi
