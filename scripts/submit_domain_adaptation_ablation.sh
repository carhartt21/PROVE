#!/bin/bash
# PROVE Domain Adaptation Ablation - Job Submission Script
#
# This script submits evaluation jobs for the domain adaptation ablation study.
# It evaluates models trained on BDD10k/IDD-AW/MapillaryVistas on the ACDC dataset.
#
# Key features:
# - Evaluates cross-dataset domain adaptation capability
# - Filters out ACDC reference images (_ref_) with mismatched labels
# - Excludes clear_day and dawn_dusk domains (no valid images)
# - Reports per-domain (weather condition) metrics
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

# Models to evaluate
MODELS=("deeplabv3plus_r50" "pspnet_r50" "segformer_mit-b5")

# Default LSF settings
DEFAULT_QUEUE="BatchGPU"
DEFAULT_GPU_MEM="16G"
DEFAULT_GPU_MODE="shared"
DEFAULT_NUM_CPUS=4

# Paths
WEIGHTS_ROOT="${PROVE_WEIGHTS_ROOT:-/scratch/aaa_exchange/AWARE/WEIGHTS}"
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
    echo "  Target: ACDC (adverse weather evaluation)"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --all               Submit jobs for all source/model combinations (9 jobs)"
    echo "  --source-dataset <name>"
    echo "                      Specific source dataset: ${SOURCE_DATASETS[*]}"
    echo "  --model <name>      Specific model: ${MODELS[*]}"
    echo "  --dry-run           Show commands without executing"
    echo "  --list              List all jobs that would be submitted"
    echo "  --queue <name>      LSF queue name (default: $DEFAULT_QUEUE)"
    echo "  --gpu-mem <size>    GPU memory requirement (default: $DEFAULT_GPU_MEM)"
    echo "  --gpu-mode <mode>   GPU mode: shared or exclusive_process (default: $DEFAULT_GPU_MODE)"
    echo "  --num-cpus <n>      Number of CPUs per job (default: $DEFAULT_NUM_CPUS)"
    echo "  --skip-existing     Skip jobs with existing results"
    echo ""
    echo "Examples:"
    echo "  # Submit all 9 evaluation jobs"
    echo "  $0 --all"
    echo ""
    echo "  # Submit single job"
    echo "  $0 --source-dataset BDD10k --model deeplabv3plus_r50"
    echo ""
    echo "  # Dry run - show what would be submitted"
    echo "  $0 --all --dry-run"
    echo ""
    echo "  # List available checkpoints"
    echo "  $0 --list"
    echo ""
}

# Check if checkpoint exists
check_checkpoint() {
    local source_dataset="$1"
    local model="$2"
    
    local checkpoint_dir="${WEIGHTS_ROOT}/baseline/${source_dataset,,}/${model}"
    
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

# Check if results already exist
check_results_exist() {
    local source_dataset="$1"
    local model="$2"
    
    local result_file="${OUTPUT_ROOT}/${source_dataset,,}/${model}/acdc_evaluation.json"
    [ -f "$result_file" ]
}

# Submit a single evaluation job
submit_job() {
    local source_dataset="$1"
    local model="$2"
    local queue="$3"
    local gpu_mem="$4"
    local gpu_mode="$5"
    local num_cpus="$6"
    local dry_run="$7"
    
    # Check for checkpoint
    local checkpoint=$(check_checkpoint "$source_dataset" "$model")
    if [ -z "$checkpoint" ]; then
        echo "  SKIP: No checkpoint found for ${source_dataset}/${model}"
        return 1
    fi
    
    # Job name
    local jobname="da_${source_dataset,,}_${model}_to_acdc"
    
    # Log directory
    local log_dir="${PROJECT_ROOT}/logs/domain_adaptation"
    mkdir -p "$log_dir"
    
    # Build the evaluation command
    local eval_cmd="python ${PROJECT_ROOT}/tools/evaluate_domain_adaptation.py \
        --source-dataset ${source_dataset} \
        --model ${model} \
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
        \"source ${PYTHON_ENV} 2>/dev/null || conda activate ${CONDA_ENV}; ${eval_cmd}\""
    
    if [ "$dry_run" = true ]; then
        echo "  [DRY-RUN] Would submit: $jobname"
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
    echo "Source Dataset / Model / Checkpoint Status"
    echo "-------------------------------------------"
    
    for source_dataset in "${SOURCE_DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            local checkpoint=$(check_checkpoint "$source_dataset" "$model")
            local result_exists=""
            
            if check_results_exist "$source_dataset" "$model"; then
                result_exists=" [RESULTS EXIST]"
            fi
            
            if [ -n "$checkpoint" ]; then
                echo "  ✓ ${source_dataset} / ${model}${result_exists}"
                echo "    Checkpoint: ${checkpoint}"
            else
                echo "  ✗ ${source_dataset} / ${model} - NO CHECKPOINT"
            fi
        done
    done
    
    echo ""
    echo "Total configurations: $((${#SOURCE_DATASETS[@]} * ${#MODELS[@]}))"
}

# ============================================================================
# Parse Command Line Arguments
# ============================================================================

ALL_MODE=false
DRY_RUN=false
LIST_MODE=false
SKIP_EXISTING=false
QUEUE="$DEFAULT_QUEUE"
GPU_MEM="$DEFAULT_GPU_MEM"
GPU_MODE="$DEFAULT_GPU_MODE"
NUM_CPUS="$DEFAULT_NUM_CPUS"
FILTER_SOURCE=""
FILTER_MODEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            ALL_MODE=true
            shift
            ;;
        --source-dataset)
            FILTER_SOURCE="$2"
            shift 2
            ;;
        --model)
            FILTER_MODEL="$2"
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
elif [ -n "$FILTER_SOURCE" ] && [ -n "$FILTER_MODEL" ]; then
    SELECTED_SOURCES=("$FILTER_SOURCE")
    SELECTED_MODELS=("$FILTER_MODEL")
elif [ -n "$FILTER_SOURCE" ]; then
    SELECTED_SOURCES=("$FILTER_SOURCE")
    SELECTED_MODELS=("${MODELS[@]}")
elif [ -n "$FILTER_MODEL" ]; then
    SELECTED_SOURCES=("${SOURCE_DATASETS[@]}")
    SELECTED_MODELS=("$FILTER_MODEL")
else
    print_usage
    echo "ERROR: Specify --all or --source-dataset and/or --model"
    exit 1
fi

echo "========================================================================"
echo "PROVE Domain Adaptation Ablation - Job Submission"
echo "========================================================================"
echo ""
echo "Source Datasets: ${SELECTED_SOURCES[*]}"
echo "Models: ${SELECTED_MODELS[*]}"
echo "Target: ACDC (adverse weather)"
echo "Queue: $QUEUE"
echo "GPU Memory: $GPU_MEM"
echo "Dry Run: $DRY_RUN"
echo ""

# Submit jobs
submitted_count=0
skipped_count=0

for source_dataset in "${SELECTED_SOURCES[@]}"; do
    for model in "${SELECTED_MODELS[@]}"; do
        echo "Processing: ${source_dataset} / ${model}"
        
        # Check if results already exist
        if [ "$SKIP_EXISTING" = true ] && check_results_exist "$source_dataset" "$model"; then
            echo "  SKIP: Results already exist"
            ((skipped_count++))
            continue
        fi
        
        # Submit job
        if submit_job "$source_dataset" "$model" "$QUEUE" "$GPU_MEM" "$GPU_MODE" "$NUM_CPUS" "$DRY_RUN"; then
            ((submitted_count++))
        else
            ((skipped_count++))
        fi
    done
done

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
