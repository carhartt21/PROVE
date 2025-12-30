#!/bin/bash
# PROVE Ratio Ablation Study - Test Job Submission Script
#
# This script submits evaluation/testing jobs for the ratio ablation study.
# It tests trained models with varying ratios of generated to real images.
#
# The script scans for available checkpoints in WEIGHTS_RATIO_ABLATION
# and submits test jobs for each configuration found.
#
# Usage:
#   ./submit_test_ratio_ablation.sh [options]
#
# Options:
#   --dry-run           Show commands without executing
#   --list              List all jobs that would be submitted
#   --dataset <name>    Filter to specific dataset
#   --model <name>      Filter to specific model
#   --strategy <name>   Filter to specific strategy
#   --ratio <value>     Filter to specific ratio
#   --queue <name>      LSF queue name (default: BatchGPU)
#   --gpu-mem <size>    GPU memory requirement (default: 16G)
#   --limit <n>         Limit number of jobs to submit
#   --weights-root <path> Custom weights root directory

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ============================================================================
# Configuration
# ============================================================================

# Top 5 gen_* strategies used in ratio ablation
TOP_5_GEN_STRATEGIES=(
    "gen_LANIT"
    "gen_step1x_new"
    "gen_automold"
    "gen_TSIT"
    "gen_NST"
)

# Ratios tested in ablation (excluding 0.5 which is in regular WEIGHTS)
RATIOS=(0.125 0.25 0.375 0.625 0.75 0.875 1.0)

# Datasets and models
DATASETS=("ACDC" "BDD10k" "IDD-AW" "MapillaryVistas" "OUTSIDE15k")
MODELS=("deeplabv3plus_r50" "pspnet_r50" "segformer_mit-b5")

# Default LSF settings
DEFAULT_QUEUE="BatchGPU"
DEFAULT_GPU_MEM="16G"
DEFAULT_GPU_MODE="shared"
DEFAULT_NUM_CPUS=4
DEFAULT_WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION"

# ============================================================================
# Helper Functions
# ============================================================================

print_usage() {
    echo "PROVE Ratio Ablation Study - Test Job Submission Script"
    echo "======================================================="
    echo ""
    echo "Submits evaluation/testing jobs for ratio ablation study models."
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --dry-run           Show commands without executing"
    echo "  --list              List all jobs that would be submitted"
    echo "  --dataset <name>    Filter to specific dataset"
    echo "                      Available: ${DATASETS[*]}"
    echo "  --model <name>      Filter to specific model"
    echo "                      Available: ${MODELS[*]}"
    echo "  --strategy <name>   Filter to specific strategy"
    echo "                      Available: ${TOP_5_GEN_STRATEGIES[*]}"
    echo "  --ratio <value>     Filter to specific ratio"
    echo "                      Available: ${RATIOS[*]}"
    echo "  --queue <name>      LSF queue name (default: $DEFAULT_QUEUE)"
    echo "  --gpu-mem <size>    GPU memory requirement (default: $DEFAULT_GPU_MEM)"
    echo "  --gpu-mode <mode>   GPU mode: shared or exclusive_process (default: $DEFAULT_GPU_MODE)"
    echo "  --num-cpus <n>      Number of CPUs per job (default: $DEFAULT_NUM_CPUS)"
    echo "  --limit <n>         Limit number of jobs to submit"
    echo "  --weights-root <path>"
    echo "                      Custom weights root directory"
    echo "                      (default: $DEFAULT_WEIGHTS_ROOT)"
    echo "  --skip-tested       Skip configurations that already have test results"
    echo ""
    echo "Examples:"
    echo "  # List all available test jobs"
    echo "  $0 --list"
    echo ""
    echo "  # Dry run for all jobs"
    echo "  $0 --dry-run"
    echo ""
    echo "  # Submit tests for specific dataset"
    echo "  $0 --dataset ACDC"
    echo ""
    echo "  # Submit tests for specific ratio"
    echo "  $0 --ratio 0.25"
    echo ""
    echo "  # Submit first 10 jobs"
    echo "  $0 --limit 10"
    echo ""
    echo "  # Skip already tested configurations"
    echo "  $0 --skip-tested"
}

# Convert ratio to directory suffix format (e.g., 0.25 -> _ratio0p25)
ratio_to_suffix() {
    local ratio="$1"
    if [[ "$ratio" == "1.0" ]]; then
        echo ""
    else
        echo "_ratio${ratio//.p/p}" | sed 's/\./_/g'
    fi
}

# Check if checkpoint exists for a configuration
checkpoint_exists() {
    local weights_root="$1"
    local strategy="$2"
    local dataset="$3"
    local model="$4"
    local ratio="$5"
    
    local ratio_suffix=$(ratio_to_suffix "$ratio")
    local checkpoint_dir="${weights_root}/${strategy}/${dataset,,}/${model}${ratio_suffix}"
    
    # Check for best checkpoint or iter checkpoint
    if [ -f "${checkpoint_dir}/best_mIoU_iter_"*.pth ] 2>/dev/null || \
       [ -f "${checkpoint_dir}/iter_"*.pth ] 2>/dev/null || \
       [ -f "${checkpoint_dir}/latest.pth" ]; then
        return 0
    fi
    return 1
}

# Check if test results exist for a configuration
test_results_exist() {
    local weights_root="$1"
    local strategy="$2"
    local dataset="$3"
    local model="$4"
    local ratio="$5"
    
    local ratio_suffix=$(ratio_to_suffix "$ratio")
    local results_dir="${weights_root}/${strategy}/${dataset,,}/${model}${ratio_suffix}/test_results/test"
    
    # Check for any timestamp directories with JSON results
    if [ -d "$results_dir" ]; then
        local json_count=$(find "$results_dir" -name "*.json" 2>/dev/null | wc -l)
        if [ "$json_count" -gt 0 ]; then
            return 0
        fi
    fi
    return 1
}

# ============================================================================
# Main Script
# ============================================================================

# Parse arguments
DRY_RUN=false
LIST_ONLY=false
FILTER_DATASET=""
FILTER_MODEL=""
FILTER_STRATEGY=""
FILTER_RATIO=""
QUEUE="$DEFAULT_QUEUE"
GPU_MEM="$DEFAULT_GPU_MEM"
GPU_MODE="$DEFAULT_GPU_MODE"
NUM_CPUS="$DEFAULT_NUM_CPUS"
JOB_LIMIT=""
WEIGHTS_ROOT="$DEFAULT_WEIGHTS_ROOT"
SKIP_TESTED=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --list)
            LIST_ONLY=true
            shift
            ;;
        --dataset)
            FILTER_DATASET="$2"
            shift 2
            ;;
        --model)
            FILTER_MODEL="$2"
            shift 2
            ;;
        --strategy)
            FILTER_STRATEGY="$2"
            shift 2
            ;;
        --ratio)
            FILTER_RATIO="$2"
            shift 2
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
        --limit)
            JOB_LIMIT="$2"
            shift 2
            ;;
        --weights-root)
            WEIGHTS_ROOT="$2"
            shift 2
            ;;
        --skip-tested)
            SKIP_TESTED=true
            shift
            ;;
        --help|-h)
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

# Create logs directory
mkdir -p logs

# Count jobs
TOTAL_POSSIBLE=0
FOUND=0
SUBMITTED=0
SKIPPED=0

echo "=============================================="
echo "PROVE Ratio Ablation - Test Job Submission"
echo "=============================================="
echo ""
echo "Weights root: $WEIGHTS_ROOT"
echo "Queue: $QUEUE"
echo "GPU memory: $GPU_MEM"
echo ""

if [ "$LIST_ONLY" = true ]; then
    echo "Mode: LIST (showing available configurations)"
elif [ "$DRY_RUN" = true ]; then
    echo "Mode: DRY RUN (no jobs will be submitted)"
else
    echo "Mode: SUBMIT (jobs will be submitted)"
fi
echo ""
echo "----------------------------------------------"

# Iterate through configurations
for strategy in "${TOP_5_GEN_STRATEGIES[@]}"; do
    # Apply strategy filter
    if [ -n "$FILTER_STRATEGY" ] && [ "$strategy" != "$FILTER_STRATEGY" ]; then
        continue
    fi
    
    for dataset in "${DATASETS[@]}"; do
        # Apply dataset filter
        if [ -n "$FILTER_DATASET" ] && [ "$dataset" != "$FILTER_DATASET" ]; then
            continue
        fi
        
        for model in "${MODELS[@]}"; do
            # Apply model filter
            if [ -n "$FILTER_MODEL" ] && [ "$model" != "$FILTER_MODEL" ]; then
                continue
            fi
            
            for ratio in "${RATIOS[@]}"; do
                # Apply ratio filter
                if [ -n "$FILTER_RATIO" ] && [ "$ratio" != "$FILTER_RATIO" ]; then
                    continue
                fi
                
                TOTAL_POSSIBLE=$((TOTAL_POSSIBLE + 1))
                
                # Check if checkpoint exists
                if ! checkpoint_exists "$WEIGHTS_ROOT" "$strategy" "$dataset" "$model" "$ratio"; then
                    continue
                fi
                
                FOUND=$((FOUND + 1))
                
                # Check if already tested (if skip option enabled)
                if [ "$SKIP_TESTED" = true ]; then
                    if test_results_exist "$WEIGHTS_ROOT" "$strategy" "$dataset" "$model" "$ratio"; then
                        SKIPPED=$((SKIPPED + 1))
                        continue
                    fi
                fi
                
                # Check job limit
                if [ -n "$JOB_LIMIT" ] && [ "$SUBMITTED" -ge "$JOB_LIMIT" ]; then
                    echo "Job limit ($JOB_LIMIT) reached."
                    break 4
                fi
                
                # Format ratio for job name and command
                local_ratio_str="${ratio//./_}"
                job_name="ratio_test_${dataset}_${model}_${strategy}_${local_ratio_str}"
                
                # Build test command
                test_cmd="./test_unified.sh single --dataset $dataset --model $model --strategy $strategy --ratio $ratio --work-dir $WEIGHTS_ROOT"
                
                if [ "$LIST_ONLY" = true ]; then
                    echo "[$FOUND] $strategy / $dataset / $model / ratio=$ratio"
                elif [ "$DRY_RUN" = true ]; then
                    echo "[$FOUND] Would submit: $job_name"
                    echo "    Command: $test_cmd"
                    echo ""
                    SUBMITTED=$((SUBMITTED + 1))
                else
                    # Submit job
                    echo "[$FOUND] Submitting: $job_name"
                    
                    bsub -gpu "num=1:mode=${GPU_MODE}:gmem=${GPU_MEM}" \
                        -q "${QUEUE}" \
                        -R "span[hosts=1]" \
                        -n "${NUM_CPUS}" \
                        -oo "logs/${job_name}_%J.log" \
                        -eo "logs/${job_name}_%J.err" \
                        -L /bin/bash \
                        -J "${job_name}" \
                        "${test_cmd}"
                    
                    SUBMITTED=$((SUBMITTED + 1))
                fi
            done
        done
    done
done

echo ""
echo "=============================================="
echo "Summary"
echo "=============================================="
echo "Total possible configurations: $TOTAL_POSSIBLE"
echo "Checkpoints found: $FOUND"
if [ "$SKIP_TESTED" = true ]; then
    echo "Skipped (already tested): $SKIPPED"
fi
if [ "$LIST_ONLY" = true ]; then
    echo "Listed: $FOUND"
elif [ "$DRY_RUN" = true ]; then
    echo "Would submit: $SUBMITTED"
else
    echo "Jobs submitted: $SUBMITTED"
fi
echo "=============================================="
