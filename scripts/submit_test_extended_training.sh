#!/bin/bash
# PROVE Extended Training Study - Test Job Submission Script
#
# This script submits evaluation/testing jobs for the extended training study.
# It tests models at different iteration checkpoints (80k, 120k, 160k, etc.)
#
# The script scans for available checkpoints in WEIGHTS_EXTENDED
# and submits test jobs for each checkpoint found.
#
# Usage:
#   ./submit_test_extended_training.sh [options]
#
# Options:
#   --dry-run           Show commands without executing
#   --list              List all jobs that would be submitted
#   --dataset <name>    Filter to specific dataset
#   --model <name>      Filter to specific model
#   --strategy <name>   Filter to specific strategy
#   --iteration <n>     Filter to specific iteration (e.g., 80000, 160000)
#   --queue <name>      LSF queue name (default: BatchGPU)
#   --gpu-mem <size>    GPU memory requirement (default: 16G)
#   --limit <n>         Limit number of jobs to submit
#   --weights-root <path> Custom weights root directory

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# ============================================================================
# Configuration
# ============================================================================

# Available strategies in WEIGHTS_EXTENDED (gen_* strategies from extended training)
AVAILABLE_STRATEGIES=(
    "gen_cyclediffusion"
    "gen_flux_kontext"
    "gen_step1x_new"
    "std_randaugment"
    "gen_step1x_v1p2"
)

# Iteration checkpoints to test
ITERATIONS=(80000 120000 160000 200000 240000 280000 320000)

# Datasets and models
DATASETS=("BDD10k" "IDD-AW" "MapillaryVistas" "OUTSIDE15k")
MODELS=("pspnet_r50" "segformer_mit-b5")

# Default LSF settings
DEFAULT_QUEUE="BatchGPU"
DEFAULT_GPU_MEM="16G"
DEFAULT_GPU_MODE="shared"
DEFAULT_NUM_CPUS=10
DEFAULT_WEIGHTS_ROOT="${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED"

# ============================================================================
# Helper Functions
# ============================================================================

print_usage() {
    echo "PROVE Extended Training Study - Test Job Submission Script"
    echo "==========================================================="
    echo ""
    echo "Submits evaluation/testing jobs for extended training study models."
    echo "Tests checkpoints at different training iterations."
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
    echo "  --iteration <n>     Filter to specific iteration"
    echo "                      Common: ${ITERATIONS[*]}"
    echo "  --queue <name>      LSF queue name (default: $DEFAULT_QUEUE)"
    echo "  --gpu-mem <size>    GPU memory requirement (default: $DEFAULT_GPU_MEM)"
    echo "  --gpu-mode <mode>   GPU mode: shared or exclusive_process (default: $DEFAULT_GPU_MODE)"
    echo "  --num-cpus <n>      Number of CPUs per job (default: $DEFAULT_NUM_CPUS)"
    echo "  --limit <n>         Limit number of jobs to submit"
    echo "  --weights-root <path>"
    echo "                      Custom weights root directory"
    echo "                      (default: $DEFAULT_WEIGHTS_ROOT)"
    echo "  --skip-tested       Skip configurations that already have test results"
    echo "  --all-checkpoints   Test all available checkpoints, not just standard iterations"
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
    echo "  # Submit tests for 160k iteration only"
    echo "  $0 --iteration 160000"
    echo ""
    echo "  # Submit first 10 jobs"
    echo "  $0 --limit 10"
    echo ""
    echo "  # Test all found checkpoints"
    echo "  $0 --all-checkpoints --list"
}

# Find available checkpoints in a model directory
find_checkpoints() {
    local model_dir="$1"
    local filter_iter="$2"
    
    if [ ! -d "$model_dir" ]; then
        return
    fi
    
    # Find all iter_*.pth files
    for ckpt in "$model_dir"/iter_*.pth; do
        if [ -f "$ckpt" ]; then
            # Extract iteration number
            local iter=$(basename "$ckpt" | sed 's/iter_\([0-9]*\)\.pth/\1/')
            
            # Apply filter if specified
            if [ -n "$filter_iter" ] && [ "$iter" != "$filter_iter" ]; then
                continue
            fi
            
            echo "$iter"
        fi
    done
}

# Check if test results exist for a specific iteration
test_results_exist_for_iter() {
    local weights_root="$1"
    local strategy="$2"
    local dataset="$3"
    local model="$4"
    local iteration="$5"
    
    local results_dir="${weights_root}/${strategy}/${dataset,,}/${model}/test_results/test"
    
    # Check for result file with this iteration
    if [ -d "$results_dir" ]; then
        # Look for any JSON that mentions this iteration
        local result_file="${results_dir}/test_results_iter_${iteration}.json"
        if [ -f "$result_file" ]; then
            return 0
        fi
        
        # Also check timestamped directories for results
        for ts_dir in "$results_dir"/*/; do
            if [ -d "$ts_dir" ]; then
                # Check log files for this iteration
                if grep -q "iter_${iteration}" "$ts_dir"/*.log 2>/dev/null; then
                    return 0
                fi
            fi
        done
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
FILTER_ITERATION=""
QUEUE="$DEFAULT_QUEUE"
GPU_MEM="$DEFAULT_GPU_MEM"
GPU_MODE="$DEFAULT_GPU_MODE"
NUM_CPUS="$DEFAULT_NUM_CPUS"
JOB_LIMIT=""
WEIGHTS_ROOT="$DEFAULT_WEIGHTS_ROOT"
SKIP_TESTED=false
ALL_CHECKPOINTS=true  # Test all available checkpoints by default

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
        --iteration)
            FILTER_ITERATION="$2"
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
        --all-checkpoints)
            ALL_CHECKPOINTS=true
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
FOUND=0
SUBMITTED=0
SKIPPED=0

echo "=============================================="
echo "PROVE Extended Training - Test Job Submission"
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

# Check if weights root exists
if [ ! -d "$WEIGHTS_ROOT" ]; then
    echo "Warning: Weights root does not exist: $WEIGHTS_ROOT"
    echo "No checkpoints found."
    exit 0
fi

# Iterate through configurations
for strategy in "${AVAILABLE_STRATEGIES[@]}"; do
    # Apply strategy filter
    if [ -n "$FILTER_STRATEGY" ] && [ "$strategy" != "$FILTER_STRATEGY" ]; then
        continue
    fi
    
    strategy_dir="${WEIGHTS_ROOT}/${strategy}"
    if [ ! -d "$strategy_dir" ]; then
        continue
    fi
    
    for dataset in "${DATASETS[@]}"; do
        # Apply dataset filter
        if [ -n "$FILTER_DATASET" ] && [ "$dataset" != "$FILTER_DATASET" ]; then
            continue
        fi
        
        dataset_dir="${strategy_dir}/${dataset,,}"
        if [ ! -d "$dataset_dir" ]; then
            continue
        fi
        
        for model in "${MODELS[@]}"; do
            # Apply model filter
            if [ -n "$FILTER_MODEL" ] && [ "$model" != "$FILTER_MODEL" ]; then
                continue
            fi
            
            model_dir="${dataset_dir}/${model}_ratio0p50"
            if [ ! -d "$model_dir" ]; then
                continue
            fi
            
            # Find checkpoints in this directory
            if [ "$ALL_CHECKPOINTS" = true ]; then
                # Find all checkpoints
                checkpoints=$(find_checkpoints "$model_dir" "$FILTER_ITERATION")
            else
                # Only check standard iterations
                checkpoints=""
                for iter in "${ITERATIONS[@]}"; do
                    if [ -n "$FILTER_ITERATION" ] && [ "$iter" != "$FILTER_ITERATION" ]; then
                        continue
                    fi
                    if [ -f "${model_dir}/iter_${iter}.pth" ]; then
                        checkpoints="$checkpoints $iter"
                    fi
                done
            fi
            
            for iteration in $checkpoints; do
                FOUND=$((FOUND + 1))
                
                # Check if already tested (if skip option enabled)
                if [ "$SKIP_TESTED" = true ]; then
                    if test_results_exist_for_iter "$WEIGHTS_ROOT" "$strategy" "$dataset" "$model" "$iteration"; then
                        SKIPPED=$((SKIPPED + 1))
                        continue
                    fi
                fi
                
                # Check job limit
                if [ -n "$JOB_LIMIT" ] && [ "$SUBMITTED" -ge "$JOB_LIMIT" ]; then
                    echo "Job limit ($JOB_LIMIT) reached."
                    break 4
                fi
                
                # Format iteration for job name (e.g., 160000 -> 160k)
                iter_k=$((iteration / 1000))k
                job_name="ext_test_${dataset}_${model}_${strategy}_${iter_k}"
                
                # Build test command with specific checkpoint
                checkpoint_path="${model_dir}/iter_${iteration}.pth"
                test_cmd="$SCRIPT_DIR/test_unified.sh single --dataset $dataset --model $model --strategy $strategy --checkpoint $checkpoint_path --work-dir $WEIGHTS_ROOT"
                
                if [ "$LIST_ONLY" = true ]; then
                    echo "[$FOUND] $strategy / $dataset / $model / iter=${iteration}"
                elif [ "$DRY_RUN" = true ]; then
                    echo "[$FOUND] Would submit: $job_name"
                    echo "    Checkpoint: $checkpoint_path"
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
