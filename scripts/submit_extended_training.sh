#!/bin/bash
# PROVE Extended Training Ablation Study - Job Submission Script
#
# This script continues training from the latest checkpoint for an extended
# number of iterations to study the effect of longer training.
#
# Top 15 Strategies (by average mIoU):
#   1.  std_randaugment+std_mixup    (56.06)
#   2.  gen_LANIT                    (55.71)
#   3.  gen_step1x_new               (55.70)
#   4.  std_mixup+std_autoaugment    (55.67)
#   5.  gen_automold                 (55.62)
#   6.  gen_TSIT                     (55.61)
#   7.  std_randaugment              (55.58)
#   8.  gen_NST                      (55.55)
#   9.  gen_CUT                      (55.52)
#   10. gen_Attribute_Hallucination  (55.46)
#   11. gen_UniControl               (55.46)
#   12. std_cutmix+std_autoaugment   (55.45)
#   13. gen_Img2Img                  (55.43)
#   14. gen_flux1_kontext            (55.40)
#   15. gen_SUSTechGAN               (55.37)
#
# This script:
#   - Finds the latest checkpoint for each configuration
#   - Resumes training with extended max_iters
#   - Disables early stopping to train for full duration
#   - Stores weights in a separate folder for ablation results
#
# Usage:
#   ./submit_extended_training.sh [options]
#
# Options:
#   --dry-run           Show commands without executing
#   --list              List all jobs that would be submitted
#   --dataset <name>    Filter to specific dataset
#   --model <name>      Filter to specific model
#   --strategy <name>   Filter to specific strategy
#   --max-iters <n>     Maximum iterations (default: 160000)
#   --queue <name>      LSF queue name (default: BatchGPU)
#   --gpu-mem <size>    GPU memory requirement (default: 24G)
#   --gpu-mode <mode>   GPU mode: shared or exclusive_process (default: shared)
#   --num-cpus <n>      Number of CPUs per job (default: 8)
#   --limit <n>         Limit number of jobs to submit
#   --weights-root <path>   Source weights root (default: /scratch/aaa_exchange/AWARE/WEIGHTS)
#   --output-root <path>    Output weights root (default: /scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# ============================================================================
# Configuration
# ============================================================================

# Top 15 strategies by average mIoU (including combined std strategies)
TOP_15_STRATEGIES=(
    "std_randaugment+std_mixup"
    "gen_LANIT"
    "gen_step1x_new"
    "std_mixup+std_autoaugment"
    "gen_automold"
    "gen_TSIT"
    "std_randaugment"
    "gen_NST"
    "gen_CUT"
    "gen_Attribute_Hallucination"
    "gen_UniControl"
    "std_cutmix+std_autoaugment"
    "gen_Img2Img"
    "gen_flux1_kontext"
    "gen_SUSTechGAN"
)

# Datasets and models
DATASETS=("ACDC" "BDD10k" "IDD-AW" "MapillaryVistas" "OUTSIDE15k")
MODELS=("deeplabv3plus_r50" "pspnet_r50" "segformer_mit-b5")

# Default settings
DEFAULT_MAX_ITERS=160000  # 2x the default 80000
DEFAULT_QUEUE="BatchGPU"
DEFAULT_GPU_MEM="24G"
DEFAULT_GPU_MODE="shared"
DEFAULT_NUM_CPUS=8
DEFAULT_WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS"
DEFAULT_OUTPUT_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED"

# ============================================================================
# Helper Functions
# ============================================================================

print_usage() {
    echo "PROVE Extended Training Ablation Study - Job Submission Script"
    echo "==============================================================="
    echo ""
    echo "Continues training from the latest checkpoint for extended iterations"
    echo "using the top 15 best performing strategies."
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
    echo "  --strategy <name>   Filter to specific strategy (use quotes for combined)"
    echo "  --max-iters <n>     Maximum iterations (default: $DEFAULT_MAX_ITERS)"
    echo "  --queue <name>      LSF queue name (default: $DEFAULT_QUEUE)"
    echo "  --gpu-mem <size>    GPU memory requirement (default: $DEFAULT_GPU_MEM)"
    echo "  --gpu-mode <mode>   GPU mode: shared or exclusive_process (default: $DEFAULT_GPU_MODE)"
    echo "  --num-cpus <n>      Number of CPUs per job (default: $DEFAULT_NUM_CPUS)"
    echo "  --limit <n>         Limit number of jobs to submit"
    echo "  --weights-root <path>"
    echo "                      Source weights root (default: $DEFAULT_WEIGHTS_ROOT)"
    echo "  --output-root <path>"
    echo "                      Output weights root (default: $DEFAULT_OUTPUT_ROOT)"
    echo "  --help              Show this help message"
    echo ""
    echo "Top 15 Strategies (by average mIoU):"
    for i in "${!TOP_15_STRATEGIES[@]}"; do
        printf "  %2d. %s\n" $((i+1)) "${TOP_15_STRATEGIES[$i]}"
    done
    echo ""
    echo "Training Extension:"
    echo "  - Default: 80,000 → 160,000 iterations (2x)"
    echo "  - Early stopping is DISABLED for full training duration"
    echo "  - Resumes from latest checkpoint to continue training"
    echo ""
    echo "Iteration Recommendations:"
    echo "  - 160,000 (2x):   Moderate extension, ~2x training time"
    echo "  - 240,000 (3x):   Significant extension, ~3x training time"  
    echo "  - 320,000 (4x):   Large extension, recommended max"
    echo "  - 400,000 (5x):   Very long, may see diminishing returns"
    echo ""
    echo "Examples:"
    echo "  $0 --list                                    # List all jobs"
    echo "  $0 --dry-run                                 # Show commands"
    echo "  $0 --dataset ACDC --model deeplabv3plus_r50  # Single config"
    echo "  $0 --strategy gen_LANIT --max-iters 240000   # One strategy, 3x iters"
    echo "  $0 --limit 10                                # Submit first 10 jobs"
    echo ""
    echo "Total jobs (full ablation): 15 strategies × 5 datasets × 3 models = 225 jobs"
}

# Find latest checkpoint for a given model directory
find_latest_checkpoint() {
    local model_dir="$1"
    
    if [ ! -d "$model_dir" ]; then
        echo ""
        return
    fi
    
    # Find all iter_*.pth files and get the one with highest iteration
    local latest=""
    local max_iter=0
    
    for ckpt in "$model_dir"/iter_*.pth; do
        if [ -f "$ckpt" ]; then
            # Extract iteration number from filename
            local iter_num=$(basename "$ckpt" | sed 's/iter_\([0-9]*\)\.pth/\1/')
            if [ "$iter_num" -gt "$max_iter" ]; then
                max_iter=$iter_num
                latest="$ckpt"
            fi
        fi
    done
    
    echo "$latest"
}

# Get iteration number from checkpoint path
get_checkpoint_iter() {
    local ckpt_path="$1"
    if [ -z "$ckpt_path" ]; then
        echo "0"
        return
    fi
    basename "$ckpt_path" | sed 's/iter_\([0-9]*\)\.pth/\1/'
}

# Parse strategy into main strategy and std_strategy
parse_strategy() {
    local strategy="$1"
    
    # Check if it's a combined std+std strategy
    if [[ "$strategy" == std_*+std_* ]]; then
        # Split on + and return both parts
        local part1="${strategy%%+*}"
        local part2="${strategy#*+}"
        echo "$part1|$part2"
    # Check if it's a gen+std combination
    elif [[ "$strategy" == gen_*+std_* ]]; then
        local gen_part="${strategy%%+*}"
        local std_part="${strategy#*+}"
        echo "$gen_part|$std_part"
    else
        # Single strategy
        echo "$strategy|"
    fi
}

# ============================================================================
# Parse Arguments
# ============================================================================

DRY_RUN=false
LIST_ONLY=false
FILTER_DATASET=""
FILTER_MODEL=""
FILTER_STRATEGY=""
MAX_ITERS="$DEFAULT_MAX_ITERS"
QUEUE="$DEFAULT_QUEUE"
GPU_MEM="$DEFAULT_GPU_MEM"
GPU_MODE="$DEFAULT_GPU_MODE"
NUM_CPUS="$DEFAULT_NUM_CPUS"
LIMIT=0
WEIGHTS_ROOT="$DEFAULT_WEIGHTS_ROOT"
OUTPUT_ROOT="$DEFAULT_OUTPUT_ROOT"

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
        --max-iters)
            MAX_ITERS="$2"
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
            LIMIT="$2"
            shift 2
            ;;
        --weights-root)
            WEIGHTS_ROOT="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
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

# ============================================================================
# Main Logic
# ============================================================================

echo "PROVE Extended Training Ablation Study"
echo "======================================="
echo ""
echo "Configuration:"
echo "  Source Weights: $WEIGHTS_ROOT"
echo "  Output Weights: $OUTPUT_ROOT"
echo "  Max Iterations: $MAX_ITERS"
echo "  Queue:          $QUEUE"
echo "  GPU Memory:     $GPU_MEM"
echo "  GPU Mode:       $GPU_MODE"
echo "  CPUs/Job:       $NUM_CPUS"

if [ -n "$FILTER_DATASET" ]; then
    echo "  Dataset Filter:  $FILTER_DATASET"
fi
if [ -n "$FILTER_MODEL" ]; then
    echo "  Model Filter:    $FILTER_MODEL"
fi
if [ -n "$FILTER_STRATEGY" ]; then
    echo "  Strategy Filter: $FILTER_STRATEGY"
fi
if [ $LIMIT -gt 0 ]; then
    echo "  Job Limit:       $LIMIT"
fi
echo ""

# Create logs directory
mkdir -p logs

# Collect jobs to submit
declare -a JOBS=()
declare -a SKIPPED_NO_CHECKPOINT=()

for strategy in "${TOP_15_STRATEGIES[@]}"; do
    # Apply strategy filter
    if [ -n "$FILTER_STRATEGY" ] && [ "$strategy" != "$FILTER_STRATEGY" ]; then
        continue
    fi
    
    # Parse strategy into main and std parts
    IFS='|' read -r main_strategy std_strategy <<< "$(parse_strategy "$strategy")"
    
    # Build the directory path for this strategy
    if [ -n "$std_strategy" ]; then
        strategy_dir="${main_strategy}+${std_strategy}"
    else
        strategy_dir="$main_strategy"
    fi
    
    for dataset in "${DATASETS[@]}"; do
        # Apply dataset filter
        if [ -n "$FILTER_DATASET" ] && [ "$dataset" != "$FILTER_DATASET" ]; then
            continue
        fi
        
        dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
        
        for model in "${MODELS[@]}"; do
            # Apply model filter
            if [ -n "$FILTER_MODEL" ] && [ "$model" != "$FILTER_MODEL" ]; then
                continue
            fi
            
            # Build model directory path
            model_dir="${WEIGHTS_ROOT}/${strategy_dir}/${dataset_lower}/${model}"
            
            # Find latest checkpoint
            latest_ckpt=$(find_latest_checkpoint "$model_dir")
            
            if [ -z "$latest_ckpt" ]; then
                SKIPPED_NO_CHECKPOINT+=("${strategy}|${dataset}|${model}")
                continue
            fi
            
            # Get current iteration
            current_iter=$(get_checkpoint_iter "$latest_ckpt")
            
            # Skip if already at or beyond max_iters
            if [ "$current_iter" -ge "$MAX_ITERS" ]; then
                continue
            fi
            
            # Build job entry: strategy|std_strategy|dataset|model|checkpoint|current_iter
            JOBS+=("${main_strategy}|${std_strategy}|${dataset}|${model}|${latest_ckpt}|${current_iter}")
        done
    done
done

TOTAL_JOBS=${#JOBS[@]}
TOTAL_SKIPPED=${#SKIPPED_NO_CHECKPOINT[@]}

echo "Found $TOTAL_JOBS jobs to process"
if [ $TOTAL_SKIPPED -gt 0 ]; then
    echo "Skipped $TOTAL_SKIPPED configurations (no checkpoint found)"
fi
echo ""

if [ $TOTAL_JOBS -eq 0 ]; then
    echo "No jobs match the specified filters or all are already at max iterations."
    if [ $TOTAL_SKIPPED -gt 0 ]; then
        echo ""
        echo "Configurations without checkpoints:"
        for skipped in "${SKIPPED_NO_CHECKPOINT[@]:0:10}"; do
            IFS='|' read -r s d m <<< "$skipped"
            echo "  - $s / $d / $m"
        done
        if [ $TOTAL_SKIPPED -gt 10 ]; then
            echo "  ... and $((TOTAL_SKIPPED - 10)) more"
        fi
    fi
    exit 0
fi

# Apply limit if specified
if [ $LIMIT -gt 0 ] && [ $LIMIT -lt $TOTAL_JOBS ]; then
    echo "Limiting to first $LIMIT jobs"
    JOBS=("${JOBS[@]:0:$LIMIT}")
fi

# Process jobs
if [ "$LIST_ONLY" = true ]; then
    echo "Jobs to submit:"
    echo "---------------"
    printf "%-35s %-18s %-25s %-12s %-10s\n" "Strategy" "Dataset" "Model" "Current" "Target"
    printf "%-35s %-18s %-25s %-12s %-10s\n" "--------" "-------" "-----" "-------" "------"
    
    for job in "${JOBS[@]}"; do
        IFS='|' read -r main_strat std_strat dataset model ckpt current_iter <<< "$job"
        if [ -n "$std_strat" ]; then
            full_strategy="${main_strat}+${std_strat}"
        else
            full_strategy="$main_strat"
        fi
        printf "%-35s %-18s %-25s %-12s %-10s\n" "$full_strategy" "$dataset" "$model" "$current_iter" "$MAX_ITERS"
    done
    
    echo ""
    echo "Total: ${#JOBS[@]} jobs"
    exit 0
fi

# Submit jobs
SUBMITTED=0
SKIPPED=0

for job in "${JOBS[@]}"; do
    IFS='|' read -r main_strategy std_strategy dataset model checkpoint current_iter <<< "$job"
    
    # Build full strategy name for display
    if [ -n "$std_strategy" ]; then
        full_strategy="${main_strategy}+${std_strategy}"
    else
        full_strategy="$main_strategy"
    fi
    
    # Build job name
    local_iters_k=$((MAX_ITERS / 1000))
    job_name="ext_${dataset}_${model}_${full_strategy}_${local_iters_k}k"
    # Sanitize job name (replace + with _)
    job_name="${job_name//+/_}"
    
    # Build training command
    train_cmd="PROVE_WEIGHTS_ROOT='${OUTPUT_ROOT}' $SCRIPT_DIR/train_unified.sh single --dataset $dataset --model $model --strategy $main_strategy --no-early-stop --max-iters $MAX_ITERS --resume-from $checkpoint"
    
    # Add std_strategy if present
    if [ -n "$std_strategy" ]; then
        train_cmd="$train_cmd --std-strategy $std_strategy"
    fi
    
    # Build bsub command
    bsub_cmd="bsub -gpu \"num=1:mode=${GPU_MODE}:gmem=${GPU_MEM}\" \
        -q ${QUEUE} \
        -R \"span[hosts=1]\" \
        -n ${NUM_CPUS} \
        -oo \"logs/${job_name}_%J.log\" \
        -eo \"logs/${job_name}_%J.err\" \
        -L /bin/bash \
        -J \"${job_name}\" \
        \"${train_cmd}\""
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] $job_name"
        echo "  Strategy:   $full_strategy"
        echo "  Dataset:    $dataset"
        echo "  Model:      $model"
        echo "  Resume:     $checkpoint"
        echo "  Progress:   $current_iter → $MAX_ITERS"
        echo "  Command:    $train_cmd"
        echo ""
        SUBMITTED=$((SUBMITTED + 1))
    else
        echo "Submitting: $job_name ($current_iter → $MAX_ITERS)"
        if eval $bsub_cmd; then
            SUBMITTED=$((SUBMITTED + 1))
        else
            echo "  ERROR: Failed to submit job"
            SKIPPED=$((SKIPPED + 1))
        fi
    fi
done

echo ""
echo "=========================================="
if [ "$DRY_RUN" = true ]; then
    echo "[DRY-RUN] Would submit $SUBMITTED jobs"
else
    echo "Submitted: $SUBMITTED jobs"
    if [ $SKIPPED -gt 0 ]; then
        echo "Failed:    $SKIPPED jobs"
    fi
fi
echo ""
echo "Extended weights will be saved to:"
echo "  $OUTPUT_ROOT/{strategy}/{dataset}/{model}/"
echo ""
echo "To monitor jobs: bjobs -w"
echo "To check logs:   ls -la logs/ext_*.log"
