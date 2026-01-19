#!/bin/bash
# PROVE Ratio Ablation Study - Job Submission Script
#
# This script submits training jobs to vary the ratio of generated to real images
# using the top 5 best performing gen_* strategies.
#
# Top 5 Gen Strategies (by average mIoU, with 4/4 datasets complete):
#   1. gen_TSIT                  (48.8)
#   2. gen_albumentations_weather (48.8)
#   3. gen_cycleGAN              (48.5)
#   4. gen_UniControl            (48.5)
#   5. gen_automold              (47.5)
#
# Ratios: 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0
#   - ratio 1.0 = 100% real images (baseline for that strategy)
#   - ratio 0.5 = 50% real, 50% generated
#   - ratio 0.125 = 12.5% real, 87.5% generated
#
# Usage:
#   ./submit_ratio_ablation.sh [options]
#
# Options:
#   --dry-run           Show commands without executing
#   --list              List all jobs that would be submitted
#   --dataset <name>    Filter to specific dataset (default: all)
#   --model <name>      Filter to specific model (default: all)
#   --strategy <name>   Filter to specific strategy (default: all top 5)
#   --ratio <value>     Filter to specific ratio (default: all)
#   --queue <name>      LSF queue name (default: BatchGPU)
#   --gpu-mem <size>    GPU memory requirement (default: 24G)
#   --gpu-mode <mode>   GPU mode: shared or exclusive_process (default: shared)
#   --num-cpus <n>      Number of CPUs per job (default: 8)
#   --limit <n>         Limit number of jobs to submit
#   --weights-root <path> Custom weights root directory
#                       (default: /scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# ============================================================================
# Configuration
# ============================================================================

# Top 5 gen_* strategies by average mIoU (with 4/4 datasets complete)
# Updated 2026-01-14 based on TESTING_COVERAGE.md results
# NOTE: gen_flux_kontext has 3/4 datasets, excluded for now
TOP_5_GEN_STRATEGIES=(
    "gen_cyclediffusion"
    "gen_step1x_new"
    "gen_step1x_v1p2"
    "gen_stargan_v2"
    "gen_TSIT"
)

# Ratios to test (0.0 to 1.0 in 0.125 increments)
# Note: 0.0 = 100% synthetic, 0.5 is excluded (same as standard gen_* training in WEIGHTS)
RATIOS=(0.0 0.125 0.25 0.375 0.625 0.75 0.875)

# Datasets and models for ablation
# Note: DeepLabV3+ intentionally excluded from ratio ablation study
DATASETS=("BDD10k" "IDD-AW" "MapillaryVistas" "OUTSIDE15k")
MODELS=("pspnet_r50" "segformer_mit-b5")

# Default LSF settings
DEFAULT_QUEUE="BatchGPU"
DEFAULT_GPU_MEM="24G"
DEFAULT_GPU_MODE="shared"
DEFAULT_NUM_CPUS=8
DEFAULT_WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION"

# ============================================================================
# Helper Functions
# ============================================================================

print_usage() {
    echo "PROVE Ratio Ablation Study - Job Submission Script"
    echo "=================================================="
    echo ""
    echo "Submits training jobs varying the ratio of generated to real images"
    echo "using the top 5 best performing gen_* strategies."
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
    echo "  --help              Show this help message"
    echo ""
    echo "Top 5 Gen Strategies (by average mIoU, with 4/4 datasets complete):"
    echo "  1. gen_TSIT                  (48.8)"
    echo "  2. gen_albumentations_weather (48.8)"
    echo "  3. gen_cycleGAN              (48.5)"
    echo "  4. gen_UniControl            (48.5)"
    echo "  5. gen_automold              (47.5)"
    echo ""
    echo "Ratios:"
    echo "  - 1.0   = 100% real images (strategy baseline)"
    echo "  - 0.875 = 87.5% real, 12.5% generated"
    echo "  - 0.75  = 75% real, 25% generated"
    echo "  - 0.625 = 62.5% real, 37.5% generated"
    echo "  - 0.375 = 37.5% real, 62.5% generated"
    echo "  - 0.25  = 25% real, 75% generated"
    echo "  - 0.125 = 12.5% real, 87.5% generated"
    echo ""
    echo "Examples:"
    echo "  $0 --list                                    # List all jobs"
    echo "  $0 --dry-run                                 # Show commands"
    echo "  $0 --dataset ACDC --model deeplabv3plus_r50  # Single config"
    echo "  $0 --strategy gen_LANIT --ratio 0.5          # One strategy/ratio"
    echo "  $0 --limit 10                                # Submit first 10 jobs"
    echo ""
    echo "Total jobs (full ablation): 5 strategies × 8 ratios × 5 datasets × 3 models = 600 jobs"
}

# ============================================================================
# Parse Arguments
# ============================================================================

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
LIMIT=0
WEIGHTS_ROOT="$DEFAULT_WEIGHTS_ROOT"

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
            LIMIT="$2"
            shift 2
            ;;
        --weights-root)
            WEIGHTS_ROOT="$2"
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

echo "PROVE Ratio Ablation Study"
echo "=========================="
echo ""
echo "Configuration:"
echo "  Weights Root: $WEIGHTS_ROOT"
echo "  Queue:        $QUEUE"
echo "  GPU Memory:   $GPU_MEM"
echo "  GPU Mode:     $GPU_MODE"
echo "  CPUs/Job:     $NUM_CPUS"

if [ -n "$FILTER_DATASET" ]; then
    echo "  Dataset Filter: $FILTER_DATASET"
fi
if [ -n "$FILTER_MODEL" ]; then
    echo "  Model Filter:   $FILTER_MODEL"
fi
if [ -n "$FILTER_STRATEGY" ]; then
    echo "  Strategy Filter: $FILTER_STRATEGY"
fi
if [ -n "$FILTER_RATIO" ]; then
    echo "  Ratio Filter:   $FILTER_RATIO"
fi
if [ $LIMIT -gt 0 ]; then
    echo "  Job Limit:      $LIMIT"
fi
echo ""

# Create logs directory
mkdir -p logs

# Collect jobs to submit
declare -a JOBS=()

for strategy in "${TOP_5_GEN_STRATEGIES[@]}"; do
    # Apply strategy filter
    if [ -n "$FILTER_STRATEGY" ] && [ "$strategy" != "$FILTER_STRATEGY" ]; then
        continue
    fi
    
    for ratio in "${RATIOS[@]}"; do
        # Apply ratio filter
        if [ -n "$FILTER_RATIO" ] && [ "$ratio" != "$FILTER_RATIO" ]; then
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
                
                # Build job entry
                JOBS+=("${strategy}|${ratio}|${dataset}|${model}")
            done
        done
    done
done

TOTAL_JOBS=${#JOBS[@]}
echo "Found $TOTAL_JOBS jobs to process"
echo ""

if [ $TOTAL_JOBS -eq 0 ]; then
    echo "No jobs match the specified filters."
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
    printf "%-20s %-8s %-18s %-25s\n" "Strategy" "Ratio" "Dataset" "Model"
    printf "%-20s %-8s %-18s %-25s\n" "--------" "-----" "-------" "-----"
    
    for job in "${JOBS[@]}"; do
        IFS='|' read -r strategy ratio dataset model <<< "$job"
        printf "%-20s %-8s %-18s %-25s\n" "$strategy" "$ratio" "$dataset" "$model"
    done
    
    echo ""
    echo "Total: ${#JOBS[@]} jobs"
    exit 0
fi

# Submit jobs
SUBMITTED=0
SKIPPED=0

for job in "${JOBS[@]}"; do
    IFS='|' read -r strategy ratio dataset model <<< "$job"
    
    # Build job name with ratio
    # Convert ratio to integer percentage for job name (e.g., 0.125 -> r12)
    ratio_pct=$(echo "$ratio * 100" | bc | cut -d. -f1)
    job_name="ratio_${dataset}_${model}_${strategy}_r${ratio_pct}"
    
    # Build training command with custom weights root via environment variable
    train_cmd="PROVE_WEIGHTS_ROOT='${WEIGHTS_ROOT}' $SCRIPT_DIR/train_unified.sh single --dataset $dataset --model $model --strategy $strategy --ratio $ratio"
    
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
        echo "  Strategy: $strategy"
        echo "  Ratio:    $ratio"
        echo "  Dataset:  $dataset"
        echo "  Model:    $model"
        echo "  Command:  $train_cmd"
        echo ""
        SUBMITTED=$((SUBMITTED + 1))
    else
        echo "Submitting: $job_name"
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
        echo "Skipped:   $SKIPPED jobs (errors)"
    fi
fi
echo ""
echo "Weights will be saved to:"
echo "  $WEIGHTS_ROOT/{strategy}/{dataset}/{model}_ratio{X}p{YY}/"
echo "  e.g., $WEIGHTS_ROOT/gen_LANIT/acdc/deeplabv3plus_r50_ratio0p50/"
echo ""
echo "To monitor jobs: bjobs -w"
echo "To check logs:   ls -la logs/ratio_*.log"
