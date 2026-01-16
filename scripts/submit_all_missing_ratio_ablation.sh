#!/bin/bash
# PROVE Ratio Ablation Study - Submit ONLY Missing Jobs
#
# This script scans the WEIGHTS_RATIO_ABLATION directory and submits
# only the jobs that haven't completed yet (missing iter_80000.pth).
#
# Usage:
#   ./submit_all_missing_ratio_ablation.sh [options]
#
# Options:
#   --dry-run           Show commands without executing
#   --list              List all missing jobs without submitting  
#   --strategy <name>   Filter to specific strategy
#   --exclude-ratio0    Skip ratio 0.0 jobs (already running)
#   --help              Show this help message

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Configuration
WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION"
QUEUE="BatchGPU"
GPU_MEM="24G"
GPU_MODE="shared"
NUM_CPUS=8

# Top 5 gen_* strategies by average mIoU (with 4/4 datasets complete)
STRATEGIES=("gen_TSIT" "gen_albumentations_weather" "gen_cycleGAN" "gen_UniControl" "gen_automold")
DATASETS=("ACDC" "BDD10k" "IDD-AW" "MapillaryVistas" "OUTSIDE15k")
DATASETS_LOWER=("acdc" "bdd10k" "idd-aw" "mapillaryvistas" "outside15k")
MODELS=("deeplabv3plus_r50" "pspnet_r50" "segformer_mit-b5")
# Exclude ratio 0.5 (standard training) and 1.0 (same as baseline)
RATIOS=(0.125 0.25 0.375 0.625 0.75 0.875)

# Modes
DRY_RUN=false
LIST_ONLY=false
FILTER_STRATEGY=""
EXCLUDE_RATIO0=false
INCLUDE_RATIO0=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

usage() {
    echo "PROVE Ratio Ablation Study - Submit Missing Jobs"
    echo "================================================="
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dry-run           Show commands without executing"
    echo "  --list              List all missing jobs without submitting"
    echo "  --strategy NAME     Filter to specific strategy only"
    echo "  --exclude-ratio0    Skip ratio 0.0 jobs (if already submitted)"
    echo "  --include-ratio0    Include ratio 0.0 jobs"
    echo "  --help              Show this help message"
    echo ""
    echo "Strategies: ${STRATEGIES[*]}"
    echo "Ratios: ${RATIOS[*]} (+ 0.0 if --include-ratio0)"
    echo ""
    echo "Examples:"
    echo "  $0 --list                      # List all missing jobs"
    echo "  $0 --dry-run                   # Show what would be submitted"
    echo "  $0 --strategy gen_TSIT         # Submit only gen_TSIT jobs"
    echo "  $0 --include-ratio0            # Include ratio 0.0 in submission"
}

# Parse arguments
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
        --strategy)
            FILTER_STRATEGY="$2"
            shift 2
            ;;
        --exclude-ratio0)
            EXCLUDE_RATIO0=true
            shift
            ;;
        --include-ratio0)
            INCLUDE_RATIO0=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Add ratio 0.0 if requested
if [[ "$INCLUDE_RATIO0" == "true" ]]; then
    RATIOS=(0.0 "${RATIOS[@]}")
fi

# Check bsub availability
if [[ "$LIST_ONLY" != "true" && "$DRY_RUN" != "true" ]]; then
    if ! command -v bsub &> /dev/null; then
        echo -e "${RED}ERROR: bsub command not found. Run from an HPC login node.${NC}"
        exit 1
    fi
fi

echo "=============================================="
echo "PROVE Ratio Ablation - Submit Missing Jobs"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Weights Root: $WEIGHTS_ROOT"
echo "  Queue: $QUEUE"
echo "  GPU Memory: $GPU_MEM"
echo ""

if [[ -n "$FILTER_STRATEGY" ]]; then
    echo "  Filter: $FILTER_STRATEGY only"
fi
echo "  Ratios: ${RATIOS[*]}"
echo ""

# Function to format ratio string (e.g., 0.125 -> 0p12)
format_ratio() {
    local ratio=$1
    # Use printf to format to 2 decimal places, then replace . with p
    printf "%.2f" "$ratio" | sed 's/\./p/'
}

# Function to check if job is complete
is_complete() {
    local strategy=$1
    local dataset=$2
    local model=$3
    local ratio=$4
    
    # Format ratio: 0.125 -> 0p12, 0.375 -> 0p38, etc.
    local ratio_str=$(format_ratio "$ratio")
    
    local ckpt_path="${WEIGHTS_ROOT}/${strategy}/${dataset}/${model}_ratio${ratio_str}/iter_80000.pth"
    
    [[ -f "$ckpt_path" ]]
}

# Collect missing jobs
declare -a MISSING_JOBS=()
declare -A MISSING_BY_STRATEGY

for strategy in "${STRATEGIES[@]}"; do
    if [[ -n "$FILTER_STRATEGY" && "$strategy" != "$FILTER_STRATEGY" ]]; then
        continue
    fi
    
    MISSING_BY_STRATEGY[$strategy]=0
    
    for ratio in "${RATIOS[@]}"; do
        for i in "${!DATASETS[@]}"; do
            dataset=${DATASETS[$i]}
            dataset_lower=${DATASETS_LOWER[$i]}
            
            for model in "${MODELS[@]}"; do
                if ! is_complete "$strategy" "$dataset_lower" "$model" "$ratio"; then
                    MISSING_JOBS+=("${strategy}|${ratio}|${dataset}|${dataset_lower}|${model}")
                    MISSING_BY_STRATEGY[$strategy]=$((${MISSING_BY_STRATEGY[$strategy]} + 1))
                fi
            done
        done
    done
done

# Summary
echo "Missing Jobs Summary:"
echo "---------------------"
total=0
for strategy in "${STRATEGIES[@]}"; do
    if [[ -n "$FILTER_STRATEGY" && "$strategy" != "$FILTER_STRATEGY" ]]; then
        continue
    fi
    
    count=${MISSING_BY_STRATEGY[$strategy]:-0}
    total=$((total + count))
    
    if [[ $count -gt 0 ]]; then
        echo -e "  $strategy: ${RED}$count missing${NC}"
    else
        echo -e "  $strategy: ${GREEN}✓ Complete${NC}"
    fi
done
echo ""
echo -e "Total missing: ${RED}$total${NC}"
echo ""

if [[ $total -eq 0 ]]; then
    echo "No missing jobs found!"
    exit 0
fi

# List mode
if [[ "$LIST_ONLY" == "true" ]]; then
    echo "Missing Jobs:"
    echo "-------------"
    printf "%-20s %-8s %-18s %-25s\n" "Strategy" "Ratio" "Dataset" "Model"
    printf "%-20s %-8s %-18s %-25s\n" "--------" "-----" "-------" "-----"
    
    for job in "${MISSING_JOBS[@]}"; do
        IFS='|' read -r strategy ratio dataset dataset_lower model <<< "$job"
        printf "%-20s %-8s %-18s %-25s\n" "$strategy" "$ratio" "$dataset" "$model"
    done
    
    echo ""
    echo "Total: ${#MISSING_JOBS[@]} jobs"
    exit 0
fi

# Confirmation
if [[ "$DRY_RUN" != "true" ]]; then
    echo "Ready to submit ${#MISSING_JOBS[@]} jobs."
    read -p "Continue? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Submit/dry-run
echo ""
echo "Processing jobs..."
echo "------------------"

mkdir -p logs
submitted=0

for job in "${MISSING_JOBS[@]}"; do
    IFS='|' read -r strategy ratio dataset dataset_lower model <<< "$job"
    
    # Build job name
    ratio_pct=$(echo "$ratio * 100" | bc | cut -d. -f1)
    job_name="ratio_${dataset}_${model}_${strategy}_r${ratio_pct}"
    
    # Build training command
    train_cmd="PROVE_WEIGHTS_ROOT='${WEIGHTS_ROOT}' $SCRIPT_DIR/train_unified.sh single --dataset $dataset --model $model --strategy $strategy --ratio $ratio"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] $job_name"
        echo "  Strategy: $strategy, Ratio: $ratio"
        echo "  Dataset: $dataset, Model: $model"
        echo "  Command: $train_cmd"
        echo ""
        submitted=$((submitted + 1))
    else
        echo "Submitting: $job_name"
        
        bsub -gpu "num=1:mode=${GPU_MODE}:gmem=${GPU_MEM}" \
            -q "${QUEUE}" \
            -R "span[hosts=1]" \
            -n "${NUM_CPUS}" \
            -oo "logs/${job_name}_%J.log" \
            -eo "logs/${job_name}_%J.err" \
            -L /bin/bash \
            -J "${job_name}" \
            "${train_cmd}"
        
        submitted=$((submitted + 1))
    fi
done

echo ""
echo "=============================================="
if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] Would submit: $submitted jobs"
else
    echo "Submitted: $submitted jobs"
fi
echo "=============================================="
echo ""
echo "Monitor with: bjobs -w"
echo "Logs in: logs/ratio_*.log"
