#!/bin/bash
#
# Domain Adaptation Ablation Study - LSF Job Submission Script
#
# This script submits domain adaptation evaluation jobs to the LSF cluster.
# Each job tests one model on Cityscapes (clear_day) + ACDC (foggy, night, rainy, snowy).
#
# Usage:
#   ./scripts/submit_domain_adaptation_ablation.sh --all           # Submit all 9 baseline jobs
#   ./scripts/submit_domain_adaptation_ablation.sh --strategy gen_cycleGAN --all  # All gen_cycleGAN jobs
#   ./scripts/submit_domain_adaptation_ablation.sh --source-dataset bdd10k --model pspnet_r50
#   ./scripts/submit_domain_adaptation_ablation.sh --dry-run --all  # Show what would be submitted
#
# LSF Resource Requirements:
#   - GPU: 1x RTX 3090 or A5000 (24GB)
#   - Runtime: ~30-60 minutes per job (depends on dataset size)
#   - Memory: 32GB
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_ROOT}/logs/domain_adaptation"

# Defaults
STRATEGY="baseline"
DRY_RUN=false
ALL=false
SOURCE_DATASET=""
MODEL=""

# Source datasets and models
SOURCE_DATASETS=("bdd10k" "idd-aw" "mapillaryvistas")
MODELS=("pspnet_r50" "segformer_mit-b5" "deeplabv3plus_r50")

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --source-dataset)
            SOURCE_DATASET="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --all)
            ALL=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --strategy STRATEGY      Training strategy (default: baseline)"
            echo "  --source-dataset DATASET Source dataset (bdd10k, idd-aw, mapillaryvistas)"
            echo "  --model MODEL            Model architecture (pspnet_r50, segformer_mit-b5, deeplabv3plus_r50)"
            echo "  --all                    Run all source datasets and models"
            echo "  --dry-run                Show what would be submitted without actually submitting"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create log directory
mkdir -p "$LOG_DIR"

# Function to submit a single job
submit_job() {
    local source_ds="$1"
    local model="$2"
    local strategy="$3"
    
    # Job name
    local job_name="DA_${strategy}_${source_ds}_${model}"
    job_name="${job_name//+/_}"  # Replace + with _
    
    # Check if checkpoint exists
    local checkpoint="/scratch/aaa_exchange/AWARE/WEIGHTS/${strategy}/${source_ds}/${model}/iter_80000.pth"
    if [[ ! -f "$checkpoint" ]]; then
        echo "[SKIP] Checkpoint not found: $checkpoint"
        return 0
    fi
    
    local log_file="${LOG_DIR}/${job_name}.log"
    local err_file="${LOG_DIR}/${job_name}.err"
    
    # Build command
    local cmd="python ${PROJECT_ROOT}/scripts/run_domain_adaptation_tests.py \
        --source-dataset ${source_ds} \
        --model ${model} \
        --strategy ${strategy} \
        --batch-size 4"
    
    if $DRY_RUN; then
        echo "[DRY RUN] Would submit: $job_name"
        echo "          Command: $cmd"
        echo "          Log: $log_file"
        return 0
    fi
    
    # Submit to LSF
    echo "Submitting: $job_name"
    bsub -J "$job_name" \
        -n 4 \
        -W 2:00 \
        -R "rusage[mem=8000,ngpus_excl_p=1]" \
        -R "select[gpu_model0==NVIDIAGeForceRTX3090||gpu_model0==NVIDIARTXA5000]" \
        -o "$log_file" \
        -e "$err_file" \
        "source ~/.bashrc && conda activate prove && cd ${PROJECT_ROOT} && $cmd"
}

# Main logic
echo "=============================================="
echo "Domain Adaptation Ablation Study - Job Submission"
echo "=============================================="
echo "Strategy: $STRATEGY"
echo "Dry Run: $DRY_RUN"
echo ""

if $ALL; then
    # Submit all combinations
    submitted=0
    skipped=0
    for source_ds in "${SOURCE_DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            if submit_job "$source_ds" "$model" "$STRATEGY"; then
                ((submitted++)) || true
            else
                ((skipped++)) || true
            fi
        done
    done
    echo ""
    echo "=============================================="
    if $DRY_RUN; then
        echo "Would submit: $submitted jobs"
    else
        echo "Submitted: $submitted jobs"
    fi
    echo "Skipped: $skipped (missing checkpoints)"
elif [[ -n "$SOURCE_DATASET" && -n "$MODEL" ]]; then
    # Submit single job
    submit_job "$SOURCE_DATASET" "$MODEL" "$STRATEGY"
else
    echo "Error: Either --all or both --source-dataset and --model are required"
    echo "Run with --help for usage information"
    exit 1
fi

echo ""
echo "To monitor jobs: bjobs -w"
echo "To view logs: ls -la $LOG_DIR/"
