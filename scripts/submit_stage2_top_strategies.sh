#!/bin/bash
# Submit Stage 2 training jobs for top 10 strategies
# These are the top performing strategies from Stage 1 testing

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WEIGHTS_ROOT="/scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2"

# Default is dry run
DRY_RUN=true
LIMIT=0
SUBMITTED=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --execute) DRY_RUN=false; shift ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--execute] [--limit N]"
            echo ""
            echo "Options:"
            echo "  --execute   Actually submit jobs (default: dry run)"
            echo "  --limit N   Limit number of jobs to submit"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "Stage 2 Training - Top 10 Strategies"
echo "============================================================"
echo "Dry run: $DRY_RUN"
[ "$LIMIT" -gt 0 ] && echo "Limit: $LIMIT jobs"
echo ""

# Top 10 strategies (ordered by Stage 1 performance)
declare -A STRATEGIES
STRATEGIES=(
    ["gen_step1x_new"]="gen"
    ["std_autoaugment"]="std"
    ["gen_step1x_v1p2"]="gen"
    ["std_randaugment"]="std"
    ["photometric_distort"]="std"
    ["gen_stargan_v2"]="gen"
    ["gen_Qwen_Image_Edit"]="gen"
    ["gen_LANIT"]="gen"
    ["gen_Attribute_Hallucination"]="gen"
    ["gen_albumentations_weather"]="gen"
)

# Datasets and models
DATASETS=("BDD10k" "IDD-AW" "MapillaryVistas" "OUTSIDE15k")
MODELS=("deeplabv3plus_r50" "pspnet_r50" "segformer_mit-b5")

# Dataset directory names
declare -A DS_DIRS
DS_DIRS=(
    ["BDD10k"]="bdd10k"
    ["IDD-AW"]="idd-aw"
    ["MapillaryVistas"]="mapillaryvistas"
    ["OUTSIDE15k"]="outside15k"
)

submit_job() {
    local strategy=$1
    local dataset=$2
    local model=$3
    local type=$4
    
    local ds_dir="${DS_DIRS[$dataset]}"
    local model_dir
    
    if [ "$type" = "gen" ]; then
        model_dir="${model}_ratio0p50"
    else
        model_dir="$model"
    fi
    
    local ckpt_path="${WEIGHTS_ROOT}/${strategy}/${ds_dir}/${model_dir}/iter_80000.pth"
    
    # Check if already trained
    if [ -f "$ckpt_path" ]; then
        return 1  # Already exists
    fi
    
    # Check limit
    if [ "$LIMIT" -gt 0 ] && [ "$SUBMITTED" -ge "$LIMIT" ]; then
        return 2  # Limit reached
    fi
    
    echo "  $strategy / $dataset / $model"
    
    if [ "$DRY_RUN" = false ]; then
        if [ "$type" = "gen" ]; then
            "$SCRIPT_DIR/submit_training.sh" \
                --dataset "$dataset" \
                --model "$model" \
                --strategy "$strategy" \
                --ratio 0.5
        else
            "$SCRIPT_DIR/submit_training.sh" \
                --dataset "$dataset" \
                --model "$model" \
                --strategy "$strategy"
        fi
    fi
    
    SUBMITTED=$((SUBMITTED + 1))
    return 0
}

echo "Scanning for missing training configurations..."
echo ""

for strategy in "${!STRATEGIES[@]}"; do
    type="${STRATEGIES[$strategy]}"
    
    for dataset in "${DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            submit_job "$strategy" "$dataset" "$model" "$type"
            result=$?
            if [ $result -eq 2 ]; then
                echo ""
                echo "Limit reached. Submitted: $SUBMITTED"
                exit 0
            fi
        done
    done
done

echo ""
echo "============================================================"
echo "Summary"
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Would submit: $SUBMITTED jobs"
    echo ""
    echo "Run with --execute to actually submit"
else
    echo "Submitted: $SUBMITTED jobs"
fi
