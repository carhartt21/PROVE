#!/bin/bash
# Retrain Stage 2 (all domains) for MapillaryVistas and OUTSIDE15k using native classes.
# Uses WEIGHTS_STAGE_2 strategy list and submits all 3 models per strategy.
# Skips gen_EDICT (no generated images for MapillaryVistas/OUTSIDE15k).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROVE_DIR="$(dirname "$SCRIPT_DIR")"
WEIGHTS_STAGE_2="/scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2"

MODELS=(deeplabv3plus_r50 pspnet_r50 segformer_mit-b5)
DATASETS=(MapillaryVistas OUTSIDE15k)
SKIP_STRATEGIES=(gen_EDICT)
DRY_RUN="${1:-false}"

submit_job() {
    local dataset=$1
    local strategy=$2
    local model=$3
    local ratio=$4

    echo "========================================"
    echo "Dataset: $dataset | Strategy: $strategy | Model: $model"

    if [ "$DRY_RUN" = "--dry-run" ]; then
        if [ -n "$ratio" ]; then
            echo "[DRY-RUN] $SCRIPT_DIR/submit_training.sh --dataset $dataset --model $model --strategy $strategy --ratio $ratio --dry-run"
        else
            echo "[DRY-RUN] $SCRIPT_DIR/submit_training.sh --dataset $dataset --model $model --strategy $strategy --dry-run"
        fi
    else
        if [ -n "$ratio" ]; then
            "$SCRIPT_DIR/submit_training.sh" --dataset $dataset --model $model --strategy $strategy --ratio $ratio
        else
            "$SCRIPT_DIR/submit_training.sh" --dataset $dataset --model $model --strategy $strategy
        fi
    fi
}

if [ ! -d "$WEIGHTS_STAGE_2" ]; then
    echo "ERROR: WEIGHTS_STAGE_2 not found at $WEIGHTS_STAGE_2"
    exit 1
fi

# Build strategy list from Stage 2 weights root
STRATEGIES=()
while IFS= read -r name; do
    STRATEGIES+=("$name")
done < <(ls -1 "$WEIGHTS_STAGE_2" | sort)

# Filter out skip strategies
FILTERED_STRATEGIES=()
for s in "${STRATEGIES[@]}"; do
    skip=false
    for k in "${SKIP_STRATEGIES[@]}"; do
        if [ "$s" = "$k" ]; then
            skip=true
            break
        fi
    done
    if [ "$skip" = false ]; then
        FILTERED_STRATEGIES+=("$s")
    fi
done

TOTAL_JOBS=0

echo "========================================"
echo "Stage 2 Retraining (Native Classes)"
echo "Strategies: ${#FILTERED_STRATEGIES[@]} (skipping gen_EDICT)"
echo "Datasets: ${DATASETS[*]}"
echo "Models: ${MODELS[*]}"
echo "========================================"
echo ""

for dataset in "${DATASETS[@]}"; do
    echo "### ${dataset} ###"
    echo ""
    for strategy in "${FILTERED_STRATEGIES[@]}"; do
        # Use ratio for gen_* strategies
        if [[ "$strategy" == gen_* ]]; then
            ratio="0.5"
        else
            ratio=""
        fi

        for model in "${MODELS[@]}"; do
            submit_job "$dataset" "$strategy" "$model" "$ratio"
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
        done
    done
    echo ""
done

echo "========================================"
echo "Total jobs submitted: ${TOTAL_JOBS}"
echo "========================================"
