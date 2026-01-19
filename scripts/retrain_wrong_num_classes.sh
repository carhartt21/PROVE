#!/bin/bash
# Retraining script for models with wrong num_classes
# MapillaryVistas: Need 66 classes (was 19)
# OUTSIDE15k: Need 24 classes (was 19)
# 
# Native classes is now the default (66 for MapillaryVistas, 24 for OUTSIDE15k)
# All are Stage 1 (clear_day domain filter)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROVE_DIR="$(dirname "$SCRIPT_DIR")"

MODELS="deeplabv3plus_r50 pspnet_r50 segformer_mit-b5"
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
            echo "[DRY-RUN] $SCRIPT_DIR/submit_training.sh --dataset $dataset --model $model --strategy $strategy --ratio $ratio --domain-filter clear_day --dry-run"
        else
            echo "[DRY-RUN] $SCRIPT_DIR/submit_training.sh --dataset $dataset --model $model --strategy $strategy --domain-filter clear_day --dry-run"
        fi
    else
        if [ -n "$ratio" ]; then
            "$SCRIPT_DIR/submit_training.sh" --dataset $dataset --model $model --strategy $strategy --ratio $ratio --domain-filter clear_day
        else
            "$SCRIPT_DIR/submit_training.sh" --dataset $dataset --model $model --strategy $strategy --domain-filter clear_day
        fi
    fi
}

echo "========================================"
echo "Retraining Jobs for Wrong num_classes Models"
echo "========================================"
echo ""

# MapillaryVistas strategies (66 classes)
echo "### MapillaryVistas (66 classes) ###"
echo ""

for model in $MODELS; do
    submit_job MapillaryVistas gen_cyclediffusion $model 0.5
done

for model in $MODELS; do
    submit_job MapillaryVistas gen_TSIT $model 0.5
done

# OUTSIDE15k strategies (24 classes)
echo ""
echo "### OUTSIDE15k (24 classes) ###"
echo ""

for model in $MODELS; do
    submit_job OUTSIDE15k std_cutmix $model
done

for model in $MODELS; do
    submit_job OUTSIDE15k std_mixup $model
done

for model in $MODELS; do
    submit_job OUTSIDE15k gen_cyclediffusion $model 0.5
done

for model in $MODELS; do
    submit_job OUTSIDE15k gen_flux_kontext $model 0.5
done

for model in $MODELS; do
    submit_job OUTSIDE15k gen_TSIT $model 0.5
done

echo ""
echo "========================================"
echo "Total jobs to submit: 21"
echo " - MapillaryVistas: 6 jobs (2 strategies × 3 models)"
echo " - OUTSIDE15k: 15 jobs (5 strategies × 3 models)"
echo "========================================"
