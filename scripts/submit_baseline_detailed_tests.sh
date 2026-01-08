#!/bin/bash
# Submit detailed tests for baseline strategy
# This script submits per-domain/per-class tests for all baseline configurations

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# Datasets and models to test
DATASETS="bdd10k idd-aw mapillaryvistas outside15k"
MODELS="deeplabv3plus_r50 pspnet_r50 segformer_mit-b5"
STRATEGY="baseline"

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
fi

echo "==================================="
echo "Submit Baseline Detailed Tests"
echo "==================================="
echo ""

COUNT=0
for dataset in $DATASETS; do
    for model in $MODELS; do
        COUNT=$((COUNT + 1))
        echo "[$COUNT] ${STRATEGY} / ${dataset} / ${model}"
        
        if [ "$DRY_RUN" = true ]; then
            $SCRIPT_DIR/test_unified.sh submit-detailed --dataset "$dataset" --model "$model" --strategy "$STRATEGY" --dry-run
        else
            $SCRIPT_DIR/test_unified.sh submit-detailed --dataset "$dataset" --model "$model" --strategy "$STRATEGY"
        fi
        
        sleep 1  # Small delay between submissions
        echo ""
    done
done

echo "==================================="
echo "Total jobs submitted: $COUNT"
echo "==================================="
