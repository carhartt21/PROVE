#!/bin/bash
# PROVE - Run All Missing Detailed Tests Sequentially (Single Job)
#
# This script runs fine-grained (detailed) tests sequentially within a single
# batch job. It processes all configurations that have basic test results
# but are missing detailed per-domain metrics.
#
# Usage:
#   As a batch job: bsub -q BatchGPU ... ./run_detailed_tests_sequential.sh
#   Or directly: ./run_detailed_tests_sequential.sh [--dataset NAME]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# Options
FILTER_DATASET=""  # empty = all datasets
DATA_ROOT="${PROVE_DATA_ROOT:-/scratch/aaa_exchange/AWARE/FINAL_SPLITS}"
WEIGHTS_ROOT="${PROVE_WEIGHTS_ROOT:-/scratch/aaa_exchange/AWARE/WEIGHTS}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            FILTER_DATASET="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --weights-root)
            WEIGHTS_ROOT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dataset NAME] [--data-root PATH] [--weights-root PATH]"
            exit 1
            ;;
    esac
done

echo "PROVE - Run Detailed Tests Sequentially"
echo "========================================"
echo ""
echo "Start time: $(date)"
echo "Data root: $DATA_ROOT"
echo "Weights root: $WEIGHTS_ROOT"
if [ -n "$FILTER_DATASET" ]; then
    echo "Dataset filter: $FILTER_DATASET"
fi
echo ""

# Activate conda environment
echo "Activating prove conda environment..."
source ~/.bashrc
conda activate prove

# Update weights summary
echo "Analyzing configurations..."
python weights_analyzer.py --format json >/dev/null 2>&1

# Get list of configurations to test
CONFIGS=$(python3 << PYTHON_SCRIPT
import json

with open('weights_summary.json') as f:
    data = json.load(f)

# Filter for configs that have basic tests but no detailed tests
missing = [c for c in data if c.get('has_test_results') and not c.get('has_detailed_test_results')]

# Apply dataset filter if specified
filter_dataset = "${FILTER_DATASET}".lower()
if filter_dataset:
    missing = [c for c in missing if c['dataset'].lower() == filter_dataset]

# Sort by strategy, dataset, model
missing.sort(key=lambda x: (x['strategy'], x['dataset'], x['model']))

# Dataset name mapping
dataset_map = {
    'acdc': 'ACDC',
    'bdd10k': 'BDD10k', 
    'bdd100k': 'BDD100k',
    'idd-aw': 'IDD-AW',
    'mapillaryvistas': 'MapillaryVistas',
    'outside15k': 'OUTSIDE15k'
}

# Output configs as CSV
for c in missing:
    dataset_proper = dataset_map.get(c['dataset'].lower(), c['dataset'])
    print(f"{c['strategy']},{dataset_proper},{c['model']}")
PYTHON_SCRIPT
)

TOTAL=$(echo "$CONFIGS" | wc -l)
echo "Found $TOTAL configurations to test"
echo ""

if [ "$TOTAL" -eq 0 ]; then
    echo "No configurations need detailed testing."
    exit 0
fi

# Track progress
COMPLETED=0
FAILED=0

# Process each configuration
echo "$CONFIGS" | while IFS=',' read -r STRATEGY DATASET MODEL; do
    COMPLETED=$((COMPLETED + 1))
    
    echo ""
    echo "=============================================="
    echo "[$COMPLETED/$TOTAL] $STRATEGY / $DATASET / $MODEL"
    echo "=============================================="
    echo "Time: $(date)"
    
    # Find config and checkpoint paths
    MODEL_DIR="$WEIGHTS_ROOT/$STRATEGY/${DATASET,,}/$MODEL"
    
    if [ ! -d "$MODEL_DIR" ]; then
        echo "ERROR: Model directory not found: $MODEL_DIR"
        FAILED=$((FAILED + 1))
        continue
    fi
    
    CONFIG_FILE="$MODEL_DIR/configs/training_config.py"
    if [ ! -f "$CONFIG_FILE" ]; then
        CONFIG_FILE="$MODEL_DIR/training_config.py"
    fi
    
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "ERROR: Config file not found in $MODEL_DIR"
        FAILED=$((FAILED + 1))
        continue
    fi
    
    # Find best checkpoint (highest iteration)
    CHECKPOINT=$(ls -1 "$MODEL_DIR"/iter_*.pth 2>/dev/null | sort -V | tail -1)
    if [ -z "$CHECKPOINT" ]; then
        echo "ERROR: No checkpoint found in $MODEL_DIR"
        FAILED=$((FAILED + 1))
        continue
    fi
    
    OUTPUT_DIR="$MODEL_DIR/test_results_detailed"
    
    echo "Config: $CONFIG_FILE"
    echo "Checkpoint: $CHECKPOINT"
    echo "Output: $OUTPUT_DIR"
    
    # Run fine-grained test
    python fine_grained_test.py \
        --config "$CONFIG_FILE" \
        --checkpoint "$CHECKPOINT" \
        --output-dir "$OUTPUT_DIR" \
        --dataset "$DATASET" \
        --data-root "$DATA_ROOT" \
        --test-split test
    
    if [ $? -eq 0 ]; then
        echo "SUCCESS: Test completed for $STRATEGY/$DATASET/$MODEL"
    else
        echo "FAILED: Test failed for $STRATEGY/$DATASET/$MODEL"
        FAILED=$((FAILED + 1))
    fi
    
done

echo ""
echo "=============================================="
echo "ALL TESTS COMPLETED"
echo "=============================================="
echo "End time: $(date)"
echo "Total: $TOTAL"
echo "Completed: $((TOTAL - FAILED))"
echo "Failed: $FAILED"
