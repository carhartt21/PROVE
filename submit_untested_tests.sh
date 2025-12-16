#!/bin/bash
# PROVE - Submit Untested Configuration Tests
#
# This script analyzes trained model weights and submits test jobs
# for configurations that haven't been tested yet.
#
# Usage:
#   ./submit_untested_tests.sh [options]
#
# Options:
#   --dry-run           Show what would be submitted without actually submitting
#   --include-clear-day Include models trained with --domain-filter clear_day
#   --include-multi     Include multi-dataset configurations
#   --strategy <name>   Only submit tests for specific strategy
#   --dataset <name>    Only submit tests for specific dataset
#   --model <name>      Only submit tests for specific model
#   --queue <name>      LSF queue name (default: BatchGPU)
#   --gpu-mem <size>    GPU memory requirement (default: 24G)
#   --limit <n>         Maximum number of jobs to submit
#   --batch-size <n>    Number of jobs per batch before pause (default: 10)
#   --batch-delay <s>   Seconds to pause between batches (default: 60)
#   --help              Show this help message

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Default options
DRY_RUN=false
INCLUDE_CLEAR_DAY=true
INCLUDE_MULTI=false
FILTER_STRATEGY=""
FILTER_DATASET=""
FILTER_MODEL=""
QUEUE="BatchGPU"
GPU_MEM="24G"
LIMIT=0  # 0 = no limit
BATCH_SIZE=10
BATCH_DELAY=60

print_usage() {
    echo "PROVE - Submit Untested Configuration Tests"
    echo "==========================================="
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --dry-run           Show what would be submitted without actually submitting"
    echo "  --include-clear-day Include models trained with --domain-filter clear_day"
    echo "  --include-multi     Include multi-dataset configurations"
    echo "  --strategy <name>   Only submit tests for specific strategy"
    echo "  --dataset <name>    Only submit tests for specific dataset"
    echo "  --model <name>      Only submit tests for specific model"
    echo "  --queue <name>      LSF queue name (default: BatchGPU)"
    echo "  --gpu-mem <size>    GPU memory requirement (default: 24G)"
    echo "  --limit <n>         Maximum number of jobs to submit (default: no limit)"
    echo "  --batch-size <n>    Jobs per batch before pause (default: 10)"
    echo "  --batch-delay <s>   Seconds between batches (default: 60)"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --dry-run                                    # Preview all jobs"
    echo "  $0 --strategy baseline                          # Only baseline tests"
    echo "  $0 --include-clear-day --limit 10               # Include clear_day, max 10 jobs"
    echo "  $0 --dataset ACDC --model deeplabv3plus_r50     # Specific config"
    echo "  $0 --batch-size 5 --batch-delay 120             # 5 jobs, 2min pause"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --include-clear-day)
            INCLUDE_CLEAR_DAY=true
            shift
            ;;
        --include-multi)
            INCLUDE_MULTI=true
            shift
            ;;
        --strategy)
            FILTER_STRATEGY="$2"
            shift 2
            ;;
        --dataset)
            FILTER_DATASET="$2"
            shift 2
            ;;
        --model)
            FILTER_MODEL="$2"
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
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --batch-delay)
            BATCH_DELAY="$2"
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

echo "PROVE - Submit Untested Tests"
echo "============================="
echo ""
echo "Batch size: $BATCH_SIZE jobs"
echo "Batch delay: $BATCH_DELAY seconds"
echo ""

# Run weights analyzer to get JSON
echo "Analyzing trained configurations..."
mamba run -n PROVE python weights_analyzer.py --format json 2>/dev/null

# Check if JSON was created
if [ ! -f "weights_summary.json" ]; then
    echo "Error: weights_summary.json not found"
    exit 1
fi

# Use Python to parse JSON and generate submission commands
echo ""
echo "Identifying untested configurations..."
echo ""

SUBMISSION_SCRIPT=$(mktemp)

mamba run -n PROVE python3 << PYTHON_SCRIPT
import json
import sys

# Read weights summary
with open('weights_summary.json', 'r') as f:
    data = json.load(f)

# Filter options
include_clear_day = "${INCLUDE_CLEAR_DAY}" == "true"
include_multi = "${INCLUDE_MULTI}" == "true"
filter_strategy = "${FILTER_STRATEGY}"
filter_dataset = "${FILTER_DATASET}"
filter_model = "${FILTER_MODEL}"
limit = ${LIMIT}
queue = "${QUEUE}"
gpu_mem = "${GPU_MEM}"
batch_size = ${BATCH_SIZE}
batch_delay = ${BATCH_DELAY}

# Find untested configurations
untested = []
for config in data:
    if config['has_test_results']:
        continue
    
    model = config['model']
    dataset = config['dataset']
    strategy = config['strategy']
    
    # Skip clear_day variants unless requested
    if '_clear_day' in model and not include_clear_day:
        continue
    
    # Skip multi-dataset unless requested
    if 'multi_' in dataset and not include_multi:
        continue
    
    # Apply filters
    if filter_strategy and strategy != filter_strategy:
        continue
    if filter_dataset and dataset.lower() != filter_dataset.lower():
        continue
    if filter_model and model != filter_model:
        continue
    
    untested.append(config)

# Sort by strategy, dataset, model
untested.sort(key=lambda x: (x['strategy'], x['dataset'], x['model']))

# Apply limit
if limit > 0:
    untested = untested[:limit]

print(f"Found {len(untested)} untested configurations to submit")
print()

if len(untested) == 0:
    print("No untested configurations found matching criteria.")
    sys.exit(0)

# Generate submission commands
with open('${SUBMISSION_SCRIPT}', 'w') as f:
    for i, config in enumerate(untested):
        strategy = config['strategy']
        dataset = config['dataset']
        model = config['model']
        
        # Handle dataset naming for test_unified.sh
        # Dataset names need proper casing
        dataset_map = {
            'acdc': 'ACDC',
            'bdd10k': 'BDD10k',
            'bdd100k': 'BDD100k',
            'idd-aw': 'IDD-AW',
            'mapillaryvistas': 'MapillaryVistas',
            'outside15k': 'OUTSIDE15k'
        }
        dataset_proper = dataset_map.get(dataset.lower(), dataset)
        
        # Handle multi-dataset
        if 'multi_' in dataset:
            # Extract datasets from multi_ds1+ds2+ds3 format
            ds_part = dataset.replace('multi_', '')
            datasets_list = ds_part.split('+')
            datasets_proper = ' '.join(dataset_map.get(d.lower(), d) for d in datasets_list)
            
            cmd = f'./test_unified.sh submit --datasets {datasets_proper} --model {model} --strategy {strategy} --queue {queue} --gpu-mem {gpu_mem}'
        else:
            cmd = f'./test_unified.sh submit --dataset {dataset_proper} --model {model} --strategy {strategy} --queue {queue} --gpu-mem {gpu_mem}'
        
        f.write(f"echo '[{i+1}/{len(untested)}] {strategy}/{dataset}/{model}'\n")
        f.write(f"{cmd}\n")
        f.write("sleep 0.5\n")
        
        # Add batch delay after every batch_size jobs (except at the end)
        if (i + 1) % batch_size == 0 and i < len(untested) - 1:
            f.write(f"echo ''\n")
            f.write(f"echo '=== Batch complete ({i+1}/{len(untested)} jobs). Pausing {batch_delay} seconds... ==='\n")
            f.write(f"echo ''\n")
            f.write(f"sleep {batch_delay}\n")

# Print summary by strategy
print("Summary by strategy:")
from collections import Counter
strategy_counts = Counter(c['strategy'] for c in untested)
for strategy, count in sorted(strategy_counts.items()):
    print(f"  {strategy}: {count}")
print()

# Print list
print("Configurations to test:")
for i, config in enumerate(untested):
    print(f"  [{i+1:3d}] {config['strategy']:20s} | {config['dataset']:35s} | {config['model']}")

PYTHON_SCRIPT

echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Commands that would be executed:"
    echo "==========================================="
    cat "$SUBMISSION_SCRIPT" | grep -v "^sleep" | grep -v "^echo"
    echo ""
    echo "[DRY RUN] No jobs submitted."
else
    echo "Submitting jobs..."
    echo ""
    source "$SUBMISSION_SCRIPT"
    echo ""
    echo "Job submission complete."
fi

rm -f "$SUBMISSION_SCRIPT"
