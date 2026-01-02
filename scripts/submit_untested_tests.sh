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
#   --dry-run             Show what would be submitted without actually submitting
#   --include-clear-day   Include models trained with --domain-filter clear_day
#   --include-multi       Include multi-dataset configurations
#   --strategy <name>     Only submit tests for specific strategy
#   --dataset <name>      Only submit tests for specific dataset
#   --model <name>        Only submit tests for specific model
#   --queue <name>        LSF queue name (default: BatchGPU)
#   --gpu-mem <size>      GPU memory requirement (default: 24G)
#   --limit <n>           Maximum number of jobs to submit
#   --batch-size <n>      Number of jobs per batch before pause (default: 10)
#   --batch-delay <s>     Seconds to pause between batches (default: 60)
#   --detailed            Submit detailed (fine-grained) tests instead of basic tests
#   --missing-detailed    Filter for configs that have basic tests but missing detailed tests
#   --help                Show this help message

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

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
DETAILED_MODE=false
MISSING_DETAILED=false

print_usage() {
    echo "PROVE - Submit Untested Configuration Tests"
    echo "==========================================="
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --dry-run             Show what would be submitted without actually submitting"
    echo "  --include-clear-day   Include models trained with --domain-filter clear_day"
    echo "  --include-multi       Include multi-dataset configurations"
    echo "  --strategy <name>     Only submit tests for specific strategy"
    echo "  --dataset <name>      Only submit tests for specific dataset"
    echo "  --model <name>        Only submit tests for specific model"
    echo "  --queue <name>        LSF queue name (default: BatchGPU)"
    echo "  --gpu-mem <size>      GPU memory requirement (default: 24G)"
    echo "  --limit <n>           Maximum number of jobs to submit (default: no limit)"
    echo "  --batch-size <n>      Jobs per batch before pause (default: 10)"
    echo "  --batch-delay <s>     Seconds between batches (default: 60)"
    echo "  --detailed            Submit detailed (fine-grained) tests instead of basic tests"
    echo "  --missing-detailed    Filter for configs missing detailed tests (has basic, no detailed)"
    echo "  --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --dry-run                                    # Preview all untested jobs"
    echo "  $0 --strategy baseline                          # Only baseline tests"
    echo "  $0 --include-clear-day --limit 10               # Include clear_day, max 10 jobs"
    echo "  $0 --dataset ACDC --model deeplabv3plus_r50     # Specific config"
    echo "  $0 --batch-size 5 --batch-delay 120             # 5 jobs, 2min pause"
    echo "  $0 --missing-detailed --detailed                # Submit detailed tests for configs missing them"
    echo "  $0 --missing-detailed --detailed --dry-run      # Preview missing detailed tests"
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
        --detailed)
            DETAILED_MODE=true
            shift
            ;;
        --missing-detailed)
            MISSING_DETAILED=true
            shift
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

# Determine mode
if [ "$MISSING_DETAILED" = true ]; then
    # When looking for missing detailed tests, always submit detailed tests
    MODE_DESC="missing detailed tests"
    TEST_CMD="submit-detailed"
elif [ "$DETAILED_MODE" = true ]; then
    MODE_DESC="untested (detailed mode)"
    TEST_CMD="submit-detailed"
else
    MODE_DESC="untested"
    TEST_CMD="submit"
fi

echo "PROVE - Submit Tests"
echo "===================="
echo ""
echo "Mode: $MODE_DESC"
echo "Test command: $TEST_CMD"
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
echo "Identifying configurations to test..."
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
missing_detailed = "${MISSING_DETAILED}" == "true"
detailed_mode = "${DETAILED_MODE}" == "true"
test_cmd = "${TEST_CMD}"

# Find configurations to test
to_test = []
for config in data:
    model = config['model']
    dataset = config['dataset']
    strategy = config['strategy']
    
    # Determine if this config needs testing based on mode
    if missing_detailed:
        # Look for configs that have basic test results but no detailed results
        has_basic = config.get('has_test_results', False)
        has_detailed = config.get('has_detailed_test_results', False)
        
        if not (has_basic and not has_detailed):
            continue
    else:
        # Standard mode: find untested configs
        if config.get('has_test_results', False):
            continue
    
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
    
    to_test.append(config)

# Sort by strategy, dataset, model
to_test.sort(key=lambda x: (x['strategy'], x['dataset'], x['model']))

# Apply limit
if limit > 0:
    to_test = to_test[:limit]

mode_desc = "missing detailed tests" if missing_detailed else "untested configurations"
print(f"Found {len(to_test)} {mode_desc} to submit")
print()

if len(to_test) == 0:
    print(f"No {mode_desc} found matching criteria.")
    sys.exit(0)

# Generate submission commands
with open('${SUBMISSION_SCRIPT}', 'w') as f:
    for i, config in enumerate(to_test):
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
            
            cmd = f'${SCRIPT_DIR}/test_unified.sh {test_cmd} --datasets {datasets_proper} --model {model} --strategy {strategy} --queue {queue} --gpu-mem {gpu_mem}'
        else:
            cmd = f'${SCRIPT_DIR}/test_unified.sh {test_cmd} --dataset {dataset_proper} --model {model} --strategy {strategy} --queue {queue} --gpu-mem {gpu_mem}'
        
        f.write(f"echo '[{i+1}/{len(to_test)}] {strategy}/{dataset}/{model}'\n")
        f.write(f"{cmd}\n")
        f.write("sleep 0.5\n")
        
        # Add batch delay after every batch_size jobs (except at the end)
        if (i + 1) % batch_size == 0 and i < len(to_test) - 1:
            f.write(f"echo ''\n")
            f.write(f"echo '=== Batch complete ({i+1}/{len(to_test)} jobs). Pausing {batch_delay} seconds... ==='\n")
            f.write(f"echo ''\n")
            f.write(f"sleep {batch_delay}\n")

# Print summary by strategy
print("Summary by strategy:")
from collections import Counter
strategy_counts = Counter(c['strategy'] for c in to_test)
for strategy, count in sorted(strategy_counts.items()):
    print(f"  {strategy}: {count}")
print()

# Print list
print("Configurations to test:")
for i, config in enumerate(to_test):
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
