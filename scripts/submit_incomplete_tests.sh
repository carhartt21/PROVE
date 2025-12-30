#!/bin/bash
# PROVE - Resubmit Tests with Incomplete/Intermediate Weights
#
# This script analyzes test results to find tests that were performed on 
# intermediate weights and resubmits them if newer weights are available.
#
# The script checks the log files in test_results/test folders to identify
# which tests used older weights, and checks if newer checkpoints exist.
#
# Usage:
#   ./submit_incomplete_tests.sh [options]
#
# Options:
#   --dry-run             Show what would be submitted without actually submitting
#   --strategy <name>     Only check specific strategy
#   --dataset <name>      Only check specific dataset
#   --model <name>        Only check specific model
#   --queue <name>        LSF queue name (default: BatchGPU)
#   --gpu-mem <size>      GPU memory requirement (default: 24G)
#   --limit <n>           Maximum number of jobs to submit
#   --detailed            Resubmit detailed tests instead of basic tests
#   --list                Only list incomplete tests, don't submit
#   --help                Show this help message

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Default options
DRY_RUN=false
FILTER_STRATEGY=""
FILTER_DATASET=""
FILTER_MODEL=""
QUEUE="BatchGPU"
GPU_MEM="24G"
GPU_MODE="shared"
NUM_CPUS=4
LIMIT=0  # 0 = no limit
DETAILED_MODE=false
LIST_ONLY=false
WEIGHTS_ROOT="${PROVE_WEIGHTS_ROOT:-/scratch/aaa_exchange/AWARE/WEIGHTS}"

print_usage() {
    echo "PROVE - Resubmit Tests with Incomplete/Intermediate Weights"
    echo "============================================================"
    echo ""
    echo "This script finds tests that were performed on intermediate weights"
    echo "and resubmits them if newer weights are available."
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --dry-run             Show what would be submitted without actually submitting"
    echo "  --strategy <name>     Only check specific strategy"
    echo "  --dataset <name>      Only check specific dataset"
    echo "  --model <name>        Only check specific model"
    echo "  --queue <name>        LSF queue name (default: BatchGPU)"
    echo "  --gpu-mem <size>      GPU memory requirement (default: 24G)"
    echo "  --limit <n>           Maximum number of jobs to submit (default: no limit)"
    echo "  --detailed            Resubmit detailed tests instead of basic tests"
    echo "  --list                Only list incomplete tests, don't submit"
    echo "  --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --list                              # List all incomplete tests"
    echo "  $0 --dry-run                           # Preview what would be submitted"
    echo "  $0                                     # Submit all incomplete tests"
    echo "  $0 --strategy gen_cycleGAN             # Only for gen_cycleGAN strategy"
    echo "  $0 --dataset ACDC --limit 10           # Only ACDC, max 10 jobs"
    echo "  $0 --detailed                          # Resubmit detailed tests"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
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
        --detailed)
            DETAILED_MODE=true
            shift
            ;;
        --list)
            LIST_ONLY=true
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

echo "PROVE - Resubmit Incomplete Tests"
echo "=================================="
echo ""
echo "Weights root: $WEIGHTS_ROOT"
echo ""

# Determine test directory name based on mode
if [ "$DETAILED_MODE" = true ]; then
    TEST_DIR="test_results_detailed"
    TEST_CMD="submit-detailed"
else
    TEST_DIR="test_results"
    TEST_CMD="submit"
fi

# Export variables for Python script
export WEIGHTS_ROOT
export FILTER_STRATEGY
export FILTER_DATASET
export FILTER_MODEL
export TEST_DIR
export LIST_ONLY
export DRY_RUN
export LIMIT
export QUEUE
export GPU_MEM
export GPU_MODE
export NUM_CPUS
export TEST_CMD
export SCRIPT_DIR

# Find all incomplete tests using Python for reliable parsing
python3 << 'PYTHON_SCRIPT'
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

# Get environment variables
weights_root = os.environ.get('WEIGHTS_ROOT', '/scratch/aaa_exchange/AWARE/WEIGHTS')
filter_strategy = os.environ.get('FILTER_STRATEGY', '')
filter_dataset = os.environ.get('FILTER_DATASET', '')
filter_model = os.environ.get('FILTER_MODEL', '')
test_dir_name = os.environ.get('TEST_DIR', 'test_results')
list_only = os.environ.get('LIST_ONLY', 'false') == 'true'
dry_run = os.environ.get('DRY_RUN', 'false') == 'true'
limit = int(os.environ.get('LIMIT', '0'))
queue = os.environ.get('QUEUE', 'BatchGPU')
gpu_mem = os.environ.get('GPU_MEM', '24G')
gpu_mode = os.environ.get('GPU_MODE', 'shared')
num_cpus = int(os.environ.get('NUM_CPUS', '4'))
test_cmd = os.environ.get('TEST_CMD', 'submit')
script_dir = os.environ.get('SCRIPT_DIR', '.')

root = Path(weights_root)
incomplete_tests = []

# Pattern to extract iteration from load_from line
load_from_pattern = re.compile(r"load_from\s*=\s*['\"](.+?)['\"]")
iter_pattern = re.compile(r"iter_(\d+)\.pth")

def get_latest_test_run(test_dir):
    """Get the latest test run directory based on timestamp in name."""
    test_path = Path(test_dir) / "test"
    if not test_path.exists():
        return None
    
    # Find all timestamped directories (format: YYYYMMDD_HHMMSS)
    timestamp_dirs = []
    for item in test_path.iterdir():
        if item.is_dir() and re.match(r'\d{8}_\d{6}', item.name):
            timestamp_dirs.append(item)
    
    if not timestamp_dirs:
        return None
    
    # Sort by name (which is timestamp) and return latest
    timestamp_dirs.sort(key=lambda x: x.name, reverse=True)
    return timestamp_dirs[0]

def get_iteration_from_log(log_dir):
    """Extract iteration number from the log file in a test run directory."""
    if not log_dir or not log_dir.exists():
        return None
    
    # Find log file
    log_files = list(log_dir.glob("*.log"))
    if not log_files:
        return None
    
    log_file = log_files[0]
    
    try:
        with open(log_file, 'r', errors='ignore') as f:
            content = f.read()
            
            # Find load_from line
            match = load_from_pattern.search(content)
            if match:
                checkpoint_path = match.group(1)
                iter_match = iter_pattern.search(checkpoint_path)
                if iter_match:
                    return int(iter_match.group(1))
    except Exception as e:
        print(f"Warning: Could not read {log_file}: {e}", file=sys.stderr)
    
    return None

def get_max_available_iter(model_dir):
    """Find the maximum iteration checkpoint available in the model directory."""
    iter_files = list(model_dir.glob("iter_*.pth"))
    if not iter_files:
        return None
    
    iterations = []
    for f in iter_files:
        match = iter_pattern.search(f.name)
        if match:
            iterations.append(int(match.group(1)))
    
    return max(iterations) if iterations else None

print(f"Scanning {weights_root} for incomplete tests...")
print()

# Walk through directory structure: strategy/dataset/model
for strategy_dir in sorted(root.iterdir()):
    if not strategy_dir.is_dir():
        continue
    
    strategy = strategy_dir.name
    
    # Apply strategy filter
    if filter_strategy and strategy != filter_strategy:
        continue
    
    for dataset_dir in sorted(strategy_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        
        dataset = dataset_dir.name
        
        # Apply dataset filter (case-insensitive)
        if filter_dataset and dataset.lower() != filter_dataset.lower():
            continue
        
        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            
            model = model_dir.name
            
            # Apply model filter
            if filter_model and model != filter_model:
                continue
            
            # Check for test results directory
            test_results_dir = model_dir / test_dir_name
            if not test_results_dir.exists():
                continue
            
            # Get latest test run
            latest_run = get_latest_test_run(test_results_dir)
            if not latest_run:
                continue
            
            # Get iteration from log
            test_iter = get_iteration_from_log(latest_run)
            if test_iter is None:
                continue
            
            # Get the maximum available iteration
            max_iter = get_max_available_iter(model_dir)
            if max_iter is None:
                continue
            
            # Check if test was performed on intermediate weights (not the latest available)
            if test_iter < max_iter:
                incomplete_tests.append({
                    'strategy': strategy,
                    'dataset': dataset,
                    'model': model,
                    'test_iter': test_iter,
                    'max_iter': max_iter,
                    'log_dir': str(latest_run)
                })

# Sort by strategy, dataset, model
incomplete_tests.sort(key=lambda x: (x['strategy'], x['dataset'], x['model']))

# Print results
print(f"Found {len(incomplete_tests)} tests performed on intermediate weights (newer checkpoints available)")
print()

if not incomplete_tests:
    print("All tests were performed with the latest available weights. Nothing to resubmit.")
    sys.exit(0)

# All tests in the list have newer weights available, so all are ready to retest
ready_to_retest = incomplete_tests

print(f"Ready to retest: {len(ready_to_retest)}")
print()

# Print table header
print(f"{'Strategy':<40} {'Dataset':<15} {'Model':<30} {'Tested@':<10} {'Max Avail':<10}")
print("-" * 115)

for test in incomplete_tests:
    print(f"{test['strategy']:<40} {test['dataset']:<15} {test['model']:<30} {test['test_iter']:<10} {test['max_iter']:<10}")

print()

if list_only:
    print("List mode - not submitting jobs.")
    sys.exit(0)

# Submit tests - all are ready since newer weights are available
to_submit = ready_to_retest
if limit > 0:
    to_submit = to_submit[:limit]

print(f"Submitting {len(to_submit)} tests...")
print()

# Generate submission commands
for test in to_submit:
    strategy = test['strategy']
    dataset = test['dataset']
    model = test['model']  # Full model name including _clear_day if present
    
    # Build job name (sanitize for LSF)
    job_name = f"test_{dataset}_{model}_{strategy}".replace('+', '_')
    
    # Build test command - use full model name since checkpoint dir includes it
    test_cmd_full = f"{script_dir}/test_unified.sh {test_cmd} --dataset {dataset} --model {model} --strategy {strategy}"
    
    # Build bsub command
    bsub_cmd = f'''bsub -gpu "num=1:mode={gpu_mode}:gmem={gpu_mem}" \\
    -q {queue} \\
    -R "span[hosts=1]" \\
    -n {num_cpus} \\
    -oo "logs/{job_name}_%J.out" \\
    -eo "logs/{job_name}_%J.err" \\
    -L /bin/bash \\
    -J "{job_name}" \\
    "{test_cmd_full}"'''
    
    if dry_run:
        print(f"[DRY-RUN] {job_name}")
        print(f"  Command: {test_cmd_full}")
    else:
        print(f"Submitting: {job_name}")
        os.system(bsub_cmd)

print()
if dry_run:
    print(f"[DRY-RUN] Would submit {len(to_submit)} jobs")
else:
    print(f"Submitted {len(to_submit)} jobs")

PYTHON_SCRIPT

echo ""
echo "Done."
