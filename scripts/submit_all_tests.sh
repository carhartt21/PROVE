#!/bin/bash

# submit_all_tests.sh
# Submits test jobs for all available checkpoints of a given strategy.
# Tests are grouped by model and run sequentially in a single job.

# Auto-detect script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

STRATEGY=$1
DATASET=$2
MODEL=$3

if [ -z "$STRATEGY" ]; then
    echo "Usage: $0 <strategy> [dataset] [model]"
    echo "Example: $0 gen_LANIT acdc deeplabv3plus_r50"
    exit 1
fi

BASE_DIR="/scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED"
STRATEGY_DIR="$BASE_DIR/$STRATEGY"

if [ ! -d "$STRATEGY_DIR" ]; then
    echo "Strategy directory not found: $STRATEGY_DIR"
    exit 1
fi

# Find all model directories
if [ -n "$DATASET" ] && [ -n "$MODEL" ]; then
    MODEL_DIRS=$(find "$STRATEGY_DIR/$DATASET/$MODEL" -maxdepth 0 -type d 2>/dev/null)
elif [ -n "$DATASET" ]; then
    MODEL_DIRS=$(find "$STRATEGY_DIR/$DATASET" -mindepth 1 -maxdepth 1 -type d 2>/dev/null)
else
    MODEL_DIRS=$(find "$STRATEGY_DIR" -mindepth 2 -maxdepth 2 -type d 2>/dev/null)
fi

if [ -z "$MODEL_DIRS" ]; then
    echo "No model directories found for strategy $STRATEGY"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/logs"

for model_dir in $MODEL_DIRS; do
    # Extract dataset and model names from path
    # Path is .../STRATEGY/DATASET/MODEL
    rel_path=${model_dir#$BASE_DIR/}
    strat=$(echo $rel_path | cut -d'/' -f1)
    ds=$(echo $rel_path | cut -d'/' -f2)
    mod=$(echo $rel_path | cut -d'/' -f3)
    
    echo "Processing $strat/$ds/$mod..."
    
    # Find all checkpoints
    checkpoints=$(ls $model_dir/iter_*.pth 2>/dev/null | sort -V)
    
    if [ -z "$checkpoints" ]; then
        echo "  No checkpoints found in $model_dir"
        continue
    fi
    
    # Find config file
    config_file="$model_dir/training_config.py"
    if [ ! -f "$config_file" ]; then
        # Try to find any .py file in configs/
        config_file=$(ls $model_dir/configs/*.py 2>/dev/null | head -n 1)
    fi
    
    if [ ! -f "$config_file" ]; then
        echo "  No config file found in $model_dir"
        continue
    fi
    
    # Build a single command to run all tests sequentially
    test_cmd="export PROJECT_ROOT=$PROJECT_ROOT && cd \$PROJECT_ROOT"
    
    count=0
    for ckpt in $checkpoints; do
        iter=$(basename $ckpt | sed 's/iter_\([0-9]*\).pth/\1/')
        
        # Skip very small iterations if desired (e.g., < 10000)
        if [ "$iter" -lt 10000 ]; then
            continue
        fi
        
        test_cmd="$test_cmd && echo 'Testing iter $iter...' && mamba run -n prove python fine_grained_test.py --config $config_file --checkpoint $ckpt --output-dir $model_dir/test_results --dataset ${ds^^}"
        count=$((count + 1))
    done
    
    if [ $count -eq 0 ]; then
        echo "  No valid checkpoints to test."
        continue
    fi
    
    job_name="test_seq_${strat}_${ds}_${mod}"
    log_file="$PROJECT_ROOT/logs/${job_name}_%J.log"
    err_file="$PROJECT_ROOT/logs/${job_name}_%J.err"
    
    # Check if already submitted or running
    if bjobs -J "$job_name" 2>/dev/null | grep -q "$job_name"; then
        echo "  Job $job_name already in queue, skipping."
        continue
    fi
    
    echo "  Submitting sequential test job for $count checkpoints..."
    
    # Increase walltime since it's sequential (e.g., 24 hours)
    bsub -q BatchGPU -gpu "num=1:mode=shared:j_exclusive=yes" -n 4 -W 24:00 \
        -o "$log_file" -e "$err_file" -J "$job_name" \
        "$test_cmd"
done
