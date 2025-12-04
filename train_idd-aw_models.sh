#!/bin/bash
# Multi-Model Training Script for IDD-AW Dataset
# Trains multiple models with the same dataset configuration

set -e  # Exit on any error

# Configuration
DATASET="IDD-AW"
CONFIG_DIR="./multi_model_configs/${DATASET^^}"
WORK_DIR_BASE="/scratch/aaa_exchange/AWARE/WEIGHTS/${DATASET,,}"
LOG_DIR="./logs/${DATASET,,}"

# Create directories
mkdir -p "$LOG_DIR"

echo "IDD-AW Multi-Model Training Script"
echo "=================================="

# Function to train a single model
train_model() {
    local model_name=$1
    local config_file="$CONFIG_DIR/${model_name}_config.py"
    local work_dir="$WORK_DIR_BASE/${model_name}"
    local log_file="$LOG_DIR/${model_name}_training.log"

    echo "Training $model_name..."
    echo "Config: $config_file"
    echo "Work dir: $work_dir"
    echo "Log: $log_file"

    # Create work directory
    mkdir -p "$work_dir"

    # Run training with logging
    if python tools/train.py "$config_file" --work-dir "$work_dir" > "$log_file" 2>&1; then
        echo "✓ $model_name training completed successfully"
    else
        echo "✗ $model_name training failed. Check log: $log_file"
        return 1
    fi
}

# Check if config directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config directory $CONFIG_DIR not found!"
    echo "Run 'python multi_model_training.py' first to generate configs."
    exit 1
fi

# Get list of available configs
configs="deeplabv3plus_r50 pspnet_r50 segformer_mit-b5"

if [ -z "$configs" ]; then
    echo "No config files found in $CONFIG_DIR"
    exit 1
fi

echo "Found configs for models: $configs"
echo

# Parse command line arguments
PARALLEL=false
if [ "$1" = "--parallel" ]; then
    PARALLEL=true
    echo "Running training in parallel mode"
fi

# Train models
if [ "$PARALLEL" = true ]; then
    echo "Starting parallel training..."

    # Run training jobs in parallel
    pids=()
    for model in $configs; do
        train_model "$model" &
        pids+=($!)
    done

    # Wait for all jobs to complete
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    echo "All parallel training jobs completed"
else
    echo "Starting sequential training..."

    # Train models sequentially
    for model in $configs; do
        if ! train_model "$model"; then
            echo "Stopping due to training failure"
            exit 1
        fi
        echo
    done

    echo "All sequential training completed"
fi

echo
echo "Training summary for IDD-AW:"
echo "- Logs saved in: $LOG_DIR"
echo "- Models saved in: $WORK_DIR_BASE"
echo "- Configs used from: $CONFIG_DIR"
