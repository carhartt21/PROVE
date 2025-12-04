#!/usr/bin/env python3
"""
Multi-Model Training Script for PROVE Pipeline
Demonstrates how to train multiple models with the same dataset configuration
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from prove_config import PROVEConfig

# Try to import MMCV/mmengine, but continue if not available
try:
    from mmengine import Config
    MMCV_AVAILABLE = True
except ImportError:
    try:
        from mmcv import Config
        MMCV_AVAILABLE = True
    except ImportError:
        MMCV_AVAILABLE = False
        print("Warning: MMCV/mmengine not available. Configs will be saved as Python dictionaries.")


def save_configs(configs_dict, output_dir='./configs'):
    """Save multiple configs to files"""
    os.makedirs(output_dir, exist_ok=True)

    saved_files = {}
    for model_name, config in configs_dict.items():
        config_file = os.path.join(output_dir, f'{model_name}_config.py')

        try:
            # Try to use MMCV Config if available
            from mmcv import Config
            mmcv_config = Config(config)
            mmcv_config.dump(config_file)
        except ImportError:
            # Fallback: save as Python dict
            with open(config_file, 'w') as f:
                f.write("# PROVE Configuration File\n")
                f.write("# Generated automatically - convert to MMCV format if needed\n\n")
                f.write("config = {\n")
                for key, value in config.items():
                    f.write(f"    '{key}': {repr(value)},\n")
                f.write("}\n")

        saved_files[model_name] = config_file
        print(f"Saved config for {model_name}: {config_file}")

    return saved_files


def generate_training_script(dataset_name, config_files):
    """Generate a training script for a specific dataset"""
    script_content = f'''#!/bin/bash
# Multi-Model Training Script for {dataset_name} Dataset
# Trains multiple models with the same dataset configuration

set -e  # Exit on any error

# Configuration
DATASET="{dataset_name}"
CONFIG_DIR="./multi_model_configs/${{DATASET^^}}"
WORK_DIR_BASE="/scratch/aaa_exchange/AWARE/WEIGHTS/${{DATASET,,}}"
LOG_DIR="./logs/${{DATASET,,}}"

# Create directories
mkdir -p "$LOG_DIR"

echo "{dataset_name} Multi-Model Training Script"
echo "=================================="

# Function to train a single model
train_model() {{
    local model_name=$1
    local config_file="$CONFIG_DIR/${{model_name}}_config.py"
    local work_dir="$WORK_DIR_BASE/${{model_name}}"
    local log_file="$LOG_DIR/${{model_name}}_training.log"

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
}}

# Check if config directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config directory $CONFIG_DIR not found!"
    echo "Run 'python multi_model_training.py' first to generate configs."
    exit 1
fi

# Get list of available configs
configs="{' '.join([name.split('_', 1)[1] for name in config_files.keys()])}"

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
    for pid in "${{pids[@]}}"; do
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
echo "Training summary for {dataset_name}:"
echo "- Logs saved in: $LOG_DIR"
echo "- Models saved in: $WORK_DIR_BASE"
echo "- Configs used from: $CONFIG_DIR"
'''
    return script_content


def main():
    """Main function generating multi-model configs for all AWARE datasets"""
    print("PROVE Multi-Model Config Generation for AWARE Datasets")
    print("=" * 60)

    # Initialize PROVE config
    pipeline = PROVEConfig()

    # Define datasets and their configurations
    datasets = {
        'ACDC': {
            'task_type': 'semantic_segmentation',
            'dataset_format': 'cityscapes',  # ACDC follows Cityscapes format
            'path': '/scratch/aaa_exchange/AWARE/FINAL_SPLITS',
            'models': ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']
        },
        'BDD100k': {
            'task_type': 'object_detection',
            'dataset_format': 'bdd100k_json',
            'path': '/scratch/aaa_exchange/AWARE/FINAL_SPLITS',
            'models': ['faster_rcnn_r50_fpn_1x', 'yolox_l', 'rtmdet_l']
        },
        'BDD10k': {
            'task_type': 'semantic_segmentation',
            'dataset_format': 'cityscapes',  # BDD10k follows Cityscapes format
            'path': '/scratch/aaa_exchange/AWARE/FINAL_SPLITS',
            'models': ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']
        },
        'IDD-AW': {
            'task_type': 'semantic_segmentation',
            'dataset_format': 'cityscapes',  # IDD-AW follows Cityscapes format
            'path': '/scratch/aaa_exchange/AWARE/FINAL_SPLITS',
            'models': ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']
        },
        'MapillaryVistas': {
            'task_type': 'semantic_segmentation',
            'dataset_format': 'mapillary_vistas',
            'path': '/scratch/aaa_exchange/AWARE/FINAL_SPLITS',
            'models': ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']
        },
        'OUTSIDE15k': {
            'task_type': 'semantic_segmentation',
            'dataset_format': 'outside15k',
            'path': '/scratch/aaa_exchange/AWARE/FINAL_SPLITS',
            'models': ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']
        }
    }

    # Generate configs for all datasets
    all_configs = {}
    total_configs = 0

    for dataset_name, config_info in datasets.items():
        print(f"\n{'='*50}")
        print(f"Generating configs for {dataset_name}")
        print(f"Task: {config_info['task_type']}")
        print(f"Format: {config_info['dataset_format']}")
        print(f"Models: {', '.join(config_info['models'])}")
        print(f"{'='*50}")

        try:
            # Generate multi-model configs for this dataset
            dataset_configs = pipeline.generate_multi_model_config(
                task_type=config_info['task_type'],
                dataset_format=config_info['dataset_format'],
                dataset_path=config_info['path'],
                model_names=config_info['models']
            )

            # Prefix config names with dataset name for uniqueness
            prefixed_configs = {}
            for model_name, config in dataset_configs.items():
                prefixed_name = f"{dataset_name.lower()}_{model_name}"
                prefixed_configs[prefixed_name] = config

            all_configs.update(prefixed_configs)
            print(f"✓ Generated {len(prefixed_configs)} configs for {dataset_name}")

            total_configs += len(prefixed_configs)

        except Exception as e:
            print(f"✗ Failed to generate configs for {dataset_name}: {str(e)}")
            continue

    # Save all configs organized by dataset
    print(f"\n{'='*60}")
    print("Saving configurations...")
    print(f"{'='*60}")

    config_files = {}
    for config_name, config in all_configs.items():
        # Extract dataset name from config name
        dataset_name = config_name.split('_')[0].upper()

        # Create dataset-specific output directory
        output_dir = f'./multi_model_configs/{dataset_name}'
        os.makedirs(output_dir, exist_ok=True)

        config_file = os.path.join(output_dir, f'{config_name}_config.py')

        try:
            # Convert config dict to MMCV Config object and save
            mmcv_config = Config(config)
            mmcv_config.dump(config_file)

            config_files[config_name] = config_file
            print(f"✓ Saved {config_name}: {config_file}")

        except Exception as e:
            print(f"✗ Failed to save {config_name}: {str(e)}")

    # Generate training script for each dataset
    print(f"\n{'='*60}")
    print("Generating training scripts...")
    print(f"{'='*60}")

    for dataset_name in datasets.keys():
        dataset_configs = {k: v for k, v in config_files.items()
                          if k.startswith(dataset_name.lower())}

        if dataset_configs:
            script_content = generate_training_script(dataset_name, dataset_configs)
            script_file = f'./train_{dataset_name.lower()}_models.sh'

            with open(script_file, 'w') as f:
                f.write(script_content)

            # Make script executable
            os.chmod(script_file, 0o755)

            print(f"✓ Generated training script: {script_file}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total datasets processed: {len(datasets)}")
    print(f"Total configs generated: {total_configs}")
    print(f"Total configs saved: {len(config_files)}")

    for dataset_name in datasets.keys():
        dataset_count = len([k for k in config_files.keys()
                           if k.startswith(dataset_name.lower())])
        print(f"- {dataset_name}: {dataset_count} configs")

    print("\nConfig files saved in: ./multi_model_configs/")
    print("Training scripts generated in current directory")
    print("\nTo train models for a specific dataset:")
    print("  ./train_<dataset>_models.sh          # Sequential training")
    print("  ./train_<dataset>_models.sh --parallel # Parallel training")


if __name__ == "__main__":
    main()