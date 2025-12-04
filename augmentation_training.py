#!/usr/bin/env python3
"""
PROVE Augmentation Training Pipeline

This script provides an integrated workflow for training models with
different augmentation strategies:

1. Baseline: No augmentation (already in multi_model_configs/baseline/)
2. PhotoMetricDistort: Classical image augmentation
3. Generated Images: Augmentation using generative model outputs

Usage:
    # Generate configs for a specific strategy
    python augmentation_training.py --generate-configs --strategy photometric_distort
    
    # Generate configs for a specific generative model
    python augmentation_training.py --generate-configs --strategy gen_cycleGAN
    
    # Train all models for a strategy
    python augmentation_training.py --train --strategy photometric_distort --dataset ACDC
    
    # List available strategies and models
    python augmentation_training.py --list
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from augmentation_config_generator import (
    AugmentationConfigGenerator,
    DATASETS,
    SEGMENTATION_MODELS,
    DETECTION_MODELS,
    GENERATIVE_MODELS,
    TRAINING_CONFIG,
    GENERATED_IMAGES_ROOT,
)


# ============================================================================
# Configuration Templates
# ============================================================================

def get_photometric_distort_pipeline(task: str) -> List[dict]:
    """Get training pipeline with PhotoMetricDistortion"""
    pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', scale=(512, 512), keep_ratio=True),
        dict(type='PhotoMetricDistortion'),  # Random brightness, contrast, saturation
    ]
    
    if task == 'segmentation':
        pipeline.append(dict(type='PackSegInputs'))
    else:
        pipeline.append(dict(type='PackDetInputs'))
    
    return pipeline


def get_generated_images_pipeline(task: str, generative_model: str) -> List[dict]:
    """
    Get training pipeline for generated image augmentation.
    
    Note: The actual image mixing happens at the dataset level,
    not in the pipeline transforms.
    """
    pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', scale=(512, 512), keep_ratio=True),
    ]
    
    if task == 'segmentation':
        pipeline.append(dict(type='PackSegInputs'))
    else:
        pipeline.append(dict(type='PackDetInputs'))
    
    return pipeline


# ============================================================================
# Config Generation Functions
# ============================================================================

def generate_photometric_configs(output_dir: str, datasets: List[str] = None):
    """Generate configs with PhotoMetricDistort augmentation"""
    
    datasets = datasets or list(DATASETS.keys())
    output_path = Path(output_dir) / 'photometric_distort'
    output_path.mkdir(parents=True, exist_ok=True)
    
    configs_generated = 0
    
    for dataset in datasets:
        dataset_info = DATASETS[dataset]
        task = dataset_info['task']
        
        # Get models for this task
        models = SEGMENTATION_MODELS if task == 'segmentation' else DETECTION_MODELS
        metric = 'mIoU' if task == 'segmentation' else 'bbox'
        
        dataset_dir = output_path / dataset.upper()
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        for model in models:
            config = {
                '_base_': [f'../_base_/models/{model}.py'] if 'r50' in model else [],
                'runner': dict(
                    type='IterBasedRunner',
                    max_iters=TRAINING_CONFIG['max_iters']
                ),
                'checkpoint_config': dict(interval=TRAINING_CONFIG['checkpoint_interval']),
                'evaluation': dict(
                    interval=TRAINING_CONFIG['eval_interval'],
                    metric=metric
                ),
                'log_config': dict(
                    interval=TRAINING_CONFIG['log_interval'],
                    hooks=[
                        dict(type='TextLoggerHook'),
                        dict(type='TensorboardLoggerHook'),
                    ]
                ),
                'data': dict(
                    samples_per_gpu=TRAINING_CONFIG['batch_size'],
                    workers_per_gpu=4,
                ),
                'train_pipeline': get_photometric_distort_pipeline(task),
                'work_dir': f'/scratch/aaa_exchange/AWARE/WEIGHTS/photometric_distort/{dataset.lower()}/{model}',
                'seed': 42,
                'deterministic': True,
            }
            
            config_path = dataset_dir / f'{dataset.lower()}_{model}_config.py'
            
            with open(config_path, 'w') as f:
                f.write(f"# PROVE Augmentation Config: PhotoMetricDistort\n")
                f.write(f"# Dataset: {dataset}\n")
                f.write(f"# Model: {model}\n\n")
                for key, value in config.items():
                    f.write(f"{key} = {repr(value)}\n")
            
            print(f"✓ Generated: {config_path}")
            configs_generated += 1
    
    return configs_generated


def generate_generative_model_configs(
    output_dir: str,
    generative_model: str,
    datasets: List[str] = None
):
    """Generate configs for a specific generative model augmentation"""
    
    datasets = datasets or list(DATASETS.keys())
    
    # Check if generative model has manifest
    gen_path = os.path.join(GENERATED_IMAGES_ROOT, generative_model)
    manifest_path = os.path.join(gen_path, 'manifest.csv')
    
    if not os.path.exists(manifest_path):
        print(f"Warning: No manifest found for {generative_model}")
        print(f"Expected: {manifest_path}")
        return 0
    
    output_path = Path(output_dir) / f'gen_{generative_model}'
    output_path.mkdir(parents=True, exist_ok=True)
    
    configs_generated = 0
    
    for dataset in datasets:
        dataset_info = DATASETS[dataset]
        task = dataset_info['task']
        
        # Skip detection datasets for now (focus on segmentation)
        if task == 'detection':
            print(f"Skipping {dataset} (detection) - generated images use segmentation labels")
            continue
        
        models = SEGMENTATION_MODELS if task == 'segmentation' else DETECTION_MODELS
        metric = 'mIoU' if task == 'segmentation' else 'bbox'
        
        dataset_dir = output_path / dataset.upper()
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        for model in models:
            config = {
                '_base_': [f'../_base_/models/{model}.py'] if 'r50' in model else [],
                'runner': dict(
                    type='IterBasedRunner',
                    max_iters=TRAINING_CONFIG['max_iters']
                ),
                'checkpoint_config': dict(interval=TRAINING_CONFIG['checkpoint_interval']),
                'evaluation': dict(
                    interval=TRAINING_CONFIG['eval_interval'],
                    metric=metric
                ),
                'log_config': dict(
                    interval=TRAINING_CONFIG['log_interval'],
                    hooks=[
                        dict(type='TextLoggerHook'),
                        dict(type='TensorboardLoggerHook'),
                    ]
                ),
                'data': dict(
                    samples_per_gpu=TRAINING_CONFIG['batch_size'],
                    workers_per_gpu=4,
                ),
                'train_pipeline': get_generated_images_pipeline(task, generative_model),
                
                # Generated images configuration
                'generated_augmentation': dict(
                    enabled=True,
                    generative_model=generative_model,
                    manifest_path=manifest_path,
                    gen_root=gen_path,
                    conditions=['cloudy', 'dawn_dusk', 'fog', 'night', 'rainy', 'snowy'],
                    augmentation_multiplier=7,  # 1 original + 6 conditions
                ),
                
                'work_dir': f'/scratch/aaa_exchange/AWARE/WEIGHTS/gen_{generative_model}/{dataset.lower()}/{model}',
                'seed': 42,
                'deterministic': True,
            }
            
            config_path = dataset_dir / f'{dataset.lower()}_{model}_config.py'
            
            with open(config_path, 'w') as f:
                f.write(f"# PROVE Augmentation Config: Generated Images ({generative_model})\n")
                f.write(f"# Dataset: {dataset}\n")
                f.write(f"# Model: {model}\n")
                f.write(f"# Augmentation: 7x (original + 6 adverse conditions)\n\n")
                for key, value in config.items():
                    f.write(f"{key} = {repr(value)}\n")
            
            print(f"✓ Generated: {config_path}")
            configs_generated += 1
    
    return configs_generated


# ============================================================================
# Training Script Generation
# ============================================================================

def generate_training_script(
    strategy: str,
    output_dir: str,
    datasets: List[str] = None
):
    """Generate shell script for training all models with a strategy"""
    
    datasets = datasets or list(DATASETS.keys())
    
    script_content = f'''#!/bin/bash
# PROVE Training Script: {strategy}
# Auto-generated - trains all models for this augmentation strategy

set -e

STRATEGY="{strategy}"
CONFIG_DIR="./multi_model_configs/$STRATEGY"
WORK_DIR_BASE="/scratch/aaa_exchange/AWARE/WEIGHTS/$STRATEGY"
LOG_DIR="./logs/$STRATEGY"

mkdir -p "$LOG_DIR"

echo "PROVE Training: $STRATEGY"
echo "{'='*50}"

# Function to train a model
train_model() {{
    local dataset=$1
    local model=$2
    local config="$CONFIG_DIR/${{dataset^^}}/${{dataset,,}}_${{model}}_config.py"
    local work_dir="$WORK_DIR_BASE/${{dataset,,}}/$model"
    local log_file="$LOG_DIR/${{dataset,,}}_${{model}}.log"
    
    if [ ! -f "$config" ]; then
        echo "Config not found: $config"
        return 1
    fi
    
    echo "Training: $dataset / $model"
    echo "Config: $config"
    echo "Work dir: $work_dir"
    
    mkdir -p "$work_dir"
    
    python tools/train.py "$config" --work-dir "$work_dir" > "$log_file" 2>&1
    
    echo "✓ Completed: $dataset / $model"
}}

# Train all datasets
'''
    
    for dataset in datasets:
        dataset_info = DATASETS[dataset]
        task = dataset_info['task']
        models = SEGMENTATION_MODELS if task == 'segmentation' else DETECTION_MODELS
        
        script_content += f"\n# {dataset}\n"
        for model in models:
            script_content += f'train_model "{dataset}" "{model}"\n'
    
    script_content += '''
echo
echo "Training complete for strategy: $STRATEGY"
echo "Results saved in: $WORK_DIR_BASE"
'''
    
    script_path = os.path.join(output_dir, f'train_{strategy}.sh')
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"✓ Generated training script: {script_path}")
    
    return script_path


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PROVE Augmentation Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available strategies and generative models
  python augmentation_training.py --list
  
  # Generate configs for PhotoMetricDistort
  python augmentation_training.py --generate-configs --strategy photometric_distort
  
  # Generate configs for cycleGAN generated images
  python augmentation_training.py --generate-configs --strategy gen_cycleGAN
  
  # Generate configs for all available generative models
  python augmentation_training.py --generate-configs --all-generative
        """
    )
    
    parser.add_argument('--list', action='store_true',
                       help='List available strategies and generative models')
    parser.add_argument('--generate-configs', action='store_true',
                       help='Generate configuration files')
    parser.add_argument('--strategy', type=str,
                       help='Strategy name (e.g., photometric_distort, gen_cycleGAN)')
    parser.add_argument('--all-generative', action='store_true',
                       help='Generate configs for all generative models with manifests')
    parser.add_argument('--datasets', type=str, nargs='+',
                       help='Specific datasets to process')
    parser.add_argument('--output-dir', type=str, default='./multi_model_configs',
                       help='Output directory for configs')
    
    args = parser.parse_args()
    
    if args.list:
        print("\n" + "=" * 60)
        print("PROVE Augmentation Strategies")
        print("=" * 60)
        
        print("\n📋 Available Augmentation Strategies:")
        print("  - baseline (already in multi_model_configs/baseline/)")
        print("  - photometric_distort")
        print("  - gen_<model> (for each generative model)")
        
        print("\n🎨 Available Generative Models:")
        from generated_images_dataset import list_available_generative_models
        models = list_available_generative_models(GENERATED_IMAGES_ROOT)
        
        ready_models = []
        pending_models = []
        
        for model in models:
            if model['has_manifest']:
                ready_models.append(model)
            else:
                pending_models.append(model)
        
        print(f"\n  Ready ({len(ready_models)}):")
        for m in ready_models:
            count = m.get('total_generated', 'N/A')
            print(f"    ✓ {m['name']}: {count} images")
        
        print(f"\n  Pending manifest ({len(pending_models)}):")
        for m in pending_models:
            print(f"    ○ {m['name']}")
        
        print("\n📊 Datasets:")
        for name, info in DATASETS.items():
            print(f"  - {name}: {info['task']}")
        
        return
    
    if args.generate_configs:
        datasets = args.datasets or list(DATASETS.keys())
        
        if args.all_generative:
            # Generate configs for all generative models with manifests
            from generated_images_dataset import list_available_generative_models
            models = list_available_generative_models(GENERATED_IMAGES_ROOT)
            
            total = 0
            for model in models:
                if model['has_manifest']:
                    print(f"\n{'='*60}")
                    print(f"Generating configs for: {model['name']}")
                    print(f"{'='*60}")
                    count = generate_generative_model_configs(
                        args.output_dir, model['name'], datasets
                    )
                    total += count
                    
                    # Generate training script
                    generate_training_script(
                        f"gen_{model['name']}", args.output_dir, datasets
                    )
            
            print(f"\n{'='*60}")
            print(f"Total configs generated: {total}")
            
        elif args.strategy:
            if args.strategy == 'photometric_distort':
                print(f"\n{'='*60}")
                print("Generating PhotoMetricDistort configs")
                print(f"{'='*60}")
                count = generate_photometric_configs(args.output_dir, datasets)
                generate_training_script('photometric_distort', args.output_dir, datasets)
                print(f"\nGenerated {count} configs")
                
            elif args.strategy.startswith('gen_'):
                gen_model = args.strategy[4:]  # Remove 'gen_' prefix
                print(f"\n{'='*60}")
                print(f"Generating configs for: {gen_model}")
                print(f"{'='*60}")
                count = generate_generative_model_configs(
                    args.output_dir, gen_model, datasets
                )
                generate_training_script(args.strategy, args.output_dir, datasets)
                print(f"\nGenerated {count} configs")
                
            else:
                print(f"Unknown strategy: {args.strategy}")
                print("Use --list to see available strategies")
                return 1
        else:
            print("Please specify --strategy or --all-generative")
            return 1
    
    if not (args.list or args.generate_configs):
        parser.print_help()


if __name__ == "__main__":
    main()
