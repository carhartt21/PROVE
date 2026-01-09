#!/usr/bin/env python3
"""
Retrain models affected by the label transformation bug.

This script generates LSF job scripts for retraining and retesting affected models.
Each strategy gets its own job script that processes models sequentially.

Affected datasets: BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k

BDD10k, IDD-AW: Fixed bug - were incorrectly processed with CityscapesLabelIdToTrainId
MapillaryVistas: Now uses native 66 classes (was unified to 19 classes)
OUTSIDE15k: Now uses native 24 classes (was incorrectly mapped)

Usage:
    python scripts/retrain_affected_models.py --generate-scripts
    python scripts/retrain_affected_models.py --submit-all
    python scripts/retrain_affected_models.py --submit-strategy baseline
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Configuration
WEIGHTS_ROOT = Path("/scratch/aaa_exchange/AWARE/WEIGHTS")
SCRIPTS_DIR = Path("/home/mima2416/repositories/PROVE/scripts/retrain_jobs")
LOGS_DIR = Path("/home/mima2416/repositories/PROVE/logs/retrain")

AFFECTED_DATASETS = ['bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k']

# Dataset to config name mapping
DATASET_CONFIG_MAP = {
    'bdd10k': 'BDD10k',
    'idd-aw': 'IDD-AW', 
    'mapillaryvistas': 'MapillaryVistas',
    'outside15k': 'OUTSIDE15k',
}

# Model directory to model config name mapping
MODEL_CONFIG_MAP = {
    'deeplabv3plus_r50': 'deeplabv3plus_r50',
    'deeplabv3plus_r50_clear_day': 'deeplabv3plus_r50',  # Same model, different domain filter
    'pspnet_r50': 'pspnet_r50',
    'pspnet_r50_clear_day': 'pspnet_r50',
    'segformer_mit-b5': 'segformer_mit-b5',
    'segformer_mit-b5_clear_day': 'segformer_mit-b5',
}

# LSF job template
JOB_TEMPLATE = '''#!/bin/bash
#BSUB -J {job_name}
#BSUB -o {log_dir}/{job_name}_%J.out
#BSUB -e {log_dir}/{job_name}_%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=20G"
#BSUB -W 72:00
#BSUB -q BatchGPU

# Activate environment
source ~/.bashrc
conda activate prove

cd /home/mima2416/repositories/PROVE

echo "========================================"
echo "Retraining job: {job_name}"
echo "Strategy: {strategy}"
echo "Started: $(date)"
echo "========================================"

# Process each model configuration sequentially
{model_commands}

echo "========================================"
echo "Job completed: $(date)"
echo "========================================"
'''

# Training command template
TRAIN_COMMAND = '''
echo "----------------------------------------"
echo "Training: {dataset}/{model}"
echo "Strategy: {strategy}"
echo "Real/Gen Ratio: {real_gen_ratio}"
echo "Started: $(date)"
echo "----------------------------------------"

# Train model
python unified_training.py \\
    --dataset {dataset_config} \\
    --model {model_config} \\
    --strategy {strategy} \\
    --real-gen-ratio {real_gen_ratio} \\
    {domain_filter} \\
    --max-iters 80000

# Test model
if [ -f "{weights_path}/iter_80000.pth" ]; then
    echo "Training complete. Running test..."
    python fine_grained_test.py \\
        --checkpoint {weights_path}/iter_80000.pth \\
        --config {weights_path}/training_config.py \\
        --dataset {dataset_config} \\
        --output-dir {weights_path}/test_results_detailed
else
    echo "ERROR: Training failed - checkpoint not found"
fi

echo "Finished: {dataset}/{model} at $(date)"
'''


def get_affected_configurations():
    """Get all configurations affected by the label bug."""
    configs = []
    
    # Define known strategies (based on existing weights directories)
    # Includes both currently used strategies and newly added ones from GENERATED_IMAGES
    STRATEGIES = [
        'baseline',
        # Existing gen_* strategies
        'gen_Attribute_Hallucination',
        'gen_augmenters',
        'gen_automold',
        'gen_CUT',
        'gen_cycleGAN',
        'gen_EDICT',
        'gen_flux1_kontext',  # Maps to flux_kontext in GENERATED_IMAGES
        'gen_Img2Img',
        'gen_IP2P',
        'gen_LANIT',
        'gen_NST',
        'gen_Qwen_Image_Edit',
        'gen_stargan_v2',
        'gen_step1x_new',
        'gen_StyleID',
        'gen_SUSTechGAN',
        'gen_TSIT',
        'gen_UniControl',
        'gen_Weather_Effect_Generator',
        # Newly added strategies from GENERATED_IMAGES
        'gen_AOD_Net',              # Dehazing/restoration
        'gen_CNetSeg',              # ControlNet segmentation (187,398 images)
        'gen_VisualCloze',          # Visual completion (104,427 images)
        'gen_albumentations_weather', # Weather augmentation (95,700 images)
        'gen_cyclediffusion',       # CycleDiffusion (180,783 images)
        'gen_step1x_v1p2',          # Step1X v1.2 (112,307 images)
        # Standard augmentation strategies
        'photometric_distort',
        'std_autoaugment',
        'std_cutmix',
        'std_mixup',
        'std_randaugment',
    ]
    
    # Define model variants (base models without domain filter suffix)
    MODELS = [
        'deeplabv3plus_r50',
        'pspnet_r50',
        'segformer_mit-b5',
    ]
    
    # Domain filters to test
    DOMAIN_FILTERS = [
        '',              # No filter
        'clear_day',     # Filter to clear_day domain
    ]
    
    for strategy in STRATEGIES:
        strategy_dir = WEIGHTS_ROOT / strategy
        # Allow both existing strategies (retraining) and new strategies (fresh training)
        # Skip only if the strategy directory exists but is completely empty
        # if not strategy_dir.exists():
        #     continue
            
        for dataset in AFFECTED_DATASETS:
            for model in MODELS:
                for domain_filter in DOMAIN_FILTERS:
                    # Skip Qwen for bdd10k (no data)
                    if strategy == 'gen_Qwen_Image_Edit' and dataset == 'bdd10k':
                        continue
                    
                    # Determine real_gen_ratio: 0.5 for gen_* strategies, 1.0 for others
                    if strategy.startswith('gen_'):
                        real_gen_ratio = 0.5
                        ratio_str = '_ratio0p50'
                    else:
                        real_gen_ratio = 1.0
                        ratio_str = ''
                    
                    # Build directory name matching unified_training_config.py's _set_work_dir
                    # Format: model + ratio_str + domain_str
                    domain_str = f'_{domain_filter}' if domain_filter else ''
                    model_dir = f'{model}{ratio_str}{domain_str}'
                    
                    weights_path = strategy_dir / dataset / model_dir
                    
                    # Build domain filter argument
                    domain_arg = f'--domain-filter {domain_filter}' if domain_filter else ''
                    
                    configs.append({
                        'strategy': strategy,
                        'dataset': dataset,
                        'model': model_dir,  # Full model directory name
                        'weights_path': str(weights_path),
                        'dataset_config': DATASET_CONFIG_MAP.get(dataset, dataset),
                        'model_config': model,  # Base model name for training command
                        'domain_filter': domain_arg,
                        'real_gen_ratio': real_gen_ratio,
                    })
    
    return configs


def group_by_strategy(configs):
    """Group configurations by strategy."""
    by_strategy = defaultdict(list)
    for config in configs:
        by_strategy[config['strategy']].append(config)
    return dict(by_strategy)


def generate_job_script(strategy, configs):
    """Generate LSF job script for a strategy."""
    model_commands = []
    
    for config in configs:
        cmd = TRAIN_COMMAND.format(**config)
        model_commands.append(cmd)
    
    job_name = f"retrain_{strategy}"
    script = JOB_TEMPLATE.format(
        job_name=job_name,
        log_dir=str(LOGS_DIR),
        strategy=strategy,
        model_commands='\n'.join(model_commands),
    )
    
    return script, job_name


def generate_all_scripts(configs_by_strategy):
    """Generate all job scripts."""
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    script_paths = []
    
    for strategy, configs in sorted(configs_by_strategy.items()):
        script, job_name = generate_job_script(strategy, configs)
        
        script_path = SCRIPTS_DIR / f"{job_name}.sh"
        with open(script_path, 'w') as f:
            f.write(script)
        
        script_paths.append(script_path)
        print(f"  Generated: {script_path.name} ({len(configs)} models)")
    
    return script_paths


def submit_job(script_path):
    """Submit a single job."""
    import subprocess
    result = subprocess.run(['bsub', '<', str(script_path)], 
                          shell=True, capture_output=True, text=True)
    return result.returncode == 0, result.stdout + result.stderr


def main():
    parser = argparse.ArgumentParser(description='Retrain affected models')
    parser.add_argument('--generate-scripts', action='store_true',
                       help='Generate LSF job scripts')
    parser.add_argument('--submit-all', action='store_true',
                       help='Submit all job scripts')
    parser.add_argument('--submit-strategy', type=str,
                       help='Submit job for specific strategy')
    parser.add_argument('--list', action='store_true',
                       help='List all affected configurations')
    parser.add_argument('--summary', action='store_true',
                       help='Show summary of affected configurations')
    
    args = parser.parse_args()
    
    # Get affected configurations
    configs = get_affected_configurations()
    configs_by_strategy = group_by_strategy(configs)
    
    if args.list:
        print("Affected configurations:")
        for config in configs:
            print(f"  {config['strategy']}/{config['dataset']}/{config['model']}")
        return
    
    if args.summary:
        print(f"\nAffected Configurations Summary")
        print("=" * 50)
        print(f"Total models: {len(configs)}")
        print(f"Strategies: {len(configs_by_strategy)}")
        print(f"\nBy strategy:")
        for strategy in sorted(configs_by_strategy.keys()):
            print(f"  {strategy}: {len(configs_by_strategy[strategy])} models")
        print(f"\nBy dataset:")
        by_dataset = defaultdict(int)
        for c in configs:
            by_dataset[c['dataset']] += 1
        for dataset, count in sorted(by_dataset.items()):
            print(f"  {dataset}: {count} models")
        return
    
    if args.generate_scripts:
        print(f"\nGenerating {len(configs_by_strategy)} job scripts...")
        script_paths = generate_all_scripts(configs_by_strategy)
        print(f"\nGenerated {len(script_paths)} scripts in {SCRIPTS_DIR}")
        print(f"\nTo submit all jobs:")
        print(f"  python {sys.argv[0]} --submit-all")
        print(f"\nOr submit individually:")
        print(f"  bsub < {SCRIPTS_DIR}/retrain_<strategy>.sh")
        return
    
    if args.submit_all:
        script_paths = list(SCRIPTS_DIR.glob('retrain_*.sh'))
        print(f"\nSubmitting {len(script_paths)} jobs...")
        for script_path in sorted(script_paths):
            os.system(f'bsub < {script_path}')
            print(f"  Submitted: {script_path.name}")
        return
    
    if args.submit_strategy:
        script_path = SCRIPTS_DIR / f"retrain_{args.submit_strategy}.sh"
        if script_path.exists():
            os.system(f'bsub < {script_path}')
            print(f"Submitted: {script_path.name}")
        else:
            print(f"Script not found: {script_path}")
            print(f"Run --generate-scripts first")
        return
    
    parser.print_help()


if __name__ == '__main__':
    main()
