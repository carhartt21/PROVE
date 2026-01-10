#!/usr/bin/env python3
"""
Retrain models affected by the label transformation bug.

This script generates LSF job scripts for retraining and retesting affected models.
Each strategy × dataset combination gets its own job script for parallel processing.

Affected datasets: BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k

BDD10k, IDD-AW: Fixed bug - were incorrectly processed with CityscapesLabelIdToTrainId
MapillaryVistas: Now uses native 66 classes (was unified to 19 classes)
OUTSIDE15k: Now uses native 24 classes (was incorrectly mapped)

Usage:
    python scripts/retrain_affected_models.py --generate-scripts
    python scripts/retrain_affected_models.py --submit-all
    python scripts/retrain_affected_models.py --submit-strategy baseline
    python scripts/retrain_affected_models.py --submit-strategy baseline --dataset bdd10k
    python scripts/retrain_affected_models.py --submit-pending  # Submit only jobs without completed weights
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Detect project root from script location
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parent

# Configuration - can be overridden via command line args
DEFAULT_WEIGHTS_ROOT = Path(os.environ.get('PROVE_WEIGHTS_ROOT', '/scratch/aaa_exchange/AWARE/WEIGHTS'))
DEFAULT_SCRIPTS_DIR = DEFAULT_PROJECT_ROOT / 'scripts' / 'retrain_jobs'
DEFAULT_LOGS_DIR = DEFAULT_PROJECT_ROOT / 'logs' / 'retrain'

# Global variables that will be set in main() based on args
WEIGHTS_ROOT = DEFAULT_WEIGHTS_ROOT
SCRIPTS_DIR = DEFAULT_SCRIPTS_DIR
LOGS_DIR = DEFAULT_LOGS_DIR
PROJECT_ROOT = DEFAULT_PROJECT_ROOT

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

# LSF job template - Per strategy × dataset combination
JOB_TEMPLATE = '''#!/bin/bash
#BSUB -J {job_name}
#BSUB -o {log_dir}/{job_name}_%J.out
#BSUB -e {log_dir}/{job_name}_%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=20G"
#BSUB -W 12:00
#BSUB -q BatchGPU

# Activate environment
source ~/.bashrc
conda activate prove

cd {project_root}

echo "========================================"
echo "Retraining job: {job_name}"
echo "Strategy: {strategy}"
echo "Dataset: {dataset}"
echo "Started: $(date)"
echo "========================================"

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


def get_affected_configurations(model_filter=None):
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
        # 'gen_EDICT' - EXCLUDED: 0/4 training dataset coverage (only ACDC+BDD100k)
        'gen_flux_kontext',  # flux_kontext: Only has MapillaryVistas and OUTSIDE15k
        # 'gen_flux2' - EXCLUDED: 0/4 training dataset coverage (only ACDC+BDD100k)
        'gen_Img2Img',
        'gen_IP2P',
        'gen_LANIT',
        # 'gen_NST' - EXCLUDED: Generated images missing (manifest exists but folder doesn't)
        'gen_Qwen_Image_Edit',
        'gen_stargan_v2',
        'gen_step1x_new',
        # 'gen_StyleID' - EXCLUDED: 0/4 training dataset coverage (only ACDC+BDD100k)
        'gen_SUSTechGAN',
        'gen_TSIT',
        'gen_UniControl',
        'gen_Weather_Effect_Generator',
        # Newly added strategies from GENERATED_IMAGES
        # 'gen_AOD_Net' - EXCLUDED: No manifest / permission denied
        'gen_CNetSeg',              # ControlNet segmentation (187,398 images)
        'gen_VisualCloze',          # Visual completion (104,427 images)
        'gen_albumentations_weather', # Weather augmentation (95,700 images)
        'gen_cyclediffusion',       # CycleDiffusion (180,783 images) - 2/4 datasets
        'gen_step1x_v1p2',          # Step1X v1.2 (112,307 images)
        # Standard augmentation strategies
        'photometric_distort',
        'std_autoaugment',
        'std_cutmix',
        'std_mixup',
        'std_randaugment',
    ]
    
    # Define model variants (base models without domain filter suffix)
    ALL_MODELS = [
        'deeplabv3plus_r50',
        'pspnet_r50',
        'segformer_mit-b5',
    ]
    
    # Filter models if specified
    if model_filter:
        MODELS = [m for m in ALL_MODELS if m == model_filter]
    else:
        MODELS = ALL_MODELS
    
    # Domain filters to test - Stage 1: clear_day only
    DOMAIN_FILTERS = [
        # '',              # No filter - disabled for Stage 1
        'clear_day',     # Filter to clear_day domain - Stage 1 focus
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
                    
                    # Skip configurations with incomplete generated image coverage
                    # flux_kontext: Only has MapillaryVistas and OUTSIDE15k
                    if strategy == 'gen_flux_kontext' and dataset in ['bdd10k', 'idd-aw']:
                        continue
                    
                    # cyclediffusion: Has no OUTSIDE15k images (only ACDC, BDD100k, BDD10k, IDD-AW)
                    if strategy == 'gen_cyclediffusion' and dataset == 'outside15k':
                        continue
                    
                    # step1x_new: Only has BDD10k via symlinks from BDD100k (1,212 images)
                    # Skip for now as coverage is incomplete
                    if strategy == 'gen_step1x_new' and dataset == 'bdd10k':
                        continue
                    
                    # Determine real_gen_ratio: 0.5 for gen_* strategies, 1.0 for others
                    if strategy.startswith('gen_'):
                        real_gen_ratio = 0.5
                        ratio_str = '_ratio0p50'
                    else:
                        real_gen_ratio = 1.0
                        ratio_str = ''
                    
                    # Build directory names matching unified_training_config.py's _set_work_dir
                    # New format: strategy/dataset_cd/model_ratio
                    # Domain filter is now part of dataset directory (e.g., bdd10k_cd for clear_day)
                    domain_abbrev = {'clear_day': 'cd', 'clear_night': 'cn', 'rainy_day': 'rd', 'rainy_night': 'rn', 'fog': 'fg', 'snow': 'sn'}
                    domain_suffix = f'_{domain_abbrev.get(domain_filter, domain_filter[:2])}' if domain_filter else ''
                    dataset_dir = f'{dataset}{domain_suffix}'
                    model_dir = f'{model}{ratio_str}'
                    
                    weights_path = strategy_dir / dataset_dir / model_dir
                    
                    # Build domain filter argument
                    domain_arg = f'--domain-filter {domain_filter}' if domain_filter else ''
                    
                    configs.append({
                        'strategy': strategy,
                        'dataset': dataset,
                        'dataset_dir': dataset_dir,  # Dataset directory with domain suffix
                        'model': model_dir,  # Model directory name (without domain suffix)
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


def group_by_strategy_dataset(configs):
    """Group configurations by strategy and dataset."""
    by_key = defaultdict(list)
    for config in configs:
        key = (config['strategy'], config['dataset'])
        by_key[key].append(config)
    return dict(by_key)


def generate_job_script(strategy, dataset, configs):
    """Generate LSF job script for a strategy × dataset combination."""
    model_commands = []
    
    for config in configs:
        cmd = TRAIN_COMMAND.format(**config)
        model_commands.append(cmd)
    
    job_name = f"retrain_{strategy}_{dataset}"
    script = JOB_TEMPLATE.format(
        job_name=job_name,
        log_dir=str(LOGS_DIR),
        project_root=str(PROJECT_ROOT),
        strategy=strategy,
        dataset=dataset,
        model_commands='\n'.join(model_commands),
    )
    
    return script, job_name


def generate_all_scripts(configs_by_key):
    """Generate all job scripts (one per strategy × dataset)."""
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    script_paths = []
    
    for (strategy, dataset), configs in sorted(configs_by_key.items()):
        script, job_name = generate_job_script(strategy, dataset, configs)
        
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


def check_weights_exist(strategy, dataset, domain_filter='clear_day', model='deeplabv3plus_r50'):
    """Check if weights already exist for this combination."""
    # Build dataset directory with domain suffix
    domain_abbrev = {'clear_day': 'cd', 'clear_night': 'cn', 'rainy_day': 'rd', 'rainy_night': 'rn', 'fog': 'fg', 'snow': 'sn'}
    domain_suffix = f'_{domain_abbrev.get(domain_filter, domain_filter[:2])}' if domain_filter else ''
    dataset_dir = f'{dataset}{domain_suffix}'
    
    # Determine model directory
    if strategy == 'baseline' or not strategy.startswith('gen_'):
        model_dir = model
    else:
        model_dir = f'{model}_ratio0p50'
    
    checkpoint = WEIGHTS_ROOT / strategy / dataset_dir / model_dir / "iter_80000.pth"
    return checkpoint.exists()


def main():
    global WEIGHTS_ROOT, SCRIPTS_DIR, LOGS_DIR, PROJECT_ROOT
    
    parser = argparse.ArgumentParser(description='Retrain affected models')
    parser.add_argument('--generate-scripts', action='store_true',
                       help='Generate LSF job scripts')
    parser.add_argument('--submit-all', action='store_true',
                       help='Submit all job scripts')
    parser.add_argument('--submit-pending', action='store_true',
                       help='Submit only jobs without completed weights')
    parser.add_argument('--submit-strategy', type=str,
                       help='Submit job for specific strategy')
    parser.add_argument('--dataset', type=str,
                       help='Filter by specific dataset (use with --submit-strategy)')
    parser.add_argument('--model', type=str,
                       choices=['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5'],
                       help='Filter by specific model (e.g., pspnet_r50)')
    parser.add_argument('--list', action='store_true',
                       help='List all affected configurations')
    parser.add_argument('--summary', action='store_true',
                       help='Show summary of affected configurations')
    parser.add_argument('--domain-filter', type=str, default='clear_day',
                       help='Domain filter for training (default: clear_day)')
    # Path configuration arguments
    parser.add_argument('--project-root', type=str,
                       help=f'Project root directory (default: auto-detected as {DEFAULT_PROJECT_ROOT})')
    parser.add_argument('--weights-root', type=str,
                       help=f'Weights root directory (default: {DEFAULT_WEIGHTS_ROOT})')
    parser.add_argument('--scripts-dir', type=str,
                       help=f'Scripts output directory (default: <project-root>/scripts/retrain_jobs)')
    parser.add_argument('--logs-dir', type=str,
                       help=f'Logs directory (default: <project-root>/logs/retrain)')
    
    args = parser.parse_args()
    
    # Set up paths based on arguments or defaults
    PROJECT_ROOT = Path(args.project_root) if args.project_root else DEFAULT_PROJECT_ROOT
    WEIGHTS_ROOT = Path(args.weights_root) if args.weights_root else DEFAULT_WEIGHTS_ROOT
    SCRIPTS_DIR = Path(args.scripts_dir) if args.scripts_dir else PROJECT_ROOT / 'scripts' / 'retrain_jobs'
    LOGS_DIR = Path(args.logs_dir) if args.logs_dir else PROJECT_ROOT / 'logs' / 'retrain'
    
    # Get affected configurations (with optional model filter)
    configs = get_affected_configurations(model_filter=args.model)
    configs_by_key = group_by_strategy_dataset(configs)
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
        print(f"Jobs to submit: {len(configs_by_key)} (strategy × dataset)")
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
        print(f"\nGenerating {len(configs_by_key)} job scripts...")
        script_paths = generate_all_scripts(configs_by_key)
        print(f"\nGenerated {len(script_paths)} scripts in {SCRIPTS_DIR}")
        print(f"\nTo submit all jobs:")
        print(f"  python {sys.argv[0]} --submit-all")
        print(f"\nOr submit pending only:")
        print(f"  python {sys.argv[0]} --submit-pending")
        print(f"\nOr submit individually:")
        print(f"  python {sys.argv[0]} --submit-strategy baseline --dataset bdd10k")
        return
    
    if args.submit_all:
        script_paths = sorted(SCRIPTS_DIR.glob('retrain_*_*.sh'))
        print(f"\nSubmitting {len(script_paths)} jobs...")
        for script_path in script_paths:
            os.system(f'bsub < {script_path}')
            print(f"  Submitted: {script_path.name}")
        return
    
    if args.submit_pending:
        # Only get new-format scripts (with dataset suffix)
        script_paths = []
        for dataset in AFFECTED_DATASETS:
            script_paths.extend(sorted(SCRIPTS_DIR.glob(f'retrain_*_{dataset}.sh')))
        
        pending_count = 0
        skipped_count = 0
        
        print(f"\nChecking {len(script_paths)} jobs for pending training...")
        for script_path in sorted(script_paths):
            # Parse strategy and dataset from filename
            # Format: retrain_<strategy>_<dataset>.sh
            name = script_path.stem  # retrain_<strategy>_<dataset>
            parts = name.split('_')
            if len(parts) >= 3:
                # Handle strategies with underscores (e.g., gen_Attribute_Hallucination)
                dataset = parts[-1]  # Last part is dataset
                strategy = '_'.join(parts[1:-1])  # Everything between 'retrain_' and dataset
                
                if check_weights_exist(strategy, dataset, args.domain_filter):
                    skipped_count += 1
                    continue
                
                os.system(f'bsub < {script_path}')
                print(f"  Submitted: {script_path.name}")
                pending_count += 1
        
        print(f"\nSubmitted {pending_count} jobs, skipped {skipped_count} (already complete)")
        return
    
    if args.submit_strategy:
        if args.dataset:
            # Submit specific strategy × dataset
            script_path = SCRIPTS_DIR / f"retrain_{args.submit_strategy}_{args.dataset}.sh"
            if script_path.exists():
                os.system(f'bsub < {script_path}')
                print(f"Submitted: {script_path.name}")
            else:
                print(f"Script not found: {script_path}")
                print(f"Run --generate-scripts first")
        else:
            # Submit all datasets for this strategy
            script_paths = sorted(SCRIPTS_DIR.glob(f'retrain_{args.submit_strategy}_*.sh'))
            if script_paths:
                print(f"Submitting {len(script_paths)} jobs for {args.submit_strategy}...")
                for script_path in script_paths:
                    os.system(f'bsub < {script_path}')
                    print(f"  Submitted: {script_path.name}")
            else:
                print(f"No scripts found for strategy: {args.submit_strategy}")
                print(f"Run --generate-scripts first")
        return
    
    parser.print_help()


if __name__ == '__main__':
    main()
