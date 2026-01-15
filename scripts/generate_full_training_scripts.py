#!/usr/bin/env python3
"""
Generate LSF job scripts for training on FULL datasets (not clear_day filtered).

This script generates training jobs for the top N strategies identified from 
clear_day training results. The purpose is to validate whether strategies that
perform well on clear_day data also perform well when trained on full datasets.

Usage:
    # Generate jobs for top 10 strategies
    python scripts/generate_full_training_scripts.py --top-n 10
    
    # Generate jobs for specific strategies
    python scripts/generate_full_training_scripts.py --strategies gen_TSIT gen_CNetSeg baseline
    
    # Generate for all datasets
    python scripts/generate_full_training_scripts.py --top-n 10 --datasets bdd10k idd-aw mapillaryvistas outside15k
    
    # Dry run - show what would be generated
    python scripts/generate_full_training_scripts.py --top-n 10 --dry-run
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Set, Tuple
import json

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
WEIGHTS_ROOT = Path(os.environ.get('PROVE_WEIGHTS_ROOT', '/scratch/aaa_exchange/AWARE/WEIGHTS'))
OUTPUT_DIR = SCRIPT_DIR / 'full_train_jobs'
LOG_DIR = PROJECT_ROOT / 'logs' / 'full_train'

# Dataset configurations
DATASETS = {
    'bdd10k': {
        'name': 'BDD10K',
        'models': ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5'],
        'has_generated': True,
    },
    'idd-aw': {
        'name': 'IDD-AW',
        'models': ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5'],
        'has_generated': True,
    },
    'mapillaryvistas': {
        'name': 'MapillaryVistas',
        'models': ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5'],
        'has_generated': True,
        'native_classes': True,  # Uses 66 native classes
    },
    'outside15k': {
        'name': 'OUTSIDE15k',
        'models': ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5'],
        'has_generated': True,
        'native_classes': True,  # Uses 24 native classes
    },
}

# Strategy configurations
# Strategies that don't have generated images for certain datasets
SKIP_COMBOS = {
    ('gen_Qwen_Image_Edit', 'bdd10k'),
    ('gen_flux_kontext', 'bdd10k'),
    ('gen_flux_kontext', 'idd-aw'),
    ('gen_step1x_new', 'bdd10k'),
    ('gen_cyclediffusion', 'outside15k'),
    ('gen_cyclediffusion', 'mapillaryvistas'),
}

# LSF Job Template
LSF_TEMPLATE = '''#!/bin/bash
#BSUB -J train_full_{strategy}_{dataset}_{model_short}
#BSUB -o {log_dir}/train_full_{strategy}_{dataset}_{model_short}_%J.out
#BSUB -e {log_dir}/train_full_{strategy}_{dataset}_{model_short}_%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=20G"
#BSUB -W 12:00
#BSUB -q BatchGPU

# Activate environment
source ~/.bashrc
mamba activate prove

cd {project_root}

echo "========================================"
echo "Full Dataset Training Job"
echo "Strategy: {strategy}"
echo "Dataset: {dataset} (full, no domain filter)"
echo "Model: {model}"
echo "Started: $(date)"
echo "========================================"

# Pre-flight checks
WEIGHTS_PATH="{weights_path}"
LOCK_FILE="${{WEIGHTS_PATH}}/.training_lock"
CHECKPOINT="${{WEIGHTS_PATH}}/iter_80000.pth"

# Check if final checkpoint already exists
if [ -f "$CHECKPOINT" ]; then
    echo "SKIP: Final checkpoint already exists: $CHECKPOINT"
else
    # Try to create lock file (atomic operation)
    mkdir -p "$WEIGHTS_PATH"
    if ( set -o noclobber; echo "$$" > "$LOCK_FILE" ) 2>/dev/null; then
        trap "rm -f '$LOCK_FILE'" EXIT
        echo "Lock acquired. Starting training..."
        
        # Train model (NO domain filter = full dataset)
        python unified_training.py \\
            --dataset {dataset_name} \\
            --model {model} \\
            --strategy {strategy} \\
            --real-gen-ratio {ratio} \\
            --max-iters 80000{native_classes_flag}
        
        # Remove lock
        rm -f "$LOCK_FILE"
        
        if [ -f "$CHECKPOINT" ]; then
            echo "Training complete. Running test..."
            python fine_grained_test.py \\
                --checkpoint $CHECKPOINT \\
                --config ${{WEIGHTS_PATH}}/training_config.py \\
                --dataset {dataset_name} \\
                --output-dir ${{WEIGHTS_PATH}}/test_results_detailed \\
                --batch-size 8{native_classes_flag}
        else
            echo "ERROR: Training failed - checkpoint not found"
        fi
    else
        LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null)
        echo "SKIP: Another job (PID: $LOCK_PID) is already training this model"
    fi
fi

echo "========================================"
echo "Job completed: $(date)"
echo "========================================"
'''


def get_top_strategies_from_clearday(n: int = 10) -> List[str]:
    """Get top N strategies from clear_day results, sorted by overall mIoU.
    
    Uses test_results_summary.csv to rank strategies by average mIoU across all
    datasets with valid results. Falls back to predefined list if data incomplete.
    
    Returns:
        List of strategy names, sorted by performance (best first).
    """
    import pandas as pd
    
    # Primary source: test_results_summary.csv
    csv_path = PROJECT_ROOT / 'test_results_summary.csv'
    
    try:
        df = pd.read_csv(csv_path)
        
        # Filter to _cd datasets (clear_day trained) and fixed test results
        df = df[df['dataset'].str.endswith('_cd')]
        df = df[df['test_type'] == 'test_results_detailed_fixed']
        
        # Drop rows without mIoU values
        df = df.dropna(subset=['mIoU'])
        
        if len(df) < 10:
            print(f"Warning: Only {len(df)} test results available in CSV")
        
        # Group by strategy and compute average mIoU across all models and datasets
        strategy_miou = df.groupby('strategy')['mIoU'].mean().sort_values(ascending=False)
        
        # Exclude baseline from the ranking (it's always a reference)
        strategy_miou = strategy_miou[strategy_miou.index != 'baseline']
        
        if len(strategy_miou) >= n:
            top_strategies = strategy_miou.head(n).index.tolist()
            
            print(f"\nTop {n} strategies from clear_day test results:")
            for i, strategy in enumerate(top_strategies, 1):
                print(f"  {i}. {strategy}: {strategy_miou[strategy]:.2f}")
            
            return top_strategies
        else:
            print(f"Warning: Only {len(strategy_miou)} strategies with results, supplementing with fallback")
            
    except Exception as e:
        print(f"Warning: Could not load test results CSV: {e}")
    
    # Try leaderboard as secondary source
    try:
        from analysis_scripts.generate_strategy_leaderboard import (
            generate_comprehensive_leaderboards, WEIGHTS_ROOT as LB_WEIGHTS_ROOT
        )
        
        # Generate leaderboards
        _, clear_df = generate_comprehensive_leaderboards(LB_WEIGHTS_ROOT)
        
        # Filter out baseline entries and sort by Overall mIoU
        strategy_df = clear_df[~clear_df['Strategy'].str.startswith('baseline')]
        strategy_df = strategy_df.dropna(subset=['Overall mIoU'])
        strategy_df = strategy_df.sort_values('Overall mIoU', ascending=False)
        
        top_strategies = strategy_df['Strategy'].head(n).tolist()
        
        print(f"\nTop {n} strategies from clear_day results:")
        for i, row in strategy_df.head(n).iterrows():
            print(f"  {row['Strategy']}: {row['Overall mIoU']:.2f}")
        
        return top_strategies
    except Exception as e:
        print(f"Warning: Could not load leaderboard data: {e}")
        print("Falling back to manual strategy list.")
        
        # Fallback: Use a reasonable default list based on expected top performers
        fallback = [
            'gen_TSIT',
            'gen_albumentations_weather',
            'gen_VisualCloze',
            'gen_UniControl',
            'gen_CNetSeg',
            'gen_cycleGAN',
            'photometric_distort',
            'std_autoaugment',
            'gen_Attribute_Hallucination',
            'gen_automold',
        ]
        return fallback[:n]


def get_model_short_name(model: str) -> str:
    """Convert model name to short version for job names."""
    return (model
            .replace('deeplabv3plus_r50', 'dlv3p')
            .replace('pspnet_r50', 'pspn')
            .replace('segformer_mit-b5', 'segf'))


def get_weights_path(strategy: str, dataset: str, model: str) -> Path:
    """Get the weights path for a full-dataset training configuration.
    
    Full dataset training does NOT use the _cd suffix (that's for clear_day).
    """
    # Determine model subdirectory name based on strategy type
    if strategy == 'baseline' or strategy.startswith('std_') or strategy == 'photometric_distort':
        model_subdir = model
    else:
        model_subdir = f"{model}_ratio0p50"
    
    return WEIGHTS_ROOT / strategy / dataset / model_subdir


def check_existing_weights(weights_path: Path) -> bool:
    """Check if weights already exist."""
    checkpoint = weights_path / 'iter_80000.pth'
    return checkpoint.exists()


def generate_job_script(
    strategy: str,
    dataset: str,
    model: str,
    dry_run: bool = False
) -> Optional[Path]:
    """Generate a single LSF job script.
    
    Returns:
        Path to the generated script, or None if skipped.
    """
    dataset_config = DATASETS.get(dataset)
    if not dataset_config:
        print(f"  Unknown dataset: {dataset}")
        return None
    
    # Check if this combination should be skipped
    if (strategy, dataset) in SKIP_COMBOS:
        print(f"  SKIP (no data): {strategy}/{dataset}")
        return None
    
    # Get paths
    weights_path = get_weights_path(strategy, dataset, model)
    model_short = get_model_short_name(model)
    
    # Check if weights already exist
    if check_existing_weights(weights_path):
        print(f"  EXISTS: {strategy}/{dataset}/{model}")
        return None
    
    # Determine ratio based on strategy type
    if strategy == 'baseline' or strategy.startswith('std_') or strategy == 'photometric_distort':
        ratio = 1.0
    else:
        ratio = 0.5
    
    # Determine if native classes should be used
    native_classes_flag = ''
    if dataset_config.get('native_classes', False):
        native_classes_flag = ' \\\n            --use-native-classes'
    
    # Generate script content
    script_content = LSF_TEMPLATE.format(
        strategy=strategy,
        dataset=dataset,
        dataset_name=dataset_config['name'],
        model=model,
        model_short=model_short,
        weights_path=weights_path,
        ratio=ratio,
        native_classes_flag=native_classes_flag,
        project_root=PROJECT_ROOT,
        log_dir=LOG_DIR,
    )
    
    if dry_run:
        print(f"  Would create: train_full_{strategy}_{dataset}_{model_short}.sh")
        return None
    
    # Create script file
    script_name = f"train_full_{strategy}_{dataset}_{model_short}.sh"
    script_path = OUTPUT_DIR / script_name
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    
    print(f"  Created: {script_name}")
    return script_path


def generate_all_scripts(
    strategies: List[str],
    datasets: List[str],
    dry_run: bool = False
) -> Tuple[List[Path], Dict[str, int]]:
    """Generate all job scripts for the given strategies and datasets.
    
    Returns:
        Tuple of (list of script paths, stats dict).
    """
    # Create output directories
    if not dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    scripts = []
    stats = {
        'created': 0,
        'existing': 0,
        'skipped': 0,
    }
    
    for strategy in strategies:
        print(f"\nStrategy: {strategy}")
        
        for dataset in datasets:
            dataset_config = DATASETS.get(dataset)
            if not dataset_config:
                continue
            
            for model in dataset_config['models']:
                script = generate_job_script(strategy, dataset, model, dry_run)
                if script:
                    scripts.append(script)
                    stats['created'] += 1
                else:
                    # Check why it was skipped
                    weights_path = get_weights_path(strategy, dataset, model)
                    if check_existing_weights(weights_path):
                        stats['existing'] += 1
                    else:
                        stats['skipped'] += 1
    
    return scripts, stats


def create_submission_script(scripts: List[Path], output_path: Path):
    """Create a master script to submit all jobs."""
    with open(output_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Submit all full-dataset training jobs ({len(scripts)} jobs)\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"cd {OUTPUT_DIR}\n\n")
        
        for script in sorted(scripts):
            f.write(f"bsub < {script.name}\n")
            f.write("sleep 2\n")
        
        f.write(f"\necho 'Submitted {len(scripts)} jobs'\n")
    
    os.chmod(output_path, 0o755)


def main():
    parser = argparse.ArgumentParser(
        description="Generate LSF job scripts for full-dataset training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of top strategies to train (default: 10)')
    parser.add_argument('--strategies', nargs='+',
                        help='Specific strategies to train (overrides --top-n)')
    parser.add_argument('--datasets', nargs='+',
                        default=['bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k'],
                        help='Datasets to train on')
    parser.add_argument('--include-baseline', action='store_true',
                        help='Include baseline in training (always useful as reference)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be generated without creating files')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Full Dataset Training Script Generator")
    print("=" * 70)
    
    # Determine strategies to train
    if args.strategies:
        strategies = args.strategies
        print(f"\nUsing specified strategies: {strategies}")
    else:
        strategies = get_top_strategies_from_clearday(args.top_n)
    
    # Optionally include baseline
    if args.include_baseline and 'baseline' not in strategies:
        strategies = ['baseline'] + strategies
    
    # Validate datasets
    datasets = [d for d in args.datasets if d in DATASETS]
    if not datasets:
        print("ERROR: No valid datasets specified")
        return 1
    
    print(f"\nDatasets: {datasets}")
    print(f"Total configurations: {len(strategies)} strategies × {len(datasets)} datasets × 3 models")
    
    if args.dry_run:
        print("\n[DRY RUN - No files will be created]")
    
    # Generate scripts
    scripts, stats = generate_all_scripts(strategies, datasets, args.dry_run)
    
    # Create submission script
    if scripts and not args.dry_run:
        submit_script = OUTPUT_DIR / 'submit_all_full_train.sh'
        create_submission_script(scripts, submit_script)
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Created:  {stats['created']} job scripts")
    print(f"  Existing: {stats['existing']} (weights already exist)")
    print(f"  Skipped:  {stats['skipped']} (no generated data)")
    
    if scripts and not args.dry_run:
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print(f"\nTo submit all jobs:")
        print(f"  bash {OUTPUT_DIR / 'submit_all_full_train.sh'}")
    
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
