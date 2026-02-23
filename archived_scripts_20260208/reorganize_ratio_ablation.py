#!/usr/bin/env python3
"""
Reorganize WEIGHTS_RATIO_ABLATION into stage1/ and stage2/ directories.

This script:
1. Creates stage1/ and stage2/ directories with 775 permissions
2. Moves all model directories to the appropriate stage based on domain_filter
3. Normalizes dataset naming (removes _ad suffix, standardizes idd-aw)
4. Merges directories with same config into unified paths
"""

import os
import shutil
from pathlib import Path
import argparse

ROOT = Path('${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION')

def get_stage_from_config(config_path: Path) -> str:
    """Determine stage from training config's domain_filter setting."""
    content = config_path.read_text()
    if 'domain_filter=None' in content or 'domain_filter = None' in content:
        return 'stage2'
    elif "domain_filter='clear_day'" in content or 'domain_filter="clear_day"' in content:
        return 'stage1'
    else:
        return 'unknown'

def normalize_dataset_name(dataset: str) -> str:
    """Normalize dataset name by removing _ad suffix and standardizing."""
    # Remove _ad suffix
    clean = dataset.replace('_ad', '')
    # Standardize idd-aw naming
    clean = clean.replace('iddaw', 'idd-aw')
    return clean

def plan_moves(dry_run: bool = True):
    """Plan all directory moves."""
    moves = []
    
    for strategy_dir in ROOT.glob('gen_*'):
        strategy = strategy_dir.name
        for dataset_dir in strategy_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            
            # Determine stage from first model's config
            stage = None
            for model_dir in dataset_dir.iterdir():
                if model_dir.is_dir() and 'ratio' in model_dir.name:
                    config_file = model_dir / 'training_config.py'
                    if config_file.exists():
                        stage = get_stage_from_config(config_file)
                        break
            
            if stage and stage != 'unknown':
                clean_dataset = normalize_dataset_name(dataset)
                
                for model_dir in dataset_dir.iterdir():
                    if model_dir.is_dir() and 'ratio' in model_dir.name:
                        old_path = model_dir
                        new_path = ROOT / stage / strategy / clean_dataset / model_dir.name
                        moves.append((old_path, new_path))
    
    return moves

def execute_moves(moves: list, dry_run: bool = True):
    """Execute the planned moves."""
    # Create stage directories with 775 permissions
    for stage in ['stage1', 'stage2']:
        stage_dir = ROOT / stage
        if not stage_dir.exists():
            if dry_run:
                print(f"[DRY-RUN] Would create: {stage_dir}")
            else:
                stage_dir.mkdir(mode=0o775, exist_ok=True)
                print(f"Created: {stage_dir}")
    
    # Execute moves
    moved = 0
    skipped = 0
    
    for old_path, new_path in sorted(moves):
        if new_path.exists():
            print(f"SKIP (exists): {new_path}")
            skipped += 1
            continue
        
        if dry_run:
            print(f"[DRY-RUN] mv {old_path}")
            print(f"         -> {new_path}")
        else:
            # Create parent directories with 775 permissions
            new_path.parent.mkdir(parents=True, mode=0o775, exist_ok=True)
            
            # Move the directory
            shutil.move(str(old_path), str(new_path))
            print(f"Moved: {old_path.name} -> {new_path}")
            moved += 1
    
    print()
    print(f"Summary: {moved} moved, {skipped} skipped")
    return moved, skipped

def cleanup_empty_dirs():
    """Remove empty directories after moves."""
    for strategy_dir in ROOT.glob('gen_*'):
        for dataset_dir in strategy_dir.iterdir():
            if dataset_dir.is_dir() and not any(dataset_dir.iterdir()):
                print(f"Removing empty: {dataset_dir}")
                dataset_dir.rmdir()
        
        if strategy_dir.is_dir() and not any(strategy_dir.iterdir()):
            print(f"Removing empty: {strategy_dir}")
            strategy_dir.rmdir()

def main():
    parser = argparse.ArgumentParser(description='Reorganize ratio ablation directories')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without doing it')
    parser.add_argument('--no-cleanup', action='store_true', help='Do not remove empty directories')
    args = parser.parse_args()
    
    print("="*70)
    print("RATIO ABLATION DIRECTORY REORGANIZATION")
    print("="*70)
    print()
    
    if args.dry_run:
        print("*** DRY RUN MODE - No changes will be made ***")
        print()
    
    # Plan moves
    moves = plan_moves()
    
    stage1 = [m for m in moves if '/stage1/' in str(m[1])]
    stage2 = [m for m in moves if '/stage2/' in str(m[1])]
    
    print(f"Found {len(moves)} directories to move:")
    print(f"  Stage 1 (clear_day only): {len(stage1)}")
    print(f"  Stage 2 (all domains): {len(stage2)}")
    print()
    
    # Execute moves
    moved, skipped = execute_moves(moves, dry_run=args.dry_run)
    
    # Cleanup empty directories
    if not args.dry_run and not args.no_cleanup:
        print()
        print("Cleaning up empty directories...")
        cleanup_empty_dirs()
    
    print()
    print("Done!")

if __name__ == '__main__':
    main()
