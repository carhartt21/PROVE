#!/usr/bin/env python3
"""
Clean the WEIGHTS directory by keeping only the latest training and test runs.

This script:
1. For each model directory, keeps only the latest timestamped training run directory
2. For test_results directories, keeps only the latest test run
3. Optionally removes intermediate checkpoints, keeping only the final one (iter_80000.pth)
4. Reports space savings

Usage:
    python tools/clean_weights_directory.py --dry-run                    # Preview what would be deleted
    python tools/clean_weights_directory.py                              # Delete old runs only
    python tools/clean_weights_directory.py --clean-checkpoints          # Also delete intermediate checkpoints
    python tools/clean_weights_directory.py --clean-checkpoints --dry-run # Preview checkpoint cleanup
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime
import re

WEIGHTS_DIR = "${AWARE_DATA_ROOT}/WEIGHTS"
FINAL_CHECKPOINT = "iter_80000.pth"  # The final checkpoint to keep

def get_size(path):
    """Get total size of a directory in bytes."""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += get_size(entry.path)
    except PermissionError:
        pass
    return total

def format_size(size_bytes):
    """Format size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"

def is_timestamp_dir(name):
    """Check if directory name matches timestamp pattern YYYYMMDD_HHMMSS."""
    return bool(re.match(r'^20\d{6}_\d{6}$', name))

def find_timestamp_dirs(path):
    """Find all timestamp-named directories in a path."""
    timestamp_dirs = []
    try:
        for entry in os.scandir(path):
            if entry.is_dir() and is_timestamp_dir(entry.name):
                timestamp_dirs.append(entry.path)
    except PermissionError:
        pass
    return sorted(timestamp_dirs)

def get_latest_timestamp_dir(dirs):
    """Return the latest (most recent) timestamp directory from a list."""
    if not dirs:
        return None
    # Sort by name (which is timestamp-based, so lexicographic sort works)
    return sorted(dirs)[-1]

def clean_model_directory(model_dir, dry_run=True, clean_checkpoints=False):
    """Clean a model directory, keeping only the latest training run and optionally removing intermediate checkpoints."""
    deleted_size = 0
    deleted_count = 0
    
    # Find timestamp directories (training runs)
    timestamp_dirs = find_timestamp_dirs(model_dir)
    
    if len(timestamp_dirs) > 1:
        latest = get_latest_timestamp_dir(timestamp_dirs)
        for ts_dir in timestamp_dirs:
            if ts_dir != latest:
                size = get_size(ts_dir)
                deleted_size += size
                deleted_count += 1
                if dry_run:
                    print(f"  [DRY-RUN] Would delete: {ts_dir} ({format_size(size)})")
                else:
                    print(f"  Deleting: {ts_dir} ({format_size(size)})")
                    shutil.rmtree(ts_dir)
    
    # Clean intermediate checkpoints if requested
    if clean_checkpoints:
        checkpoint_pattern = re.compile(r'^iter_(\d+)\.pth$')
        try:
            for entry in os.scandir(model_dir):
                if entry.is_file():
                    match = checkpoint_pattern.match(entry.name)
                    if match and entry.name != FINAL_CHECKPOINT:
                        size = entry.stat().st_size
                        deleted_size += size
                        deleted_count += 1
                        if dry_run:
                            print(f"  [DRY-RUN] Would delete checkpoint: {entry.name} ({format_size(size)})")
                        else:
                            print(f"  Deleting checkpoint: {entry.name} ({format_size(size)})")
                            os.remove(entry.path)
        except PermissionError:
            pass
    
    return deleted_size, deleted_count

def clean_test_results_directory(test_dir, dry_run=True):
    """Clean test results directory, keeping only the latest test run."""
    deleted_size = 0
    deleted_count = 0
    
    # Test results have subdirectories like 'test', 'val', etc.
    try:
        for entry in os.scandir(test_dir):
            if entry.is_dir():
                # Find timestamp directories inside test/val directories
                timestamp_dirs = find_timestamp_dirs(entry.path)
                
                if len(timestamp_dirs) > 1:
                    latest = get_latest_timestamp_dir(timestamp_dirs)
                    for ts_dir in timestamp_dirs:
                        if ts_dir != latest:
                            size = get_size(ts_dir)
                            deleted_size += size
                            deleted_count += 1
                            if dry_run:
                                print(f"  [DRY-RUN] Would delete test: {ts_dir} ({format_size(size)})")
                            else:
                                print(f"  Deleting test: {ts_dir} ({format_size(size)})")
                                shutil.rmtree(ts_dir)
    except PermissionError:
        pass
    
    return deleted_size, deleted_count

def main():
    parser = argparse.ArgumentParser(description='Clean WEIGHTS directory')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Preview what would be deleted without actually deleting')
    parser.add_argument('--clean-checkpoints', action='store_true',
                        help='Also remove intermediate checkpoints (keep only iter_80000.pth)')
    args = parser.parse_args()
    
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN MODE - No files will be deleted")
        if args.clean_checkpoints:
            print("(Including intermediate checkpoint cleanup)")
        print("=" * 60)
    else:
        print("=" * 60)
        print("LIVE MODE - Files will be deleted!")
        if args.clean_checkpoints:
            print("(Including intermediate checkpoints)")
        print("=" * 60)
        response = input("Are you sure you want to proceed? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            sys.exit(0)
    
    total_deleted_size = 0
    total_deleted_count = 0
    
    # Walk through all strategy directories
    for strategy in sorted(os.listdir(WEIGHTS_DIR)):
        strategy_path = os.path.join(WEIGHTS_DIR, strategy)
        if not os.path.isdir(strategy_path):
            continue
        
        print(f"\n{'=' * 60}")
        print(f"Strategy: {strategy}")
        print('=' * 60)
        
        # Walk through dataset directories
        for dataset in sorted(os.listdir(strategy_path)):
            dataset_path = os.path.join(strategy_path, dataset)
            if not os.path.isdir(dataset_path):
                continue
            
            # Walk through model directories
            for model in sorted(os.listdir(dataset_path)):
                model_path = os.path.join(dataset_path, model)
                if not os.path.isdir(model_path):
                    continue
                
                print(f"\n  {dataset}/{model}:")
                
                # Clean training run directories (and optionally checkpoints)
                size, count = clean_model_directory(model_path, args.dry_run, args.clean_checkpoints)
                total_deleted_size += size
                total_deleted_count += count
                
                # Clean test_results directories
                test_results_path = os.path.join(model_path, 'test_results')
                if os.path.isdir(test_results_path):
                    size, count = clean_test_results_directory(test_results_path, args.dry_run)
                    total_deleted_size += size
                    total_deleted_count += count
                
                # Clean test_results_detailed directories
                test_detailed_path = os.path.join(model_path, 'test_results_detailed')
                if os.path.isdir(test_detailed_path):
                    size, count = clean_test_results_directory(test_detailed_path, args.dry_run)
                    total_deleted_size += size
                    total_deleted_count += count
    
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print('=' * 60)
    print(f"Total items to delete: {total_deleted_count}")
    print(f"Total space to reclaim: {format_size(total_deleted_size)}")
    
    if args.dry_run:
        print("\nRun without --dry-run to actually delete these items.")

if __name__ == '__main__':
    main()
