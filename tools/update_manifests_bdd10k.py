#!/usr/bin/env python3
"""
Script to update manifests to properly identify BDD10k images.

The issue: Some manifests label images as 'BDD100k' but the original_path 
references files that exist in both BDD100k and BDD10k. When training uses 
BDD10k dataset, it looks for 'BDD10k' in original_path but finds 'BDD100k'.

Solution: For entries where the filename exists in BDD10k, update both:
1. The 'dataset' column to 'BDD10k' 
2. The 'original_path' to point to the BDD10k location

Usage:
    # Dry run (show what would be changed)
    python tools/update_manifests_bdd10k.py --dry-run
    
    # Update all manifests
    python tools/update_manifests_bdd10k.py --all
    
    # Update specific strategy
    python tools/update_manifests_bdd10k.py --strategy EDICT
"""

import argparse
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil


def load_bdd10k_files():
    """Load all BDD10k filenames and their full paths."""
    bdd10k_path = "${AWARE_DATA_ROOT}/FINAL_SPLITS/train/images/BDD10k"
    
    bdd10k_files = {}  # filename -> full_path
    for root, dirs, files in os.walk(bdd10k_path):
        for f in files:
            if f.endswith(('.jpg', '.png', '.jpeg')):
                full_path = os.path.join(root, f)
                # Get relative path from images/
                rel_path = full_path.replace('${AWARE_DATA_ROOT}/FINAL_SPLITS/', '')
                bdd10k_files[f] = '${AWARE_DATA_ROOT}/FINAL_SPLITS/' + rel_path
    
    return bdd10k_files


def update_manifest(manifest_path, bdd10k_files, dry_run=False):
    """Update a manifest to properly identify BDD10k images."""
    if not os.path.exists(manifest_path):
        print(f"  Manifest not found: {manifest_path}")
        return 0
    
    df = pd.read_csv(manifest_path)
    
    if 'dataset' not in df.columns or 'original_path' not in df.columns:
        print(f"  Missing required columns in {manifest_path}")
        return 0
    
    # Find BDD100k entries
    bdd100k_mask = df['dataset'] == 'BDD100k'
    bdd100k_count = bdd100k_mask.sum()
    
    if bdd100k_count == 0:
        print(f"  No BDD100k entries to update")
        return 0
    
    # Check which entries should be BDD10k
    updates = 0
    new_datasets = df['dataset'].copy()
    new_paths = df['original_path'].copy()
    
    for idx in df[bdd100k_mask].index:
        original_path = df.loc[idx, 'original_path']
        filename = os.path.basename(original_path)
        
        if filename in bdd10k_files:
            updates += 1
            new_datasets.loc[idx] = 'BDD10k'
            new_paths.loc[idx] = bdd10k_files[filename]
    
    if updates == 0:
        print(f"  No matching BDD10k files found")
        return 0
    
    print(f"  Found {updates} entries to update to BDD10k")
    
    if not dry_run:
        # Backup original
        backup_path = manifest_path + f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        shutil.copy(manifest_path, backup_path)
        print(f"  Backed up to: {backup_path}")
        
        # Update dataframe
        df['dataset'] = new_datasets
        df['original_path'] = new_paths
        
        # Save updated manifest
        df.to_csv(manifest_path, index=False)
        print(f"  Updated manifest saved")
    else:
        print(f"  [DRY RUN] Would update {updates} entries")
    
    return updates


def main():
    parser = argparse.ArgumentParser(description='Update manifests to properly identify BDD10k images')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without making changes')
    parser.add_argument('--strategy', type=str, help='Update specific strategy manifest')
    parser.add_argument('--all', action='store_true', help='Update all manifests')
    args = parser.parse_args()
    
    if not args.strategy and not args.all:
        parser.error("Must specify either --strategy or --all")
    
    gen_images_root = "${AWARE_DATA_ROOT}/GENERATED_IMAGES"
    
    print("Loading BDD10k files...")
    bdd10k_files = load_bdd10k_files()
    print(f"Found {len(bdd10k_files)} BDD10k files")
    print()
    
    strategies = []
    if args.strategy:
        strategies = [args.strategy]
    else:
        # Get all directories
        strategies = [d for d in os.listdir(gen_images_root) 
                     if os.path.isdir(os.path.join(gen_images_root, d))]
    
    total_updates = 0
    
    for strategy in sorted(strategies):
        manifest_path = os.path.join(gen_images_root, strategy, 'manifest.csv')
        
        if os.path.exists(manifest_path):
            print(f"\nProcessing: {strategy}")
            updates = update_manifest(manifest_path, bdd10k_files, args.dry_run)
            total_updates += updates
    
    print(f"\n{'='*60}")
    if args.dry_run:
        print(f"[DRY RUN] Would update {total_updates} entries across all manifests")
    else:
        print(f"Updated {total_updates} entries across all manifests")


if __name__ == '__main__':
    main()
