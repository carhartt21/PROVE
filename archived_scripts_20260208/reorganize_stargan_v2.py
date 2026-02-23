#!/usr/bin/env python3
"""
Reorganize stargan_v2 directory structure.

Current structure:
  stargan_v2/
  ├── cloudy/              # 31,233 mixed files (BDD, MapVistas, OUTSIDE15k, IDD-AW, ACDC)
  ├── dawn_dusk/           # same
  ├── foggy/               # same
  ├── night/               # same
  ├── rainy/               # same
  ├── snowy/               # same
  ├── Cityscapes/          # organized: {domain}/{files} - 2,975 per domain
  │   ├── cloudy/
  │   └── ...
  └── Cityscapes_from_lat/ # EXACT DUPLICATE of Cityscapes/ - to be removed
      └── Cityscapes/
          ├── cloudy/
          └── ...

Target structure:
  stargan_v2/
  ├── cloudy/
  │   ├── ACDC/
  │   ├── BDD10k/
  │   ├── BDD100k/
  │   ├── Cityscapes/
  │   ├── IDD-AW/
  │   ├── MapillaryVistas/
  │   └── OUTSIDE15k/
  ├── dawn_dusk/
  │   └── ... (same)
  └── ...
"""

import argparse
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path


# Base paths
GENERATED_IMAGES_BASE = Path("${AWARE_DATA_ROOT}/GENERATED_IMAGES")
STARGAN_V2_DIR = GENERATED_IMAGES_BASE / "stargan_v2"
FINAL_SPLITS_DIR = Path("${AWARE_DATA_ROOT}/FINAL_SPLITS/train/images")
CITYSCAPES_ORIGINALS_DIR = Path("${AWARE_DATA_ROOT}/CITYSCAPES/leftImg8bit/train")

WEATHER_DOMAINS = ["cloudy", "dawn_dusk", "foggy", "night", "rainy", "snowy"]

DATASETS = ["ACDC", "BDD10k", "BDD100k", "Cityscapes", "IDD-AW", "MapillaryVistas", "OUTSIDE15k"]


def build_filename_index():
    """Build a mapping from base filename (stem) → dataset name.
    
    This indexes ALL original images from FINAL_SPLITS and Cityscapes,
    allowing us to classify stargan_v2 generated files by matching 
    their stems (minus the _lat suffix) to originals.
    """
    print("Building filename → dataset index from FINAL_SPLITS...")
    index = {}  # stem → dataset
    conflicts = defaultdict(set)
    
    for dataset in DATASETS:
        if dataset == "Cityscapes":
            # Cityscapes originals are in a different location
            base = CITYSCAPES_ORIGINALS_DIR
        else:
            base = FINAL_SPLITS_DIR / dataset
        
        if not base.exists():
            print(f"  SKIP: {base} does not exist")
            continue
        
        count = 0
        for root, dirs, files in os.walk(base):
            for f in files:
                stem = Path(f).stem  # e.g., "aachen_000000_000019_leftImg8bit"
                if stem in index and index[stem] != dataset:
                    conflicts[stem].add(index[stem])
                    conflicts[stem].add(dataset)
                    # BDD10k takes priority over BDD100k (more specific)
                    if dataset == "BDD10k":
                        index[stem] = dataset
                else:
                    index[stem] = dataset
                count += 1
        
        print(f"  {dataset}: {count:,} files indexed")
    
    if conflicts:
        # Count unique conflicts (excluding BDD10k/BDD100k overlap which is expected)
        real_conflicts = {k: v for k, v in conflicts.items() 
                         if v != {"BDD10k", "BDD100k"}}
        print(f"  BDD10k/BDD100k overlaps: {len(conflicts) - len(real_conflicts)}")
        if real_conflicts:
            print(f"  WARNING: {len(real_conflicts)} non-BDD filename conflicts")
            for stem, datasets in list(real_conflicts.items())[:5]:
                print(f"    {stem}: {datasets}")
    
    print(f"  Total index: {len(index):,} unique stems")
    return index


def strip_lat_suffix(filename):
    """Remove _lat suffix from generated filename to get original stem.
    
    Examples:
        'aachen_000000_000019_leftImg8bit_lat.jpg' → 'aachen_000000_000019_leftImg8bit'
        '0000f77c-6257be58_lat.jpg' → '0000f77c-6257be58'
        'ADE_train_00000564_lat.jpg' → 'ADE_train_00000564'
    """
    stem = Path(filename).stem  # Remove .jpg
    if stem.endswith("_lat"):
        stem = stem[:-4]  # Remove _lat
    return stem


def classify_file(filename, index):
    """Classify a generated file to its source dataset using the filename index."""
    stem = strip_lat_suffix(filename)
    return index.get(stem, None)


def analyze_flat_dir(domain_dir, index):
    """Analyze a flat domain directory and classify files by dataset."""
    classification = defaultdict(list)  # dataset → [filenames]
    unclassified = []
    
    for f in sorted(os.listdir(domain_dir)):
        if not os.path.isfile(domain_dir / f):
            continue
        dataset = classify_file(f, index)
        if dataset:
            classification[dataset].append(f)
        else:
            unclassified.append(f)
    
    return classification, unclassified


def run_reorganization(dry_run=True, skip_duplicate_removal=False):
    """Main reorganization logic."""
    
    # Step 0: Verify stargan_v2 exists
    if not STARGAN_V2_DIR.exists():
        print(f"ERROR: {STARGAN_V2_DIR} does not exist!")
        sys.exit(1)
    
    # Step 1: Build filename index
    index = build_filename_index()
    
    # Step 2: Analyze flat domain directories
    print("\n" + "=" * 70)
    print("ANALYZING FLAT DOMAIN DIRECTORIES")
    print("=" * 70)
    
    total_moves = 0
    total_unclassified = 0
    all_classifications = {}
    
    for domain in WEATHER_DOMAINS:
        domain_dir = STARGAN_V2_DIR / domain
        if not domain_dir.exists():
            print(f"\n  SKIP: {domain}/ does not exist")
            continue
        
        # Check if already reorganized (has subdirectories)
        subdirs = [d for d in domain_dir.iterdir() if d.is_dir()]
        flat_files = [f for f in domain_dir.iterdir() if f.is_file()]
        
        if subdirs and not flat_files:
            print(f"\n  {domain}/: Already reorganized ({len(subdirs)} subdirs, 0 flat files)")
            continue
        
        if subdirs and flat_files:
            print(f"\n  {domain}/: Partially reorganized ({len(subdirs)} subdirs, {len(flat_files)} flat files)")
        
        classification, unclassified = analyze_flat_dir(domain_dir, index)
        all_classifications[domain] = (classification, unclassified)
        
        print(f"\n  {domain}/ ({sum(len(v) for v in classification.values()) + len(unclassified):,} files):")
        for ds in sorted(classification.keys()):
            print(f"    {ds}: {len(classification[ds]):,}")
            total_moves += len(classification[ds])
        if unclassified:
            print(f"    UNCLASSIFIED: {len(unclassified):,}")
            total_unclassified += len(unclassified)
            # Show sample unclassified
            for f in unclassified[:3]:
                print(f"      e.g., {f}")
    
    # Step 3: Analyze Cityscapes merge
    print("\n" + "=" * 70)
    print("CITYSCAPES MERGE PLAN")
    print("=" * 70)
    
    cityscapes_dir = STARGAN_V2_DIR / "Cityscapes"
    cityscapes_lat_dir = STARGAN_V2_DIR / "Cityscapes_from_lat"
    
    cityscapes_merge_count = 0
    if cityscapes_dir.exists():
        for domain in WEATHER_DOMAINS:
            src = cityscapes_dir / domain
            if src.exists():
                count = len(list(src.iterdir()))
                print(f"  Cityscapes/{domain}/ → {domain}/Cityscapes/ ({count:,} files)")
                cityscapes_merge_count += count
    
    print(f"\n  Total Cityscapes files to merge: {cityscapes_merge_count:,}")
    
    if cityscapes_lat_dir.exists():
        lat_count = sum(1 for _ in cityscapes_lat_dir.rglob("*") if _.is_file())
        print(f"  Cityscapes_from_lat/ ({lat_count:,} files) - DUPLICATE, will be removed")
    
    # Step 4: Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Flat files to move into dataset subdirs: {total_moves:,}")
    print(f"  Cityscapes files to merge: {cityscapes_merge_count:,}")
    print(f"  Unclassified files: {total_unclassified:,}")
    print(f"  Duplicate dir to remove: Cityscapes_from_lat/ ({lat_count if cityscapes_lat_dir.exists() else 0:,} files)")
    print(f"  Total operations: {total_moves + cityscapes_merge_count:,} moves + 1 dir removal")
    
    if dry_run:
        print("\n  *** DRY RUN - no changes made ***")
        return True
    
    # Step 5: Execute reorganization
    print("\n" + "=" * 70)
    print("EXECUTING REORGANIZATION")
    print("=" * 70)
    
    # 5a: Move flat files into dataset subdirectories
    for domain in WEATHER_DOMAINS:
        domain_dir = STARGAN_V2_DIR / domain
        if domain not in all_classifications:
            continue
        
        classification, unclassified = all_classifications[domain]
        
        for dataset, files in sorted(classification.items()):
            target_dir = domain_dir / dataset
            target_dir.mkdir(exist_ok=True)
            
            print(f"  Moving {len(files):,} files → {domain}/{dataset}/...", end="", flush=True)
            moved = 0
            for f in files:
                src = domain_dir / f
                dst = target_dir / f
                if src.exists():
                    src.rename(dst)
                    moved += 1
            print(f" done ({moved:,})")
        
        # Handle unclassified files - put in _unclassified/
        if unclassified:
            unc_dir = domain_dir / "_unclassified"
            unc_dir.mkdir(exist_ok=True)
            print(f"  Moving {len(unclassified):,} unclassified → {domain}/_unclassified/...", end="", flush=True)
            for f in unclassified:
                src = domain_dir / f
                dst = unc_dir / f
                if src.exists():
                    src.rename(dst)
            print(" done")
    
    # 5b: Merge Cityscapes/ into domain dirs
    if cityscapes_dir.exists():
        print(f"\n  Merging Cityscapes/ into domain subdirs...")
        for domain in WEATHER_DOMAINS:
            src_dir = cityscapes_dir / domain
            if not src_dir.exists():
                continue
            
            target_dir = STARGAN_V2_DIR / domain / "Cityscapes"
            target_dir.mkdir(exist_ok=True)
            
            files = list(src_dir.iterdir())
            print(f"  Moving {len(files):,} files → {domain}/Cityscapes/...", end="", flush=True)
            moved = 0
            for f in files:
                dst = target_dir / f.name
                if not dst.exists():
                    f.rename(dst)
                    moved += 1
                else:
                    # File already exists (shouldn't happen, but be safe)
                    pass
            print(f" done ({moved:,})")
        
        # Remove empty Cityscapes directory
        try:
            shutil.rmtree(cityscapes_dir)
            print(f"  Removed empty Cityscapes/ directory")
        except Exception as e:
            print(f"  WARNING: Could not remove Cityscapes/: {e}")
    
    # 5c: Remove Cityscapes_from_lat (exact duplicate)
    if not skip_duplicate_removal and cityscapes_lat_dir.exists():
        print(f"\n  Removing duplicate Cityscapes_from_lat/...")
        try:
            shutil.rmtree(cityscapes_lat_dir)
            print(f"  Removed Cityscapes_from_lat/ successfully")
        except Exception as e:
            print(f"  WARNING: Could not remove Cityscapes_from_lat/: {e}")
    
    print("\n  REORGANIZATION COMPLETE!")
    return True


def verify_reorganization():
    """Verify the reorganization was successful."""
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    total = 0
    for domain in WEATHER_DOMAINS:
        domain_dir = STARGAN_V2_DIR / domain
        if not domain_dir.exists():
            print(f"  {domain}/: MISSING!")
            continue
        
        flat_files = [f for f in domain_dir.iterdir() if f.is_file()]
        subdirs = [d for d in domain_dir.iterdir() if d.is_dir()]
        
        if flat_files:
            print(f"  WARNING: {domain}/ still has {len(flat_files)} flat files!")
        
        domain_total = 0
        for sd in sorted(subdirs, key=lambda x: x.name):
            count = len([f for f in sd.iterdir() if f.is_file()])
            domain_total += count
            print(f"  {domain}/{sd.name}/: {count:,}")
        
        total += domain_total
    
    cityscapes_dir = STARGAN_V2_DIR / "Cityscapes"
    cityscapes_lat_dir = STARGAN_V2_DIR / "Cityscapes_from_lat"
    
    if cityscapes_dir.exists():
        print(f"\n  WARNING: Cityscapes/ still exists (should be merged)")
    if cityscapes_lat_dir.exists():
        print(f"\n  WARNING: Cityscapes_from_lat/ still exists (should be removed)")
    
    print(f"\n  Total files in reorganized structure: {total:,}")


def main():
    parser = argparse.ArgumentParser(description="Reorganize stargan_v2 directory structure")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Show what would be done without making changes (default)")
    parser.add_argument("--execute", action="store_true",
                        help="Actually perform the reorganization")
    parser.add_argument("--verify", action="store_true",
                        help="Verify reorganization was successful")
    parser.add_argument("--keep-duplicate", action="store_true",
                        help="Keep Cityscapes_from_lat (don't remove duplicate)")
    args = parser.parse_args()
    
    if args.verify:
        verify_reorganization()
        return
    
    dry_run = not args.execute
    
    if not dry_run:
        print("WARNING: This will reorganize files in:")
        print(f"  {STARGAN_V2_DIR}")
        response = input("Continue? [y/N] ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    run_reorganization(dry_run=dry_run, skip_duplicate_removal=args.keep_duplicate)


if __name__ == "__main__":
    main()
