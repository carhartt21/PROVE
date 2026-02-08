#!/usr/bin/env python3
"""
Fix class names in downstream_results.csv for MapillaryVistas and OUTSIDE15k.

The per_class_metrics column contains class names that may be incorrectly labeled
with Cityscapes names instead of native dataset names.

This script:
1. Reads the CSV
2. Identifies rows for MapillaryVistas and OUTSIDE15k
3. Remaps class names from Cityscapes-style to native names
4. Writes the corrected CSV

Usage:
    python scripts/fix_class_names_in_csv.py [--stage 1|2|both] [--dry-run]
"""

import argparse
import ast
import re
from pathlib import Path

# Native class names for MapillaryVistas (66 classes)
MAPILLARY_CLASSES = [
    'Bird', 'Ground Animal', 'Curb', 'Fence', 'Guard Rail', 'Barrier', 'Wall',
    'Bike Lane', 'Crosswalk - Plain', 'Curb Cut', 'Parking', 'Pedestrian Area',
    'Rail Track', 'Road', 'Service Lane', 'Sidewalk', 'Bridge', 'Building', 'Tunnel',
    'Person', 'Bicyclist', 'Motorcyclist', 'Other Rider', 'Lane Marking - Crosswalk',
    'Lane Marking - General', 'Mountain', 'Sand', 'Sky', 'Snow', 'Terrain', 'Vegetation',
    'Water', 'Banner', 'Bench', 'Bike Rack', 'Billboard', 'Catch Basin', 'CCTV Camera',
    'Fire Hydrant', 'Junction Box', 'Mailbox', 'Manhole', 'Phone Booth', 'Pothole',
    'Street Light', 'Pole', 'Traffic Sign Frame', 'Utility Pole', 'Traffic Light',
    'Traffic Sign (Back)', 'Traffic Sign (Front)', 'Trash Can', 'Bicycle', 'Boat',
    'Bus', 'Car', 'Caravan', 'Motorcycle', 'On Rails', 'Other Vehicle', 'Trailer',
    'Truck', 'Wheeled Slow', 'Car Mount', 'Ego Vehicle', 'Unlabeled'
]

# Native class names for OUTSIDE15k (24 classes)
OUTSIDE15K_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
    'construction', 'animal', 'water', 'other', 'ignore'
]

# Cityscapes class names (19 classes) - for reference
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]


def fix_class_names_in_dict(per_class_str: str, native_classes: list) -> str:
    """Fix class names in per_class_metrics string.
    
    Args:
        per_class_str: String representation of dict with class metrics
        native_classes: List of correct class names
        
    Returns:
        Fixed string with correct class names
    """
    if not per_class_str or per_class_str == 'nan':
        return per_class_str
    
    try:
        # Parse the string as a dict
        per_class = ast.literal_eval(per_class_str)
    except (ValueError, SyntaxError):
        return per_class_str
    
    if not isinstance(per_class, dict):
        return per_class_str
    
    # Check if already has correct names
    first_key = list(per_class.keys())[0] if per_class else None
    if first_key == native_classes[0]:
        return per_class_str  # Already correct
    
    # Build new dict with correct class names
    new_per_class = {}
    old_keys = list(per_class.keys())
    
    for i, old_key in enumerate(old_keys):
        if i < len(native_classes):
            new_key = native_classes[i]
        else:
            new_key = old_key  # Keep if no mapping available
        new_per_class[new_key] = per_class[old_key]
    
    return repr(new_per_class)


def process_csv(input_path: Path, output_path: Path, dry_run: bool = False) -> dict:
    """Process CSV and fix class names.
    
    Returns dict with statistics.
    """
    import csv
    
    stats = {
        'total_rows': 0,
        'mapillary_fixed': 0,
        'outside_fixed': 0,
        'already_correct': 0,
        'skipped': 0
    }
    
    # Read and process
    with open(input_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    stats['total_rows'] = len(rows)
    
    # Process each row
    for row in rows:
        dataset = row.get('dataset', '').lower()
        per_class = row.get('per_class_metrics', '')
        
        if not per_class or per_class == 'nan':
            stats['skipped'] += 1
            continue
        
        if dataset == 'mapillaryvistas':
            fixed = fix_class_names_in_dict(per_class, MAPILLARY_CLASSES)
            if fixed != per_class:
                row['per_class_metrics'] = fixed
                stats['mapillary_fixed'] += 1
            else:
                stats['already_correct'] += 1
        elif dataset == 'outside15k':
            fixed = fix_class_names_in_dict(per_class, OUTSIDE15K_CLASSES)
            if fixed != per_class:
                row['per_class_metrics'] = fixed
                stats['outside_fixed'] += 1
            else:
                stats['already_correct'] += 1
        else:
            stats['skipped'] += 1
    
    # Write output
    if not dry_run:
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Fix class names in downstream_results.csv')
    parser.add_argument('--stage', type=str, default='1',
                       choices=['1', '2', 'both'],
                       help='Which stage CSV to process (default: 1)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be changed without writing')
    parser.add_argument('--backup', action='store_true', default=True,
                       help='Create backup before modifying (default: True)')
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    
    csvs_to_process = []
    if args.stage in ['1', 'both']:
        csvs_to_process.append(base_dir / 'downstream_results.csv')
    if args.stage in ['2', 'both']:
        csvs_to_process.append(base_dir / 'downstream_results_stage2.csv')
    
    for csv_path in csvs_to_process:
        if not csv_path.exists():
            print(f"CSV not found: {csv_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {csv_path.name}")
        print('='*60)
        
        # Create backup
        if args.backup and not args.dry_run:
            backup_path = csv_path.with_suffix('.csv.bak')
            import shutil
            shutil.copy2(csv_path, backup_path)
            print(f"Backup created: {backup_path}")
        
        # Process
        output_path = csv_path if not args.dry_run else None
        stats = process_csv(csv_path, output_path if output_path else csv_path, dry_run=args.dry_run)
        
        print(f"\nStatistics:")
        print(f"  Total rows: {stats['total_rows']}")
        print(f"  MapillaryVistas fixed: {stats['mapillary_fixed']}")
        print(f"  OUTSIDE15k fixed: {stats['outside_fixed']}")
        print(f"  Already correct: {stats['already_correct']}")
        print(f"  Skipped (no per_class): {stats['skipped']}")
        
        if args.dry_run:
            print("\n[DRY RUN] No changes written")
        else:
            print(f"\nChanges written to: {csv_path}")


if __name__ == '__main__':
    main()
