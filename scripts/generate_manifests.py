#!/usr/bin/env python3
"""
Generate/update manifest files for generated image directories.

This script scans generated image directories and creates/updates manifest.json files
with accurate counts of images per domain and dataset.

Usage:
    python scripts/generate_manifests.py --all                     # Generate all missing/outdated manifests
    python scripts/generate_manifests.py --dir AOD-Net             # Generate for specific directory
    python scripts/generate_manifests.py --check                   # Check which manifests need updating
"""

import os
import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Configuration
GENERATED_IMAGES_BASE = Path("/scratch/aaa_exchange/AWARE/GENERATED_IMAGES")
ORIGINAL_DIR = "/scratch/aaa_exchange/AWARE/FINAL_SPLITS/train/images"
TARGET_DIR = "/scratch/aaa_exchange/AWARE/AWACS/train"

# Known domains
KNOWN_DOMAINS = ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy']

# Known datasets
KNOWN_DATASETS = ['ACDC', 'BDD100k', 'BDD10k', 'IDD-AW', 'MapillaryVistas', 'OUTSIDE15k']

# Special restoration domains (defogging, dehazing, etc.)
RESTORATION_DOMAINS = ['defogged', 'dehazed', 'restored', 'clear']


def count_images(directory: Path) -> int:
    """Count image files in directory."""
    count = 0
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        count += len(list(directory.glob(f'**/{ext}')))
    return count


def detect_structure(gen_dir: Path) -> Tuple[str, Dict]:
    """
    Detect the directory structure type and return structure info.
    
    Returns:
        Tuple of (structure_type, structure_info)
        structure_type: 'flat_domain', 'domain_dataset', 'dataset_domain', 'dataset_flat', 'restoration'
    """
    subdirs = [d for d in gen_dir.iterdir() if d.is_dir()]
    subdir_names = [d.name for d in subdirs]
    
    # Check if subdirectories are domains or datasets
    has_domains = any(d in KNOWN_DOMAINS or d in RESTORATION_DOMAINS for d in subdir_names)
    has_datasets = any(d in KNOWN_DATASETS for d in subdir_names)
    
    if RESTORATION_DOMAINS[0] in subdir_names or any(d in RESTORATION_DOMAINS for d in subdir_names):
        # Restoration structure: domain/dataset/
        return 'restoration', {'restoration_domain': next((d for d in subdir_names if d in RESTORATION_DOMAINS), None)}
    
    if has_domains and not has_datasets:
        # Check if subdirs of domains are datasets
        for subdir in subdirs:
            if subdir.name in KNOWN_DOMAINS:
                sub_subdirs = [d.name for d in subdir.iterdir() if d.is_dir()]
                if any(d in KNOWN_DATASETS for d in sub_subdirs):
                    return 'domain_dataset', {}
                else:
                    # Could be flat domain structure
                    return 'flat_domain', {}
    
    if has_datasets and not has_domains:
        # Check if subdirs of datasets are domains
        for subdir in subdirs:
            if subdir.name in KNOWN_DATASETS:
                sub_subdirs = [d.name for d in subdir.iterdir() if d.is_dir()]
                if any(d in KNOWN_DOMAINS for d in sub_subdirs):
                    return 'dataset_domain', {}
                else:
                    # Dataset flat structure
                    return 'dataset_flat', {}
    
    if has_datasets and has_domains:
        # Mixed - could be transitional, treat as domain_dataset + dataset folders
        return 'mixed', {}
    
    return 'unknown', {}


def scan_directory(gen_dir: Path) -> Dict:
    """Scan a generated images directory and return manifest data."""
    method_name = gen_dir.name
    
    # Detect structure
    structure_type, structure_info = detect_structure(gen_dir)
    
    # Initialize manifest structure
    manifest = {
        "method": method_name,
        "generated_dir": str(gen_dir),
        "original_dir": ORIGINAL_DIR,
        "target_dir": TARGET_DIR,
        "manifest_path": str(gen_dir / "manifest.csv"),
        "structure_type": structure_type,
        "task_type": "restoration" if structure_type == 'restoration' else "generation",
        "total_generated": 0,
        "total_matched": 0,
        "total_unmatched": 0,
        "overall_match_rate": 100.0,
        "domains": {},
        "generated_at": datetime.now().isoformat(),
    }
    
    if structure_type == 'restoration':
        # Restoration methods like AOD-Net: restoration_domain/dataset/
        restoration_domain = structure_info.get('restoration_domain', 'defogged')
        restoration_dir = gen_dir / restoration_domain
        
        manifest["domains"][restoration_domain] = {
            "total": 0,
            "matched": 0,
            "unmatched": 0,
            "match_rate": 100.0,
            "is_restoration": True,
            "restoration_source_weather": "foggy",
            "datasets": {}
        }
        
        if restoration_dir.exists():
            for ds_dir in restoration_dir.iterdir():
                if ds_dir.is_dir() and ds_dir.name in KNOWN_DATASETS:
                    count = count_images(ds_dir)
                    manifest["domains"][restoration_domain]["datasets"][ds_dir.name] = {
                        "matched": count,
                        "unmatched": 0,
                        "total": count
                    }
                    manifest["domains"][restoration_domain]["total"] += count
                    manifest["domains"][restoration_domain]["matched"] += count
        
        manifest["total_generated"] = manifest["domains"][restoration_domain]["total"]
        manifest["total_matched"] = manifest["total_generated"]
        
    elif structure_type == 'domain_dataset' or structure_type == 'flat_domain':
        # domain/dataset/ structure
        for domain_dir in gen_dir.iterdir():
            if domain_dir.is_dir() and domain_dir.name in KNOWN_DOMAINS:
                domain_name = domain_dir.name
                manifest["domains"][domain_name] = {
                    "total": 0,
                    "matched": 0,
                    "unmatched": 0,
                    "match_rate": 100.0,
                    "is_restoration": False,
                    "restoration_source_weather": None,
                    "datasets": {}
                }
                
                for ds_dir in domain_dir.iterdir():
                    if ds_dir.is_dir():
                        ds_name = ds_dir.name
                        if ds_name in KNOWN_DATASETS:
                            count = count_images(ds_dir)
                            manifest["domains"][domain_name]["datasets"][ds_name] = {
                                "matched": count,
                                "unmatched": 0,
                                "total": count
                            }
                            manifest["domains"][domain_name]["total"] += count
                            manifest["domains"][domain_name]["matched"] += count
                
                manifest["total_generated"] += manifest["domains"][domain_name]["total"]
                manifest["total_matched"] += manifest["domains"][domain_name]["matched"]
    
    elif structure_type == 'dataset_domain':
        # dataset/domain/ structure - need to reorganize for manifest
        domain_data = defaultdict(lambda: {"total": 0, "matched": 0, "unmatched": 0, "datasets": {}})
        
        for ds_dir in gen_dir.iterdir():
            if ds_dir.is_dir() and ds_dir.name in KNOWN_DATASETS:
                ds_name = ds_dir.name
                for domain_dir in ds_dir.iterdir():
                    if domain_dir.is_dir() and domain_dir.name in KNOWN_DOMAINS:
                        domain_name = domain_dir.name
                        count = count_images(domain_dir)
                        
                        if ds_name not in domain_data[domain_name]["datasets"]:
                            domain_data[domain_name]["datasets"][ds_name] = {"matched": 0, "unmatched": 0, "total": 0}
                        
                        domain_data[domain_name]["datasets"][ds_name]["matched"] += count
                        domain_data[domain_name]["datasets"][ds_name]["total"] += count
                        domain_data[domain_name]["total"] += count
                        domain_data[domain_name]["matched"] += count
        
        for domain_name, domain_info in domain_data.items():
            manifest["domains"][domain_name] = {
                "total": domain_info["total"],
                "matched": domain_info["matched"],
                "unmatched": 0,
                "match_rate": 100.0,
                "is_restoration": False,
                "restoration_source_weather": None,
                "datasets": domain_info["datasets"]
            }
            manifest["total_generated"] += domain_info["total"]
            manifest["total_matched"] += domain_info["matched"]
    
    elif structure_type == 'dataset_flat':
        # dataset/ with images directly
        for ds_dir in gen_dir.iterdir():
            if ds_dir.is_dir() and ds_dir.name in KNOWN_DATASETS:
                ds_name = ds_dir.name
                count = count_images(ds_dir)
                
                # Put under a synthetic 'default' domain
                if 'default' not in manifest["domains"]:
                    manifest["domains"]["default"] = {
                        "total": 0,
                        "matched": 0,
                        "unmatched": 0,
                        "match_rate": 100.0,
                        "is_restoration": False,
                        "restoration_source_weather": None,
                        "datasets": {}
                    }
                
                manifest["domains"]["default"]["datasets"][ds_name] = {
                    "matched": count,
                    "unmatched": 0,
                    "total": count
                }
                manifest["domains"]["default"]["total"] += count
                manifest["domains"]["default"]["matched"] += count
                manifest["total_generated"] += count
                manifest["total_matched"] += count
    
    elif structure_type == 'mixed':
        # Handle mixed structure: some domain folders, some dataset folders
        domain_data = defaultdict(lambda: {"total": 0, "matched": 0, "unmatched": 0, "datasets": {}})
        
        for subdir in gen_dir.iterdir():
            if not subdir.is_dir():
                continue
            
            if subdir.name in KNOWN_DOMAINS:
                # Domain folder - scan for datasets
                domain_name = subdir.name
                for ds_dir in subdir.iterdir():
                    if ds_dir.is_dir() and ds_dir.name in KNOWN_DATASETS:
                        count = count_images(ds_dir)
                        if ds_dir.name not in domain_data[domain_name]["datasets"]:
                            domain_data[domain_name]["datasets"][ds_dir.name] = {"matched": 0, "unmatched": 0, "total": 0}
                        domain_data[domain_name]["datasets"][ds_dir.name]["matched"] += count
                        domain_data[domain_name]["datasets"][ds_dir.name]["total"] += count
                        domain_data[domain_name]["total"] += count
                        domain_data[domain_name]["matched"] += count
            
            elif subdir.name in KNOWN_DATASETS:
                # Dataset folder - scan for domains
                ds_name = subdir.name
                for domain_dir in subdir.iterdir():
                    if domain_dir.is_dir() and domain_dir.name in KNOWN_DOMAINS:
                        domain_name = domain_dir.name
                        count = count_images(domain_dir)
                        if ds_name not in domain_data[domain_name]["datasets"]:
                            domain_data[domain_name]["datasets"][ds_name] = {"matched": 0, "unmatched": 0, "total": 0}
                        domain_data[domain_name]["datasets"][ds_name]["matched"] += count
                        domain_data[domain_name]["datasets"][ds_name]["total"] += count
                        domain_data[domain_name]["total"] += count
                        domain_data[domain_name]["matched"] += count
        
        for domain_name, domain_info in domain_data.items():
            manifest["domains"][domain_name] = {
                "total": domain_info["total"],
                "matched": domain_info["matched"],
                "unmatched": 0,
                "match_rate": 100.0,
                "is_restoration": False,
                "restoration_source_weather": None,
                "datasets": domain_info["datasets"]
            }
            manifest["total_generated"] += domain_info["total"]
            manifest["total_matched"] += domain_info["matched"]
    
    # Calculate overall match rate
    if manifest["total_generated"] > 0:
        manifest["overall_match_rate"] = round((manifest["total_matched"] / manifest["total_generated"]) * 100, 2)
    
    return manifest


def write_manifest(gen_dir: Path, manifest: Dict, dry_run: bool = False) -> None:
    """Write manifest.json file."""
    manifest_path = gen_dir / "manifest.json"
    
    if dry_run:
        print(f"  Would write: {manifest_path}")
        print(f"  Total images: {manifest['total_generated']}")
        print(f"  Domains: {list(manifest['domains'].keys())}")
        return
    
    # Backup existing manifest
    if manifest_path.exists():
        backup_path = gen_dir / f"manifest.json.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.rename(manifest_path, backup_path)
        print(f"  Backed up existing manifest to {backup_path.name}")
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"  Created: {manifest_path}")
    print(f"  Total images: {manifest['total_generated']}")
    print(f"  Domains: {list(manifest['domains'].keys())}")


def check_manifests() -> Dict[str, Dict]:
    """Check which manifests need updating."""
    results = {
        'missing': [],
        'outdated': [],
        'ok': []
    }
    
    for gen_dir in sorted(GENERATED_IMAGES_BASE.iterdir()):
        if not gen_dir.is_dir():
            continue
        
        manifest_path = gen_dir / "manifest.json"
        actual_count = count_images(gen_dir)
        
        if not manifest_path.exists():
            results['missing'].append({
                'name': gen_dir.name,
                'actual_count': actual_count,
                'manifest_count': None
            })
        else:
            with open(manifest_path) as f:
                manifest = json.load(f)
            manifest_count = manifest.get('total_generated', 0)
            
            if actual_count != manifest_count:
                results['outdated'].append({
                    'name': gen_dir.name,
                    'actual_count': actual_count,
                    'manifest_count': manifest_count,
                    'diff': actual_count - manifest_count
                })
            else:
                results['ok'].append({
                    'name': gen_dir.name,
                    'actual_count': actual_count
                })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Generate/update manifest files for generated images')
    parser.add_argument('--all', action='store_true', help='Generate all missing/outdated manifests')
    parser.add_argument('--missing', action='store_true', help='Generate only missing manifests')
    parser.add_argument('--outdated', action='store_true', help='Update only outdated manifests')
    parser.add_argument('--dir', type=str, help='Generate manifest for specific directory')
    parser.add_argument('--check', action='store_true', help='Check which manifests need updating')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without writing')
    
    args = parser.parse_args()
    
    if args.check:
        print("Checking manifests...")
        print("=" * 70)
        results = check_manifests()
        
        if results['missing']:
            print(f"\nMissing manifests ({len(results['missing'])}):")
            for item in results['missing']:
                print(f"  {item['name']}: {item['actual_count']} images")
        
        if results['outdated']:
            print(f"\nOutdated manifests ({len(results['outdated'])}):")
            for item in results['outdated']:
                diff = f"+{item['diff']}" if item['diff'] > 0 else str(item['diff'])
                print(f"  {item['name']}: manifest={item['manifest_count']}, actual={item['actual_count']} ({diff})")
        
        if results['ok']:
            print(f"\nUp-to-date manifests ({len(results['ok'])}):")
            for item in results['ok']:
                print(f"  {item['name']}: {item['actual_count']} images")
        
        print(f"\nSummary: {len(results['missing'])} missing, {len(results['outdated'])} outdated, {len(results['ok'])} ok")
        return
    
    if args.dir:
        gen_dir = GENERATED_IMAGES_BASE / args.dir
        if not gen_dir.exists():
            print(f"Directory not found: {gen_dir}")
            return
        
        print(f"Generating manifest for {args.dir}...")
        manifest = scan_directory(gen_dir)
        write_manifest(gen_dir, manifest, dry_run=args.dry_run)
        return
    
    if args.all or args.missing or args.outdated:
        results = check_manifests()
        dirs_to_process = []
        
        if args.all or args.missing:
            dirs_to_process.extend([item['name'] for item in results['missing']])
        
        if args.all or args.outdated:
            dirs_to_process.extend([item['name'] for item in results['outdated']])
        
        if not dirs_to_process:
            print("No manifests need updating.")
            return
        
        print(f"Processing {len(dirs_to_process)} directories...")
        print("=" * 70)
        
        for dir_name in sorted(set(dirs_to_process)):
            gen_dir = GENERATED_IMAGES_BASE / dir_name
            print(f"\n{dir_name}:")
            manifest = scan_directory(gen_dir)
            write_manifest(gen_dir, manifest, dry_run=args.dry_run)
        
        print(f"\nProcessed {len(dirs_to_process)} directories.")
        return
    
    parser.print_help()


if __name__ == '__main__':
    main()
