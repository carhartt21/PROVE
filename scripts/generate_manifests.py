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
GENERATED_IMAGES_BASE = Path("${AWARE_DATA_ROOT}/GENERATED_IMAGES")
ORIGINAL_DIR = "${AWARE_DATA_ROOT}/FINAL_SPLITS/train/images"
TARGET_DIR = "${AWARE_DATA_ROOT}/AWACS/train"

# Known domains (standard weather domains)
KNOWN_DOMAINS = ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy']

# Alternative domain naming used by various generation methods
ALTERNATIVE_DOMAINS = [
    'fog',  # CUT, TSIT use 'fog' instead of 'foggy'
    # clear_day2X pattern (cycleGAN)
    'clear_day2cloudy', 'clear_day2dawn_dusk', 'clear_day2fog', 'clear_day2night', 'clear_day2rainy', 'clear_day2snowy',
    # clear_day_to_X pattern (SUSTechGAN)
    'clear_day_to_cloudy', 'clear_day_to_dawn_dusk', 'clear_day_to_foggy', 'clear_day_to_night', 'clear_day_to_rainy', 'clear_day_to_snowy',
    # sunny_day2X pattern (EDICT, CNetSeg)
    'sunny_day2cloudy', 'sunny_day2dawn_dusk', 'sunny_day2foggy', 'sunny_day2night', 'sunny_day2rainy', 'sunny_day2snowy',
]

# Combined set for efficient lookup
ALL_DOMAINS = set(KNOWN_DOMAINS + ALTERNATIVE_DOMAINS)

# Known datasets (includes Cityscapes naming variants across methods)
KNOWN_DATASETS = ['ACDC', 'BDD100k', 'BDD10k', 'IDD-AW', 'MapillaryVistas', 'OUTSIDE15k',
                  'Cityscapes', 'Cityscapes_from_lat']

# Restoration-style domains (used by AOD-Net style methods)
RESTORATION_DOMAINS = ['defogged', 'dehazed', 'restored', 'clear']

# Domain name normalization: maps non-standard on-disk names to standard names
# Used when physical dirs can't be renamed (permission issues)
DOMAIN_REMAP = {
    'fog': 'foggy',
    'rain': 'rainy',
    'snow': 'snowy',
    'clouds': 'cloudy',
    'clear_day2cloudy': 'cloudy',
    'clear_day2dawn_dusk': 'dawn_dusk',
    'clear_day2fog': 'foggy',
    'clear_day2night': 'night',
    'clear_day2rainy': 'rainy',
    'clear_day2snowy': 'snowy',
    'sunny_day2cloudy': 'cloudy',
    'sunny_day2dawn_dusk': 'dawn_dusk',
    'sunny_day2foggy': 'foggy',
    'sunny_day2night': 'night',
    'sunny_day2rainy': 'rainy',
    'sunny_day2snowy': 'snowy',
    'clear_day_to_cloudy': 'cloudy',
    'clear_day_to_dawn_dusk': 'dawn_dusk',
    'clear_day_to_foggy': 'foggy',
    'clear_day_to_night': 'night',
    'clear_day_to_rainy': 'rainy',
    'clear_day_to_snowy': 'snowy',
}

# Local manifest directory (for repo copies)
LOCAL_MANIFEST_DIR = Path(__file__).parent.parent / 'generated_manifests'


def count_images(directory: Path) -> int:
    """Count image files in directory."""
    count = 0
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        count += len(list(directory.glob(f'**/{ext}')))
    return count


def _find_datasets_deeper(parent_dir: Path, max_depth: int = 2) -> Optional[Dict]:
    """Look for known dataset directories up to max_depth levels deeper.
    
    Handles patterns like: domain/test_latest/images/{dataset}/
    Returns dict with 'path' (relative intermediate path) and 'datasets' found, or None.
    """
    def _search(current: Path, depth: int, rel_parts: list) -> Optional[Dict]:
        if depth > max_depth:
            return None
        try:
            children = [d for d in current.iterdir() if d.is_dir()]
        except PermissionError:
            return None
        child_names = [d.name for d in children]
        found_datasets = [n for n in child_names if n in KNOWN_DATASETS]
        if found_datasets:
            return {'path': '/'.join(rel_parts), 'datasets': found_datasets}
        # Recurse into non-dataset children
        for child in children:
            if child.name not in KNOWN_DATASETS and child.name not in ALL_DOMAINS:
                result = _search(child, depth + 1, rel_parts + [child.name])
                if result:
                    return result
        return None
    return _search(parent_dir, 0, [])


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
    has_domains = any(d in ALL_DOMAINS or d in RESTORATION_DOMAINS for d in subdir_names)
    has_datasets = any(d in KNOWN_DATASETS for d in subdir_names)
    
    if RESTORATION_DOMAINS[0] in subdir_names or any(d in RESTORATION_DOMAINS for d in subdir_names):
        # Restoration structure: domain/dataset/
        return 'restoration', {'restoration_domain': next((d for d in subdir_names if d in RESTORATION_DOMAINS), None)}
    
    if has_domains and not has_datasets:
        # Check if subdirs of domains are datasets
        for subdir in subdirs:
            if subdir.name in ALL_DOMAINS:
                try:
                    sub_subdirs = [d.name for d in subdir.iterdir() if d.is_dir()]
                except PermissionError:
                    continue
                if any(d in KNOWN_DATASETS for d in sub_subdirs):
                    return 'domain_dataset', {}
                else:
                    # Check for intermediate dirs like test_latest/images/ containing datasets
                    deeper_datasets = _find_datasets_deeper(subdir, max_depth=2)
                    if deeper_datasets:
                        return 'domain_dataset_nested', {'intermediate_path': deeper_datasets['path']}
                    # Truly flat domain structure
                    return 'flat_domain', {}
    
    if has_datasets and not has_domains:
        # Check if subdirs of datasets are domains
        for subdir in subdirs:
            if subdir.name in KNOWN_DATASETS:
                try:
                    sub_subdirs = [d.name for d in subdir.iterdir() if d.is_dir()]
                except PermissionError:
                    continue
                if any(d in ALL_DOMAINS for d in sub_subdirs):
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
        
    elif structure_type in ('domain_dataset', 'domain_dataset_nested', 'flat_domain'):
        # domain/dataset/ structure (or flat domain with images directly in domain dirs)
        # For nested: domain/test_latest/images/dataset/
        intermediate_path = structure_info.get('intermediate_path', '')
        
        for domain_dir in gen_dir.iterdir():
            if domain_dir.is_dir() and domain_dir.name in ALL_DOMAINS:
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
                
                # Determine the actual parent of dataset dirs
                if intermediate_path:
                    scan_dir = domain_dir / intermediate_path
                else:
                    scan_dir = domain_dir
                
                dataset_counted = 0
                if scan_dir.exists():
                    try:
                        for ds_dir in scan_dir.iterdir():
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
                                    dataset_counted += count
                    except PermissionError:
                        print(f"  WARNING: Permission denied scanning {scan_dir}")
                
                # Also check for datasets directly under domain_dir (no intermediate)
                if intermediate_path:
                    try:
                        for ds_dir in domain_dir.iterdir():
                            if ds_dir.is_dir() and ds_dir.name in KNOWN_DATASETS:
                                ds_name = ds_dir.name
                                if ds_name not in manifest["domains"][domain_name]["datasets"]:
                                    count = count_images(ds_dir)
                                    manifest["domains"][domain_name]["datasets"][ds_name] = {
                                        "matched": count,
                                        "unmatched": 0,
                                        "total": count
                                    }
                                    manifest["domains"][domain_name]["total"] += count
                                    manifest["domains"][domain_name]["matched"] += count
                                    dataset_counted += count
                    except PermissionError:
                        pass
                
                # For flat_domain: count images directly in domain dir (not in dataset subdirs)
                try:
                    total_domain = count_images(domain_dir)
                except PermissionError:
                    total_domain = dataset_counted
                remaining = total_domain - dataset_counted
                if remaining > 0:
                    manifest["domains"][domain_name]["datasets"]["flat"] = {
                        "matched": remaining,
                        "unmatched": 0,
                        "total": remaining
                    }
                    manifest["domains"][domain_name]["total"] += remaining
                    manifest["domains"][domain_name]["matched"] += remaining
                
                manifest["total_generated"] += manifest["domains"][domain_name]["total"]
                manifest["total_matched"] += manifest["domains"][domain_name]["matched"]
    
    elif structure_type == 'dataset_domain':
        # dataset/domain/ structure - need to reorganize for manifest
        domain_data = defaultdict(lambda: {"total": 0, "matched": 0, "unmatched": 0, "datasets": {}})
        
        for ds_dir in gen_dir.iterdir():
            if ds_dir.is_dir() and ds_dir.name in KNOWN_DATASETS:
                ds_name = ds_dir.name
                domain_counted = 0
                try:
                    for domain_dir in ds_dir.iterdir():
                        if domain_dir.is_dir() and domain_dir.name in ALL_DOMAINS:
                            domain_name = domain_dir.name
                            count = count_images(domain_dir)
                            
                            if ds_name not in domain_data[domain_name]["datasets"]:
                                domain_data[domain_name]["datasets"][ds_name] = {"matched": 0, "unmatched": 0, "total": 0}
                            
                            domain_data[domain_name]["datasets"][ds_name]["matched"] += count
                            domain_data[domain_name]["datasets"][ds_name]["total"] += count
                            domain_data[domain_name]["total"] += count
                            domain_data[domain_name]["matched"] += count
                            domain_counted += count
                except PermissionError:
                    print(f"  WARNING: Permission denied accessing {ds_dir}")
                    continue
                
                # Handle datasets with non-standard subdirectory naming (e.g., Cityscapes transforms)
                try:
                    total_ds = count_images(ds_dir)
                except PermissionError:
                    total_ds = domain_counted
                remaining = total_ds - domain_counted
                if remaining > 0:
                    fallback_domain = 'other_transforms'
                    if ds_name not in domain_data[fallback_domain]["datasets"]:
                        domain_data[fallback_domain]["datasets"][ds_name] = {"matched": 0, "unmatched": 0, "total": 0}
                    domain_data[fallback_domain]["datasets"][ds_name]["matched"] += remaining
                    domain_data[fallback_domain]["datasets"][ds_name]["total"] += remaining
                    domain_data[fallback_domain]["total"] += remaining
                    domain_data[fallback_domain]["matched"] += remaining
        
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
            
            if subdir.name in ALL_DOMAINS:
                # Domain folder - scan for datasets
                domain_name = subdir.name
                dataset_counted = 0
                for ds_dir in subdir.iterdir():
                    if ds_dir.is_dir() and ds_dir.name in KNOWN_DATASETS:
                        count = count_images(ds_dir)
                        if ds_dir.name not in domain_data[domain_name]["datasets"]:
                            domain_data[domain_name]["datasets"][ds_dir.name] = {"matched": 0, "unmatched": 0, "total": 0}
                        domain_data[domain_name]["datasets"][ds_dir.name]["matched"] += count
                        domain_data[domain_name]["datasets"][ds_dir.name]["total"] += count
                        domain_data[domain_name]["total"] += count
                        domain_data[domain_name]["matched"] += count
                        dataset_counted += count
                
                # Flat domain fallback: count images not in dataset subdirs
                try:
                    total_domain = count_images(subdir)
                except PermissionError:
                    total_domain = dataset_counted
                remaining = total_domain - dataset_counted
                if remaining > 0:
                    if "flat" not in domain_data[domain_name]["datasets"]:
                        domain_data[domain_name]["datasets"]["flat"] = {"matched": 0, "unmatched": 0, "total": 0}
                    domain_data[domain_name]["datasets"]["flat"]["matched"] += remaining
                    domain_data[domain_name]["datasets"]["flat"]["total"] += remaining
                    domain_data[domain_name]["total"] += remaining
                    domain_data[domain_name]["matched"] += remaining
            
            elif subdir.name in KNOWN_DATASETS:
                # Dataset folder - scan for domains
                ds_name = subdir.name
                domain_counted = 0
                try:
                    for domain_dir in subdir.iterdir():
                        if domain_dir.is_dir() and domain_dir.name in ALL_DOMAINS:
                            domain_name = domain_dir.name
                            count = count_images(domain_dir)
                            if ds_name not in domain_data[domain_name]["datasets"]:
                                domain_data[domain_name]["datasets"][ds_name] = {"matched": 0, "unmatched": 0, "total": 0}
                            domain_data[domain_name]["datasets"][ds_name]["matched"] += count
                            domain_data[domain_name]["datasets"][ds_name]["total"] += count
                            domain_data[domain_name]["total"] += count
                            domain_data[domain_name]["matched"] += count
                            domain_counted += count
                except PermissionError:
                    print(f"  WARNING: Permission denied accessing {subdir}")
                    continue
                
                # Handle datasets with non-standard subdirectory naming
                try:
                    total_ds = count_images(subdir)
                except PermissionError:
                    total_ds = domain_counted
                remaining = total_ds - domain_counted
                if remaining > 0:
                    fallback_domain = 'other_transforms'
                    if ds_name not in domain_data[fallback_domain]["datasets"]:
                        domain_data[fallback_domain]["datasets"][ds_name] = {"matched": 0, "unmatched": 0, "total": 0}
                    domain_data[fallback_domain]["datasets"][ds_name]["matched"] += remaining
                    domain_data[fallback_domain]["datasets"][ds_name]["total"] += remaining
                    domain_data[fallback_domain]["total"] += remaining
                    domain_data[fallback_domain]["matched"] += remaining
        
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
    
    elif structure_type == 'unknown':
        # Fallback: count all images recursively under the entire directory
        try:
            total = count_images(gen_dir)
        except PermissionError:
            total = 0
            print(f"  WARNING: Permission denied counting images in {gen_dir}")
        
        if total > 0:
            manifest["domains"]["all"] = {
                "total": total,
                "matched": total,
                "unmatched": 0,
                "match_rate": 100.0,
                "is_restoration": False,
                "restoration_source_weather": None,
                "datasets": {"unknown": {"matched": total, "unmatched": 0, "total": total}}
            }
            manifest["total_generated"] = total
            manifest["total_matched"] = total
    
    # Calculate overall match rate
    if manifest["total_generated"] > 0:
        manifest["overall_match_rate"] = round((manifest["total_matched"] / manifest["total_generated"]) * 100, 2)
    
    return manifest


def normalize_domain_names(manifest: Dict) -> Dict:
    """Normalize domain names in manifest using DOMAIN_REMAP.
    
    Merges data from non-standard domain names into their standard equivalents.
    For example, 'fog' data gets merged into 'foggy', 'sunny_day2cloudy' into 'cloudy'.
    """
    if not manifest.get('domains'):
        return manifest
    
    domains = manifest['domains']
    remapped = {}
    
    for old_name, domain_data in domains.items():
        new_name = DOMAIN_REMAP.get(old_name, old_name)
        
        if new_name in remapped:
            # Merge into existing domain
            existing = remapped[new_name]
            existing['total'] += domain_data.get('total', 0)
            existing['matched'] += domain_data.get('matched', 0)
            existing['unmatched'] += domain_data.get('unmatched', 0)
            # Merge datasets
            for ds_name, ds_data in domain_data.get('datasets', {}).items():
                if ds_name in existing['datasets']:
                    existing['datasets'][ds_name]['total'] += ds_data.get('total', 0)
                    existing['datasets'][ds_name]['matched'] += ds_data.get('matched', 0)
                    existing['datasets'][ds_name]['unmatched'] += ds_data.get('unmatched', 0)
                else:
                    existing['datasets'][ds_name] = dict(ds_data)
        else:
            remapped[new_name] = dict(domain_data)
            if 'datasets' in domain_data:
                remapped[new_name]['datasets'] = {k: dict(v) for k, v in domain_data['datasets'].items()}
    
    manifest['domains'] = remapped
    return manifest


def write_manifest(gen_dir: Path, manifest: Dict, dry_run: bool = False, save_local: bool = True) -> None:
    """Write manifest.json file and optionally save a local copy."""
    # Normalize domain names before writing
    manifest = normalize_domain_names(manifest)
    
    manifest_path = gen_dir / "manifest.json"
    method_name = manifest.get('method', gen_dir.name)
    
    if dry_run:
        print(f"  Would write: {manifest_path}")
        print(f"  Total images: {manifest['total_generated']}")
        print(f"  Domains: {list(manifest['domains'].keys())}")
        if save_local:
            print(f"  Would save local copy: {LOCAL_MANIFEST_DIR / f'{method_name}_manifest.json'}")
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
    
    # Save local copy to generated_manifests/
    if save_local:
        LOCAL_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        local_path = LOCAL_MANIFEST_DIR / f"{method_name}_manifest.json"
        with open(local_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"  Local copy: {local_path}")


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
        try:
            actual_count = count_images(gen_dir)
        except PermissionError:
            print(f"  WARNING: Permission denied for {gen_dir.name}, skipping")
            continue
        
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
    parser.add_argument('--force-all', action='store_true', help='Regenerate ALL manifests regardless of status')
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
    
    if args.force_all:
        # Regenerate ALL manifests regardless of current status
        dirs_to_process = []
        for gen_dir in sorted(GENERATED_IMAGES_BASE.iterdir()):
            if gen_dir.is_dir():
                dirs_to_process.append(gen_dir.name)
        
        if not dirs_to_process:
            print("No directories found.")
            return
        
        print(f"Force-regenerating ALL {len(dirs_to_process)} manifests...")
        print("=" * 70)
        
        for dir_name in dirs_to_process:
            gen_dir = GENERATED_IMAGES_BASE / dir_name
            print(f"\n{dir_name}:")
            try:
                manifest = scan_directory(gen_dir)
                write_manifest(gen_dir, manifest, dry_run=args.dry_run)
            except PermissionError:
                print(f"  SKIPPED: Permission denied")
            except Exception as e:
                print(f"  ERROR: {e}")
        
        print(f"\nProcessed {len(dirs_to_process)} directories.")
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
