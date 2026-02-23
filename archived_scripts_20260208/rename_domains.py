#!/usr/bin/env python3
"""Rename domain directories to use consistent standard names.

Standard domain names: clear_day, cloudy, dawn_dusk, foggy, night, rainy, snowy

This script handles:
- Group 1: Top-level domain dir renames (CUT, cycleGAN, EDICT, SUSTechGAN, stargan_v2)
- Group 2: Nested domain dir renames under datasets (CNetSeg, Img2Img, IP2P, UniControl)
- Group 3: Classical augmentation methods (albumentations_weather, augmenters, automold,
           Weather_Effect_Generator) - reports permission issues

Usage:
    python scripts/rename_domains.py --dry-run       # Preview all renames
    python scripts/rename_domains.py                  # Execute renames
    python scripts/rename_domains.py --group 1        # Only top-level domain dirs
    python scripts/rename_domains.py --group 2        # Only nested under datasets
    python scripts/rename_domains.py --group 3        # Only classical augmentation
"""

import argparse
import os
import sys
from pathlib import Path

BASE = Path('${AWARE_DATA_ROOT}/GENERATED_IMAGES')

# Mapping: old_name -> new_name
# For prefix patterns, we strip the prefix and normalize fog->foggy
DOMAIN_REMAP = {
    # Simple renames
    'fog': 'foggy',
    'rain': 'rainy',
    'snow': 'snowy',
    'clouds': 'cloudy',
    # clear_day2X pattern (cycleGAN)
    'clear_day2cloudy': 'cloudy',
    'clear_day2dawn_dusk': 'dawn_dusk',
    'clear_day2fog': 'foggy',
    'clear_day2night': 'night',
    'clear_day2rainy': 'rainy',
    'clear_day2snowy': 'snowy',
    # sunny_day2X pattern (CNetSeg, EDICT, Img2Img, IP2P, UniControl)
    'sunny_day2cloudy': 'cloudy',
    'sunny_day2dawn_dusk': 'dawn_dusk',
    'sunny_day2foggy': 'foggy',
    'sunny_day2night': 'night',
    'sunny_day2rainy': 'rainy',
    'sunny_day2snowy': 'snowy',
    # clear_day_to_X pattern (SUSTechGAN)
    'clear_day_to_cloudy': 'cloudy',
    'clear_day_to_dawn_dusk': 'dawn_dusk',
    'clear_day_to_foggy': 'foggy',
    'clear_day_to_night': 'night',
    'clear_day_to_rainy': 'rainy',
    'clear_day_to_snowy': 'snowy',
}

# Group 1: Methods with domain dirs at top level
GROUP1_METHODS = {
    'CUT': ['fog'],  # fog -> foggy
    'cycleGAN': [
        'clear_day2cloudy', 'clear_day2dawn_dusk', 'clear_day2fog',
        'clear_day2night', 'clear_day2rainy', 'clear_day2snowy',
    ],
    'EDICT': [
        'sunny_day2cloudy', 'sunny_day2dawn_dusk', 'sunny_day2foggy',
        'sunny_day2night', 'sunny_day2rainy', 'sunny_day2snowy',
    ],
    'SUSTechGAN': [
        'clear_day_to_cloudy', 'clear_day_to_dawn_dusk', 'clear_day_to_foggy',
        'clear_day_to_night', 'clear_day_to_rainy', 'clear_day_to_snowy',
    ],
    'stargan_v2': ['fog'],  # fog -> foggy
}

# Group 2: Methods with dataset dirs at top, domain dirs nested underneath
GROUP2_METHODS = ['CNetSeg', 'Img2Img', 'IP2P', 'UniControl']
GROUP2_DOMAINS = [
    'sunny_day2cloudy', 'sunny_day2dawn_dusk', 'sunny_day2foggy',
    'sunny_day2night', 'sunny_day2rainy', 'sunny_day2snowy',
]

KNOWN_DATASETS = ['ACDC', 'BDD100k', 'BDD10k', 'IDD-AW', 'MapillaryVistas', 'OUTSIDE15k', 'Cityscapes']

# Group 3: Classical augmentation methods (dataset dirs at top, domain dirs nested)
# These need write permission on the dataset dirs (owned by chge7185: drwxr-xr-x)
GROUP3_METHODS = {
    'albumentations_weather': {
        # Cityscapes has 'fog' (others have 'foggy')
        # All datasets: rain->rainy, snow->snowy
        'all_datasets': ['rain', 'snow'],
        'Cityscapes': ['fog'],
    },
    'augmenters': {
        'all_datasets': ['fog', 'rain', 'snow', 'clouds'],
        # snow_no_flakes is left as-is (no clean mapping)
    },
    'automold': {
        # Only Cityscapes has non-standard names (fog, rain, snow)
        # Other datasets already use foggy, rainy, snowy
        # Cityscapes also has sub-variants: fog_heavy, fog_light, rain_drizzle, etc. - left as-is
        'Cityscapes': ['fog', 'rain', 'snow'],
    },
    'Weather_Effect_Generator': {
        # Cityscapes uses fog, rain, snow; others use foggy, rainy, snowy
        'Cityscapes': ['fog', 'rain', 'snow'],
    },
}


def check_writable(parent_dir: Path) -> bool:
    """Check if we can rename dirs inside parent_dir."""
    return os.access(parent_dir, os.W_OK)


def plan_renames(groups=None):
    """Build list of (old_path, new_path) rename operations."""
    renames = []
    skipped = []

    if groups is None:
        groups = [1, 2, 3]

    # Group 1: Top-level domain dirs
    if 1 in groups:
        for method, domains in GROUP1_METHODS.items():
            method_dir = BASE / method
            if not method_dir.exists():
                print(f"  SKIP: {method}/ does not exist")
                continue
            for old_domain in domains:
                new_domain = DOMAIN_REMAP[old_domain]
                old_path = method_dir / old_domain
                new_path = method_dir / new_domain
                if not old_path.exists():
                    continue
                if new_path.exists():
                    print(f"  CONFLICT: {method}/{new_domain} already exists! Skipping {old_domain}")
                    skipped.append((str(old_path), f"target {new_domain} already exists"))
                    continue
                if not check_writable(method_dir):
                    skipped.append((str(old_path), "no write permission on parent"))
                    continue
                renames.append((old_path, new_path))

    # Group 2: Nested domain dirs
    if 2 in groups:
        for method in GROUP2_METHODS:
            method_dir = BASE / method
            if not method_dir.exists():
                continue
            for dataset in KNOWN_DATASETS:
                dataset_dir = method_dir / dataset
                if not dataset_dir.exists():
                    continue
                if not check_writable(dataset_dir):
                    skipped.append((str(dataset_dir), "no write permission"))
                    continue
                for old_domain in GROUP2_DOMAINS:
                    new_domain = DOMAIN_REMAP[old_domain]
                    old_path = dataset_dir / old_domain
                    new_path = dataset_dir / new_domain
                    if not old_path.exists():
                        continue
                    if new_path.exists():
                        skipped.append((str(old_path), f"target {new_domain} already exists"))
                        continue
                    renames.append((old_path, new_path))

    # Group 3: Classical augmentation methods
    if 3 in groups:
        for method, spec in GROUP3_METHODS.items():
            method_dir = BASE / method
            if not method_dir.exists():
                continue

            # Handle 'all_datasets' key
            all_ds_domains = spec.get('all_datasets', [])
            per_ds_domains = {k: v for k, v in spec.items() if k != 'all_datasets'}

            datasets_to_process = set()
            if all_ds_domains:
                datasets_to_process = set(KNOWN_DATASETS)
            datasets_to_process.update(per_ds_domains.keys())

            for dataset in datasets_to_process:
                dataset_dir = method_dir / dataset
                if not dataset_dir.exists():
                    continue

                # Collect domains to rename for this dataset
                domains = list(all_ds_domains) + per_ds_domains.get(dataset, [])
                domains = list(set(domains))  # deduplicate

                if not check_writable(dataset_dir):
                    for d in domains:
                        old_path = dataset_dir / d
                        if old_path.exists():
                            skipped.append((str(old_path), "no write permission on dataset dir"))
                    continue

                for old_domain in domains:
                    new_domain = DOMAIN_REMAP.get(old_domain)
                    if not new_domain:
                        continue
                    old_path = dataset_dir / old_domain
                    new_path = dataset_dir / new_domain
                    if not old_path.exists():
                        continue
                    if new_path.exists():
                        skipped.append((str(old_path), f"target {new_domain} already exists"))
                        continue
                    renames.append((old_path, new_path))

    return renames, skipped


def main():
    parser = argparse.ArgumentParser(description='Rename domain directories to standard names')
    parser.add_argument('--dry-run', action='store_true', help='Preview renames without executing')
    parser.add_argument('--group', type=int, nargs='+', choices=[1, 2, 3],
                        help='Only process specific groups (1=top-level, 2=nested, 3=classical)')
    args = parser.parse_args()

    groups = args.group if args.group else [1, 2, 3]

    print(f"Planning renames for groups: {groups}")
    print(f"Base: {BASE}\n")

    renames, skipped = plan_renames(groups)

    # Print summary
    if renames:
        print(f"{'DRY RUN - ' if args.dry_run else ''}Renames to execute ({len(renames)}):")
        # Group by method for readability
        current_method = None
        for old_path, new_path in renames:
            method = old_path.relative_to(BASE).parts[0]
            if method != current_method:
                current_method = method
                print(f"\n  {method}/")
            rel_old = old_path.relative_to(BASE / method)
            rel_new = new_path.relative_to(BASE / method)
            print(f"    {rel_old} → {rel_new}")
    else:
        print("No renames needed.")

    if skipped:
        print(f"\nSkipped ({len(skipped)}):")
        for path, reason in skipped:
            rel = Path(path).relative_to(BASE) if Path(path).is_relative_to(BASE) else path
            print(f"  {rel}: {reason}")

    if args.dry_run or not renames:
        return

    # Execute renames
    print(f"\nExecuting {len(renames)} renames...")
    success = 0
    failed = 0
    for old_path, new_path in renames:
        try:
            old_path.rename(new_path)
            success += 1
        except PermissionError:
            print(f"  PERMISSION DENIED: {old_path.relative_to(BASE)}")
            failed += 1
        except OSError as e:
            print(f"  ERROR: {old_path.relative_to(BASE)}: {e}")
            failed += 1

    print(f"\nDone: {success} renamed, {failed} failed")

    if success > 0:
        print("\nRemember to regenerate manifests for affected methods!")
        methods = set()
        for old_path, _ in renames:
            methods.add(old_path.relative_to(BASE).parts[0])
        dirs_arg = ' '.join(f'--dir {m}' for m in sorted(methods))
        print(f"  python scripts/generate_manifests.py {dirs_arg}")


if __name__ == '__main__':
    main()
