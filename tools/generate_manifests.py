#!/usr/bin/env python3
"""
Standalone manifest generation script for PROVE.

This script creates CSV manifests that map generated images to their
original counterparts, enabling training with generative augmentation.

Usage:
    # Check manifest status
    python tools/generate_manifests.py --status
    
    # Generate manifests for all methods missing them
    python tools/generate_manifests.py --all-missing
    
    # Generate manifest for a specific method
    python tools/generate_manifests.py --method cycleGAN
    
    # Dry run (show what would be done)
    python tools/generate_manifests.py --all-missing --dry-run

Outputs per method directory:
    - manifest.csv: Paired generated and original images
    - manifest.json: Summary statistics and metadata
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

# =============================================================================
# Constants and Configuration
# =============================================================================

CANONICAL_GENERATION_DOMAINS = {"snowy", "rainy", "foggy", "night", "cloudy", "dawn_dusk", "clear_day"}
CANONICAL_RESTORATION_DOMAINS = {"derained", "dehazed", "desnowed", "night2day"}
RESTORATION_SOURCE_MAPPING = {
    "derained": "rainy",
    "dehazed": "foggy",
    "desnowed": "snowy",
    "night2day": "night",
}

# Common domain name mappings
DOMAIN_MAPPING = {
    "fog": "foggy",
    "foggy": "foggy",
    "rain": "rainy",
    "rainy": "rainy",
    "snow": "snowy",
    "snowy": "snowy",
    "sunny": "clear_day",
    "sunny_day": "clear_day",
    "clear": "clear_day",
    "clear_day": "clear_day",
    "overcast": "cloudy",
    "cloudy": "cloudy",
    "dusk": "dawn_dusk",
    "dawn": "dawn_dusk",
    "dawn_dusk": "dawn_dusk",
    "night": "night",
    # Automold-specific domain names
    "bright": "clear_day",
    "dark": "night",
    "fog_heavy": "foggy",
    "fog_light": "foggy",
    "gravel": "clear_day",  # road condition, map to clear_day
    "rain_drizzle": "rainy",
    "rain_heavy": "rainy",
    "rain_torrential": "rainy",
    "shadow": "clear_day",  # lighting condition, map to clear_day
    "snow_heavy": "snowy",
    "sun_flare": "clear_day",  # lighting condition, map to clear_day
    # Augmenters-specific domain names
    "clouds": "cloudy",
    "snow_no_flakes": "snowy",
}

KNOWN_DATASETS = {'ACDC', 'BDD100k', 'BDD10k', 'IDD-AW', 'MapillaryVistas', 'OUTSIDE15k', 'Cityscapes', 'CITYSCAPES', 'Cityscapes_2'}
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp'}

# Default paths (from environment or hardcoded)
DEFAULT_GENERATED_BASE = Path(os.environ.get(
    'PROVE_GEN_ROOT', 
    '${AWARE_DATA_ROOT}/GENERATED_IMAGES'
))
DEFAULT_ORIGINAL_DIR = Path(os.environ.get(
    'PROVE_DATA_ROOT',
    '${AWARE_DATA_ROOT}/FINAL_SPLITS'
)) / 'train' / 'images'

# Additional original image directories for datasets not in FINAL_SPLITS
ADDITIONAL_ORIGINAL_DIRS = [
    Path('${AWARE_DATA_ROOT}/CITYSCAPES/leftImg8bit/train'),
]

# =============================================================================
# Helper Functions
# =============================================================================

def find_image_files(directory: Path, recursive: bool = True, follow_symlinks: bool = True) -> List[Path]:
    """Find all supported image files in a directory.
    
    Args:
        directory: Directory to search in
        recursive: Whether to search recursively in subdirectories
        follow_symlinks: Whether to follow symbolic links (default True)
    """
    image_files = []
    try:
        if recursive and follow_symlinks:
            # os.walk follows symlinks by default, unlike pathlib.rglob
            for root, dirs, files in os.walk(directory, followlinks=True):
                root_path = Path(root)
                for f in files:
                    if any(f.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                        image_files.append(root_path / f)
        else:
            glob_func = directory.rglob if recursive else directory.glob
            for ext in SUPPORTED_EXTENSIONS:
                image_files.extend(glob_func(f"*{ext}"))
                image_files.extend(glob_func(f"*{ext.upper()}"))
    except PermissionError:
        pass
    return sorted(image_files)


def normalize_filename(filename: str) -> str:
    """
    Normalize a filename by removing common suffixes and prefixes.
    
    Handles:
    - Dataset prefixes (e.g., 'ACDC_image.png' -> 'image.png')
    - Generation suffixes (_fake, _translated, _output, _gen, _generated)
    - Style transfer suffixes (_lat, _ref, _stylized, _styled)
    - Weather effect suffixes (-fsyn, -rsyn, -ssyn for fog/rain/snow synthesis)
    - NST pattern (_sa_<number>)
    """
    stem = Path(filename).stem
    
    # Remove dataset prefix if present
    for dataset in KNOWN_DATASETS:
        if stem.startswith(dataset + '_'):
            stem = stem[len(dataset) + 1:]
            break
    
    # NST style: ends with _sa_<number>
    nst_pattern = re.compile(r'_sa_\d+$')
    stem = nst_pattern.sub('', stem)
    
    # Weather Effect Generator patterns: -fsyn, -rsyn, -ssyn
    weather_effect_pattern = re.compile(r'-[frs]syn$')
    stem = weather_effect_pattern.sub('', stem)
    
    # style0 suffix from tunit and similar methods
    style_pattern = re.compile(r'_style\d+$')
    stem = style_pattern.sub('', stem)
    
    # Remove common generation suffixes (order matters - check longer suffixes first)
    suffixes_to_remove = [
        '_fake', '_translated', '_output', '_gen', '_generated',
        '_lat', '_ref', '_stylized', '_styled',
    ]
    
    # Keep removing suffixes until none match
    changed = True
    while changed:
        changed = False
        for suffix in suffixes_to_remove:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                changed = True
                break
    
    return stem


def normalize_domain(domain_name: str) -> Optional[str]:
    """
    Map a domain name to its canonical form.
    
    Returns None if the domain is not recognized.
    """
    # Direct lookup
    if domain_name in DOMAIN_MAPPING:
        return DOMAIN_MAPPING[domain_name]
    
    # Try lowercase
    lower = domain_name.lower()
    if lower in DOMAIN_MAPPING:
        return DOMAIN_MAPPING[lower]
    
    # Try extracting target from translation patterns
    # Handle both "source2target" and "source_to_target" formats
    target = None
    if '2' in domain_name:
        parts = domain_name.split('2', 1)
        if len(parts) > 1:
            target = parts[1]
    elif '_to_' in domain_name:
        parts = domain_name.split('_to_', 1)
        if len(parts) > 1:
            target = parts[1]
    
    if target:
        if target in DOMAIN_MAPPING:
            return DOMAIN_MAPPING[target]
        if target.lower() in DOMAIN_MAPPING:
            return DOMAIN_MAPPING[target.lower()]
        # Check if target is already canonical
        all_canonical = CANONICAL_GENERATION_DOMAINS | CANONICAL_RESTORATION_DOMAINS
        if target in all_canonical:
            return target
        if target.lower() in all_canonical:
            return target.lower()
    
    # Check if it's already a canonical domain
    all_canonical = CANONICAL_GENERATION_DOMAINS | CANONICAL_RESTORATION_DOMAINS
    if domain_name in all_canonical:
        return domain_name
    if domain_name.lower() in all_canonical:
        return domain_name.lower()
    
    return None


def extract_source_domain(domain_name: str) -> Optional[str]:
    """Extract source domain from translation folder name (e.g., 'clear_day2cloudy' -> 'clear_day')."""
    if '2' in domain_name:
        parts = domain_name.split('2', 1)
        if parts[0]:
            return parts[0]
    elif '_to_' in domain_name:
        parts = domain_name.split('_to_', 1)
        if parts[0]:
            return parts[0]
    return None


def extract_target_domain(domain_name: str) -> str:
    """Extract target domain from translation folder name (e.g., 'clear_day2cloudy' -> 'cloudy')."""
    if '2' in domain_name:
        parts = domain_name.split('2', 1)
        if len(parts) > 1 and parts[1]:
            return parts[1]
    elif '_to_' in domain_name:
        parts = domain_name.split('_to_', 1)
        if len(parts) > 1 and parts[1]:
            return parts[1]
    return domain_name


def is_restoration_domain(canonical_domain: str) -> bool:
    """Check if a canonical domain is a restoration task."""
    return canonical_domain in CANONICAL_RESTORATION_DOMAINS


def get_restoration_source_domain(canonical_domain: str) -> Optional[str]:
    """Get the source weather domain for a restoration task."""
    return RESTORATION_SOURCE_MAPPING.get(canonical_domain)


def detect_directory_structure(method_dir: Path) -> str:
    """
    Detect the directory structure type for a method.
    
    Returns one of:
    - 'domain_dataset': domain/dataset hierarchy (e.g., foggy/ACDC/)
    - 'dataset_domain': dataset/domain hierarchy (e.g., ACDC/foggy/)
    - 'flat_domain': domain folders with images directly in them
    - 'flat_dataset': dataset folders with images directly in them
    - 'unknown': could not determine structure
    """
    try:
        subdirs = [d for d in method_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    except PermissionError:
        return 'unknown'
    
    if not subdirs:
        return 'unknown'
    
    # Try multiple subdirs - first recognized one determines structure
    # (skip unrecognized dirs like Cityscapes_from_lat)
    first_subdir = None
    first_name = None
    for sd in subdirs:
        if sd.name in KNOWN_DATASETS or normalize_domain(sd.name) is not None:
            first_subdir = sd
            first_name = sd.name
            break
    
    if first_subdir is None:
        # No recognized subdirs - try the first one anyway
        first_subdir = subdirs[0]
        first_name = first_subdir.name
    
    # Check if first level is datasets
    if first_name in KNOWN_DATASETS:
        try:
            second_level = [d for d in first_subdir.iterdir() if d.is_dir()]
        except PermissionError:
            return 'flat_dataset'
        
        if second_level:
            second_name = second_level[0].name
            if normalize_domain(second_name) is not None:
                return 'dataset_domain'
            elif second_name in KNOWN_DATASETS:
                return 'flat_dataset'
            else:
                images = find_image_files(first_subdir, recursive=False)
                if images:
                    return 'flat_dataset'
        return 'flat_dataset'
    
    # Check if first level is domains
    normalized = normalize_domain(first_name)
    if normalized is not None:
        try:
            second_level = [d for d in first_subdir.iterdir() if d.is_dir()]
        except PermissionError:
            return 'flat_domain'
        
        if second_level:
            second_name = second_level[0].name
            if second_name in KNOWN_DATASETS:
                return 'domain_dataset'
            else:
                for nested in second_level:
                    if nested.name in KNOWN_DATASETS:
                        return 'domain_dataset'
                    try:
                        images_dir = nested / "images"
                        if images_dir.exists():
                            return 'flat_domain'
                    except PermissionError:
                        continue
        
        images = find_image_files(first_subdir, recursive=False)
        if images:
            return 'flat_domain'
        
        return 'flat_domain'
    
    return 'unknown'


def get_dataset_from_filename(filename: str) -> Optional[str]:
    """Try to determine dataset from filename patterns."""
    stem = Path(filename).stem
    
    # Check for dataset prefix
    for dataset in KNOWN_DATASETS:
        if stem.startswith(dataset + '_'):
            return dataset
    
    # ACDC pattern: GOPR or GP\d+ prefix
    if re.match(r'^(GOPR|GP)\d+', stem):
        return 'ACDC'
    
    # BDD pattern: UUID-like with dashes
    if re.match(r'^[a-f0-9]{8}-[a-f0-9]{4}', stem):
        return 'BDD100k'  # or BDD10k
    
    # Mapillary pattern: long alphanumeric IDs
    if re.match(r'^[A-Za-z0-9_-]{20,}$', stem):
        return 'MapillaryVistas'
    
    return None


# =============================================================================
# Image Entry and Index Building
# =============================================================================

@dataclass
class ImageEntry:
    """Represents a generated image with its metadata."""
    gen_path: Path
    original_path: Optional[Path] = None
    name: str = ""
    dataset: str = ""
    domain_raw: str = ""
    domain_canonical: str = ""
    source_domain: Optional[str] = None
    is_restoration: bool = False
    restoration_source_weather: Optional[str] = None


def build_original_index(original_dir: Path, verbose: bool = False) -> Tuple[Dict[str, Path], Dict[str, str], Dict[str, Dict[str, Path]]]:
    """
    Build an index of original images by normalized filename.
    
    Returns:
        Tuple of:
        - Dict mapping normalized filename stem to full path (for generation tasks)
        - Dict mapping filename stem to dataset name
        - Dict mapping weather domain -> {stem -> path} for restoration source matching
    """
    # Collect files from main original_dir and additional directories
    all_original_files = []
    all_original_files.extend(find_image_files(original_dir, recursive=True))
    
    # Add Cityscapes and other additional directories
    for additional_dir in ADDITIONAL_ORIGINAL_DIRS:
        if additional_dir.exists():
            additional_files = find_image_files(additional_dir, recursive=True)
            all_original_files.extend(additional_files)
            if verbose:
                logging.info("  Added %d images from %s", len(additional_files), additional_dir)
    
    if verbose:
        logging.info("Found %d original images", len(all_original_files))
    
    # Main index for generation tasks
    index: Dict[str, Path] = {}
    stem_to_dataset: Dict[str, str] = {}
    duplicates: Dict[str, List[Path]] = defaultdict(list)
    
    # Weather domain indices for restoration tasks
    weather_indices: Dict[str, Dict[str, Path]] = defaultdict(dict)
    
    for path in all_original_files:
        stem = path.stem
        
        # Extract dataset and domain from path
        dataset = None
        domain = None
        parts = path.parts
        for i, part in enumerate(parts):
            if part in KNOWN_DATASETS:
                dataset = part
                if i + 1 < len(parts) - 1:
                    domain = parts[i + 1]
                break
        
        # For Cityscapes images (no domain subfolder), assign dataset from path
        if dataset is None and 'CITYSCAPES' in str(path).upper():
            dataset = 'Cityscapes'
        
        # Add to weather domain index if domain is recognized
        if domain:
            canonical = normalize_domain(domain)
            if canonical and canonical in CANONICAL_GENERATION_DOMAINS:
                weather_indices[canonical][stem] = path
        
        # Main index handling
        if stem in index:
            duplicates[stem].append(path)
            if len(duplicates[stem]) == 1:
                duplicates[stem].insert(0, index[stem])
        else:
            index[stem] = path
            if dataset:
                stem_to_dataset[stem] = dataset
    
    if verbose and duplicates:
        logging.warning("  %d filenames appear multiple times", len(duplicates))
        for stem, paths in list(duplicates.items())[:3]:
            logging.warning("    '%s': %s...", stem, [str(p) for p in paths[:2]])
    
    return index, stem_to_dataset, dict(weather_indices)


# =============================================================================
# Processing Functions
# =============================================================================

def process_flat_structure(
    method_dir: Path,
    original_index: Dict[str, Path],
    stem_to_dataset: Dict[str, str],
    weather_indices: Dict[str, Dict[str, Path]],
) -> Tuple[List[ImageEntry], Dict]:
    """Process a method with flat structure (images directly in domain/dataset folders)."""
    entries = []
    stats = defaultdict(lambda: defaultdict(lambda: {"matched": 0, "unmatched": 0}))
    
    try:
        subdirs = list(method_dir.iterdir())
    except PermissionError:
        return entries, dict(stats)
    
    for subdir in subdirs:
        if not subdir.is_dir() or subdir.name.startswith('.'):
            continue
        
        dir_name = subdir.name
        domain_canonical = normalize_domain(dir_name)
        
        if domain_canonical is None:
            continue
        
        domain_raw = dir_name
        source_domain = extract_source_domain(dir_name)
        is_restoration = is_restoration_domain(domain_canonical)
        restoration_source = get_restoration_source_domain(domain_canonical) if is_restoration else None
        
        # Select appropriate index for matching
        if is_restoration and restoration_source and restoration_source in weather_indices:
            match_index = weather_indices[restoration_source]
        else:
            match_index = original_index
        
        images = find_image_files(subdir, recursive=True)
        
        for img_path in images:
            normalized_stem = normalize_filename(img_path.name)
            
            dataset = stem_to_dataset.get(normalized_stem)
            if not dataset:
                dataset = get_dataset_from_filename(img_path.name) or "unknown"
            
            entry = ImageEntry(
                gen_path=img_path,
                name=normalized_stem,
                dataset=dataset,
                domain_raw=domain_raw,
                domain_canonical=domain_canonical,
                source_domain=source_domain,
                is_restoration=is_restoration,
                restoration_source_weather=restoration_source,
            )
            
            if normalized_stem in match_index:
                entry.original_path = match_index[normalized_stem]
                stats[domain_canonical][dataset]["matched"] += 1
            else:
                stats[domain_canonical][dataset]["unmatched"] += 1
            
            entries.append(entry)
    
    return entries, dict(stats)


def process_dataset_domain_structure(
    method_dir: Path,
    original_index: Dict[str, Path],
    stem_to_dataset: Dict[str, str],
    weather_indices: Dict[str, Dict[str, Path]],
) -> Tuple[List[ImageEntry], Dict]:
    """Process a method with dataset/domain hierarchy."""
    entries = []
    stats = defaultdict(lambda: defaultdict(lambda: {"matched": 0, "unmatched": 0}))
    
    try:
        dataset_dirs = list(method_dir.iterdir())
    except PermissionError:
        return entries, dict(stats)
    
    for dataset_dir in dataset_dirs:
        if not dataset_dir.is_dir() or dataset_dir.name.startswith('.'):
            continue
        
        if dataset_dir.name not in KNOWN_DATASETS:
            continue
        
        dataset = dataset_dir.name
        
        try:
            domain_dirs = list(dataset_dir.iterdir())
        except PermissionError:
            continue
        
        for domain_dir in domain_dirs:
            if not domain_dir.is_dir():
                continue
            
            domain_raw = domain_dir.name
            domain_canonical = normalize_domain(domain_raw)
            source_domain = extract_source_domain(domain_raw)
            
            if domain_canonical is None:
                # Check for intermediary directories (e.g., 'generated', 'test_latest', 'images')
                # that contain domain subdirs one level deeper
                INTERMEDIARY_DIRS = {'generated', 'test_latest', 'images', 'output', 'results'}
                if domain_dir.name.lower() in INTERMEDIARY_DIRS:
                    try:
                        nested_dirs = list(domain_dir.iterdir())
                    except PermissionError:
                        continue
                    for nested_dir in nested_dirs:
                        if not nested_dir.is_dir():
                            continue
                        nested_domain_raw = nested_dir.name
                        nested_domain_canonical = normalize_domain(nested_domain_raw)
                        if nested_domain_canonical is None:
                            continue
                        nested_source = extract_source_domain(nested_domain_raw)
                        nested_is_restoration = is_restoration_domain(nested_domain_canonical)
                        nested_restoration_source = get_restoration_source_domain(nested_domain_canonical) if nested_is_restoration else None
                        if nested_is_restoration and nested_restoration_source and nested_restoration_source in weather_indices:
                            nested_match_index = weather_indices[nested_restoration_source]
                        else:
                            nested_match_index = original_index
                        images = find_image_files(nested_dir, recursive=True)
                        for img_path in images:
                            normalized_stem = normalize_filename(img_path.name)
                            entry = ImageEntry(
                                gen_path=img_path,
                                name=normalized_stem,
                                dataset=dataset,
                                domain_raw=nested_domain_raw,
                                domain_canonical=nested_domain_canonical,
                                source_domain=nested_source,
                                is_restoration=nested_is_restoration,
                                restoration_source_weather=nested_restoration_source,
                            )
                            if normalized_stem in nested_match_index:
                                entry.original_path = nested_match_index[normalized_stem]
                                stats[nested_domain_canonical][dataset]["matched"] += 1
                            else:
                                stats[nested_domain_canonical][dataset]["unmatched"] += 1
                            entries.append(entry)
                continue
            
            is_restoration = is_restoration_domain(domain_canonical)
            restoration_source = get_restoration_source_domain(domain_canonical) if is_restoration else None
            
            # Select appropriate index for matching
            if is_restoration and restoration_source and restoration_source in weather_indices:
                match_index = weather_indices[restoration_source]
            else:
                match_index = original_index
            
            images = find_image_files(domain_dir, recursive=True)
            
            for img_path in images:
                normalized_stem = normalize_filename(img_path.name)
                entry = ImageEntry(
                    gen_path=img_path,
                    name=normalized_stem,
                    dataset=dataset,
                    domain_raw=domain_raw,
                    domain_canonical=domain_canonical,
                    source_domain=source_domain,
                    is_restoration=is_restoration,
                    restoration_source_weather=restoration_source,
                )
                
                if normalized_stem in match_index:
                    entry.original_path = match_index[normalized_stem]
                    stats[domain_canonical][dataset]["matched"] += 1
                else:
                    stats[domain_canonical][dataset]["unmatched"] += 1
                
                entries.append(entry)
    
    return entries, dict(stats)


def process_domain_dataset_structure(
    method_dir: Path,
    original_index: Dict[str, Path],
    stem_to_dataset: Dict[str, str],
    weather_indices: Dict[str, Dict[str, Path]],
) -> Tuple[List[ImageEntry], Dict]:
    """Process a method with domain/dataset hierarchy."""
    entries = []
    stats = defaultdict(lambda: defaultdict(lambda: {"matched": 0, "unmatched": 0}))
    
    try:
        domain_dirs = list(method_dir.iterdir())
    except PermissionError:
        return entries, dict(stats)
    
    for domain_dir in domain_dirs:
        if not domain_dir.is_dir() or domain_dir.name.startswith('.'):
            continue
        
        domain_raw = domain_dir.name
        domain_canonical = normalize_domain(domain_raw)
        source_domain = extract_source_domain(domain_raw)
        
        if domain_canonical is None:
            continue
        
        is_restoration = is_restoration_domain(domain_canonical)
        restoration_source = get_restoration_source_domain(domain_canonical) if is_restoration else None
        
        # Select appropriate index for matching
        if is_restoration and restoration_source and restoration_source in weather_indices:
            match_index = weather_indices[restoration_source]
        else:
            match_index = original_index
        
        try:
            subdirs = list(domain_dir.iterdir())
        except PermissionError:
            continue
        
        for dataset_dir in subdirs:
            if not dataset_dir.is_dir():
                continue
            
            if dataset_dir.name in KNOWN_DATASETS:
                dataset = dataset_dir.name
                images = find_image_files(dataset_dir, recursive=True)
                
                for img_path in images:
                    normalized_stem = normalize_filename(img_path.name)
                    entry = ImageEntry(
                        gen_path=img_path,
                        name=normalized_stem,
                        dataset=dataset,
                        domain_raw=domain_raw,
                        domain_canonical=domain_canonical,
                        source_domain=source_domain,
                        is_restoration=is_restoration,
                        restoration_source_weather=restoration_source,
                    )
                    
                    if normalized_stem in match_index:
                        entry.original_path = match_index[normalized_stem]
                        stats[domain_canonical][dataset]["matched"] += 1
                    else:
                        stats[domain_canonical][dataset]["unmatched"] += 1
                    
                    entries.append(entry)
            else:
                # Nested structure like test_latest/images
                images = find_image_files(dataset_dir, recursive=True)
                for img_path in images:
                    normalized_stem = normalize_filename(img_path.name)
                    
                    dataset = stem_to_dataset.get(normalized_stem)
                    if not dataset:
                        dataset = get_dataset_from_filename(img_path.name) or "unknown"
                    
                    entry = ImageEntry(
                        gen_path=img_path,
                        name=normalized_stem,
                        dataset=dataset,
                        domain_raw=domain_raw,
                        domain_canonical=domain_canonical,
                        source_domain=source_domain,
                        is_restoration=is_restoration,
                        restoration_source_weather=restoration_source,
                    )
                    
                    if normalized_stem in match_index:
                        entry.original_path = match_index[normalized_stem]
                        stats[domain_canonical][dataset]["matched"] += 1
                    else:
                        stats[domain_canonical][dataset]["unmatched"] += 1
                    
                    entries.append(entry)
    
    return entries, dict(stats)


def process_method(
    method_dir: Path,
    original_index: Dict[str, Path],
    stem_to_dataset: Dict[str, str],
    weather_indices: Dict[str, Dict[str, Path]],
    verbose: bool = False,
) -> Tuple[List[ImageEntry], Dict, str]:
    """Process a method directory and return image entries and stats."""
    structure = detect_directory_structure(method_dir)
    
    if verbose:
        logging.info("  Detected structure: %s", structure)
    
    if structure == 'domain_dataset':
        entries, stats = process_domain_dataset_structure(
            method_dir, original_index, stem_to_dataset, weather_indices
        )
    elif structure == 'dataset_domain':
        entries, stats = process_dataset_domain_structure(
            method_dir, original_index, stem_to_dataset, weather_indices
        )
    elif structure in ('flat_domain', 'flat_dataset'):
        entries, stats = process_flat_structure(
            method_dir, original_index, stem_to_dataset, weather_indices
        )
    else:
        entries, stats = [], {}
    
    return entries, stats, structure


# =============================================================================
# Manifest Writing
# =============================================================================

def write_manifest(
    entries: List[ImageEntry],
    stats: Dict,
    method_name: str,
    method_dir: Path,
    original_dir: Path,
    output_dir: Path,
    structure_type: str,
) -> Dict:
    """Write manifest CSV and JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "manifest.csv"
    json_path = output_dir / "manifest.json"
    
    # Write CSV
    matched_entries = [e for e in entries if e.original_path is not None]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "gen_path", "original_path", "name", "domain", "dataset", "target_domain"
        ])
        writer.writeheader()
        for entry in matched_entries:
            writer.writerow({
                "gen_path": str(entry.gen_path),
                "original_path": str(entry.original_path) if entry.original_path else "",
                "name": entry.name,
                "domain": entry.domain_canonical,
                "dataset": entry.dataset,
                "target_domain": extract_target_domain(entry.domain_raw),
            })
    
    # Aggregate statistics
    total_matched = sum(
        s["matched"] for domain_stats in stats.values() 
        for s in domain_stats.values()
    )
    total_unmatched = sum(
        s["unmatched"] for domain_stats in stats.values() 
        for s in domain_stats.values()
    )
    total = total_matched + total_unmatched
    
    # Build domain summary
    domain_summary = {}
    for domain, dataset_stats in stats.items():
        domain_matched = sum(s["matched"] for s in dataset_stats.values())
        domain_unmatched = sum(s["unmatched"] for s in dataset_stats.values())
        domain_total = domain_matched + domain_unmatched
        
        is_restoration = is_restoration_domain(domain)
        restoration_source = get_restoration_source_domain(domain) if is_restoration else None
        
        domain_summary[domain] = {
            "total": domain_total,
            "matched": domain_matched,
            "unmatched": domain_unmatched,
            "match_rate": domain_matched / domain_total * 100 if domain_total else 0,
            "is_restoration": is_restoration,
            "restoration_source_weather": restoration_source,
            "datasets": {
                ds: {
                    "matched": s["matched"],
                    "unmatched": s["unmatched"],
                    "total": s["matched"] + s["unmatched"],
                }
                for ds, s in dataset_stats.items()
            }
        }
    
    # Build dataset summary
    dataset_summary = defaultdict(lambda: {"matched": 0, "unmatched": 0, "total": 0})
    for domain_stats in stats.values():
        for ds, s in domain_stats.items():
            dataset_summary[ds]["matched"] += s["matched"]
            dataset_summary[ds]["unmatched"] += s["unmatched"]
            dataset_summary[ds]["total"] += s["matched"] + s["unmatched"]
    
    # Determine task type
    has_restoration = any(is_restoration_domain(d) for d in domain_summary.keys())
    has_generation = any(d in CANONICAL_GENERATION_DOMAINS for d in domain_summary.keys())
    if has_restoration and has_generation:
        task_type = "mixed"
    elif has_restoration:
        task_type = "restoration"
    else:
        task_type = "generation"
    
    # Write JSON
    summary = {
        "method": method_name,
        "generated_dir": str(method_dir),
        "original_dir": str(original_dir),
        "manifest_path": str(csv_path),
        "structure_type": structure_type,
        "task_type": task_type,
        "total_generated": total,
        "total_matched": total_matched,
        "total_unmatched": total_unmatched,
        "overall_match_rate": total_matched / total * 100 if total else 0,
        "domains": domain_summary,
        "datasets_aggregate": dict(dataset_summary),
    }
    
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Write unmatched files if any
    unmatched_entries = [e for e in entries if e.original_path is None]
    if unmatched_entries:
        unmatched_path = output_dir / "manifest_unmatched.txt"
        with open(unmatched_path, 'w') as f:
            for entry in unmatched_entries:
                f.write(f"{entry.gen_path}\n")
    
    return summary


# =============================================================================
# Status Checking
# =============================================================================

def check_status(generated_base: Path) -> dict:
    """Check manifest status for all methods."""
    status = {}
    
    if not generated_base.exists():
        logging.error("Generated images directory not found: %s", generated_base)
        return status
    
    for method_dir in sorted(generated_base.iterdir()):
        if not method_dir.is_dir():
            continue
        if method_dir.name.startswith('.'):
            continue
        
        manifest_csv = method_dir / "manifest.csv"
        manifest_json = method_dir / "manifest.json"
        
        has_csv = manifest_csv.exists()
        has_json = manifest_json.exists()
        
        # Count images if manifest exists
        image_count = 0
        if has_csv:
            try:
                with open(manifest_csv, 'r') as f:
                    image_count = sum(1 for line in f) - 1  # Subtract header
            except Exception:
                pass
        
        # Check write access
        writable = os.access(method_dir, os.W_OK)
        
        status[method_dir.name] = {
            'has_manifest': has_csv,
            'has_json': has_json,
            'image_count': image_count,
            'writable': writable,
            'path': method_dir,
        }
    
    return status


def print_status(status: dict) -> None:
    """Print the status of all methods."""
    print("\nManifest Status:")
    print("=" * 90)
    print(f"{'Method':<30} {'Status':<15} {'Images':>12} {'Writable':>10}")
    print("-" * 90)
    
    missing = []
    missing_writable = []
    for method, info in sorted(status.items()):
        if info['has_manifest']:
            status_str = "✓ Ready"
            count_str = f"{info['image_count']:,}"
        else:
            status_str = "✗ Missing"
            count_str = "-"
            missing.append(method)
            if info['writable']:
                missing_writable.append(method)
        
        writable_str = "Yes" if info['writable'] else "No"
        print(f"{method:<30} {status_str:<15} {count_str:>12} {writable_str:>10}")
    
    print("-" * 90)
    print(f"Total: {len(status)} methods, {len(missing)} missing manifests")
    
    if missing:
        print(f"\nMissing manifests: {', '.join(missing)}")
        if missing_writable:
            print(f"Can generate (writable): {', '.join(missing_writable)}")
        not_writable = set(missing) - set(missing_writable)
        if not_writable:
            print(f"Cannot generate (no write access): {', '.join(not_writable)}")
        print("\nRun with --all-missing to generate missing manifests")


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate manifest files for generated images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--method", "-m",
        help="Generate manifest for a specific method (e.g., cycleGAN)"
    )
    mode_group.add_argument(
        "--all-missing", "-a", action="store_true",
        help="Generate manifests for all methods missing them"
    )
    mode_group.add_argument(
        "--all", action="store_true",
        help="Regenerate manifests for all methods"
    )
    mode_group.add_argument(
        "--status", "-s", action="store_true",
        help="Show status of manifests without generating"
    )
    
    # Directories
    parser.add_argument(
        "--generated-base", type=Path,
        default=DEFAULT_GENERATED_BASE,
        help=f"Base directory for generated images (default: {DEFAULT_GENERATED_BASE})"
    )
    parser.add_argument(
        "--original-dir", type=Path,
        default=DEFAULT_ORIGINAL_DIR,
        help=f"Directory containing original images (default: {DEFAULT_ORIGINAL_DIR})"
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        help="Output directory for manifest files (default: generated directory)"
    )
    
    # Options
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show detailed progress"
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Show what would be done without writing files"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    
    # Handle status mode (default if no other mode specified)
    if args.status or not (args.method or args.all_missing or args.all):
        status = check_status(args.generated_base)
        print_status(status)
        return 0
    
    # Validate directories
    if not args.original_dir.exists():
        logging.error("Original directory not found: %s", args.original_dir)
        return 1
    
    if not args.generated_base.exists():
        logging.error("Generated base directory not found: %s", args.generated_base)
        return 1
    
    # Build original image index
    logging.info("Building original image index from %s...", args.original_dir)
    original_index, stem_to_dataset, weather_indices = build_original_index(args.original_dir, args.verbose)
    logging.info("  Indexed %d original images (for generation tasks)", len(original_index))
    for weather_domain, weather_index in weather_indices.items():
        logging.info("  Indexed %d %s images (for restoration tasks)", len(weather_index), weather_domain)
    
    # Determine methods to process
    if args.all or args.all_missing:
        method_dirs = [d for d in args.generated_base.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        # Filter for --all-missing
        if args.all_missing:
            filtered_dirs = []
            for d in method_dirs:
                try:
                    if not (d / "manifest.csv").exists():
                        if os.access(d, os.W_OK):
                            filtered_dirs.append(d)
                        elif args.verbose:
                            logging.warning("Skipping %s (no write access)", d.name)
                except PermissionError:
                    if args.verbose:
                        logging.warning("Permission denied accessing %s", d.name)
            method_dirs = filtered_dirs
        
        logging.info("Processing %d methods...", len(method_dirs))
    else:
        # Single method mode
        method_dir = args.generated_base / args.method
        if not method_dir.exists():
            logging.error("Method directory not found: %s", method_dir)
            return 1
        method_dirs = [method_dir]
    
    # Process methods
    all_summaries = {}
    
    for method_dir in tqdm(method_dirs, desc="Methods", disable=not HAS_TQDM or len(method_dirs) == 1):
        method_name = method_dir.name
        
        if args.verbose:
            logging.info("\n=== Processing %s ===", method_name)
        
        # Process method
        entries, stats, structure = process_method(
            method_dir, original_index, stem_to_dataset, weather_indices, args.verbose
        )
        
        if not entries:
            if args.verbose:
                logging.info("  No images found, skipping")
            continue
        
        # Determine output directory
        if args.output:
            if args.all or args.all_missing:
                output_dir = args.output / method_name
            else:
                output_dir = args.output
        else:
            output_dir = method_dir
        
        if args.dry_run:
            matched = sum(1 for e in entries if e.original_path is not None)
            restoration_count = sum(1 for e in entries if e.is_restoration)
            logging.info("  Would write manifest: %d images, %d matched", len(entries), matched)
            logging.info("  Structure: %s", structure)
            domains = set(e.domain_canonical for e in entries)
            logging.info("  Domains: %s", domains)
            if restoration_count > 0:
                logging.info("  Restoration images: %d", restoration_count)
            continue
        
        # Write manifest
        try:
            summary = write_manifest(
                entries, stats, method_name, method_dir,
                args.original_dir, output_dir, structure
            )
            all_summaries[method_name] = summary
        except PermissionError:
            logging.error("  Permission denied writing to %s", output_dir)
            continue
        
        if args.verbose:
            logging.info("  Task type: %s", summary['task_type'])
            logging.info("  Total: %d images", summary['total_generated'])
            logging.info("  Matched: %d (%.1f%%)", summary['total_matched'], summary['overall_match_rate'])
            logging.info("  Domains: %s", list(summary['domains'].keys()))
        
        # Single method mode - print summary
        if not (args.all or args.all_missing):
            logging.info("\nManifest created successfully:")
            logging.info("  CSV: %s", output_dir / "manifest.csv")
            logging.info("  JSON: %s", output_dir / "manifest.json")
            logging.info("  Total images: %d", summary['total_generated'])
            logging.info("  Matched: %d (%.1f%%)", summary['total_matched'], summary['overall_match_rate'])
    
    # Write global summary for --all modes
    if (args.all or args.all_missing) and not args.dry_run and all_summaries:
        try:
            summary_path = args.generated_base / "all_manifests_summary.json"
            with open(summary_path, 'w') as f:
                json.dump({
                    "generated_base": str(args.generated_base),
                    "original_dir": str(args.original_dir),
                    "timestamp": datetime.now().isoformat(),
                    "methods_processed": len(all_summaries),
                    "methods": {
                        name: {
                            "structure_type": s["structure_type"],
                            "task_type": s["task_type"],
                            "total_generated": s["total_generated"],
                            "total_matched": s["total_matched"],
                            "match_rate": s["overall_match_rate"],
                            "domains": list(s["domains"].keys()),
                        }
                        for name, s in all_summaries.items()
                    }
                }, f, indent=2)
            logging.info("\nGlobal summary written to: %s", summary_path)
        except PermissionError:
            logging.warning("Could not write global summary (permission denied)")
    
    if args.all or args.all_missing:
        logging.info("\nProcessed %d methods successfully.", len(all_summaries))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
