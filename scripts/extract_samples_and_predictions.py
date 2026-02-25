#!/usr/bin/env python3
"""
Extract Sample Images & Generate Segmentation Predictions

This script performs two main tasks:
1. TRAINING SAMPLES: Extracts 10 training images per dataset from clear_day,
   then finds and symlinks corresponding generated images for every generative
   strategy and target weather domain.

2. TESTING SAMPLES: Extracts 10 test images per dataset (spread across domains),
   then runs segmentation inference using all available iter_15000.pth checkpoints
   from Stage 1 (WEIGHTS/) and Stage 2 (WEIGHTS_STAGE_2/).

Output Structure:
    {OUTPUT_ROOT}/
    ├── training_samples/
    │   └── {dataset}/
    │       ├── originals/              # 10 clear_day training images
    │       └── generated/
    │           └── {strategy}/
    │               └── {domain}/       # Generated versions of same images
    ├── testing_samples/
    │   └── {dataset}/
    │       ├── images/                 # 10 test images (+ domain info)
    │       ├── labels/                 # Corresponding GT labels
    │       └── predictions/
    │           └── {stage}/
    │               └── {strategy}/
    │                   └── {model}/    # Colored segmentation predictions
    └── metadata/
        ├── training_manifest.json      # Which images were selected
        ├── testing_manifest.json       # Test image selections
        └── checkpoint_inventory.json   # All available checkpoints

Usage:
    # Phase 1: Extract samples (no GPU needed)
    python scripts/extract_samples_and_predictions.py --phase extract

    # Phase 2: Submit inference jobs (needs LSF cluster)
    python scripts/extract_samples_and_predictions.py --phase inference --dry-run
    python scripts/extract_samples_and_predictions.py --phase inference --submit

    # Phase 3: Assemble results after jobs complete
    python scripts/extract_samples_and_predictions.py --phase assemble

    # All phases at once (extract + submit)
    python scripts/extract_samples_and_predictions.py --phase all --submit
"""

import os
import sys
import json
import csv
import shutil
import random
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_ROOT = Path('${AWARE_DATA_ROOT}/SAMPLE_EXTRACTION')
FINAL_SPLITS = Path('${AWARE_DATA_ROOT}/FINAL_SPLITS')
GENERATED_IMAGES = Path('${AWARE_DATA_ROOT}/GENERATED_IMAGES')
WEIGHTS_STAGE1 = Path('${AWARE_DATA_ROOT}/WEIGHTS')
WEIGHTS_STAGE2 = Path('${AWARE_DATA_ROOT}/WEIGHTS_STAGE_2')

NUM_SAMPLES = 50  # Number of images to extract per dataset
CHECKPOINT_ITER = 'iter_15000'
RANDOM_SEED = 42

# Datasets and their configurations
DATASETS = {
    'BDD10k': {
        'test_domains': ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy'],
        'train_domain': 'clear_day',
        'num_classes': 19,
        'label_type': 'cityscapes_trainid',
        'weights_dir_name': 'bdd10k',
    },
    'IDD-AW': {
        'test_domains': ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy'],
        'train_domain': 'clear_day',
        'num_classes': 19,
        'label_type': 'cityscapes_trainid',
        'weights_dir_name': 'iddaw',
    },
    'MapillaryVistas': {
        'test_domains': ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy'],
        'train_domain': 'clear_day',
        'num_classes': 66,
        'label_type': 'mapillary_rgb',
        'weights_dir_name': 'mapillaryvistas',
    },
    'OUTSIDE15k': {
        'test_domains': ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy'],
        'train_domain': 'clear_day',
        'num_classes': 24,
        'label_type': 'native',
        'weights_dir_name': 'outside15k',
    },
}

WEATHER_DOMAINS = ['cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy']

# Generative strategy names (as they appear in GENERATED_IMAGES/)
GEN_STRATEGIES = [
    'albumentations_weather', 'Attribute_Hallucination', 'augmenters', 'automold',
    'CNetSeg', 'CUT', 'cyclediffusion', 'cycleGAN', 'EDICT', 'flux2',
    'flux_kontext', 'Img2Img', 'IP2P', 'LANIT', 'magicbrush',
    'Qwen-Image-Edit', 'stargan_v2', 'step1x_new', 'step1x_v1p2',
    'StyleID', 'SUSTechGAN', 'TSIT', 'UniControl', 'VisualCloze',
    'Weather_Effect_Generator',
]

# Cityscapes palette for colorizing predictions
CITYSCAPES_PALETTE = [
    [128, 64, 128],   # road
    [244, 35, 232],   # sidewalk
    [70, 70, 70],     # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170, 30],   # traffic light
    [220, 220, 0],    # traffic sign
    [107, 142, 35],   # vegetation
    [152, 251, 152],  # terrain
    [70, 130, 180],   # sky
    [220, 20, 60],    # person
    [255, 0, 0],      # rider
    [0, 0, 142],      # car
    [0, 0, 70],       # truck
    [0, 60, 100],     # bus
    [0, 80, 100],     # train
    [0, 0, 230],      # motorcycle
    [119, 11, 32],    # bicycle
]

# Mapillary Vistas palette (66 classes) - generate programmatically
MAPILLARY_PALETTE = []
for i in range(66):
    if i < 19:
        MAPILLARY_PALETTE.append(CITYSCAPES_PALETTE[i])
    else:
        MAPILLARY_PALETTE.append([(i * 67 + 37) % 256, (i * 113 + 59) % 256, (i * 179 + 83) % 256])

# OUTSIDE15k palette (24 classes)
OUTSIDE15K_PALETTE = CITYSCAPES_PALETTE[:19] + [
    [100, 100, 100], [165, 42, 42], [0, 170, 30], [140, 140, 140], [0, 0, 0]
]


def get_palette(num_classes: int) -> list:
    """Get color palette for given number of classes."""
    if num_classes == 19:
        return CITYSCAPES_PALETTE
    elif num_classes == 66:
        return MAPILLARY_PALETTE
    elif num_classes == 24:
        return OUTSIDE15K_PALETTE
    else:
        palette = []
        for i in range(num_classes):
            palette.append([(i * 67 + 37) % 256, (i * 113 + 59) % 256, (i * 179 + 83) % 256])
        return palette


# ============================================================================
# PHASE 1: EXTRACT SAMPLES
# ============================================================================

def _get_generated_stems(strategy: str, dataset_name: str, domain: str = 'cloudy') -> Set[str]:
    """Get the set of original image stems that have generated images for a strategy.
    
    Checks the main path patterns used by each strategy (Pattern A and C).
    """
    stems: Set[str] = set()
    
    # Pattern A: strategy/dataset/domain/filename (VisualCloze, flux_kontext, etc.)
    dir_a = GENERATED_IMAGES / strategy / dataset_name / domain
    # Pattern C: strategy/domain/dataset/filename (step1x_new, CUT, etc.)
    dir_c = GENERATED_IMAGES / strategy / domain / dataset_name

    for d in [dir_a, dir_c]:
        try:
            if d.exists():
                stems.update(f.stem for f in d.iterdir() if f.suffix in ('.png', '.jpg', '.jpeg'))
        except (PermissionError, OSError):
            pass
    
    return stems


def select_training_images(dataset_name: str, config: dict) -> List[Path]:
    """Select NUM_SAMPLES training images from clear_day domain.
    
    Uses coverage-aware sampling: preferentially selects originals that have
    generated counterparts for VisualCloze and step1x_new (the two strategies
    with incomplete coverage). Falls back to random sampling if the intersection
    pool is too small.
    """
    train_dir = FINAL_SPLITS / 'train' / 'images' / dataset_name / config['train_domain']
    if not train_dir.exists():
        print(f"  WARNING: Train directory not found: {train_dir}")
        return []
    
    all_images = sorted([f for f in train_dir.iterdir() if f.suffix in ('.png', '.jpg', '.jpeg')])
    if len(all_images) == 0:
        print(f"  WARNING: No images found in {train_dir}")
        return []
    
    # Build coverage-aware eligible pool
    # Prioritize originals that have BOTH VisualCloze AND step1x_new generated images
    priority_strategies = ['VisualCloze', 'step1x_new']
    
    strategy_stems = {}
    for strat in priority_strategies:
        stems = _get_generated_stems(strat, dataset_name, domain='cloudy')
        strategy_stems[strat] = stems
        print(f"  {strat} coverage: {len(stems)} images available")
    
    # Intersection: originals covered by ALL priority strategies
    if strategy_stems:
        covered_stems = set.intersection(*strategy_stems.values())
    else:
        covered_stems = set()
    
    # Filter to images that exist in clear_day AND are covered
    all_image_map = {img.stem: img for img in all_images}
    eligible = [all_image_map[stem] for stem in sorted(covered_stems) if stem in all_image_map]
    
    print(f"  Eligible pool (covered by {', '.join(priority_strategies)}): {len(eligible)}/{len(all_images)}")
    
    # Deterministic sampling from eligible pool
    random.seed(RANDOM_SEED)
    
    if len(eligible) >= NUM_SAMPLES:
        selected = random.sample(eligible, NUM_SAMPLES)
        print(f"  Selected {NUM_SAMPLES} from coverage-aware pool")
    else:
        # Fallback: use all eligible + fill remainder from uncovered
        print(f"  WARNING: Only {len(eligible)} eligible, supplementing from uncovered pool")
        selected = list(eligible)
        remaining = [img for img in all_images if img.stem not in covered_stems]
        needed = NUM_SAMPLES - len(selected)
        if remaining and needed > 0:
            selected.extend(random.sample(remaining, min(needed, len(remaining))))
    
    return sorted(selected)


def select_testing_images(dataset_name: str, config: dict) -> List[Tuple[Path, str]]:
    """Select NUM_SAMPLES test images spread across domains.
    
    Returns list of (image_path, domain) tuples.
    """
    domains = config['test_domains']
    images_per_domain = max(1, NUM_SAMPLES // len(domains))
    remainder = NUM_SAMPLES - images_per_domain * len(domains)
    
    selected = []
    random.seed(RANDOM_SEED)
    
    for i, domain in enumerate(domains):
        test_dir = FINAL_SPLITS / 'test' / 'images' / dataset_name / domain
        if not test_dir.exists():
            print(f"  WARNING: Test directory not found: {test_dir}")
            continue
        
        all_images = sorted([f for f in test_dir.iterdir() if f.suffix in ('.png', '.jpg', '.jpeg')])
        if not all_images:
            continue
        
        n = images_per_domain + (1 if i < remainder else 0)
        domain_selected = random.sample(all_images, min(n, len(all_images)))
        for img in domain_selected:
            selected.append((img, domain))
    
    return sorted(selected, key=lambda x: (x[1], x[0].name))[:NUM_SAMPLES]


def find_generated_image_for_original(original_path: Path, strategy: str,
                                        domain: str, dataset_name: str,
                                        manifest_data: Optional[dict] = None) -> Optional[Path]:
    """Find the generated image corresponding to an original training image.
    
    Different strategies have different directory structures:
    - Pattern A (dataset/domain): strategy/dataset/domain/filename
    - Pattern B (domain only): strategy/domain/.../filename  
    - Pattern C (domain/dataset): strategy/domain/dataset/filename
    - Pattern D (augmenters): strategy/dataset/{alt_domain}/filename  (clouds→cloudy, fog→foggy, etc.)
    - Pattern E (LANIT): strategy/domain/{DATASET}_{filename}.ext  (dataset prefix in filename)
    - Pattern F (stargan_v2): strategy/domain/dataset/filename_lat.ext
    - Pattern G (SUSTechGAN): strategy/domain/test_latest/images/dataset/filename
    - Pattern H (UniControl): strategy/dataset/sunny_day2{domain}/filename
    - Pattern I (Weather_Effect_Generator): strategy/dataset/domain/filename-{f,r,s}syn.ext
    - Pattern J (TSIT): strategy/domain/filename (Cityscapes only, flat)
    """
    stem = original_path.stem
    
    # Strategy directory
    gen_root = GENERATED_IMAGES / strategy
    try:
        if not gen_root.exists():
            return None
    except PermissionError:
        return None
    
    def _safe_exists(p):
        try:
            return p.exists()
        except (PermissionError, OSError):
            return False
    
    # Domain name mapping for augmenters strategy
    AUGMENTERS_DOMAIN_MAP = {
        'cloudy': 'clouds',
        'foggy': 'fog', 
        'rainy': 'rain',
        'snowy': 'snow',
    }
    
    # Try Pattern A: strategy/dataset/domain/filename
    for ext in ['.png', '.jpg', '.jpeg']:
        candidate = gen_root / dataset_name / domain / (stem + ext)
        if _safe_exists(candidate):
            return candidate
    
    # Try Pattern D (augmenters): strategy/dataset/{alt_domain}/filename
    alt_domain = AUGMENTERS_DOMAIN_MAP.get(domain)
    if alt_domain:
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = gen_root / dataset_name / alt_domain / (stem + ext)
            if _safe_exists(candidate):
                return candidate
        # Also try snow_no_flakes as alternative for snowy
        if domain == 'snowy':
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = gen_root / dataset_name / 'snow_no_flakes' / (stem + ext)
                if _safe_exists(candidate):
                    return candidate
    
    # Try Pattern H (UniControl): strategy/dataset/sunny_day2{domain}/filename
    for ext in ['.png', '.jpg', '.jpeg']:
        candidate = gen_root / dataset_name / f'sunny_day2{domain}' / (stem + ext)
        if _safe_exists(candidate):
            return candidate
    
    # Try Pattern I (Weather_Effect_Generator): strategy/dataset/domain/filename-{f,r,s}syn.ext
    # Suffix varies by domain: foggy→-fsyn, rainy→-rsyn, snowy→-ssyn
    WEG_DOMAIN_SUFFIX = {
        'foggy': '-fsyn',
        'rainy': '-rsyn',
        'snowy': '-ssyn',
    }
    weg_suffix = WEG_DOMAIN_SUFFIX.get(domain, '-fsyn')
    for ext in ['.png', '.jpg', '.jpeg']:
        candidate = gen_root / dataset_name / domain / (stem + weg_suffix + ext)
        if _safe_exists(candidate):
            return candidate
    
    # Try Pattern C: strategy/domain/dataset/filename (step1x_new uses this)
    for ext in ['.png', '.jpg', '.jpeg']:
        candidate = gen_root / domain / dataset_name / (stem + ext)
        if _safe_exists(candidate):
            return candidate
    
    # Try Pattern F (stargan_v2): strategy/domain/dataset/filename_lat.ext
    for ext in ['.png', '.jpg', '.jpeg']:
        candidate = gen_root / domain / dataset_name / (stem + '_lat' + ext)
        if _safe_exists(candidate):
            return candidate
    
    # Try Pattern B: strategy/domain/test_latest/images/filename_fake.ext (cycleGAN)
    for ext in ['.png', '.jpg', '.jpeg']:
        candidate = gen_root / domain / 'test_latest' / 'images' / (stem + '_fake' + ext)
        if _safe_exists(candidate):
            return candidate
    
    # Try without _fake suffix in test_latest
    for ext in ['.png', '.jpg', '.jpeg']:
        candidate = gen_root / domain / 'test_latest' / 'images' / (stem + ext)
        if _safe_exists(candidate):
            return candidate
    
    # Try Pattern G (SUSTechGAN): strategy/domain/test_latest/images/dataset/filename
    for ext in ['.png', '.jpg', '.jpeg']:
        candidate = gen_root / domain / 'test_latest' / 'images' / dataset_name / (stem + ext)
        if _safe_exists(candidate):
            return candidate
    
    # Try Pattern E (LANIT): strategy/domain/{DATASET}_{filename}.ext
    for ext in ['.png', '.jpg', '.jpeg']:
        candidate = gen_root / domain / (dataset_name + '_' + stem + ext)
        if _safe_exists(candidate):
            return candidate
    
    # Try direct domain/filename (flat structure, e.g., TSIT for Cityscapes)
    for ext in ['.png', '.jpg', '.jpeg']:
        candidate = gen_root / domain / (stem + ext)
        if _safe_exists(candidate):
            return candidate
    
    return None


def find_label_for_image(img_path: Path, dataset_name: str, split: str, domain: str) -> Optional[Path]:
    """Find the ground truth label for an image."""
    label_dir = FINAL_SPLITS / split / 'labels' / dataset_name / domain
    if not label_dir.exists():
        return None
    
    # Try same filename
    for ext in ['.png', '.jpg']:
        label_path = label_dir / (img_path.stem + ext)
        if label_path.exists():
            return label_path
    
    # Cityscapes naming convention
    if '_leftImg8bit' in img_path.name:
        cs_name = img_path.stem.replace('_leftImg8bit', '_gtFine_labelIds') + '.png'
        label_path = label_dir / cs_name
        if label_path.exists():
            return label_path
    
    # ACDC naming convention
    if '_rgb_anon' in img_path.name:
        acdc_name = img_path.stem.replace('_rgb_anon', '_gt_labelIds') + '.png'
        label_path = label_dir / acdc_name
        if label_path.exists():
            return label_path
    
    return None


def extract_training_samples(output_root: Path) -> dict:
    """Extract training samples and find generated counterparts."""
    print("\n" + "=" * 70)
    print("PHASE 1a: EXTRACTING TRAINING SAMPLES")
    print("=" * 70)
    
    training_manifest = {}
    
    for dataset_name, config in DATASETS.items():
        print(f"\n--- {dataset_name} ---")
        
        selected_images = select_training_images(dataset_name, config)
        if not selected_images:
            continue
        
        print(f"  Selected {len(selected_images)} training images from {config['train_domain']}")
        
        # Create output directories
        orig_dir = output_root / 'training_samples' / dataset_name / 'originals'
        orig_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_manifest = {
            'dataset': dataset_name,
            'source_domain': config['train_domain'],
            'images': [],
        }
        
        for img_path in selected_images:
            # Symlink original image
            link_path = orig_dir / img_path.name
            if not link_path.exists():
                os.symlink(str(img_path), str(link_path))
            
            image_entry = {
                'filename': img_path.name,
                'original_path': str(img_path),
                'generated': {},
            }
            
            # Find generated counterparts for each strategy and domain
            for strategy in GEN_STRATEGIES:
                strategy_generated = {}
                for domain in WEATHER_DOMAINS:
                    gen_path = find_generated_image_for_original(
                        img_path, strategy, domain, dataset_name
                    )
                    if gen_path:
                        # Create symlink
                        gen_out_dir = output_root / 'training_samples' / dataset_name / 'generated' / strategy / domain
                        gen_out_dir.mkdir(parents=True, exist_ok=True)
                        gen_link = gen_out_dir / img_path.name
                        if not gen_link.exists():
                            # Use same extension as generated file for the symlink name
                            gen_link = gen_out_dir / (img_path.stem + gen_path.suffix)
                            if not gen_link.exists():
                                os.symlink(str(gen_path), str(gen_link))
                        strategy_generated[domain] = str(gen_path)
                
                if strategy_generated:
                    image_entry['generated'][strategy] = strategy_generated
            
            dataset_manifest['images'].append(image_entry)
            
            # Count generated images found
            total_gen = sum(len(v) for v in image_entry['generated'].values())
            print(f"  {img_path.name}: found {total_gen} generated counterparts across {len(image_entry['generated'])} strategies")
        
        training_manifest[dataset_name] = dataset_manifest
    
    return training_manifest


def extract_testing_samples(output_root: Path) -> dict:
    """Extract testing samples with their labels."""
    print("\n" + "=" * 70)
    print("PHASE 1b: EXTRACTING TESTING SAMPLES")
    print("=" * 70)
    
    testing_manifest = {}
    
    for dataset_name, config in DATASETS.items():
        print(f"\n--- {dataset_name} ---")
        
        selected = select_testing_images(dataset_name, config)
        if not selected:
            continue
        
        print(f"  Selected {len(selected)} test images across domains")
        
        # Create output directories
        img_dir = output_root / 'testing_samples' / dataset_name / 'images'
        lbl_dir = output_root / 'testing_samples' / dataset_name / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_manifest = {
            'dataset': dataset_name,
            'num_classes': config['num_classes'],
            'label_type': config['label_type'],
            'images': [],
        }
        
        for img_path, domain in selected:
            # Symlink image (prefix with domain for uniqueness)
            prefixed_name = f"{domain}__{img_path.name}"
            img_link = img_dir / prefixed_name
            if not img_link.exists():
                os.symlink(str(img_path), str(img_link))
            
            # Find and symlink label
            label_path = find_label_for_image(img_path, dataset_name, 'test', domain)
            label_name = None
            if label_path:
                label_name = f"{domain}__{label_path.name}"
                lbl_link = lbl_dir / label_name
                if not lbl_link.exists():
                    os.symlink(str(label_path), str(lbl_link))
            
            dataset_manifest['images'].append({
                'filename': prefixed_name,
                'original_path': str(img_path),
                'label_path': str(label_path) if label_path else None,
                'domain': domain,
            })
            
            print(f"  [{domain}] {img_path.name} (label: {'✓' if label_path else '✗'})")
        
        testing_manifest[dataset_name] = dataset_manifest
    
    return testing_manifest


# ============================================================================
# PHASE 2: CHECKPOINT INVENTORY & INFERENCE JOB SUBMISSION
# ============================================================================

def build_checkpoint_inventory() -> dict:
    """Discover all iter_15000 checkpoints across both stages."""
    print("\n" + "=" * 70)
    print("BUILDING CHECKPOINT INVENTORY")
    print("=" * 70)
    
    inventory = {'stage1': {}, 'stage2': {}}
    
    for stage_name, weights_root in [('stage1', WEIGHTS_STAGE1), ('stage2', WEIGHTS_STAGE2)]:
        if not weights_root.exists():
            print(f"  WARNING: {weights_root} does not exist")
            continue
        
        count = 0
        for ckpt_path in sorted(weights_root.rglob(f'{CHECKPOINT_ITER}.pth')):
            # Parse path: WEIGHTS/{strategy}/{dataset}/{model}/iter_15000.pth
            rel_path = ckpt_path.relative_to(weights_root)
            parts = rel_path.parts
            
            if len(parts) < 4:
                continue
            
            strategy = parts[0]
            dataset_dir = parts[1]
            model_dir = parts[2]
            
            # Find the training_config.py
            config_path = ckpt_path.parent / 'training_config.py'
            if not config_path.exists():
                # Try configs subdirectory
                configs_dir = ckpt_path.parent / 'configs'
                if configs_dir.exists():
                    config_candidates = list(configs_dir.glob('*.py'))
                    if config_candidates:
                        config_path = config_candidates[0]
            
            if not config_path.exists():
                continue
            
            key = f"{strategy}/{dataset_dir}/{model_dir}"
            inventory[stage_name][key] = {
                'strategy': strategy,
                'dataset_dir': dataset_dir,
                'model_dir': model_dir,
                'checkpoint_path': str(ckpt_path),
                'config_path': str(config_path),
            }
            count += 1
        
        print(f"  {stage_name}: Found {count} checkpoints at {CHECKPOINT_ITER}")
    
    return inventory


def generate_inference_script(output_root: Path, testing_manifest: dict,
                                checkpoint_info: dict, stage: str,
                                dataset_name: str) -> str:
    """Generate a Python inference script for a specific checkpoint on test images."""
    
    strategy = checkpoint_info['strategy']
    model_dir = checkpoint_info['model_dir']
    config_path = checkpoint_info['config_path']
    checkpoint_path = checkpoint_info['checkpoint_path']
    dataset_config = DATASETS.get(dataset_name, {})
    num_classes = dataset_config.get('num_classes', 19)
    
    pred_dir = output_root / 'testing_samples' / dataset_name / 'predictions' / stage / strategy / model_dir
    
    # Get the list of test images for this dataset
    test_images = testing_manifest.get(dataset_name, {}).get('images', [])
    
    script = f'''#!/usr/bin/env python3
"""Auto-generated inference script for {stage}/{strategy}/{model_dir} on {dataset_name}"""
import os, sys, json
import numpy as np
import torch
import cv2
from pathlib import Path

sys.path.insert(0, "{PROJECT_ROOT}")
from utils import custom_transforms
from utils import custom_losses

from mmengine.config import Config
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
import mmseg.models
import warnings
warnings.filterwarnings('ignore')
register_all_modules(init_default_scope=True)

# Cityscapes palette
PALETTE = {json.dumps(get_palette(num_classes))}

def colorize_mask(mask, num_classes={num_classes}):
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(min(num_classes, len(PALETTE))):
        colored[mask == c] = PALETTE[c]
    return colored

def create_side_by_side(img, pred_colored, save_path):
    """Create input | prediction side by side."""
    h = max(img.shape[0], pred_colored.shape[0])
    w1, w2 = img.shape[1], pred_colored.shape[1]
    canvas = np.zeros((h, w1 + w2 + 10, 3), dtype=np.uint8)
    canvas[:img.shape[0], :w1] = img
    canvas[:pred_colored.shape[0], w1+10:] = pred_colored
    cv2.imwrite(str(save_path), canvas)

def main():
    config_path = "{config_path}"
    checkpoint_path = "{checkpoint_path}"
    output_dir = Path("{pred_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    cfg = Config.fromfile(config_path)
    model = MODELS.build(cfg.model)
    model = revert_sync_batchnorm(model)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Detect model num_classes
    model_num_classes = {num_classes}
    if hasattr(cfg, 'model') and 'decode_head' in cfg.model:
        dh = cfg.model.decode_head
        if isinstance(dh, list):
            for h in dh:
                if 'num_classes' in h:
                    model_num_classes = h['num_classes']
                    break
        elif isinstance(dh, dict):
            model_num_classes = dh.get('num_classes', {num_classes})

    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])

    test_images = {json.dumps([img['original_path'] for img in test_images])}
    test_names = {json.dumps([img['filename'] for img in test_images])}

    results = {{}}
    for img_path_str, img_name in zip(test_images, test_names):
        try:
            img = cv2.imread(img_path_str)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to 512x512 for inference
            h_orig, w_orig = img_rgb.shape[:2]
            img_resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
            
            # Normalize
            img_norm = (img_resized.astype(np.float32) - mean) / std
            img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                img_metas = [dict(ori_shape=(h_orig, w_orig), img_shape=(512, 512),
                                  pad_shape=(512, 512), scale_factor=(1.0, 1.0))]
                result = model.inference(img_tensor, img_metas)
            
            if isinstance(result, torch.Tensor):
                if result.ndim == 4:
                    pred = result[0].argmax(dim=0).cpu().numpy()
                elif result.ndim == 3:
                    pred = result.argmax(dim=0).cpu().numpy()
                else:
                    pred = result.cpu().numpy()
            elif isinstance(result, list):
                r = result[0]
                if hasattr(r, 'pred_sem_seg'):
                    pred = r.pred_sem_seg.data.squeeze()
                    if pred.ndim == 3:
                        pred = pred.argmax(dim=0)
                    pred = pred.cpu().numpy()
                else:
                    pred = r.cpu().numpy() if isinstance(r, torch.Tensor) else np.array(r)
            else:
                pred = result.cpu().numpy() if isinstance(result, torch.Tensor) else np.array(result)
            
            pred = pred.squeeze().astype(np.uint8)
            
            # Colorize prediction
            pred_colored = colorize_mask(pred, model_num_classes)
            
            # Save raw prediction
            raw_path = output_dir / f"{{img_name.replace('.', '_')}}_pred_raw.png"
            cv2.imwrite(str(raw_path), pred)
            
            # Save colored prediction
            color_path = output_dir / f"{{img_name.replace('.', '_')}}_pred_color.png"
            cv2.imwrite(str(color_path), cv2.cvtColor(pred_colored, cv2.COLOR_RGB2BGR))
            
            # Save side-by-side (input | prediction)
            img_display = cv2.resize(img_rgb, (512, 512))
            sbs_path = output_dir / f"{{img_name.replace('.', '_')}}_sbs.png"
            create_side_by_side(
                cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(pred_colored, cv2.COLOR_RGB2BGR),
                sbs_path
            )
            
            results[img_name] = {{"status": "ok", "pred_shape": list(pred.shape)}}
            print(f"  Processed: {{img_name}}")
            
        except Exception as e:
            results[img_name] = {{"status": "error", "error": str(e)}}
            print(f"  ERROR: {{img_name}}: {{e}}")
    
    # Save results summary
    with open(output_dir / "inference_results.json", 'w') as f:
        json.dump({{
            "stage": "{stage}",
            "strategy": "{strategy}",
            "model": "{model_dir}",
            "dataset": "{dataset_name}",
            "config": config_path,
            "checkpoint": checkpoint_path,
            "model_num_classes": model_num_classes,
            "results": results
        }}, f, indent=2)
    
    print(f"Done: {{len([r for r in results.values() if r['status']=='ok'])}}/{{len(results)}} images processed")

if __name__ == '__main__':
    main()
'''
    return script


def submit_inference_jobs(output_root: Path, testing_manifest: dict,
                           inventory: dict, dry_run: bool = True,
                           limit: int = None) -> int:
    """Submit LSF inference jobs for all checkpoints."""
    print("\n" + "=" * 70)
    print("PHASE 2: SUBMITTING INFERENCE JOBS")
    print("=" * 70)
    
    jobs_dir = output_root / 'jobs'
    jobs_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = output_root / 'inference_scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    
    job_count = 0
    submitted = 0
    skipped_exists = 0
    skipped_no_dataset = 0
    
    # Map weights_dir_name back to dataset_name
    dir_to_dataset = {}
    for ds_name, ds_config in DATASETS.items():
        dir_to_dataset[ds_config['weights_dir_name']] = ds_name
    
    for stage_name, stage_checkpoints in inventory.items():
        stage_label = 'stage1' if stage_name == 'stage1' else 'stage2'
        
        for key, ckpt_info in sorted(stage_checkpoints.items()):
            dataset_dir = ckpt_info['dataset_dir']
            dataset_name = dir_to_dataset.get(dataset_dir)
            
            if not dataset_name:
                skipped_no_dataset += 1
                continue
            
            if dataset_name not in testing_manifest:
                skipped_no_dataset += 1
                continue
            
            strategy = ckpt_info['strategy']
            model_dir = ckpt_info['model_dir']
            
            # Check if predictions already exist
            pred_dir = output_root / 'testing_samples' / dataset_name / 'predictions' / stage_label / strategy / model_dir
            results_file = pred_dir / 'inference_results.json'
            if results_file.exists():
                skipped_exists += 1
                continue
            
            if limit and job_count >= limit:
                break
            
            # Generate inference script
            script_content = generate_inference_script(
                output_root, testing_manifest, ckpt_info, stage_label, dataset_name
            )
            
            script_name = f"infer_{stage_label}_{strategy}_{dataset_dir}_{model_dir}.py"
            script_path = scripts_dir / script_name
            with open(script_path, 'w') as f:
                f.write(script_content)
            os.chmod(str(script_path), 0o755)
            
            # Generate LSF job script
            job_name = f"inf_{stage_label[:2]}_{strategy[:10]}_{dataset_dir[:5]}_{model_dir[:10]}"
            log_dir = output_root / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            
            job_script = f"""#!/bin/bash
#BSUB -J {job_name}
#BSUB -o {log_dir}/{script_name.replace('.py', '.out')}
#BSUB -e {log_dir}/{script_name.replace('.py', '.err')}
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1:gmem=10G"
#BSUB -W 01:00

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate prove

cd {PROJECT_ROOT}
python {script_path}
"""
            job_path = jobs_dir / script_name.replace('.py', '.sh')
            with open(job_path, 'w') as f:
                f.write(job_script)
            os.chmod(str(job_path), 0o755)
            
            if dry_run:
                if job_count < 5:
                    print(f"  [DRY-RUN] Would submit: {job_name}")
                elif job_count == 5:
                    print(f"  ... (showing first 5 of many)")
            else:
                result = subprocess.run(
                    ['bsub', '<', str(job_path)],
                    shell=True,
                    capture_output=True, text=True,
                    cwd=str(PROJECT_ROOT),
                    input=open(str(job_path)).read()
                )
                if result.returncode == 0:
                    submitted += 1
                else:
                    print(f"  ERROR submitting {job_name}: {result.stderr}")
            
            job_count += 1
        
        if limit and job_count >= limit:
            print(f"  Reached limit of {limit} jobs")
            break
    
    print(f"\n  Summary:")
    print(f"    Total checkpoints found: {sum(len(v) for v in inventory.values())}")
    print(f"    Jobs {'prepared' if dry_run else 'submitted'}: {job_count}")
    if not dry_run:
        print(f"    Successfully submitted: {submitted}")
    print(f"    Skipped (already done): {skipped_exists}")
    print(f"    Skipped (no matching dataset): {skipped_no_dataset}")
    
    return job_count


# ============================================================================
# PHASE 3: ASSEMBLE RESULTS
# ============================================================================

def assemble_results(output_root: Path):
    """Check completion status and create summary report."""
    print("\n" + "=" * 70)
    print("PHASE 3: ASSEMBLING RESULTS")
    print("=" * 70)
    
    pred_root = output_root / 'testing_samples'
    
    total = 0
    completed = 0
    failed = 0
    missing = 0
    
    summary = {}
    
    for dataset_dir in sorted(pred_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        
        dataset_name = dataset_dir.name
        pred_dir = dataset_dir / 'predictions'
        if not pred_dir.exists():
            continue
        
        summary[dataset_name] = {'stages': {}}
        
        for stage_dir in sorted(pred_dir.iterdir()):
            if not stage_dir.is_dir():
                continue
            
            stage_name = stage_dir.name
            summary[dataset_name]['stages'][stage_name] = {}
            
            for strategy_dir in sorted(stage_dir.iterdir()):
                if not strategy_dir.is_dir():
                    continue
                
                for model_dir in sorted(strategy_dir.iterdir()):
                    if not model_dir.is_dir():
                        continue
                    
                    total += 1
                    results_file = model_dir / 'inference_results.json'
                    
                    if results_file.exists():
                        try:
                            with open(results_file) as f:
                                results = json.load(f)
                            
                            ok_count = sum(1 for r in results.get('results', {}).values() if r.get('status') == 'ok')
                            err_count = sum(1 for r in results.get('results', {}).values() if r.get('status') == 'error')
                            
                            if err_count == 0:
                                completed += 1
                                status = '✅'
                            else:
                                failed += 1
                                status = f'⚠️ ({ok_count} ok, {err_count} errors)'
                        except Exception:
                            failed += 1
                            status = '❌ (corrupt JSON)'
                    else:
                        missing += 1
                        status = '⏳ pending'
                    
                    key = f"{strategy_dir.name}/{model_dir.name}"
                    summary[dataset_name]['stages'][stage_name][key] = status
    
    print(f"\n  Overall Status:")
    print(f"    Total checkpoint-dataset combos: {total}")
    print(f"    Completed: {completed} ✅")
    print(f"    Failed: {failed} ⚠️/❌")
    print(f"    Pending: {missing} ⏳")
    
    # Save summary
    summary_path = output_root / 'metadata' / 'completion_status.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total': total,
            'completed': completed,
            'failed': failed,
            'pending': missing,
            'details': summary
        }, f, indent=2)
    print(f"  Status saved to: {summary_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract sample images and generate segmentation predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract samples only (no GPU needed):
  python scripts/extract_samples_and_predictions.py --phase extract

  # Extract + prepare inference jobs (dry run):
  python scripts/extract_samples_and_predictions.py --phase all --dry-run

  # Submit inference jobs (after extract):
  python scripts/extract_samples_and_predictions.py --phase inference --submit

  # Submit only 10 jobs (for testing):
  python scripts/extract_samples_and_predictions.py --phase inference --submit --limit 10

  # Check completion status:
  python scripts/extract_samples_and_predictions.py --phase assemble
        """
    )
    parser.add_argument('--phase', required=True,
                       choices=['extract', 'inference', 'assemble', 'all'],
                       help='Which phase to run')
    parser.add_argument('--dry-run', action='store_true',
                       help='Prepare jobs but do not submit (for inference phase)')
    parser.add_argument('--submit', action='store_true',
                       help='Actually submit LSF jobs (for inference phase)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Maximum number of inference jobs to submit')
    parser.add_argument('--output-root', type=str, default=str(OUTPUT_ROOT),
                       help=f'Output directory (default: {OUTPUT_ROOT})')
    
    args = parser.parse_args()
    output_root = Path(args.output_root)
    
    metadata_dir = output_root / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Phase 1: Extract samples
    if args.phase in ('extract', 'all'):
        training_manifest = extract_training_samples(output_root)
        testing_manifest = extract_testing_samples(output_root)
        
        # Save manifests
        with open(metadata_dir / 'training_manifest.json', 'w') as f:
            json.dump(training_manifest, f, indent=2)
        with open(metadata_dir / 'testing_manifest.json', 'w') as f:
            json.dump(testing_manifest, f, indent=2)
        
        print(f"\n  Manifests saved to {metadata_dir}")
    
    # Phase 2: Inference
    if args.phase in ('inference', 'all'):
        # Load testing manifest
        manifest_path = metadata_dir / 'testing_manifest.json'
        if not manifest_path.exists():
            print("ERROR: Run --phase extract first to create testing manifest")
            sys.exit(1)
        
        with open(manifest_path) as f:
            testing_manifest = json.load(f)
        
        # Build checkpoint inventory
        inventory = build_checkpoint_inventory()
        
        # Save inventory
        with open(metadata_dir / 'checkpoint_inventory.json', 'w') as f:
            json.dump(inventory, f, indent=2)
        
        # Submit jobs
        dry_run = not args.submit
        if args.phase == 'all' and not args.submit:
            dry_run = True
        
        submit_inference_jobs(output_root, testing_manifest, inventory,
                             dry_run=dry_run, limit=args.limit)
    
    # Phase 3: Assemble
    if args.phase in ('assemble', 'all'):
        if args.phase == 'all':
            print("\n  (Skipping assemble in 'all' mode - run separately after jobs complete)")
        else:
            assemble_results(output_root)
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
