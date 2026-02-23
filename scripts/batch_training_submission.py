#!/usr/bin/env python3
"""
Batch Training Submission System

A robust, multi-user training submission system with:
- Pre-flight checks for existing results (won't overwrite completed training)
- Training lock mechanism to prevent duplicate work across multiple users/machines
- Automatic testing after training completes
- Configurable for different evaluation stages
- LSF job submission with proper parameter handling
- File permissions: 775 for directories, 664 for files
- Resume capability to continue interrupted training

Strategies:
    - STD_STRATEGIES (5): baseline, std_autoaugment, std_cutmix, std_mixup, std_randaugment
                          (std_photometric_distort removed - essentially same as baseline)
    - GEN_STRATEGIES (21): gen_cycleGAN, gen_flux_kontext, gen_step1x_new, gen_TSIT, gen_augmenters, ...
    - ALL_STRATEGIES (26): STD_STRATEGIES + GEN_STRATEGIES

Usage:
    # Dry run to see what jobs would be submitted (ALWAYS do this first!)
    python scripts/batch_training_submission.py --stage 1 --dry-run
    
    # Submit Stage 1 jobs (all 336 jobs: 28 strategies × 4 datasets × 3 models)
    python scripts/batch_training_submission.py --stage 1
    
    # Submit Stage 1 jobs with limit
    python scripts/batch_training_submission.py --stage 1 --limit 50
    
    # Submit ONLY baseline + std_* strategies (no generative)
    python scripts/batch_training_submission.py --stage 1 --strategy-type std --dry-run
    
    # Submit ONLY generative strategies
    python scripts/batch_training_submission.py --stage 1 --strategy-type gen --dry-run
    
    # Submit specific strategies
    python scripts/batch_training_submission.py --stage 1 --strategies baseline std_minimal gen_cycleGAN
    
    # Submit jobs for specific dataset/model
    python scripts/batch_training_submission.py --stage 1 --datasets BDD10k --models deeplabv3plus_r50
    
    # Stage 2 (no domain filter, all weather conditions)
    python scripts/batch_training_submission.py --stage 2 --dry-run
    
    # Resume interrupted training (finds latest checkpoint and continues)
    python scripts/batch_training_submission.py --stage 1 --resume --dry-run
    python scripts/batch_training_submission.py --stage 1 --resume --strategies gen_step1x_new

Stages:
    Stage 1: Train on clear_day only (--domain-filter clear_day), test cross-domain robustness
    Stage 2: Train on all conditions, test domain-inclusive performance

Pre-flight Checks:
    - Skips if checkpoint already exists (iter_15000.pth)
    - Skips if training lock is held by another job
    - Skips gen_* strategies if generated images don't exist for dataset
    
Resume Mode (--resume):
    - Finds the latest iter_*.pth checkpoint in the weights directory
    - Passes --resume-from to unified_training.py to continue training
    - Skips jobs that are already complete (have final checkpoint)
    - Jobs without checkpoints will start fresh
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Set
from dataclasses import dataclass
import json

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training_lock import TrainingLock


# ============================================================================
# Configuration
# ============================================================================

# Base paths
WEIGHTS_ROOT_STAGE1 = Path('${AWARE_DATA_ROOT}/WEIGHTS')
WEIGHTS_ROOT_STAGE2 = Path('${AWARE_DATA_ROOT}/WEIGHTS_STAGE_2')
WEIGHTS_ROOT_RATIO_ABLATION = Path('${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION')
WEIGHTS_ROOT_CITYSCAPES = Path('${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES')  # Pipeline verification
WEIGHTS_ROOT_CITYSCAPES_GEN = Path('${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES_GEN')  # Cityscapes gen evaluation
WEIGHTS_ROOT_CITYSCAPES_RATIO = Path('${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES_RATIO')  # Cityscapes ratio ablation
WEIGHTS_ROOT_NOISE_ABLATION = Path('${AWARE_DATA_ROOT}/WEIGHTS_NOISE_ABLATION')  # Noise ablation study
WEIGHTS_ROOT_EXTENDED_ABLATION = Path('${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED_ABLATION')  # Extended training ablation
WEIGHTS_ROOT_FROM_SCRATCH = Path('${AWARE_DATA_ROOT}/WEIGHTS_FROM_SCRATCH')  # Training from scratch (no pretrained backbone)
GENERATED_IMAGES_ROOT = Path('${AWARE_DATA_ROOT}/GENERATED_IMAGES')

# All datasets
ALL_DATASETS = ['BDD10k', 'IDD-AW', 'MapillaryVistas', 'OUTSIDE15k']

# Cityscapes dataset for pipeline verification (not in standard ALL_DATASETS)
CITYSCAPES_DATASET = 'Cityscapes'

# All models
ALL_MODELS = [
    'deeplabv3plus_r50', 
    'pspnet_r50', 
    'segformer_mit-b3', 
    'segnext_mscan-b', 
    'hrnet_hr48',
    'mask2former_swin-b',
    ]

# Cityscapes validation models (all 5 models for full pipeline verification)
CITYSCAPES_VALIDATION_MODELS = [
    'pspnet_r50', 
    'segformer_mit-b3', 
    'segnext_mscan-b', 
    'mask2former_swin-b',
    ]  # Use all 5 models for comprehensive verification

# ============================================================================
# Cityscapes Ratio Ablation Configuration
# ============================================================================
# Selected based on cityscapes-gen performance analysis:
# - Top performing strategies from DIFFERENT generation families
# - All 4 models used in cityscapes-gen for consistency
# - Reduced ratio range (0.5 and 1.0 already available from cityscapes-gen)

CITYSCAPES_RATIO_STRATEGIES = [
    'gen_VisualCloze',      # Diffusion: 63.27% - visual cloze completion
    'gen_step1x_v1p2',      # Diffusion: 62.85% - modern diffusion editing
    'gen_flux_kontext',     # Diffusion: 62.90% - FLUX flow-matching
    'gen_TSIT',             # GAN: 63.49% - provides GAN vs Diffusion comparison
]

CITYSCAPES_RATIO_MODELS = [
    'pspnet_r50',           # Fast CNN baseline for quick ablation
    'segformer_mit-b3',     # Efficient transformer
    'segnext_mscan-b',      # MSCAN attention - different architecture family
    'mask2former_swin-b',   # Best overall (slower but important)
]

# Reduced ratio ablation values:
# - 0.5 already exists from cityscapes-gen (default ratio)
# - 1.0 baseline results available from cityscapes-gen/cityscapes baseline
# Only test: 0.0 (all generated), 0.25 (mostly generated), 0.75 (mostly real)
CITYSCAPES_RATIO_VALUES = [0.0, 0.25, 0.75]

# ============================================================================
# Stage 1 Ratio Ablation Configuration  
# ============================================================================
# Stage 1 shows 20-24% spreads (10x larger than Cityscapes!) due to:
# - Training on clear_day only → cross-domain testing on adverse conditions
# - More challenging domains (BDD10k, IDD-AW) with real-world variation

# Path for Stage 1 ratio ablation results
WEIGHTS_ROOT_STAGE1_RATIO = Path('${AWARE_DATA_ROOT}/WEIGHTS_STAGE1_RATIO')

# Focus on datasets with best signal (highest spreads in Stage 1 analysis)
STAGE1_RATIO_DATASETS = [
    'BDD10k',               # 21.54% spread - US driving scenes
    'IDD-AW',               # 20.90% spread - Indian adverse weather
]

# Top strategies from cityscapes-gen (same family diversity as cityscapes-ratio)
STAGE1_RATIO_STRATEGIES = [
    'gen_VisualCloze',      # Diffusion - top performer
    'gen_TSIT',             # GAN - provides family comparison
]

# Efficient subset of models for quicker ablation
STAGE1_RATIO_MODELS = [
    'pspnet_r50',           # Fast CNN baseline
    'segformer_mit-b3',     # Efficient transformer
]

# Same ratio values as cityscapes-ratio for comparison
STAGE1_RATIO_VALUES = [0.0, 0.25, 0.75]

# Total: 2 datasets × 2 strategies × 2 models × 3 ratios = 24 jobs

# ============================================================================
# Combination Ablation Configuration (gen_* + std_*)
# ============================================================================
# Tests synergy between generative weather augmentation and standard augmentations.
# gen_* provides weather domain shift, std_* provides additional robustness.
# Option A: std_* transforms applied to BOTH real and generated images.
#
# Strategy selection based on full S1+CG cross-stage analysis (2026-02-11):
# - gen_* chosen from top cross-stage performers representing different families
# - std_* chosen for consistency (all-positive per-dataset gains, low cross-model variance)

WEIGHTS_ROOT_COMBINATION = Path('${AWARE_DATA_ROOT}/WEIGHTS_COMBINATION_ABLATION')

# Top gen_* from cross-stage S1+CG analysis (different families)
COMBINATION_GEN_STRATEGIES = [
    'gen_Qwen_Image_Edit',  # Instruct/Edit: CG #1, S1 #4 — cross-stage champion
    'gen_Img2Img',                  # Diffusion I2I: CG #2, S1 #3 — consistent top-3
    'gen_augmenters',               # Domain-specific: CG #3, S1 #10 — best in family
]

# Top std_* from S1 cross-domain analysis (consistency + gain strength)
COMBINATION_STD_STRATEGIES = [
    'std_cutmix',               # S1 #5, all-positive per-dataset gains, best night-domain (29.19%)
    'std_autoaugment',          # S1 #2, all-positive per-dataset gains, most consistent overall
    'std_mixup',                # S1 #9, lowest cross-model variance (3.42), feature regularization
]

# Efficient model subset for quick ablation
COMBINATION_MODELS = [
    'pspnet_r50',           # Fast CNN baseline
    'segformer_mit-b3',     # Efficient transformer
]

# Total: 3 gen × 3 std × 2 models = 18 combinations
# Each runs on Cityscapes with 20k iterations

# ============================================================================
# Extended Training Ablation Configuration
# ============================================================================
# Tests whether augmentation benefits persist, grow, or diminish with more training.
# Resumes from existing checkpoints at standard iteration counts (15k for S1, 20k for CG)
# and continues training to 2x and 3x the standard duration.
#
# Research questions:
# 1. Does baseline eventually catch up to augmented models with more training?
# 2. Do augmentation gains plateau, grow, or reverse at longer training?
# 3. Which strategy family (gen_* vs std_*) shows more durable improvement?

# Two sub-stages:
# - 'extended-s1': Extends Stage 1 training (BDD10k, IDD-AW with clear_day filter)
# - 'extended-cg': Extends Cityscapes-Gen training (Cityscapes)

WEIGHTS_ROOT_EXTENDED_S1 = WEIGHTS_ROOT_EXTENDED_ABLATION / 'stage1'
WEIGHTS_ROOT_EXTENDED_CG = WEIGHTS_ROOT_EXTENDED_ABLATION / 'cityscapes_gen'

# Source checkpoint directories (where to find the base checkpoints to resume from)
EXTENDED_S1_SOURCE_ROOT = WEIGHTS_ROOT_STAGE1  # Standard S1 checkpoints at 15k
EXTENDED_CG_SOURCE_ROOT = WEIGHTS_ROOT_CITYSCAPES_GEN  # Standard CG checkpoints at 20k

# Extended S1 configuration
EXTENDED_S1_DATASETS = [
    'BDD10k',               # 21.5% strategy spread - high signal
    'IDD-AW',               # 20.9% spread - useful comparison
]

EXTENDED_S1_STRATEGIES = [
    'baseline',             # Reference: does baseline catch up?
    'gen_Img2Img',          # Top gen_* on S1 (39.99%)
    'gen_augmenters',       # Top gen_* on CG (51.47%), strong on S1 too
    'gen_cycleGAN',         # Different GAN family for comparison
    'std_randaugment',      # Top std_* on S1 (39.77%)
]

EXTENDED_S1_MODELS = [
    'pspnet_r50',           # Fast CNN baseline
    'segformer_mit-b3',     # Efficient transformer
]

# Iteration milestones for S1 (base = 15k)
# 30k = 2x, 45k = 3x standard training
EXTENDED_S1_MAX_ITERS = 45000
EXTENDED_S1_CHECKPOINT_INTERVAL = 5000   # Save every 5k → checkpoints at 20k, 25k, 30k, 35k, 40k, 45k
EXTENDED_S1_BASE_ITERS = 15000           # Standard S1 checkpoint to resume from

# Extended CG configuration
EXTENDED_CG_STRATEGIES = [
    'baseline',             # Reference
    'gen_augmenters',       # Top gen_* on CG (51.47%)
    'gen_Img2Img',          # 2nd on CG (51.42%)
    'gen_CUT',              # 3rd on CG (51.21%), GAN-based
    'std_randaugment',      # Top std_* for comparison
]

EXTENDED_CG_MODELS = [
    'pspnet_r50',           # Fast CNN baseline
    'segformer_mit-b3',     # Efficient transformer
]

# Iteration milestones for CG (base = 20k)
# 40k = 2x, 60k = 3x standard training
EXTENDED_CG_MAX_ITERS = 60000
EXTENDED_CG_CHECKPOINT_INTERVAL = 5000   # Save every 5k → checkpoints at 25k, 30k, ..., 60k
EXTENDED_CG_BASE_ITERS = 20000           # Standard CG checkpoint to resume from

# Total: S1 = 2 datasets × 5 strategies × 2 models = 20 jobs
#        CG = 1 dataset × 5 strategies × 2 models = 10 jobs
#        Grand total = 30 jobs

STAGE_1_MODELS = [
    'pspnet_r50', 
    'segformer_mit-b3', 
    'segnext_mscan-b', 
    'mask2former_swin-b',
]

STAGE_2_MODELS = [
    'pspnet_r50', 
    'segformer_mit-b3', 
    'segnext_mscan-b', 
    'mask2former_swin-b',
    ]

# 21 gen_* strategies with full dataset coverage
# (excluding gen_EDICT, gen_StyleID, gen_flux2, gen_AOD-Net - no/insufficient coverage)
GEN_STRATEGIES = [
    'gen_cycleGAN',
    'gen_flux_kontext',
    'gen_step1x_new',
    'gen_LANIT',
    'gen_albumentations_weather',
    'gen_automold',
    'gen_step1x_v1p2',
    'gen_VisualCloze',
    'gen_SUSTechGAN',
    'gen_cyclediffusion',
    'gen_IP2P',
    'gen_Attribute_Hallucination',
    'gen_UniControl',
    'gen_CUT',
    'gen_Img2Img',
    'gen_Qwen_Image_Edit',
    'gen_CNetSeg',
    'gen_stargan_v2',
    'gen_Weather_Effect_Generator',
    'gen_TSIT',           # 191,400 images with full coverage
    'gen_augmenters',     # 159,500 images with full coverage
]

# Standard augmentation strategies
STD_STRATEGIES = [
    'baseline',           # No augmentation at all
    # 'std_minimal',        # RandomCrop + RandomFlip only
    # 'std_photometric_distort',  # REMOVED: Essentially same as baseline
    'std_autoaugment',    # AutoAugment (batch-level)
    'std_cutmix',         # CutMix (batch-level)
    'std_mixup',          # MixUp (batch-level)
    'std_randaugment',    # RandAugment (batch-level)
]

# All strategies combined
ALL_STRATEGIES = STD_STRATEGIES + GEN_STRATEGIES

# LSF Job Configuration
@dataclass
class LSFConfig:
    """LSF job configuration"""
    queue: str = 'BatchGPU'
    time_limit: str = '24:00'
    memory: int = 48000  # Memory in MB
    cpu_count: int = 4


# ============================================================================
# Iteration Configuration
# ============================================================================

def get_effective_max_iters(
    stage: int,
    model: Optional[str] = None,
    override_max_iters: Optional[int] = None,
) -> int:
    """
    Compute the effective max_iters based on stage, model, and overrides.
    
    This is the single source of truth for determining the target checkpoint.
    
    Args:
        stage: Training stage (1, 2, 'cityscapes', 'cityscapes-gen', 'ratio', etc.)
        model: Model name (used for model-specific iterations)
        override_max_iters: Manual override from command line
        
    Returns:
        The target maximum iterations for training
    """
    # Manual override takes highest priority
    if override_max_iters is not None:
        return override_max_iters
    
    # Stage-specific defaults
    # Note: mask2former_swin-b uses the same target as other models per stage.
    # Cityscapes stages already default to 20k which provides sufficient training.
    if stage in ('cityscapes', 'cityscapes-gen'):
        return 20000  # 20k iters (BS=16) = 320k samples = 160k iters (BS=2)
    elif stage == 'extended-s1':
        return EXTENDED_S1_MAX_ITERS   # 45k (3x standard S1)
    elif stage == 'extended-cg':
        return EXTENDED_CG_MAX_ITERS   # 60k (3x standard CG)
    elif stage == 'from-scratch':
        return 40000  # 40k iters for from-scratch training
    else:
        # Stage 1, Stage 2, ratio ablation all use 15k iters
        return 15000  # 15k iters at BS=16 (~98% of final mIoU)


# ============================================================================
# Pre-flight Checks
# ============================================================================

def get_checkpoint_path(weights_dir: Path, max_iters: int) -> Optional[Path]:
    """Get the final checkpoint path if it exists."""
    checkpoint = weights_dir / f'iter_{max_iters}.pth'
    if checkpoint.exists():
        return checkpoint
    return None


def get_latest_checkpoint(weights_dir: Path) -> Optional[Path]:
    """Find the latest checkpoint in a weights directory for resuming training.
    
    Returns:
        Path to the latest iter_*.pth checkpoint, or None if no checkpoints exist.
    """
    if not weights_dir.exists():
        return None
    
    # Find all iter_*.pth files
    checkpoints = list(weights_dir.glob('iter_*.pth'))
    if not checkpoints:
        return None
    
    # Sort by iteration number (extract from filename)
    def get_iter_num(p: Path) -> int:
        try:
            # Extract number from iter_XXXXX.pth
            return int(p.stem.split('_')[1])
        except (IndexError, ValueError):
            return 0
    
    # Sort by iteration number descending and return the latest
    checkpoints.sort(key=get_iter_num, reverse=True)
    latest = checkpoints[0]
    
    # Verify checkpoint is valid (not empty/corrupted)
    try:
        if latest.stat().st_size < 1000:  # Less than 1KB is suspicious
            return None
    except OSError:
        return None
    
    return latest


def get_checkpoint_iteration(checkpoint_path: Path) -> int:
    """Extract iteration number from checkpoint filename."""
    try:
        return int(checkpoint_path.stem.split('_')[1])
    except (IndexError, ValueError):
        return 0


def has_valid_results(weights_dir: Path, max_iters: int) -> bool:
    """Check if valid training results already exist.
    
    Args:
        weights_dir: Path to the weights directory
        max_iters: Target iteration count (must be explicitly passed)
    """
    checkpoint = get_checkpoint_path(weights_dir, max_iters)
    if checkpoint is None:
        return False
    
    # Check if checkpoint is valid (not empty)
    try:
        if checkpoint.stat().st_size < 1000:  # Less than 1KB is suspicious
            return False
    except OSError:
        return False
    
    return True


def has_test_results(weights_dir: Path) -> bool:
    """Check if test results exist."""
    test_results_dir = weights_dir / 'test_results_detailed'
    if not test_results_dir.exists():
        return False
    
    # Check for any results.json file
    for subdir in test_results_dir.iterdir():
        if subdir.is_dir():
            results_json = subdir / 'results.json'
            if results_json.exists():
                return True
    
    return False


def strategy_to_dir_name(strategy: str) -> str:
    """Convert strategy name to directory name (handle hyphens)."""
    # Map underscores back to hyphens for specific directories
    if strategy.startswith('gen_'):
        gen_model = strategy[4:]  # Remove 'gen_' prefix
        hyphen_dirs = {'Qwen_Image_Edit': 'Qwen-Image-Edit'}
        return hyphen_dirs.get(gen_model, gen_model)
    return strategy


def has_generated_images(strategy: str, dataset: str) -> bool:
    """Check if generated images exist for this strategy/dataset combination."""
    if not strategy.startswith('gen_'):
        return True  # Non-generative strategies don't need generated images
    
    # Noise ablation uses reference manifest (cycleGAN) — no actual generated images needed
    # The actual manifest validation happens in unified_training_config.py
    if strategy == 'gen_random_noise':
        return True
    
    gen_dir = strategy_to_dir_name(strategy)
    gen_path = GENERATED_IMAGES_ROOT / gen_dir
    
    if not gen_path.exists():
        return False
    
    # Check manifest.json for this dataset (authoritative source)
    manifest_json = gen_path / 'manifest.json'
    if manifest_json.exists():
        try:
            with open(manifest_json, 'r') as f:
                manifest_data = json.load(f)
            # Check if any domain has this dataset with images
            for domain_name, domain_data in manifest_data.get('domains', {}).items():
                datasets = domain_data.get('datasets', {})
                if dataset in datasets and datasets[dataset].get('total', 0) > 0:
                    return True
            return False
        except Exception:
            pass
    
    # Fallback: check manifest.csv (legacy format)
    manifest_csv = gen_path / 'manifest.csv'
    if manifest_csv.exists():
        try:
            import csv
            dataset_lower = dataset.lower()
            with open(manifest_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_dataset = row.get('dataset', '').lower()
                    if row_dataset == dataset_lower:
                        return True
            return False
        except Exception:
            pass
    
    # Fallback: check for dataset subdirectory (try multiple case variants)
    dataset_variants = [dataset, dataset.upper(), dataset.lower(), dataset.title()]
    for variant in dataset_variants:
        dataset_path = gen_path / variant
        if dataset_path.exists():
            # Check for any images
            for subdir in dataset_path.rglob('*'):
                if subdir.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    return True
    
    return False


# ============================================================================
# Job Generation
# ============================================================================

@dataclass
class TrainingJob:
    """Represents a training job to submit."""
    strategy: str
    dataset: str
    model: str
    stage: int  # Can be 1, 2, or 'ratio' for ratio ablation
    ratio: float = 0.5
    aux_loss: Optional[str] = None
    std_strategy: Optional[str] = None  # For combination ablation (gen_* + std_*)
    weights_dir: Optional[Path] = None
    skip_reason: Optional[str] = None
    resume_from: Optional[Path] = None  # Checkpoint to resume from
    effective_max_iters: Optional[int] = None  # Target max iterations (computed per job)
    
    @property
    def job_name(self) -> str:
        """Generate LSF job name with stage prefix."""
        # Stage prefix
        if isinstance(self.stage, int):
            stage_prefix = f's{self.stage}_'
        else:
            stage_prefix = f'{self.stage}_'  # e.g., 'ratio_'
        
        dataset_short = self.dataset.lower().replace('-', '')
        model_short = self.model.split('_')[0]
        aux_tag = f"_aux-{self.aux_loss}" if self.aux_loss else ''
        std_tag = f"+{self.std_strategy}" if self.std_strategy else ''
        if self.strategy.startswith('gen_'):
            if self.ratio != 0.5:
                base = f'{self.strategy}{std_tag}_{dataset_short}_{model_short}_{self.ratio:.2f}'.replace('.', 'p')
            else:
                base = f'{self.strategy}{std_tag}_{dataset_short}_{model_short}'
        else:
            base = f'{self.strategy}_{dataset_short}_{model_short}'
        return f'{stage_prefix}{base}{aux_tag}'
    
    @property
    def is_skipped(self) -> bool:
        """Check if job should be skipped."""
        return self.skip_reason is not None


def get_weights_dir(
    strategy: str,
    dataset: str,
    model: str,
    stage: int,
    ratio: float = 0.5,
    aux_loss: Optional[str] = None,
) -> Path:
    """Get the weights directory for a training configuration."""
    # Determine base root based on stage
    if stage == 1:
        base_root = WEIGHTS_ROOT_STAGE1
    elif stage == 2:
        base_root = WEIGHTS_ROOT_STAGE2
    elif stage == 'ratio':
        base_root = WEIGHTS_ROOT_RATIO_ABLATION
    elif stage == 'cityscapes':
        base_root = WEIGHTS_ROOT_CITYSCAPES
    elif stage == 'cityscapes-gen':
        base_root = WEIGHTS_ROOT_CITYSCAPES_GEN
    elif stage == 'cityscapes-ratio':
        base_root = WEIGHTS_ROOT_CITYSCAPES_RATIO
    elif stage == 'stage1-ratio':
        base_root = WEIGHTS_ROOT_STAGE1_RATIO
    elif stage == 'combination':
        base_root = WEIGHTS_ROOT_COMBINATION
    elif stage == 'noise-ablation':
        base_root = WEIGHTS_ROOT_NOISE_ABLATION
    elif stage == 'from-scratch':
        base_root = WEIGHTS_ROOT_FROM_SCRATCH
    elif stage == 'extended-s1':
        base_root = WEIGHTS_ROOT_EXTENDED_S1
    elif stage == 'extended-cg':
        base_root = WEIGHTS_ROOT_EXTENDED_CG
    else:
        base_root = WEIGHTS_ROOT_STAGE1
    
    # Dataset directory name (lowercase, handle hyphen)
    dataset_dir = dataset.lower().replace('-', '')  # IDD-AW → iddaw for ratio ablation
    
    # Model directory name
    model_dir = model
    if strategy.startswith('gen_') and ratio != 1.0:
        model_dir = f'{model}_ratio{ratio:.2f}'.replace('.', 'p')
    if aux_loss:
        model_dir = f'{model_dir}_aux-{aux_loss}'
    
    return base_root / strategy / dataset_dir / model_dir

def generate_job_list(
    stage: int,
    strategies: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    ratios: Optional[List[float]] = None,
    aux_loss: Optional[str] = None,
    check_existing: bool = True,
    check_locks: bool = True,
    resume: bool = False,
    max_iters: Optional[int] = None,
) -> List[TrainingJob]:
    """
    Generate list of training jobs with pre-flight checks.
    
    Args:
        stage: Training stage (1, 2, or 'ratio' for ratio ablation)
        strategies: List of strategies (default: all strategies - baseline, std_*, gen_*)
        datasets: List of datasets (default: all)
        models: List of models (default: all)
        ratios: List of real/gen ratios for generative strategies (default: [0.5])
        check_existing: Skip jobs with existing results
        check_locks: Skip jobs that are currently locked
        resume: If True, look for existing checkpoints to resume from
        max_iters: Override for max training iterations (default: computed per job based on stage/model)
        
    Returns:
        List of TrainingJob objects
    """
    strategies = strategies or ALL_STRATEGIES  # Now includes baseline + std_* + gen_*
    datasets = datasets or ALL_DATASETS
    if stage == 1:
        models = models or STAGE_1_MODELS
    elif stage == 2:
        models = models or STAGE_2_MODELS
    else:
        models = models or ALL_MODELS
    
    ratios = ratios or [0.5]  # Default to single ratio of 0.5
    
    # For combination stage, also iterate over std_strategies
    std_strategies_list = COMBINATION_STD_STRATEGIES if stage == 'combination' else [None]
    
    jobs = []
    
    for strategy in strategies:
        for dataset in datasets:
            for model in models:
                # For non-generative strategies, only use ratio=1.0 (ignored anyway)
                strategy_ratios = ratios if strategy.startswith('gen_') else [1.0]
                
                for ratio in strategy_ratios:
                    for std_strategy in std_strategies_list:
                        job = TrainingJob(
                            strategy=strategy,
                            dataset=dataset,
                            model=model,
                            stage=stage,
                            ratio=ratio,
                            aux_loss=aux_loss,
                            std_strategy=std_strategy,
                        )
                        
                        # Get weights directory - for combination, include std_strategy in path
                        if std_strategy:
                            combo_strategy = f"{strategy}+{std_strategy}"
                            job.weights_dir = get_weights_dir(combo_strategy, dataset, model, stage, ratio, aux_loss)
                        else:
                            job.weights_dir = get_weights_dir(strategy, dataset, model, stage, ratio, aux_loss)
                        
                        # Compute effective max_iters for this job
                        # Uses stage and model to determine target checkpoint
                        effective_max_iters = get_effective_max_iters(stage, model, max_iters)
                        
                        # Store effective max_iters on job for later use in job script
                        job.effective_max_iters = effective_max_iters
                        
                        # Pre-flight checks
                        if check_existing and has_valid_results(job.weights_dir, effective_max_iters):
                            job.skip_reason = 'Results exist'
                        elif strategy.startswith('gen_') and not has_generated_images(strategy, dataset):
                            job.skip_reason = 'No generated images'
                        elif resume:
                            # Look for existing checkpoint to resume from
                            latest_ckpt = get_latest_checkpoint(job.weights_dir)
                            if latest_ckpt:
                                ckpt_iter = get_checkpoint_iteration(latest_ckpt)
                                if ckpt_iter >= effective_max_iters:
                                    job.skip_reason = f'Already complete (iter {ckpt_iter})'
                                else:
                                    job.resume_from = latest_ckpt
                            # If no checkpoint, will start fresh (resume_from stays None)
                        elif check_locks:
                            # Note: Don't include ratio in lock check because job scripts
                            # don't include ratio in their lock file names
                            # For combination jobs, include std_strategy in lock name
                            lock_strategy = f"{strategy}+{std_strategy}" if std_strategy else strategy
                            lock = TrainingLock(
                                lock_strategy,
                                dataset,
                                model,
                                ratio=None,  # Lock files don't include ratio
                                aux_loss=aux_loss,
                                stage=stage,  # Pass stage (int or string like 'cityscapes-gen')
                            )
                            if lock.is_locked():
                                holder = lock.get_lock_holder()
                                if holder:
                                    job.skip_reason = f"Locked by {holder.get('user', 'unknown')}@{holder.get('hostname', 'unknown')}"
                                else:
                                    job.skip_reason = 'Locked'
                        
                        jobs.append(job)
    
    return jobs


# ============================================================================
# Job Submission
# ============================================================================

def generate_job_script(
    job: TrainingJob,
    lsf_config: LSFConfig,
    max_iters: Optional[int] = None,
    batch_size: Optional[int] = None,
    checkpoint_interval: Optional[int] = None,
    eval_interval: Optional[int] = None,
    aux_loss: Optional[str] = None,
) -> str:
    """Generate LSF job script for a training job."""
    work_dir = str(job.weights_dir)
    
    # Models requiring exclusive GPU access (memory-intensive)
    EXCLUSIVE_GPU_MODELS = {'mask2former_swin-b'}
    
    # GPU memory requirements per model (at BS=16, 512x512 input, with 20% safety margin)
    # Memory estimates based on model architecture and parameter counts:
    # - ResNet-50 backbone models: ~20-24 GB at BS=16
    # - HRNet/SegFormer/SegNeXt: ~24-28 GB at BS=16  
    # - Mask2Former/Swin-B: ~48-64 GB at BS=8 (reduced batch size)
    # Reduced slightly to improve job scheduling on shared cluster
    MODEL_GMEM_REQUIREMENTS = {
        'pspnet_r50': '18G',
        'deeplabv3plus_r50': '18G',
        'hrnet_hr48': '20G',
        'segformer_mit-b3': '20G',
        'segnext_mscan-b': '20G',
        'mask2former_swin-b': '38G',
    }
    DEFAULT_GMEM = '20G'  # Safe default for unknown models
    
    # Use effective_max_iters from job if set, otherwise compute it
    # (job.effective_max_iters should already be set by generate_job_list)
    if job.effective_max_iters is not None:
        effective_max_iters = job.effective_max_iters
    elif max_iters is not None:
        effective_max_iters = max_iters
    else:
        effective_max_iters = get_effective_max_iters(job.stage, job.model)
    
    # GPU specification - use exclusive_process for memory-intensive models
    # All models get gmem specification to prevent OOM from GPU sharing
    gmem = MODEL_GMEM_REQUIREMENTS.get(job.model, DEFAULT_GMEM)
    if job.model in EXCLUSIVE_GPU_MODELS:
        gpu_spec = f'"num=1:mode=exclusive_process:gmem={gmem}"'
    else:
        gpu_spec = f'"num=1:gmem={gmem}"'
    
    # Build training command
    cmd_parts = [
        'python', str(PROJECT_ROOT / 'unified_training.py'),
        '--dataset', job.dataset,
        '--model', job.model,
        '--strategy', job.strategy,
    ]
    
    # Add domain filter for Stage 1, ratio ablation, noise ablation, and from-scratch (not Stage 2 or Cityscapes)
    if job.stage in [1, 'ratio', 'stage1-ratio', 'noise-ablation', 'extended-s1', 'from-scratch']:
        cmd_parts.extend(['--domain-filter', 'clear_day'])
    # Note: Stage 2 and Cityscapes do NOT use domain filter
    
    # Add --no-pretrained for from-scratch training
    if job.stage == 'from-scratch':
        cmd_parts.append('--no-pretrained')
    
    # Add ratio parameter for generative strategies
    if job.strategy.startswith('gen_'):
        cmd_parts.extend(['--real-gen-ratio', str(job.ratio)])
    
    # Add max iterations
    cmd_parts.extend(['--max-iters', str(effective_max_iters)])

    # Add batch size if specified
    if batch_size is not None:
        cmd_parts.extend(['--batch-size', str(batch_size)])

    # Add checkpoint and eval intervals if specified
    if checkpoint_interval is not None:
        cmd_parts.extend(['--checkpoint-interval', str(checkpoint_interval)])
    if eval_interval is not None:
        cmd_parts.extend(['--eval-interval', str(eval_interval)])

    # Add auxiliary loss if specified
    if aux_loss:
        cmd_parts.extend(['--aux-loss', aux_loss])
    
    # Add std-strategy for combination ablation
    if job.std_strategy:
        cmd_parts.extend(['--std-strategy', job.std_strategy])
    
    # Add resume-from if resuming from a checkpoint
    if job.resume_from:
        cmd_parts.extend(['--resume-from', str(job.resume_from)])
    
    # Add work-dir to ensure proper output location
    cmd_parts.extend(['--work-dir', work_dir])
    
    training_cmd = ' '.join(cmd_parts)
    
    aux_suffix = f"_aux-{aux_loss}" if aux_loss else ''
    # Ratio suffix for lock file - include ratio to differentiate ratio ablation jobs
    ratio_suffix = f"_ratio{job.ratio:.2f}".replace('.', 'p') if job.ratio != 0.5 else ''
    # std_strategy suffix for lock file - differentiates combination ablation jobs
    std_suffix = f"+{job.std_strategy}" if job.std_strategy else ''
    # Stage prefix for lock file (same as job name)
    stage_prefix = f's{job.stage}_' if isinstance(job.stage, int) else f'{job.stage}_'
    
    # Build ACDC cross-domain test section for cityscapes-gen stage
    if job.stage in ('cityscapes-gen', 'cityscapes-ratio'):
        backslash = '\\'
        acdc_test_section = f'''
        # ============================================================================
        # Cross-domain testing on ACDC (cityscapes-gen/cityscapes-ratio stages)
        # ============================================================================
        echo ""
        echo "=========================================="
        echo "Starting cross-domain testing on ACDC..."
        echo "=========================================="
        
        ACDC_TEST_OUTPUT="{work_dir}/test_results_acdc"
        
        python {PROJECT_ROOT}/fine_grained_test.py {backslash}
            --config "$CONFIG" {backslash}
            --checkpoint "$CHECKPOINT" {backslash}
            --output-dir "$ACDC_TEST_OUTPUT" {backslash}
            --dataset ACDC {backslash}
            --data-root "${AWARE_DATA_ROOT}/FINAL_SPLITS" {backslash}
            --test-split "test" {backslash}
            --batch-size 10
        
        ACDC_EXIT_CODE=$?
        
        if [ $ACDC_EXIT_CODE -eq 0 ]; then
            echo "ACDC cross-domain testing completed successfully"
        else
            echo "WARNING: ACDC testing failed with exit code: $ACDC_EXIT_CODE"
        fi
'''
    else:
        acdc_test_section = ''
    
    script = f'''#!/bin/bash
#BSUB -J {job.job_name}
#BSUB -q {lsf_config.queue}
#BSUB -o {work_dir}/train_%J.out
#BSUB -e {work_dir}/train_%J.err
#BSUB -n 2,{lsf_config.cpu_count}
#BSUB -gpu {gpu_spec}

# ============================================================================
# Environment setup
# ============================================================================

# Set permissions: 775 for directories, 664 for files
umask 002

# Force Python to not use cached bytecode (always reimport fresh code)
export PYTHONDONTWRITEBYTECODE=1

# ============================================================================
# Pre-flight checks inside the job
# ============================================================================

echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "User: $USER"
echo "Started: $(date)"
echo "Strategy: {job.strategy}"
echo "Dataset: {job.dataset}"
echo "Model: {job.model}"
echo "Stage: {job.stage}"
echo "Aux Loss: {aux_loss or 'none'}"
echo "Work Dir: {work_dir}"
echo "=========================================="

# Create work directory with proper permissions
mkdir -p {work_dir}
chmod 775 {work_dir}
cd {work_dir}

# Activate conda environment
source ~/.bashrc
mamba activate prove

# Pre-flight check: verify results don't already exist
CHECKPOINT="{work_dir}/iter_{effective_max_iters}.pth"
if [ -f "$CHECKPOINT" ]; then
    SIZE=$(stat -c%s "$CHECKPOINT" 2>/dev/null || echo 0)
    if [ "$SIZE" -gt 1000 ]; then
        echo "WARNING: Checkpoint already exists at $CHECKPOINT (size: $SIZE bytes)"
        echo "Skipping training to avoid overwriting."
        exit 0
    fi
fi

# Acquire training lock
LOCK_DIR="${AWARE_DATA_ROOT}/training_locks"
mkdir -p $LOCK_DIR
LOCK_FILE="$LOCK_DIR/{stage_prefix}{job.strategy}{std_suffix}_{job.dataset.lower().replace('-', '_')}_{job.model}{ratio_suffix}{aux_suffix}.lock"

# Try to acquire lock (non-blocking)
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "ERROR: Another job is already training this configuration"
    echo "Lock file: $LOCK_FILE"
    cat "$LOCK_FILE" 2>/dev/null || true
    exit 1
fi

# Write lock info
cat > "$LOCK_FILE" << EOF
{{
  "job_id": "$LSB_JOBID",
  "hostname": "$(hostname)",
  "user": "$USER",
  "strategy": "{job.strategy}",
  "dataset": "{job.dataset}",
  "model": "{job.model}",
  "aux_loss": "{aux_loss or ''}",
  "started": "$(date -Iseconds)"
}}
EOF

echo "Lock acquired: $LOCK_FILE"

# ============================================================================
# Training
# ============================================================================

echo ""
echo "Starting training..."
echo "Command: {training_cmd}"
echo ""

{training_cmd}

TRAIN_EXIT_CODE=$?

# ============================================================================
# Testing (if training succeeded)
# ============================================================================

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    CHECKPOINT="{work_dir}/iter_{effective_max_iters}.pth"
    CONFIG="{work_dir}/training_config.py"
    TEST_OUTPUT="{work_dir}/test_results_detailed"
    
    if [ -f "$CHECKPOINT" ] && [ -f "$CONFIG" ]; then
        echo ""
        echo "=========================================="
        echo "Starting fine-grained testing..."
        echo "=========================================="
        
        # All datasets use FINAL_SPLITS structure for fine-grained testing
        # Cityscapes test set in FINAL_SPLITS has city-based subfolders (frankfurt, lindau, munster)
        DATA_ROOT="${AWARE_DATA_ROOT}/FINAL_SPLITS"
        TEST_SPLIT="test"
        
        python {PROJECT_ROOT}/fine_grained_test.py \\
            --config "$CONFIG" \\
            --checkpoint "$CHECKPOINT" \\
            --output-dir "$TEST_OUTPUT" \\
            --dataset {job.dataset} \\
            --data-root $DATA_ROOT \\
            --test-split $TEST_SPLIT \\
            --batch-size 10
        
        TEST_EXIT_CODE=$?
        
        if [ $TEST_EXIT_CODE -eq 0 ]; then
            echo "Testing completed successfully"
        else
            echo "WARNING: Testing failed with exit code: $TEST_EXIT_CODE"
        fi
{acdc_test_section}
    else
        echo "WARNING: Checkpoint or config not found, skipping testing"
        echo "  Checkpoint: $CHECKPOINT"
        echo "  Config: $CONFIG"
    fi
else
    echo "Training failed, skipping testing"
fi

# ============================================================================
# Cleanup and permissions
# ============================================================================

# Ensure all created files/directories have proper permissions (775/664)
echo "Setting permissions on output files..."
find {work_dir} -type d -exec chmod 775 {{}} \; 2>/dev/null || true
find {work_dir} -type f -exec chmod 664 {{}} \; 2>/dev/null || true

# Release lock
flock -u 200
rm -f "$LOCK_FILE" 2>/dev/null

echo ""
echo "=========================================="
echo "Training completed with exit code: $TRAIN_EXIT_CODE"
echo "Finished: $(date)"
echo "=========================================="

exit $TRAIN_EXIT_CODE
'''
    return script


def submit_job(
    job: TrainingJob,
    lsf_config: LSFConfig,
    dry_run: bool = False,
    max_iters: Optional[int] = None,
    batch_size: Optional[int] = None,
    checkpoint_interval: Optional[int] = None,
    eval_interval: Optional[int] = None,
    aux_loss: Optional[str] = None,
) -> bool:
    """
    Submit a training job to LSF.
    
    Args:
        job: TrainingJob to submit
        lsf_config: LSF configuration
        dry_run: If True, just print what would be done
        max_iters: Optional maximum training iterations
        batch_size: Optional batch size (default: 16 for Cityscapes, 2 for others)
        checkpoint_interval: Optional checkpoint save interval
        eval_interval: Optional validation interval
        aux_loss: Optional auxiliary loss type
        
    Returns:
        True if job was submitted successfully
    """
    if job.is_skipped:
        print(f"  SKIP: {job.job_name} - {job.skip_reason}")
        return False
    
    # Ensure weights directory exists
    if job.weights_dir:
        job.weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate job script
    script = generate_job_script(
        job, lsf_config,
        max_iters=max_iters,
        batch_size=batch_size,
        checkpoint_interval=checkpoint_interval,
        eval_interval=eval_interval,
        aux_loss=aux_loss
    )
    script_path = job.weights_dir / 'submit_job.sh'
    
    if dry_run:
        print(f"  SUBMIT: {job.job_name}")
        print(f"    Dir: {job.weights_dir}")
        if job.resume_from:
            resume_iter = get_checkpoint_iteration(job.resume_from)
            print(f"    Resuming from: iter_{resume_iter}.pth")
        return True
    
    # Write script - try weights dir first, fall back to temp file
    import tempfile
    use_temp = False
    try:
        with open(script_path, 'w') as f:
            f.write(script)
        # Try to set permissions - ignore if not owner
        try:
            os.chmod(script_path, 0o755)
        except PermissionError:
            pass  # File might be owned by another user, that's OK
    except PermissionError:
        # Can't write to weights dir (owned by another user), use temp file
        use_temp = True
        fd, temp_path = tempfile.mkstemp(suffix='.sh', prefix=f'{job.job_name}_')
        script_path = Path(temp_path)
        with os.fdopen(fd, 'w') as f:
            f.write(script)
        os.chmod(script_path, 0o755)
    
    # Submit job
    result = subprocess.run(
        f'bsub < {script_path}',
        shell=True,
        capture_output=True,
        text=True
    )
    
    # Clean up temp file if used
    if use_temp:
        try:
            script_path.unlink()
        except:
            pass
    
    if result.returncode == 0:
        print(f"  SUBMIT: {job.job_name} - {result.stdout.strip()}")
        return True
    else:
        print(f"  FAILED: {job.job_name} - {result.stderr.strip()}")
        return False


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Batch Training Submission System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Dry run to see what jobs would be submitted
    python batch_training_submission.py --stage 1 --dry-run
    
    # Submit Stage 1 jobs (limit to 50)
    python batch_training_submission.py --stage 1 --limit 50
    
    # Submit specific strategies
    python batch_training_submission.py --stage 1 --strategies gen_cycleGAN gen_flux_kontext
    
    # Submit for specific dataset/model
    python batch_training_submission.py --stage 1 --datasets BDD10k --models deeplabv3plus_r50
    
    # Pipeline verification on Cityscapes
    python batch_training_submission.py --stage cityscapes --dry-run
'''
    )
    
    # Required arguments
    parser.add_argument('--stage', required=True,
                       help='Training stage: 1 (clear_day only), 2 (all domains), "ratio" for ratio ablation, or "cityscapes" for pipeline verification')
    
    # Filtering options
    parser.add_argument('--strategy-type', choices=['all', 'std', 'gen'],
                       default='all',
                       help='Strategy type: all (28), std (7 baseline+std_*), gen (21 gen_*)')
    parser.add_argument('--strategies', nargs='+', 
                       help='Specific strategies to train (overrides --strategy-type)')
    parser.add_argument('--datasets', nargs='+', choices=ALL_DATASETS + [CITYSCAPES_DATASET],
                       help='Specific datasets (default: all). Use Cityscapes for pipeline verification.')
    parser.add_argument('--models', nargs='+', choices=ALL_MODELS,
                       help='Specific models (default: all)')
    
    # Job control
    parser.add_argument('--limit', type=int, default=None,
                       help='Maximum number of jobs to submit')
    parser.add_argument('--ratios', type=float, nargs='+', default=[0.5],
                       help='Real/gen ratios for generative strategies (default: 0.5). Example: --ratios 0.0 0.25 0.5')
    parser.add_argument('--max-iters', type=int, default=None,
                       help='Maximum training iterations (default: use config default, usually 15000)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Training batch size (default: 16 for Cityscapes, 2 for others). Adjust LR and warmup automatically.')
    parser.add_argument('--checkpoint-interval', type=int, default=None,
                       help='Save checkpoint every N iterations (default: 5000)')
    parser.add_argument('--eval-interval', type=int, default=None,
                       help='Run validation every N iterations (default: 5000)')
    parser.add_argument('--aux-loss', type=str, default=None,
                       choices=['focal', 'lovasz', 'boundary'],
                       help='Auxiliary loss to add alongside CrossEntropyLoss')
    
    # Pre-flight options
    parser.add_argument('--no-check-existing', action='store_true',
                       help='Don\'t skip jobs with existing results')
    parser.add_argument('--no-check-locks', action='store_true',
                       help='Don\'t check for training locks')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from existing checkpoints. Finds latest iter_*.pth and resumes from there.')
    
    # LSF options
    parser.add_argument('--queue', default='BatchGPU',
                       help='LSF queue (default: BatchGPU)')
    parser.add_argument('--time-limit', default='24:00',
                       help='Job time limit (default: 24:00)')
    parser.add_argument('--memory', type=int, default=48000,
                       help='Memory per process in MB (default: 48000)')
    
    # Execution options
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without submitting')
    parser.add_argument('-y', '--yes', action='store_true',
                       help='Skip confirmation prompt and submit immediately')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between job submissions in seconds (default: 0.5)')
    
    args = parser.parse_args()
    
    # Create LSF config
    lsf_config = LSFConfig(
        queue=args.queue,
        time_limit=args.time_limit,
        memory=args.memory,
    )
    
    # Validate and parse stage
    stage_map = {'1': 1, '2': 2, 'ratio': 'ratio', 'cityscapes': 'cityscapes', 'cityscapes-gen': 'cityscapes-gen', 'cityscapes-ratio': 'cityscapes-ratio', 'stage1-ratio': 'stage1-ratio', 'combination': 'combination', 'noise-ablation': 'noise-ablation', 'extended-s1': 'extended-s1', 'extended-cg': 'extended-cg', 'from-scratch': 'from-scratch'}
    stage_input = str(args.stage).lower()
    if stage_input not in stage_map:
        print(f"Error: Invalid stage '{args.stage}'. Must be 1, 2, 'ratio', 'cityscapes', 'cityscapes-gen', 'cityscapes-ratio', 'stage1-ratio', 'combination', 'noise-ablation', 'extended-s1', 'extended-cg', or 'from-scratch'")
        return
    stage = stage_map.get(stage_input, args.stage if isinstance(args.stage, int) else stage_input)
    
    # For Cityscapes stage, use specific defaults
    if stage == 'cityscapes':
        # Use baseline only for Cityscapes pipeline verification
        if args.strategies is None:
            strategies = ['baseline']
        else:
            strategies = args.strategies
        # Use Cityscapes validation models
        datasets = [CITYSCAPES_DATASET]
        models = args.models or CITYSCAPES_VALIDATION_MODELS
    elif stage == 'cityscapes-gen':
        # Use all augmentation strategies for Cityscapes generative evaluation
        if args.strategies is None:
            if args.strategy_type == 'std':
                strategies = STD_STRATEGIES
            elif args.strategy_type == 'gen':
                strategies = GEN_STRATEGIES
            else:  # 'all' - default includes all augmentation strategies
                strategies = ALL_STRATEGIES
        else:
            strategies = args.strategies
        # Use Cityscapes dataset with SegFormer by default
        datasets = [CITYSCAPES_DATASET]
        models = args.models or ['segformer_mit-b3']  # SegFormer for cityscapes-gen evaluation
    elif stage == 'cityscapes-ratio':
        # Cityscapes ratio ablation: systematic study of real/gen ratios
        # Uses top-performing gen strategies from cityscapes-gen evaluation
        if args.strategies is None:
            strategies = CITYSCAPES_RATIO_STRATEGIES
        else:
            strategies = args.strategies
        datasets = [CITYSCAPES_DATASET]
        models = args.models or CITYSCAPES_RATIO_MODELS
        # Override ratios if not specified - use full ablation range
        if args.ratios == [0.5]:  # Default value means user didn't specify
            args.ratios = CITYSCAPES_RATIO_VALUES
    elif stage == 'stage1-ratio':
        # Stage 1 ratio ablation: larger effect sizes due to cross-domain testing
        # Uses clear_day domain filter (Stage 1 setup)
        if args.strategies is None:
            strategies = STAGE1_RATIO_STRATEGIES
        else:
            strategies = args.strategies
        datasets = args.datasets or STAGE1_RATIO_DATASETS
        models = args.models or STAGE1_RATIO_MODELS
        # Override ratios if not specified
        if args.ratios == [0.5]:  # Default value means user didn't specify
            args.ratios = STAGE1_RATIO_VALUES
    elif stage == 'combination':
        # Combination ablation: gen_* + std_* strategies together
        # Tests synergy between generative weather augmentation and standard augmentations
        strategies = COMBINATION_GEN_STRATEGIES  # gen_* strategies to combine
        datasets = [CITYSCAPES_DATASET]
        models = args.models or COMBINATION_MODELS
        # std_strategies will be iterated over in job generation
    elif stage == 'from-scratch':
        # From-scratch training: no pretrained backbone weights
        # Tests whether augmentation gains are genuine or masked by pretrained features
        if args.strategies is None:
            if args.strategy_type == 'std':
                strategies = STD_STRATEGIES
            elif args.strategy_type == 'gen':
                strategies = GEN_STRATEGIES
            else:
                strategies = ALL_STRATEGIES
        else:
            strategies = args.strategies
        datasets = args.datasets or ALL_DATASETS
        models = args.models or ['segformer_mit-b3']
        if args.checkpoint_interval is None:
            args.checkpoint_interval = 5000
        if args.eval_interval is None:
            args.eval_interval = 5000
    elif stage == 'noise-ablation':
        # Noise ablation: uses gen_random_noise strategy with baseline for comparison
        strategies = args.strategies or ['gen_random_noise', 'baseline']
        datasets = args.datasets or ALL_DATASETS
        models = args.models or ALL_MODELS
    elif stage == 'extended-s1':
        # Extended Stage 1 training: resume from 15k checkpoints, train to 45k
        # Tests whether augmentation gains persist/grow/diminish with more training
        strategies = args.strategies or EXTENDED_S1_STRATEGIES
        datasets = args.datasets or EXTENDED_S1_DATASETS
        models = args.models or EXTENDED_S1_MODELS
        # Set checkpoint/eval intervals for fine-grained progress tracking
        if args.checkpoint_interval is None:
            args.checkpoint_interval = EXTENDED_S1_CHECKPOINT_INTERVAL
        if args.eval_interval is None:
            args.eval_interval = EXTENDED_S1_CHECKPOINT_INTERVAL
    elif stage == 'extended-cg':
        # Extended CG training: resume from 20k checkpoints, train to 60k
        strategies = args.strategies or EXTENDED_CG_STRATEGIES
        datasets = [CITYSCAPES_DATASET]
        models = args.models or EXTENDED_CG_MODELS
        # Set checkpoint/eval intervals for fine-grained progress tracking
        if args.checkpoint_interval is None:
            args.checkpoint_interval = EXTENDED_CG_CHECKPOINT_INTERVAL
        if args.eval_interval is None:
            args.eval_interval = EXTENDED_CG_CHECKPOINT_INTERVAL
    else:
        # Determine strategies based on --strategy-type or --strategies
        strategies = args.strategies
        if strategies is None:
            if args.strategy_type == 'std':
                strategies = STD_STRATEGIES
            elif args.strategy_type == 'gen':
                strategies = GEN_STRATEGIES
            else:  # 'all'
                strategies = ALL_STRATEGIES
        datasets = args.datasets
        models = args.models
    
    # Generate job list
    print(f"\n{'='*60}")
    print(f"Batch Training Submission - Stage {stage}")
    if stage == 'ratio':
        print(f"  Ratios: {args.ratios}")
    elif stage == 'cityscapes':
        print(f"  Pipeline Verification Mode")
        print(f"  Dataset: {CITYSCAPES_DATASET}")
        print(f"  Models: {models}")
    elif stage == 'cityscapes-gen':
        print(f"  Cityscapes Generative Evaluation Mode")
        print(f"  Dataset: {CITYSCAPES_DATASET}")
        print(f"  Models: {models}")
        print(f"  Cross-domain testing: Cityscapes val + ACDC")
    elif stage == 'cityscapes-ratio':
        print(f"  Cityscapes Ratio Ablation Study")
        print(f"  Purpose: Systematic evaluation of real/gen image ratios")
        print(f"  Strategies: {strategies}")
        print(f"  Models: {models}")
        print(f"  Ratios: {args.ratios}")
        print(f"  Total configurations: {len(strategies)} × {len(models)} × {len(args.ratios)} = {len(strategies)*len(models)*len(args.ratios)}")
    elif stage == 'stage1-ratio':
        print(f"  Stage 1 Ratio Ablation Study")
        print(f"  Purpose: Ratio ablation with larger effect sizes (20-24% spreads)")
        print(f"  Datasets: {datasets}")
        print(f"  Strategies: {strategies}")
        print(f"  Models: {models}")
        print(f"  Ratios: {args.ratios}")
        print(f"  Note: Uses clear_day domain filter (Stage 1 cross-domain testing)")
        print(f"  Total configurations: {len(datasets)} × {len(strategies)} × {len(models)} × {len(args.ratios)} = {len(datasets)*len(strategies)*len(models)*len(args.ratios)}")
    elif stage == 'combination':
        print(f"  Combination Ablation Study (gen_* + std_*)")
        print(f"  Purpose: Test synergy between generative and standard augmentations")
        print(f"  gen_* strategies: {COMBINATION_GEN_STRATEGIES}")
        print(f"  std_* strategies: {COMBINATION_STD_STRATEGIES}")
        print(f"  Models: {models}")
        print(f"  Total configurations: {len(COMBINATION_GEN_STRATEGIES)} × {len(COMBINATION_STD_STRATEGIES)} × {len(models)} = {len(COMBINATION_GEN_STRATEGIES)*len(COMBINATION_STD_STRATEGIES)*len(models)}")
    elif stage == 'from-scratch':
        print(f"  From-Scratch Training (no pretrained backbone)")
        print(f"  Purpose: Test if augmentation gains are genuine or masked by pretrained features")
        print(f"  Datasets: {datasets}")
        print(f"  Models: {models}")
        print(f"  Flag: --no-pretrained (backbone init_cfg = None)")
        print(f"  Domain filter: clear_day (S1 style)")
        print(f"  Total configurations: {len(datasets)} × {len(strategies)} × {len(models)} = {len(datasets)*len(strategies)*len(models)}")
    elif stage == 'noise-ablation':
        print(f"  Noise Ablation Study")
        print(f"  Purpose: Test if models learn from image content or label layouts")
        print(f"  Datasets: {datasets}")
        print(f"  Models: {models}")
    elif stage == 'extended-s1':
        print(f"  Extended Training Ablation - Stage 1")
        print(f"  Purpose: Test if augmentation benefits persist with more training")
        print(f"  Datasets: {datasets}")
        print(f"  Strategies: {strategies}")
        print(f"  Models: {models}")
        print(f"  Iterations: {EXTENDED_S1_BASE_ITERS} → {EXTENDED_S1_MAX_ITERS} (checkpoint every {EXTENDED_S1_CHECKPOINT_INTERVAL})")
        print(f"  Source checkpoints: {EXTENDED_S1_SOURCE_ROOT}")
        print(f"  Total configurations: {len(datasets)} × {len(strategies)} × {len(models)} = {len(datasets)*len(strategies)*len(models)}")
    elif stage == 'extended-cg':
        print(f"  Extended Training Ablation - Cityscapes-Gen")
        print(f"  Purpose: Test if augmentation benefits persist with more training")
        print(f"  Dataset: Cityscapes")
        print(f"  Strategies: {strategies}")
        print(f"  Models: {models}")
        print(f"  Iterations: {EXTENDED_CG_BASE_ITERS} → {EXTENDED_CG_MAX_ITERS} (checkpoint every {EXTENDED_CG_CHECKPOINT_INTERVAL})")
        print(f"  Source checkpoints: {EXTENDED_CG_SOURCE_ROOT}")
        print(f"  Total configurations: {len(strategies)} × {len(models)} = {len(strategies)*len(models)}")
    print(f"{'='*60}")
    print(f"\nStrategy type: {args.strategy_type}")
    print(f"Strategies: {len(strategies)}")
    print(f"\nGenerating job list...")
    
    # Determine effective max_iters for job generation
    if args.max_iters is not None:
        effective_max_iters = args.max_iters
    elif stage in ('cityscapes', 'cityscapes-gen', 'cityscapes-ratio', 'combination'):
        effective_max_iters = 20000
    elif stage == 'extended-s1':
        effective_max_iters = EXTENDED_S1_MAX_ITERS
    elif stage == 'extended-cg':
        effective_max_iters = EXTENDED_CG_MAX_ITERS
    elif stage == 'stage1-ratio':
        effective_max_iters = 15000  # Standard Stage 1 iterations
    elif stage == 'from-scratch':
        effective_max_iters = 40000  # 40k iters for from-scratch training
    else:
        effective_max_iters = 15000
    
    jobs = generate_job_list(
        stage=stage,
        strategies=strategies,
        datasets=datasets,
        models=models,
        ratios=args.ratios,
        aux_loss=args.aux_loss,
        check_existing=not args.no_check_existing and not args.resume,
        check_locks=not args.no_check_locks and not args.resume,
        resume=args.resume,
        max_iters=effective_max_iters,
    )
    
    # Post-process for extended training stages:
    # Set resume_from to SOURCE checkpoint from original training dir (not the extended output dir).
    # Extended training continues from an existing checkpoint in WEIGHTS/ or WEIGHTS_CITYSCAPES_GEN/.
    if stage in ('extended-s1', 'extended-cg'):
        source_stage = 1 if stage == 'extended-s1' else 'cityscapes-gen'
        base_iters = EXTENDED_S1_BASE_ITERS if stage == 'extended-s1' else EXTENDED_CG_BASE_ITERS
        missing_source = 0
        for job in jobs:
            if job.is_skipped:
                continue
            # Construct source checkpoint path from original training
            source_dir = get_weights_dir(
                job.strategy, job.dataset, job.model, source_stage, job.ratio, job.aux_loss
            )
            source_ckpt = source_dir / f'iter_{base_iters}.pth'
            if source_ckpt.exists():
                job.resume_from = source_ckpt
            else:
                job.skip_reason = f'Source checkpoint missing: iter_{base_iters}.pth in {source_dir}'
                missing_source += 1
        if missing_source > 0:
            print(f"\n  WARNING: {missing_source} jobs skipped - source checkpoints not found")
    
    # Summary
    total = len(jobs)
    skipped = sum(1 for j in jobs if j.is_skipped)
    to_submit = total - skipped
    resuming = sum(1 for j in jobs if j.resume_from is not None)
    
    print(f"\nJob Summary:")
    print(f"  Total configurations: {total}")
    print(f"  Skipped: {skipped}")
    print(f"  To submit: {to_submit}")
    if resuming > 0:
        print(f"  Resuming from checkpoint: {resuming}")
    
    if args.limit and to_submit > args.limit:
        print(f"  Limited to: {args.limit}")
    
    # Skip reason breakdown
    skip_reasons: Dict[str, int] = {}
    for job in jobs:
        if job.skip_reason:
            skip_reasons[job.skip_reason] = skip_reasons.get(job.skip_reason, 0) + 1
    
    if skip_reasons:
        print(f"\nSkip reasons:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"  - {reason}: {count}")
    
    # Confirm submission
    if not args.dry_run and to_submit > 0 and not args.yes:
        print(f"\n{'='*60}")
        response = input(f"Submit {min(to_submit, args.limit or to_submit)} jobs? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Submit jobs
    print(f"\n{'='*60}")
    print("Submitting jobs..." if not args.dry_run else "Dry run - showing what would be submitted:")
    print(f"{'='*60}\n")
    
    submitted = 0
    for job in jobs:
        if job.is_skipped:
            continue
        
        if args.limit and submitted >= args.limit:
            print(f"\nReached limit of {args.limit} jobs")
            break
        
        if submit_job(
            job,
            lsf_config,
            dry_run=args.dry_run,
            max_iters=args.max_iters,
            batch_size=args.batch_size,
            checkpoint_interval=args.checkpoint_interval,
            eval_interval=args.eval_interval,
            aux_loss=args.aux_loss,
        ):
            submitted += 1
            if not args.dry_run and args.delay > 0:
                time.sleep(args.delay)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  {'Would submit' if args.dry_run else 'Submitted'}: {submitted} jobs")
    print(f"  Skipped: {skipped}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
