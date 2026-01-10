#!/usr/bin/env python3
"""
PROVE Unified Training Script

This script provides a streamlined training workflow using the unified
configuration system. It replaces the need for multiple training scripts
and config files by accepting all parameters via command line.

Features:
- Single entry point for all training configurations
- Support for mixed real/generated training with configurable ratios
- Compatible with MMSegmentation and MMDetection
- Automatic config generation and saving
- Job submission support for HPC clusters

Usage:
    # Train baseline model
    python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy baseline

    # Train with generated images (100% generated)
    python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN

    # Train with mixed data (50% real, 50% generated)
    python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN --real-gen-ratio 0.5

    # Generate config only (don't train)
    python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN --config-only

    # Submit as batch job (LSF)
    python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN --submit-job
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Suppress MMSegmentation deprecation warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='mmseg')

from unified_training_config import (
    UnifiedTrainingConfig,
    DATASET_CONFIGS,
    SEGMENTATION_MODELS,
    DETECTION_MODELS,
    ALL_MODELS,
    AUGMENTATION_STRATEGIES,
    ADVERSE_CONDITIONS,
)


# ============================================================================
# Training Runner
# ============================================================================

class UnifiedTrainer:
    """
    Unified training orchestrator for PROVE pipeline.
    
    Handles:
    - Configuration generation
    - Training execution with MMSeg/MMDet
    - Mixed dataloader setup
    - Logging and checkpointing
    - Pretrained weight caching
    - Early stopping
    
    Args:
        dataset: Dataset name (e.g., 'ACDC', 'BDD10k', 'BDD100k')
        model: Model name (e.g., 'deeplabv3plus_r50', 'segformer_mit-b5')
        strategy: Augmentation strategy (e.g., 'baseline', 'gen_cycleGAN')
        real_gen_ratio: Ratio of real images (0.0-1.0). Default: 1.0
        custom_conditions: Optional list of weather conditions to use
        domain_filter: Optional domain to filter training data (e.g., 'clear_day')
        work_dir: Output directory for checkpoints and logs
        cache_dir: Directory for caching pretrained weights. When specified,
            pretrained backbone weights are downloaded/stored in this directory
            instead of the default ~/.cache/torch location.
        resume_from: Checkpoint path to resume training from
        load_from: Pretrained weights path to initialize model
        seed: Random seed for reproducibility. Default: 42
        deterministic: Whether to enable deterministic mode. Default: True
        early_stop: Whether to enable early stopping. Default: True
        early_stop_patience: Number of validations without improvement before stopping. Default: 5
        gpu_ids: List of GPU IDs to use. Default: [0]
        distributed: Whether to use distributed training. Default: False
    """
    
    def __init__(
        self,
        dataset: str,
        model: str,
        strategy: str = 'baseline',
        std_strategy: Optional[str] = None,
        real_gen_ratio: float = 1.0,
        custom_conditions: Optional[List[str]] = None,
        domain_filter: Optional[str] = None,
        work_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        resume_from: Optional[str] = None,
        load_from: Optional[str] = None,
        seed: int = 42,
        deterministic: bool = True,
        early_stop: bool = True,
        early_stop_patience: int = 5,
        max_iters: Optional[int] = None,
        gpu_ids: List[int] = None,
        distributed: bool = False,
    ):
        self.dataset = dataset
        self.model = model
        self.strategy = strategy
        self.std_strategy = std_strategy
        self.real_gen_ratio = real_gen_ratio
        self.custom_conditions = custom_conditions
        self.domain_filter = domain_filter
        self.work_dir = work_dir
        self.cache_dir = cache_dir
        self.resume_from = resume_from
        self.load_from = load_from
        self.seed = seed
        self.deterministic = deterministic
        self.early_stop = early_stop
        self.early_stop_patience = early_stop_patience
        self.max_iters = max_iters
        self.gpu_ids = gpu_ids or [0]
        self.distributed = distributed
        
        # Initialize config builder
        self.config_builder = UnifiedTrainingConfig(cache_dir=cache_dir)
        
        # Build configuration
        self.config = self._build_config()
    
    def _build_config(self) -> Dict[str, Any]:
        """Build training configuration"""
        # Build custom training config to override early stopping and iteration settings
        custom_training_config = {
            'early_stop': self.early_stop,
            'early_stop_patience': self.early_stop_patience,
        }
        if self.max_iters is not None:
            custom_training_config['max_iters'] = self.max_iters
        
        config = self.config_builder.build(
            dataset=self.dataset,
            model=self.model,
            strategy=self.strategy,
            std_strategy=self.std_strategy,
            real_gen_ratio=self.real_gen_ratio,
            custom_conditions=self.custom_conditions,
            domain_filter=self.domain_filter,
            custom_training_config=custom_training_config,
        )
        
        # Override work_dir if specified
        if self.work_dir:
            config['work_dir'] = self.work_dir
        
        # Add resume/load paths
        if self.resume_from:
            config['load_from'] = self.resume_from
            config['resume'] = True
        if self.load_from:
            config['load_from'] = self.load_from
        
        # Override seed if specified
        config['seed'] = self.seed
        config['deterministic'] = self.deterministic
        config['gpu_ids'] = self.gpu_ids
        
        return config
    
    def save_config(self, filepath: Optional[str] = None) -> str:
        """Save configuration to file"""
        if filepath is None:
            config_dir = Path(self.config['work_dir']) / 'configs'
            config_dir.mkdir(parents=True, exist_ok=True)
            filepath = str(config_dir / 'training_config.py')
        
        return self.config_builder.save_config(self.config, filepath)
    
    def train(self, method: str = 'subprocess'):
        """
        Execute training.
        
        Args:
            method: Training method to use. Options:
                - 'subprocess': Run training in subprocess (recommended, avoids import conflicts)
                - 'mim': Use OpenMMLab's mim tool
                - 'direct': Direct import (may have compatibility issues)
        """
        # Save config first
        config_path = self.save_config()
        print(f"Configuration saved to: {config_path}")
        
        # Create work directory
        work_dir = Path(self.config['work_dir'])
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup mixed dataloader if needed
        if self.config.get('mixed_dataloader', {}).get('enabled', False):
            self._setup_mixed_training()
        
        # Run training
        print("\n" + "=" * 60)
        print("Starting PROVE Training")
        print("=" * 60)
        print(f"Dataset: {self.dataset}")
        print(f"Model: {self.model}")
        print(f"Strategy: {self.strategy}")
        if self.std_strategy:
            print(f"Standard Augmentation: {self.std_strategy}")
        print(f"Real/Gen Ratio: {self.real_gen_ratio}")
        print(f"Work Dir: {self.config['work_dir']}")
        print(f"Training Method: {method}")
        print("=" * 60 + "\n")
        
        if method == 'subprocess':
            return self._train_with_mmengine(config_path)
        elif method == 'mim':
            return self._train_with_mim(config_path)
        elif method == 'mmseg_tools':
            return self._train_with_mmseg_tools(config_path)
        elif method == 'direct':
            # Direct import - may have compatibility issues
            return self._train_direct(config_path)
        else:
            print(f"Unknown training method: {method}")
            print("Using subprocess method as fallback...")
            return self._train_with_mmengine(config_path)
    
    def _train_direct(self, config_path: str) -> bool:
        """Train with direct imports (may have compatibility issues)"""
        try:
            from mmengine.runner import Runner
            from mmengine.config import Config
            
            # Import custom transforms and datasets to register them
            import custom_transforms
            import unified_datasets
            
            # Import mmsegmentation to register all components
            import mmseg.models
            import mmseg.datasets  
            import mmseg.evaluation
            
            cfg = Config.fromfile(config_path)
            runner = Runner.from_cfg(cfg)
            runner.train()
            return True
        except Exception as e:
            print(f"Direct training failed: {e}")
            print("Falling back to subprocess method...")
            return self._train_with_mmengine(config_path)
    
    def _setup_mixed_training(self):
        """Setup mixed dataloader for training"""
        from mixed_dataloader import MixedDataLoader, build_mixed_dataloader_config
        
        print("Setting up mixed dataloader...")
        
        mixed_cfg = self.config['mixed_dataloader']
        print(f"  Real samples per batch: {mixed_cfg['batch_composition']['real_samples']}")
        print(f"  Generated samples per batch: {mixed_cfg['batch_composition']['generated_samples']}")
        print(f"  Sampling strategy: {mixed_cfg['sampling_strategy']}")
    
    def _train_with_mmengine(self, config_path: str) -> bool:
        """Train using MMEngine via subprocess to avoid import conflicts"""
        import subprocess
        import sys
        
        # Create a minimal training script that will be run in a subprocess
        # This avoids all the import compatibility issues between mmcv/mmseg/mmdet
        train_script = f'''
import sys
sys.path.insert(0, "{Path(__file__).parent}")

# Import custom transforms and datasets to register them BEFORE loading config
import custom_transforms  # Registers ReduceToSingleChannel transform
import unified_datasets   # Registers MapillaryLabelTransform, CityscapesLabelTransform

# Import mmsegmentation components carefully to avoid mmcv._ext issues
try:
    import mmseg.datasets  
    import mmseg.evaluation
    # Import specific model classes to register them
    from mmseg.models.segmentors import EncoderDecoder
    from mmseg.models.segmentors import CascadeEncoderDecoder
except ImportError as e:
    print(f"Warning: Some mmseg imports failed: {{e}}")

from mmengine.runner import Runner
from mmengine.config import Config

cfg = Config.fromfile("{config_path}")
runner = Runner.from_cfg(cfg)
runner.train()
'''
        
        # Write temporary training script
        script_path = Path(self.config['work_dir']) / 'train_script.py'
        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w') as f:
            f.write(train_script)
        
        # Run training in subprocess
        print(f"Starting training via subprocess...")
        print(f"Config: {config_path}")
        print(f"Work dir: {self.config['work_dir']}")
        
        env = os.environ.copy()
        # Ensure CUDA is visible
        if 'CUDA_VISIBLE_DEVICES' not in env and self.gpu_ids:
            env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.gpu_ids))
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            env=env,
            cwd=str(Path(__file__).parent),
        )
        
        return result.returncode == 0
    
    def _train_with_mmseg_tools(self, config_path: str) -> bool:
        """Train using mmsegmentation's train.py script directly"""
        import subprocess
        import sys
        
        # Find mmseg installation path
        try:
            import mmseg
            mmseg_path = Path(mmseg.__file__).parent.parent
        except ImportError:
            # Fallback: try to use mim
            print("Using mim to run training...")
            cmd = [
                sys.executable, '-m', 'mim', 'train', 'mmsegmentation',
                config_path,
                '--work-dir', self.config['work_dir'],
            ]
            if self.gpu_ids:
                cmd.extend(['--gpus', str(len(self.gpu_ids))])
            
            result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
            return result.returncode == 0
        
        # Use mmseg tools/train.py
        train_script = mmseg_path / 'tools' / 'train.py'
        if not train_script.exists():
            print(f"Warning: mmseg train.py not found at {train_script}")
            print("Falling back to mim...")
            return self._train_with_mim(config_path)
        
        cmd = [
            sys.executable, str(train_script),
            config_path,
            '--work-dir', self.config['work_dir'],
        ]
        
        env = os.environ.copy()
        if 'CUDA_VISIBLE_DEVICES' not in env and self.gpu_ids:
            env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.gpu_ids))
        
        result = subprocess.run(cmd, env=env, cwd=str(Path(__file__).parent))
        return result.returncode == 0
    
    def _train_with_mim(self, config_path: str) -> bool:
        """Train using mim (OpenMMLab package manager)"""
        import subprocess
        import sys
        
        cmd = [
            sys.executable, '-m', 'mim', 'train', 'mmsegmentation',
            config_path,
            '--work-dir', self.config['work_dir'],
        ]
        
        if self.gpu_ids:
            cmd.extend(['--gpus', str(len(self.gpu_ids))])
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
        return result.returncode == 0

    def _train_with_mmseg(self, config_path: str) -> bool:
        """Train using MMSegmentation (legacy API)"""
        from mmseg.apis import train_segmentor, init_segmentor
        from mmseg.datasets import build_dataset
        from mmseg.models import build_segmentor
        from mmcv import Config
        
        cfg = Config.fromfile(config_path)
        
        # Build dataset
        datasets = [build_dataset(cfg.data.train)]
        
        # Build model
        model = build_segmentor(cfg.model)
        
        # Train
        train_segmentor(
            model,
            datasets,
            cfg,
            distributed=self.distributed,
            validate=True,
        )
        
        return True
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training configuration"""
        # Handle both old and new config formats
        if 'train_cfg' in self.config and 'max_iters' in self.config['train_cfg']:
            max_iters = self.config['train_cfg']['max_iters']
        elif 'runner' in self.config:
            max_iters = self.config['runner']['max_iters']
        else:
            max_iters = 40000  # default
            
        if 'train_dataloader' in self.config:
            batch_size = self.config['train_dataloader']['batch_size']
        elif 'data' in self.config:
            batch_size = self.config['data']['samples_per_gpu']
        else:
            batch_size = 16  # default
            
        return {
            'dataset': self.dataset,
            'model': self.model,
            'strategy': self.strategy,
            'std_strategy': self.std_strategy,
            'real_gen_ratio': self.real_gen_ratio,
            'work_dir': self.config['work_dir'],
            'max_iters': max_iters,
            'batch_size': batch_size,
            'mixed_training': self.config.get('mixed_dataloader', {}).get('enabled', False),
            'conditions': self.custom_conditions or ADVERSE_CONDITIONS,
        }


class UnifiedMultiTrainer:
    """
    Multi-dataset training orchestrator for PROVE pipeline.
    
    Enables joint training on multiple datasets simultaneously (e.g., ACDC + MapillaryVistas).
    
    Args:
        datasets: List of dataset names (e.g., ['ACDC', 'MapillaryVistas'])
        model: Model name (e.g., 'deeplabv3plus_r50')
        strategy: Augmentation strategy (e.g., 'baseline', 'gen_cycleGAN')
        std_strategy: Optional standard augmentation to combine with gen_* strategy
                     (e.g., 'std_cutmix', 'std_mixup'). When provided with a gen_* 
                     strategy, both augmentations will be applied.
        weights: Optional sampling weights for each dataset (must sum to 1.0)
        real_gen_ratio: Ratio of real images (0.0-1.0). Default: 1.0
        custom_conditions: Optional list of weather conditions to use
        work_dir: Output directory for checkpoints and logs
        cache_dir: Directory for caching pretrained weights
        seed: Random seed for reproducibility. Default: 42
        early_stop: Whether to enable early stopping. Default: True
        early_stop_patience: Number of validations without improvement before stopping
        gpu_ids: List of GPU IDs to use. Default: [0]
        distributed: Whether to use distributed training. Default: False
    """
    
    def __init__(
        self,
        datasets: List[str],
        model: str,
        strategy: str = 'baseline',
        std_strategy: Optional[str] = None,
        weights: Optional[List[float]] = None,
        real_gen_ratio: float = 1.0,
        custom_conditions: Optional[List[str]] = None,
        work_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        seed: int = 42,
        early_stop: bool = True,
        early_stop_patience: int = 5,
        gpu_ids: List[int] = None,
        distributed: bool = False,
    ):
        self.datasets = datasets
        self.model = model
        self.strategy = strategy
        self.std_strategy = std_strategy
        self.weights = weights
        self.real_gen_ratio = real_gen_ratio
        self.custom_conditions = custom_conditions
        self.work_dir = work_dir
        self.cache_dir = cache_dir
        self.seed = seed
        self.early_stop = early_stop
        self.early_stop_patience = early_stop_patience
        self.gpu_ids = gpu_ids or [0]
        self.distributed = distributed
        
        # Initialize config builder
        self.config_builder = UnifiedTrainingConfig(cache_dir=cache_dir)
        
        # Build configuration
        self.config = self._build_config()
    
    def _build_config(self) -> Dict[str, Any]:
        """Build multi-dataset training configuration"""
        custom_training_config = {
            'early_stop': self.early_stop,
            'early_stop_patience': self.early_stop_patience,
        }
        
        config = self.config_builder.build_multi_dataset(
            datasets=self.datasets,
            model=self.model,
            strategy=self.strategy,
            std_strategy=self.std_strategy,
            real_gen_ratio=self.real_gen_ratio,
            weights=self.weights,
            custom_conditions=self.custom_conditions,
            custom_training_config=custom_training_config,
        )
        
        # Override work_dir if specified
        if self.work_dir:
            config['work_dir'] = self.work_dir
        
        # Override seed
        config['seed'] = self.seed
        config['gpu_ids'] = self.gpu_ids
        
        return config
    
    def save_config(self, filepath: Optional[str] = None) -> str:
        """Save configuration to file"""
        if filepath is None:
            config_dir = Path(self.config['work_dir']) / 'configs'
            config_dir.mkdir(parents=True, exist_ok=True)
            filepath = str(config_dir / 'training_config.py')
        
        return self.config_builder.save_config(self.config, filepath)
    
    def train(self, method: str = 'subprocess'):
        """Execute multi-dataset training."""
        # Save config first
        config_path = self.save_config()
        print(f"Configuration saved to: {config_path}")
        
        # Create work directory
        work_dir = Path(self.config['work_dir'])
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Run training
        print("\n" + "=" * 60)
        print("Starting PROVE Multi-Dataset Training")
        print("=" * 60)
        print(f"Datasets: {', '.join(self.datasets)}")
        if self.weights:
            weight_strs = [f"{d}: {w:.1%}" for d, w in zip(self.datasets, self.weights)]
            print(f"Weights: {', '.join(weight_strs)}")
        print(f"Model: {self.model}")
        if self.std_strategy:
            print(f"Strategy: {self.strategy} + {self.std_strategy}")
        else:
            print(f"Strategy: {self.strategy}")
        print(f"Real/Gen Ratio: {self.real_gen_ratio}")
        print(f"Work Dir: {self.config['work_dir']}")
        print("=" * 60 + "\n")
        
        return self._train_with_mmengine(config_path)
    
    def _train_with_mmengine(self, config_path: str) -> bool:
        """Train using MMEngine via subprocess"""
        import subprocess
        import sys
        
        train_script = f'''
import sys
sys.path.insert(0, "{Path(__file__).parent}")

# Import custom transforms and datasets to register them BEFORE loading config
import custom_transforms
import unified_datasets

try:
    import mmseg.datasets  
    import mmseg.evaluation
    from mmseg.models.segmentors import EncoderDecoder
except ImportError as e:
    print(f"Warning: Some mmseg imports failed: {{e}}")

from mmengine.runner import Runner
from mmengine.config import Config

cfg = Config.fromfile("{config_path}")
runner = Runner.from_cfg(cfg)
runner.train()
'''
        
        # Write temporary training script
        script_path = Path(self.config['work_dir']) / 'train_script.py'
        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w') as f:
            f.write(train_script)
        
        # Run training
        print(f"Starting training via subprocess...")
        
        env = os.environ.copy()
        if 'CUDA_VISIBLE_DEVICES' not in env and self.gpu_ids:
            env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.gpu_ids))
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            env=env,
            cwd=str(Path(__file__).parent),
        )
        
        return result.returncode == 0
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of multi-dataset training configuration"""
        max_iters = self.config.get('train_cfg', {}).get('max_iters', 40000)
        batch_size = self.config.get('train_dataloader', {}).get('batch_size', 2)
        
        summary = {
            'datasets': self.datasets,
            'datasets_str': '+'.join(self.datasets),
            'model': self.model,
            'strategy': self.strategy,
            'weights': self.weights,
            'real_gen_ratio': self.real_gen_ratio,
            'work_dir': self.config['work_dir'],
            'max_iters': max_iters,
            'batch_size': batch_size,
            'multi_dataset': True,
        }
        
        if self.std_strategy:
            summary['std_strategy'] = self.std_strategy
            summary['combined_strategy'] = f"{self.strategy}+{self.std_strategy}"
        
        return summary



# ============================================================================
# Job Submission
# ============================================================================

def generate_lsf_script(
    dataset: str,
    model: str,
    strategy: str,
    real_gen_ratio: float = 1.0,
    gpu_count: int = 1,
    memory: str = '32GB',
    time_limit: str = '24:00',
    queue: str = 'gpu',
) -> str:
    """Generate LSF job submission script"""
    
    job_name = f"prove_{dataset}_{model}_{strategy}"
    if real_gen_ratio < 1.0:
        job_name += f"_r{int(real_gen_ratio*100)}"
    
    script = f'''#!/bin/bash
#BSUB -J {job_name}
#BSUB -q {queue}
#BSUB -n {gpu_count}
#BSUB -R "rusage[mem={memory}]"
#BSUB -W {time_limit}
#BSUB -o logs/{job_name}_%J.out
#BSUB -e logs/{job_name}_%J.err

# PROVE Training Job
# Generated: {datetime.now().isoformat()}

# Load modules (adjust for your cluster)
module load cuda/11.7
module load python/3.9

# Activate environment
mamba activate prove

# Navigate to project
cd $HOME/repositories/PROVE

# Create log directory
mkdir -p logs

# Run training
python unified_training.py \\
    --dataset {dataset} \\
    --model {model} \\
    --strategy {strategy} \\
    --real-gen-ratio {real_gen_ratio}

echo "Training completed: {job_name}"
'''
    return script


def generate_slurm_script(
    dataset: str,
    model: str,
    strategy: str,
    real_gen_ratio: float = 1.0,
    gpu_count: int = 1,
    memory: str = '32G',
    time_limit: str = '24:00:00',
    partition: str = 'gpu',
) -> str:
    """Generate SLURM job submission script"""
    
    job_name = f"prove_{dataset}_{model}_{strategy}"
    if real_gen_ratio < 1.0:
        job_name += f"_r{int(real_gen_ratio*100)}"
    
    script = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpu_count}
#SBATCH --mem={memory}
#SBATCH --time={time_limit}
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err

# PROVE Training Job
# Generated: {datetime.now().isoformat()}

# Load modules (adjust for your cluster)
module load cuda/11.7
module load python/3.9

# Activate environment
source ~/.virtualenvs/prove/bin/activate

# Navigate to project
cd $HOME/repositories/PROVE

# Create log directory
mkdir -p logs

# Run training
python unified_training.py \\
    --dataset {dataset} \\
    --model {model} \\
    --strategy {strategy} \\
    --real-gen-ratio {real_gen_ratio}

echo "Training completed: {job_name}"
'''
    return script


# ============================================================================
# Batch Training
# ============================================================================

def run_batch_training(
    datasets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    strategies: Optional[List[str]] = None,
    ratios: Optional[List[float]] = None,
    parallel: bool = False,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run training for multiple configurations.
    
    Args:
        datasets: List of datasets to train on
        models: List of models to train
        strategies: List of augmentation strategies
        ratios: List of real-to-generated ratios
        parallel: Whether to run jobs in parallel
        dry_run: Only print commands, don't execute
        
    Returns:
        List of training results
    """
    datasets = datasets or list(DATASET_CONFIGS.keys())
    strategies = strategies or ['baseline']
    ratios = ratios or [1.0]
    
    results = []
    commands = []
    
    for dataset in datasets:
        # Get appropriate models for this dataset
        # Handle multi-dataset case (starts with "multi_")
        if dataset.startswith('multi_'):
            # Multi-datasets are always segmentation for now
            task = 'segmentation'
        else:
            task = DATASET_CONFIGS[dataset].task
        
        available_models = (
            list(SEGMENTATION_MODELS.keys()) if task == 'segmentation'
            else list(DETECTION_MODELS.keys())
        )
        dataset_models = models if models else available_models
        
        for model in dataset_models:
            if model not in ALL_MODELS:
                continue
            
            for strategy in strategies:
                for ratio in ratios:
                    # Handle multi-dataset case differently
                    if dataset.startswith('multi_'):
                        # Extract individual datasets from multi_dataset1+dataset2+... format
                        dataset_names = dataset.replace('multi_', '').split('+')
                        cmd = [
                            'python', 'unified_training.py',
                            '--multi-dataset',
                            '--datasets', *dataset_names,
                            '--model', model,
                            '--strategy', strategy,
                            '--real-gen-ratio', str(ratio),
                        ]
                    else:
                        cmd = [
                            'python', 'unified_training.py',
                            '--dataset', dataset,
                            '--model', model,
                            '--strategy', strategy,
                            '--real-gen-ratio', str(ratio),
                        ]
                    commands.append({
                        'cmd': cmd,
                        'dataset': dataset,
                        'model': model,
                        'strategy': strategy,
                        'ratio': ratio,
                    })
    
    print(f"Batch training: {len(commands)} configurations")
    print("=" * 60)
    
    for i, cmd_info in enumerate(commands):
        print(f"{i+1}. {cmd_info['dataset']}/{cmd_info['model']} - {cmd_info['strategy']} (ratio={cmd_info['ratio']})")
    
    if dry_run:
        print("\nDry run - no training executed")
        return results
    
    print("\nStarting training...")
    
    if parallel:
        # Submit all jobs
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for cmd_info in commands:
                future = executor.submit(subprocess.run, cmd_info['cmd'])
                futures.append((cmd_info, future))
            
            for cmd_info, future in futures:
                try:
                    result = future.result()
                    results.append({
                        **cmd_info,
                        'success': result.returncode == 0,
                    })
                except Exception as e:
                    results.append({
                        **cmd_info,
                        'success': False,
                        'error': str(e),
                    })
    else:
        # Run sequentially
        for cmd_info in commands:
            print(f"\nTraining: {cmd_info['dataset']}/{cmd_info['model']}")
            try:
                result = subprocess.run(cmd_info['cmd'])
                results.append({
                    **cmd_info,
                    'success': result.returncode == 0,
                })
            except Exception as e:
                results.append({
                    **cmd_info,
                    'success': False,
                    'error': str(e),
                })
    
    return results


# ============================================================================
# Command Line Interface
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='PROVE Unified Training Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train baseline
  python unified_training.py --dataset ACDC --model deeplabv3plus_r50

  # Train with cycleGAN augmentation
  python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN

  # Train with 50%% real, 50%% generated
  python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN --real-gen-ratio 0.5

  # Train with standard augmentation only (CutMix)
  python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy std_cutmix

  # Combine gen_* strategy with standard augmentation (gen + std)
  python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN --std-strategy std_cutmix

  # Train only on clear_day images
  python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --domain-filter clear_day

  # Generate config only
  python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --config-only

  # Multi-dataset training (joint training on ACDC + Mapillary)
  python unified_training.py --multi-dataset --datasets ACDC MapillaryVistas --model deeplabv3plus_r50

  # Multi-dataset with custom weights (70%% ACDC, 30%% Mapillary)
  python unified_training.py --multi-dataset --datasets ACDC MapillaryVistas --weights 0.7 0.3 --model deeplabv3plus_r50
  

  # Batch training
  python unified_training.py --batch --datasets ACDC BDD10k --strategies baseline gen_cycleGAN

  # Generate job script
  python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --generate-job-script lsf
        """
    )
    
    # Required arguments (for single training)
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--model', type=str, help='Model name')
    
    # Training configuration
    parser.add_argument('--strategy', type=str, default='baseline',
                       help='Main augmentation strategy (baseline, gen_*, or std_*)')
    parser.add_argument('--std-strategy', type=str, default=None,
                       help='Standard augmentation to combine with gen_* strategy '
                            '(e.g., std_cutmix, std_mixup, std_autoaugment, std_randaugment). '
                            'When used with a gen_* strategy, both augmentations are applied.')
    parser.add_argument('--real-gen-ratio', type=float, default=1.0,
                       help='Ratio of real images (0.0-1.0)')
    parser.add_argument('--domain-filter', type=str, default=None,
                       help='Filter training data to specific domain (e.g., clear_day)')
    parser.add_argument('--conditions', type=str, nargs='+',
                       help='Weather conditions to use')
    
    # Training options
    parser.add_argument('--work-dir', type=str, help='Output directory')
    parser.add_argument('--cache-dir', type=str, 
                       help='Directory for caching pretrained weights and checkpoints')
    parser.add_argument('--resume-from', type=str, help='Checkpoint to resume from')
    parser.add_argument('--load-from', type=str, help='Pretrained weights to load')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-early-stop', action='store_true',
                       help='Disable early stopping (enabled by default)')
    parser.add_argument('--early-stop-patience', type=int, default=5,
                       help='Early stopping patience (number of validations without improvement)')
    parser.add_argument('--max-iters', type=int, default=None,
                       help='Maximum training iterations (default: 80000 for segmentation, 40000 for detection)')
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0],
                       help='GPU IDs to use')
    parser.add_argument('--distributed', action='store_true',
                       help='Use distributed training')
    parser.add_argument('--train-method', type=str, default='subprocess',
                       choices=['subprocess', 'mim', 'mmseg_tools', 'direct'],
                       help='Training method: subprocess (recommended), mim, mmseg_tools, or direct')
    
    # Output options
    parser.add_argument('--config-only', action='store_true',
                       help='Only generate config, do not train')
    parser.add_argument('--config-output', type=str,
                       help='Path to save generated config')
    
    # Multi-dataset training
    parser.add_argument('--multi-dataset', action='store_true',
                       help='Enable multi-dataset joint training mode')
    parser.add_argument('--weights', type=float, nargs='+',
                       help='Sampling weights for each dataset (must sum to 1.0)')
    
    # Batch training
    parser.add_argument('--batch', action='store_true',
                       help='Run batch training')
    parser.add_argument('--datasets', type=str, nargs='+',
                       help='Datasets for multi-dataset or batch training')
    parser.add_argument('--models', type=str, nargs='+',
                       help='Models for batch training')
    parser.add_argument('--strategies', type=str, nargs='+',
                       help='Strategies for batch training')
    parser.add_argument('--ratios', type=float, nargs='+',
                       help='Ratios for batch training')
    parser.add_argument('--all-seg-datasets', action='store_true',
                       help='Use all segmentation datasets for batch training')
    parser.add_argument('--all-det-datasets', action='store_true',
                       help='Use all detection datasets for batch training')
    parser.add_argument('--unified-seg-dataset', action='store_true',
                       help='Use unified segmentation dataset (BDD10k+MapillaryVistas+IDD-AW, without ACDC)')
    parser.add_argument('--all-seg-models', action='store_true',
                       help='Use all segmentation models for batch training')
    parser.add_argument('--all-det-models', action='store_true',
                       help='Use all detection models for batch training')
    parser.add_argument('--parallel', action='store_true',
                       help='Run batch jobs in parallel')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without executing')
    
    # Job submission
    parser.add_argument('--generate-job-script', type=str, choices=['lsf', 'slurm'],
                       help='Generate job submission script')
    parser.add_argument('--submit-job', action='store_true',
                       help='Submit job to cluster')
    
    # Info
    parser.add_argument('--list', action='store_true',
                       help='List available options')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # List available options
    if args.list:
        config_builder = UnifiedTrainingConfig()
        options = config_builder.get_available_options()
        print("PROVE Unified Training - Available Options")
        print("=" * 60)
        print("\nDatasets:")
        for ds in options['datasets']:
            task = DATASET_CONFIGS[ds].task
            print(f"  - {ds} ({task})")
        print("\nSegmentation Models:")
        for m in options['segmentation_models']:
            print(f"  - {m}")
        print("\nDetection Models:")
        for m in options['detection_models']:
            print(f"  - {m}")
        print("\nStrategies:")
        for s in options['strategies'][:10]:  # Show first 10
            print(f"  - {s}")
        if len(options['strategies']) > 10:
            print(f"  ... and {len(options['strategies'])-10} more")
        return
    
    # Batch training
    if args.batch:
        # Expand --all-* options
        datasets = args.datasets or []
        models = args.models or []
        
        # Handle segmentation datasets and models
        if args.all_seg_datasets:
            seg_datasets = [name for name, cfg in DATASET_CONFIGS.items() if cfg.task == 'segmentation']
            datasets.extend(seg_datasets)
        
        if args.unified_seg_dataset:
            # Removed ACDC from multi-dataset training
            # ACDC uses Cityscapes label IDs which adds complexity, and was causing issues
            datasets.append('multi_mapillaryvistas+idd-aw+bdd10k')
        
        if args.all_seg_models:
            models.extend(list(SEGMENTATION_MODELS.keys()))
        
        # Handle detection datasets and models
        if args.all_det_datasets:
            det_datasets = [name for name, cfg in DATASET_CONFIGS.items() if cfg.task == 'detection']
            datasets.extend(det_datasets)
        
        if args.all_det_models:
            models.extend(list(DETECTION_MODELS.keys()))
        
        # Remove duplicates while preserving order
        datasets = list(dict.fromkeys(datasets)) if datasets else None
        models = list(dict.fromkeys(models)) if models else None
        
        results = run_batch_training(
            datasets=datasets,
            models=models,
            strategies=args.strategies,
            ratios=args.ratios,
            parallel=args.parallel,
            dry_run=args.dry_run,
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("Batch Training Summary")
        print("=" * 60)
        success = sum(1 for r in results if r.get('success', False))
        print(f"Successful: {success}/{len(results)}")
        return
    
    # Multi-dataset training mode
    if args.multi_dataset:
        if not args.datasets or len(args.datasets) < 2:
            print("Error: --multi-dataset requires at least 2 datasets via --datasets")
            print("Example: --multi-dataset --datasets ACDC MapillaryVistas --model deeplabv3plus_r50")
            return
        
        if not args.model:
            print("Error: --model is required for multi-dataset training")
            return
        
        # Validate weights if provided
        if args.weights:
            if len(args.weights) != len(args.datasets):
                print(f"Error: Number of weights ({len(args.weights)}) must match number of datasets ({len(args.datasets)})")
                return
            weight_sum = sum(args.weights)
            if not (0.99 <= weight_sum <= 1.01):
                print(f"Error: Weights must sum to 1.0, got {weight_sum}")
                return
        
        # Create multi-dataset trainer
        trainer = UnifiedMultiTrainer(
            datasets=args.datasets,
            model=args.model,
            strategy=args.strategy,
            weights=args.weights,
            real_gen_ratio=args.real_gen_ratio,
            custom_conditions=args.conditions,
            work_dir=args.work_dir,
            cache_dir=args.cache_dir,
            seed=args.seed,
            early_stop=not args.no_early_stop,
            early_stop_patience=args.early_stop_patience,
            gpu_ids=args.gpu_ids,
            distributed=args.distributed,
        )
        
        # Config only mode
        if args.config_only:
            config_path = trainer.save_config(args.config_output)
            print(f"Multi-dataset configuration saved to: {config_path}")
            
            # Print summary
            summary = trainer.get_training_summary()
            print("\nMulti-Dataset Training Summary:")
            for key, value in summary.items():
                print(f"  {key}: {value}")
            return
        
        # Run multi-dataset training
        trainer.train(method=args.train_method)
        return
    
    # Single training - validate required arguments
    if not args.dataset or not args.model:
        print("Error: --dataset and --model are required for single training")
        print("Use --multi-dataset for joint training or --list to see options")
        return
    
    # Generate job script
    if args.generate_job_script:
        if args.generate_job_script == 'lsf':
            script = generate_lsf_script(
                args.dataset, args.model, args.strategy, args.real_gen_ratio
            )
        else:
            script = generate_slurm_script(
                args.dataset, args.model, args.strategy, args.real_gen_ratio
            )
        
        script_name = f"job_{args.dataset}_{args.model}_{args.strategy}.sh"
        with open(script_name, 'w') as f:
            f.write(script)
        os.chmod(script_name, 0o755)
        print(f"Job script saved to: {script_name}")
        return
    
    # Create trainer
    trainer = UnifiedTrainer(
        dataset=args.dataset,
        model=args.model,
        strategy=args.strategy,
        std_strategy=args.std_strategy,
        real_gen_ratio=args.real_gen_ratio,
        custom_conditions=args.conditions,
        domain_filter=args.domain_filter,
        work_dir=args.work_dir,
        cache_dir=args.cache_dir,
        resume_from=args.resume_from,
        load_from=args.load_from,
        seed=args.seed,
        early_stop=not args.no_early_stop,
        early_stop_patience=args.early_stop_patience,
        max_iters=args.max_iters,
        gpu_ids=args.gpu_ids,
        distributed=args.distributed,
    )
    
    # Config only mode
    if args.config_only:
        config_path = trainer.save_config(args.config_output)
        print(f"Configuration saved to: {config_path}")
        
        # Print summary
        summary = trainer.get_training_summary()
        print("\nTraining Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        return
    
    # Submit job
    if args.submit_job:
        # Generate and submit job script
        script = generate_lsf_script(
            args.dataset, args.model, args.strategy, args.real_gen_ratio
        )
        script_name = f"/tmp/prove_job_{args.dataset}_{args.model}.sh"
        with open(script_name, 'w') as f:
            f.write(script)
        
        result = subprocess.run(['bsub', '<', script_name], shell=True)
        if result.returncode == 0:
            print("Job submitted successfully")
        else:
            print("Job submission failed")
        return
    
    # Run training
    trainer.train(method=args.train_method)


if __name__ == '__main__':
    main()
