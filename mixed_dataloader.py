#!/usr/bin/env python3
"""
PROVE Mixed Dataloader System

This module provides a custom dataloader that combines real and generated
images with configurable sampling ratios. It enables fine-grained control
over the composition of training batches.

Features:
- Dual dataloader architecture (real + generated)
- Configurable real-to-generated image ratio
- Multiple sampling strategies (ratio, alternating, batch_split)
- Compatible with MMSegmentation and MMDetection frameworks

Usage:
    from mixed_dataloader import MixedDataLoader, MixedDataset
    
    # Create mixed dataloader with 50% real, 50% generated
    dataloader = MixedDataLoader(
        real_dataset=real_dataset,
        generated_dataset=gen_dataset,
        real_gen_ratio=0.5,
        batch_size=4,
        sampling_strategy='batch_split'
    )
"""

import os
import random
import numpy as np
from typing import Dict, List, Optional, Iterator, Any, Union, Tuple
from dataclasses import dataclass

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, Sampler, IterableDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from mmseg.datasets import BaseSegDataset
    from mmseg.registry import DATASETS, DATA_SAMPLERS
    MMSEG_AVAILABLE = True
except ImportError:
    MMSEG_AVAILABLE = False
    DATASETS = None
    DATA_SAMPLERS = None


# ============================================================================
# Sampling Strategies
# ============================================================================

@dataclass
class SamplingConfig:
    """Configuration for mixed sampling"""
    real_gen_ratio: float = 0.5  # Ratio of real images (0.0 to 1.0)
    batch_size: int = 4
    strategy: str = 'batch_split'  # 'ratio', 'alternating', 'batch_split'
    shuffle: bool = True
    drop_last: bool = False


class RatioSampler:
    """
    Sampler that selects from real or generated dataset based on ratio.
    
    For each sample, randomly decides whether to use real or generated
    based on the configured ratio.
    """
    
    def __init__(
        self,
        real_dataset_size: int,
        generated_dataset_size: int,
        real_gen_ratio: float = 0.5,
        total_samples: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.real_size = real_dataset_size
        self.gen_size = generated_dataset_size
        self.real_gen_ratio = real_gen_ratio
        self.total_samples = total_samples or (real_dataset_size + generated_dataset_size)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        self._generate_indices()
    
    def _generate_indices(self):
        """Generate sampling indices for current epoch"""
        rng = random.Random(self.seed + self.epoch)
        
        self.indices = []
        real_indices = list(range(self.real_size))
        gen_indices = list(range(self.gen_size))
        
        if self.shuffle:
            rng.shuffle(real_indices)
            rng.shuffle(gen_indices)
        
        real_ptr = 0
        gen_ptr = 0
        
        for _ in range(self.total_samples):
            use_real = rng.random() < self.real_gen_ratio
            
            if use_real and real_ptr < len(real_indices):
                self.indices.append(('real', real_indices[real_ptr]))
                real_ptr += 1
            elif gen_ptr < len(gen_indices):
                self.indices.append(('generated', gen_indices[gen_ptr]))
                gen_ptr += 1
            elif real_ptr < len(real_indices):
                self.indices.append(('real', real_indices[real_ptr]))
                real_ptr += 1
            else:
                # Wrap around
                real_ptr = 0
                gen_ptr = 0
                if self.shuffle:
                    rng.shuffle(real_indices)
                    rng.shuffle(gen_indices)
    
    def __iter__(self) -> Iterator[Tuple[str, int]]:
        return iter(self.indices)
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def set_epoch(self, epoch: int):
        """Set epoch for shuffling"""
        self.epoch = epoch
        self._generate_indices()


class BatchSplitSampler:
    """
    Sampler that constructs batches with fixed numbers of real and generated samples.
    
    Each batch contains exactly N real and M generated samples, where N and M
    are determined by the ratio and batch size.
    """
    
    def __init__(
        self,
        real_dataset_size: int,
        generated_dataset_size: int,
        batch_size: int = 4,
        real_gen_ratio: float = 0.5,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.real_size = real_dataset_size
        self.gen_size = generated_dataset_size
        self.batch_size = batch_size
        self.real_gen_ratio = real_gen_ratio
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        
        # Calculate samples per batch
        # When ratio=0.0, should have 0 real samples (100% generated)
        # When ratio=1.0, should have batch_size real samples (100% real)
        self.real_per_batch = int(batch_size * real_gen_ratio)
        self.gen_per_batch = batch_size - self.real_per_batch
        
        self._generate_batches()
    
    def _generate_batches(self):
        """Generate batch indices"""
        rng = random.Random(self.seed + self.epoch)
        
        real_indices = list(range(self.real_size))
        gen_indices = list(range(self.gen_size))
        
        if self.shuffle:
            rng.shuffle(real_indices)
            rng.shuffle(gen_indices)
        
        self.batches = []
        real_ptr = 0
        gen_ptr = 0
        
        while True:
            batch = []
            
            # Add real samples
            for _ in range(self.real_per_batch):
                if real_ptr >= len(real_indices):
                    real_ptr = 0
                    if self.shuffle:
                        rng.shuffle(real_indices)
                batch.append(('real', real_indices[real_ptr]))
                real_ptr += 1
            
            # Add generated samples
            for _ in range(self.gen_per_batch):
                if gen_ptr >= len(gen_indices):
                    gen_ptr = 0
                    if self.shuffle:
                        rng.shuffle(gen_indices)
                batch.append(('generated', gen_indices[gen_ptr]))
                gen_ptr += 1
            
            self.batches.append(batch)
            
            # Check if we've covered all data
            # When real_per_batch=0 (ratio=0.0), only check generated
            # When gen_per_batch=0 (ratio=1.0), only check real
            real_done = (self.real_per_batch == 0) or (real_ptr >= len(real_indices))
            gen_done = (self.gen_per_batch == 0) or (gen_ptr >= len(gen_indices))
            if real_done and gen_done:
                break
            
            # Limit batches to avoid infinite loop
            max_batches = max(len(real_indices), len(gen_indices)) // self.batch_size + 10
            if len(self.batches) >= max_batches:
                break
        
        if self.drop_last and len(self.batches[-1]) < self.batch_size:
            self.batches.pop()
    
    def __iter__(self) -> Iterator[List[Tuple[str, int]]]:
        return iter(self.batches)
    
    def __len__(self) -> int:
        return len(self.batches)
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._generate_batches()


# ============================================================================
# Mixed Dataset
# ============================================================================

if TORCH_AVAILABLE:
    
    class MixedDataset(Dataset):
        """
        Dataset that wraps real and generated datasets for mixed sampling.
        
        This dataset provides a unified interface for accessing both real and
        generated images. The actual sampling strategy is handled by the sampler.
        
        Args:
            real_dataset: Dataset containing real images
            generated_dataset: Dataset containing generated images
            transform: Optional transform to apply to all samples
        """
        
        def __init__(
            self,
            real_dataset: Dataset,
            generated_dataset: Dataset,
            transform: Optional[Any] = None,
        ):
            self.real_dataset = real_dataset
            self.generated_dataset = generated_dataset
            self.transform = transform
        
        def __getitem__(self, index: Union[int, Tuple[str, int]]) -> Dict[str, Any]:
            """
            Get item by index.
            
            Index can be:
            - int: Traditional integer index (uses real dataset for < real_size, else generated)
            - Tuple[str, int]: ('real'/'generated', idx) for explicit dataset selection
            """
            if isinstance(index, tuple):
                source, idx = index
                if source == 'real':
                    data = self.real_dataset[idx]
                else:
                    data = self.generated_dataset[idx]
            else:
                # Fallback to simple indexing
                if index < len(self.real_dataset):
                    data = self.real_dataset[index]
                else:
                    data = self.generated_dataset[index - len(self.real_dataset)]
            
            # Mark the source
            if isinstance(data, dict):
                data['_source'] = 'real' if (isinstance(index, tuple) and index[0] == 'real') or (isinstance(index, int) and index < len(self.real_dataset)) else 'generated'
            
            if self.transform:
                data = self.transform(data)
            
            return data
        
        def __len__(self) -> int:
            return len(self.real_dataset) + len(self.generated_dataset)
        
        @property
        def real_size(self) -> int:
            return len(self.real_dataset)
        
        @property
        def generated_size(self) -> int:
            return len(self.generated_dataset)


# ============================================================================
# Mixed DataLoader
# ============================================================================

if TORCH_AVAILABLE:
    
    class MixedDataLoader:
        """
        DataLoader that combines real and generated datasets with ratio-based sampling.
        
        This class provides a drop-in replacement for standard DataLoader that
        samples from two datasets according to a configurable ratio.
        
        Args:
            real_dataset: Dataset containing real images
            generated_dataset: Dataset containing generated images
            real_gen_ratio: Ratio of real images in each batch (0.0 to 1.0)
            batch_size: Number of samples per batch
            sampling_strategy: How to sample ('ratio', 'batch_split', 'alternating')
            shuffle: Whether to shuffle data
            num_workers: Number of data loading workers
            drop_last: Whether to drop incomplete batches
            seed: Random seed for reproducibility
        """
        
        def __init__(
            self,
            real_dataset: Dataset,
            generated_dataset: Dataset,
            real_gen_ratio: float = 0.5,
            batch_size: int = 4,
            sampling_strategy: str = 'batch_split',
            shuffle: bool = True,
            num_workers: int = 4,
            drop_last: bool = False,
            seed: int = 42,
            **kwargs
        ):
            self.real_dataset = real_dataset
            self.generated_dataset = generated_dataset
            self.real_gen_ratio = real_gen_ratio
            self.batch_size = batch_size
            self.sampling_strategy = sampling_strategy
            self.shuffle = shuffle
            self.num_workers = num_workers
            self.drop_last = drop_last
            self.seed = seed
            self.epoch = 0
            
            # Create mixed dataset
            self.mixed_dataset = MixedDataset(real_dataset, generated_dataset)
            
            # Create sampler based on strategy
            self._create_sampler()
            
            # Create the actual DataLoader
            self._create_dataloader()
        
        def _create_sampler(self):
            """Create the appropriate sampler based on strategy"""
            if self.sampling_strategy == 'batch_split':
                self.sampler = BatchSplitSampler(
                    real_dataset_size=len(self.real_dataset),
                    generated_dataset_size=len(self.generated_dataset),
                    batch_size=self.batch_size,
                    real_gen_ratio=self.real_gen_ratio,
                    shuffle=self.shuffle,
                    drop_last=self.drop_last,
                    seed=self.seed,
                )
            else:  # ratio or alternating
                self.sampler = RatioSampler(
                    real_dataset_size=len(self.real_dataset),
                    generated_dataset_size=len(self.generated_dataset),
                    real_gen_ratio=self.real_gen_ratio,
                    shuffle=self.shuffle,
                    seed=self.seed,
                )
        
        def _create_dataloader(self):
            """Create the underlying DataLoader"""
            # Custom collate function that handles tuple indices
            def collate_fn(batch_indices):
                batch = []
                for idx in batch_indices:
                    if isinstance(idx, list):
                        # BatchSplitSampler returns list of tuples
                        for item_idx in idx:
                            batch.append(self.mixed_dataset[item_idx])
                    else:
                        batch.append(self.mixed_dataset[idx])
                return self._default_collate(batch)
            
            if self.sampling_strategy == 'batch_split':
                # For batch_split, sampler returns complete batches
                self.dataloader = self.sampler
            else:
                self.dataloader = DataLoader(
                    dataset=range(len(self.sampler)),
                    batch_size=self.batch_size,
                    shuffle=False,  # Sampler handles shuffling
                    num_workers=self.num_workers,
                    drop_last=self.drop_last,
                    collate_fn=collate_fn,
                )
        
        def _default_collate(self, batch: List[Dict]) -> Dict[str, Any]:
            """Default collate function for batching"""
            if not batch:
                return {}
            
            # Check if batch items are dicts
            if isinstance(batch[0], dict):
                result = {}
                for key in batch[0].keys():
                    values = [item[key] for item in batch]
                    if isinstance(values[0], torch.Tensor):
                        result[key] = torch.stack(values)
                    elif isinstance(values[0], np.ndarray):
                        result[key] = torch.from_numpy(np.stack(values))
                    else:
                        result[key] = values
                return result
            else:
                return batch
        
        def __iter__(self):
            """Iterate over batches"""
            if self.sampling_strategy == 'batch_split':
                for batch_indices in self.sampler:
                    batch = [self.mixed_dataset[idx] for idx in batch_indices]
                    yield self._default_collate(batch)
            else:
                for indices in self.dataloader:
                    batch = [self.mixed_dataset[idx] for idx in self.sampler.indices[indices[0]:indices[0]+self.batch_size]]
                    yield self._default_collate(batch)
        
        def __len__(self) -> int:
            """Return number of batches"""
            return len(self.sampler)
        
        def set_epoch(self, epoch: int):
            """Set epoch for reproducible shuffling"""
            self.epoch = epoch
            self.sampler.set_epoch(epoch)
        
        def get_batch_composition(self) -> Dict[str, int]:
            """Get the composition of each batch"""
            if self.sampling_strategy == 'batch_split':
                real_per_batch = max(1, int(self.batch_size * self.real_gen_ratio))
                return {
                    'total': self.batch_size,
                    'real': real_per_batch,
                    'generated': self.batch_size - real_per_batch,
                }
            else:
                # For ratio strategy, it's probabilistic
                return {
                    'total': self.batch_size,
                    'expected_real': int(self.batch_size * self.real_gen_ratio),
                    'expected_generated': int(self.batch_size * (1 - self.real_gen_ratio)),
                }


# ============================================================================
# MMSeg Integration
# ============================================================================

if MMSEG_AVAILABLE:
    
    @DATASETS.register_module()
    class MixedRealGeneratedDataset(BaseSegDataset):
        """
        MMSegmentation-compatible dataset for mixed real/generated training.
        
        This dataset integrates with MMSegmentation's data loading pipeline
        and provides ratio-based sampling of real and generated images.
        
        Args:
            real_dataset_cfg: Config for real images dataset
            generated_dataset_cfg: Config for generated images dataset
            real_gen_ratio: Ratio of real images (0.0-1.0)
            sampling_strategy: How to sample ('ratio', 'batch_split')
        """
        
        METAINFO = dict(
            classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                    'traffic light', 'traffic sign', 'vegetation', 'terrain',
                    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                    'motorcycle', 'bicycle'),
        )
        
        def __init__(
            self,
            real_dataset_cfg: Dict[str, Any],
            generated_dataset_cfg: Dict[str, Any],
            real_gen_ratio: float = 0.5,
            sampling_strategy: str = 'ratio',
            **kwargs
        ):
            self.real_gen_ratio = real_gen_ratio
            self.sampling_strategy = sampling_strategy
            
            # Build real dataset
            from mmseg.registry import DATASETS
            real_cfg = real_dataset_cfg.copy()
            real_type = real_cfg.pop('type')
            self.real_dataset = DATASETS.build(dict(type=real_type, **real_cfg))
            
            # Build generated dataset
            gen_cfg = generated_dataset_cfg.copy()
            gen_type = gen_cfg.pop('type')
            self.generated_dataset = DATASETS.build(dict(type=gen_type, **gen_cfg))
            
            # Don't call super().__init__ as we're wrapping existing datasets
            self._metainfo = self.real_dataset.metainfo
            
            # Build combined data list
            self.data_list = self._build_mixed_data_list()
        
        def _build_mixed_data_list(self) -> List[Dict]:
            """Build combined data list with source markers"""
            data_list = []
            
            # Add real data
            for idx, item in enumerate(self.real_dataset.data_list):
                item_copy = item.copy()
                item_copy['_source'] = 'real'
                item_copy['_source_idx'] = idx
                data_list.append(item_copy)
            
            # Add generated data
            for idx, item in enumerate(self.generated_dataset.data_list):
                item_copy = item.copy()
                item_copy['_source'] = 'generated'
                item_copy['_source_idx'] = idx
                data_list.append(item_copy)
            
            return data_list
        
        def __getitem__(self, idx: int) -> Dict[str, Any]:
            """Get item with source tracking"""
            data_info = self.data_list[idx]
            source = data_info.get('_source', 'real')
            source_idx = data_info.get('_source_idx', idx)
            
            if source == 'real':
                return self.real_dataset[source_idx]
            else:
                return self.generated_dataset[source_idx]
        
        def __len__(self) -> int:
            return len(self.data_list)
        
        @property
        def metainfo(self) -> Dict:
            return self._metainfo


# ============================================================================
# Configuration Builder for Mixed Training
# ============================================================================

def build_mixed_dataloader_config(
    real_gen_ratio: float,
    batch_size: int = 2,
    sampling_strategy: str = 'batch_split',
    **kwargs
) -> Dict[str, Any]:
    """
    Build configuration dict for mixed dataloader.
    
    This can be used to add mixed dataloader settings to an existing config.
    
    Args:
        real_gen_ratio: Ratio of real images (0.0-1.0)
        batch_size: Total batch size
        sampling_strategy: Sampling strategy
        
    Returns:
        Configuration dictionary
    """
    real_samples = max(1, int(batch_size * real_gen_ratio))
    gen_samples = batch_size - real_samples
    
    return {
        'mixed_dataloader': {
            'enabled': True,
            'real_gen_ratio': real_gen_ratio,
            'sampling_strategy': sampling_strategy,
            'batch_composition': {
                'total_batch_size': batch_size,
                'real_samples': real_samples,
                'generated_samples': gen_samples,
            },
        }
    }


def get_effective_dataset_size(
    real_size: int,
    generated_size: int,
    real_gen_ratio: float,
    epochs: int = 1,
) -> Dict[str, int]:
    """
    Calculate effective dataset sizes with mixed sampling.
    
    Args:
        real_size: Number of real images
        generated_size: Number of generated images
        real_gen_ratio: Ratio of real images
        epochs: Number of training epochs
        
    Returns:
        Dict with size statistics
    """
    total_per_epoch = max(real_size, generated_size)
    real_per_epoch = int(total_per_epoch * real_gen_ratio)
    gen_per_epoch = total_per_epoch - real_per_epoch
    
    return {
        'real_dataset_size': real_size,
        'generated_dataset_size': generated_size,
        'real_per_epoch': real_per_epoch,
        'generated_per_epoch': gen_per_epoch,
        'total_per_epoch': total_per_epoch,
        'total_iterations': total_per_epoch * epochs,
        'real_gen_ratio': real_gen_ratio,
    }


# ============================================================================
# Main - Demo and Testing
# ============================================================================

def main():
    """Demo and test the mixed dataloader"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PROVE Mixed Dataloader Demo')
    parser.add_argument('--real-size', type=int, default=1000,
                       help='Size of mock real dataset')
    parser.add_argument('--gen-size', type=int, default=6000,
                       help='Size of mock generated dataset')
    parser.add_argument('--ratio', type=float, default=0.5,
                       help='Real-to-generated ratio')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--strategy', type=str, default='batch_split',
                       choices=['ratio', 'batch_split', 'alternating'],
                       help='Sampling strategy')
    parser.add_argument('--num-batches', type=int, default=5,
                       help='Number of batches to demonstrate')
    
    args = parser.parse_args()
    
    print("PROVE Mixed Dataloader Demo")
    print("=" * 60)
    print(f"Real dataset size: {args.real_size}")
    print(f"Generated dataset size: {args.gen_size}")
    print(f"Real-to-generated ratio: {args.ratio}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sampling strategy: {args.strategy}")
    print("=" * 60)
    
    # Calculate effective sizes
    sizes = get_effective_dataset_size(
        args.real_size, args.gen_size, args.ratio
    )
    print("\nEffective dataset sizes:")
    for key, value in sizes.items():
        print(f"  {key}: {value}")
    
    # Build config
    config = build_mixed_dataloader_config(
        args.ratio, args.batch_size, args.strategy
    )
    print("\nGenerated config:")
    print(f"  {config}")
    
    if TORCH_AVAILABLE:
        print("\n" + "=" * 60)
        print("Testing with PyTorch datasets...")
        
        # Create mock datasets
        class MockDataset(Dataset):
            def __init__(self, size: int, prefix: str):
                self.size = size
                self.prefix = prefix
            
            def __getitem__(self, idx):
                return {'idx': idx, 'name': f'{self.prefix}_{idx}'}
            
            def __len__(self):
                return self.size
        
        real_ds = MockDataset(args.real_size, 'real')
        gen_ds = MockDataset(args.gen_size, 'gen')
        
        # Create mixed dataloader
        loader = MixedDataLoader(
            real_dataset=real_ds,
            generated_dataset=gen_ds,
            real_gen_ratio=args.ratio,
            batch_size=args.batch_size,
            sampling_strategy=args.strategy,
        )
        
        print(f"\nBatch composition: {loader.get_batch_composition()}")
        print(f"Total batches: {len(loader)}")
        
        print(f"\nSample batches (first {args.num_batches}):")
        for i, batch in enumerate(loader):
            if i >= args.num_batches:
                break
            sources = batch.get('_source', ['?'] * len(batch['idx']))
            real_count = sum(1 for s in sources if s == 'real')
            gen_count = len(sources) - real_count
            print(f"  Batch {i+1}: {real_count} real, {gen_count} generated")
    else:
        print("\nPyTorch not available - skipping dataloader demo")


if __name__ == '__main__':
    main()
