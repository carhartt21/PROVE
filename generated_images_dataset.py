#!/usr/bin/env python3
"""
Custom Dataset for Generated Image Augmentation

This module provides dataset classes that combine original training images
with generated adversarial weather images from various generative models.

The generated images are created by transforming clear_day images into
6 adverse conditions: cloudy, dawn_dusk, fog, night, rainy, snowy.

This results in 7x augmentation of the original clear_day subset:
- 1x original clear_day images
- 6x generated adverse condition images (using same labels)
"""

import os
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from mmseg.datasets import BaseSegDataset
    from mmseg.registry import DATASETS
    MMSEG_AVAILABLE = True
except ImportError:
    MMSEG_AVAILABLE = False
    DATASETS = None


# ============================================================================
# Generated Images Manifest Handler
# ============================================================================

class GeneratedImagesManifest:
    """
    Handler for generated images manifest files.
    
    The manifest CSV contains mappings between generated images and their
    corresponding original images, allowing us to use the original labels
    for the generated images.
    """
    
    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        self.entries = []
        self.gen_to_original = {}
        self.original_to_gen = {}
        
        self._load_manifest()
    
    def _load_manifest(self):
        """Load and parse the manifest CSV file"""
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        
        with open(self.manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.entries.append(row)
                
                gen_path = row['gen_path']
                original_path = row['original_path']
                
                self.gen_to_original[gen_path] = original_path
                
                if original_path not in self.original_to_gen:
                    self.original_to_gen[original_path] = []
                self.original_to_gen[original_path].append({
                    'gen_path': gen_path,
                    'condition': row.get('target_domain', 'unknown'),
                    'domain': row.get('domain', 'unknown')
                })
    
    def get_original_for_generated(self, gen_path: str) -> Optional[str]:
        """Get the original image path for a generated image"""
        return self.gen_to_original.get(gen_path)
    
    def get_generated_for_original(self, original_path: str) -> List[Dict]:
        """Get all generated images for an original image"""
        return self.original_to_gen.get(original_path, [])
    
    def get_all_generated_paths(self) -> List[str]:
        """Get all generated image paths"""
        return list(self.gen_to_original.keys())
    
    def get_condition_paths(self, condition: str) -> List[Tuple[str, str]]:
        """
        Get generated paths for a specific condition.
        
        Returns:
            List of (generated_path, original_path) tuples
        """
        results = []
        for entry in self.entries:
            if entry.get('target_domain') == condition:
                results.append((entry['gen_path'], entry['original_path']))
        return results
    
    def get_dataset_entries(self, dataset_name: str) -> List[dict]:
        """
        Get all entries that match a specific dataset name.
        
        Args:
            dataset_name: Dataset name to filter by (e.g., 'ACDC', 'BDD10k', 'MapillaryVistas', 'Cityscapes')
            
        Returns:
            List of manifest entries for the specified dataset
        """
        results = []
        dataset_lower = dataset_name.lower()
        for entry in self.entries:
            # Primary: check the 'dataset' column (exact, case-insensitive)
            entry_dataset = entry.get('dataset', '')
            if entry_dataset and entry_dataset.lower() == dataset_lower:
                results.append(entry)
            # Fallback: check if dataset name is in the original path (case-insensitive)
            elif dataset_lower in entry.get('original_path', '').lower():
                results.append(entry)
        return results
    
    def get_dataset_count(self, dataset_name: str) -> int:
        """
        Get the count of generated images for a specific dataset.
        
        Args:
            dataset_name: Dataset name to count (e.g., 'ACDC', 'BDD10k')
            
        Returns:
            Number of generated images for the dataset
        """
        return len(self.get_dataset_entries(dataset_name))
    
    def get_available_datasets(self) -> Dict[str, int]:
        """
        Get a count of images per dataset in the manifest.
        
        Returns:
            Dictionary mapping dataset names to image counts
        """
        dataset_counts = {}
        for entry in self.entries:
            # Primary: use the 'dataset' column if available
            entry_dataset = entry.get('dataset', '')
            if entry_dataset:
                dataset_counts[entry_dataset] = dataset_counts.get(entry_dataset, 0) + 1
            else:
                # Fallback: extract dataset name from original_path
                original_path = entry.get('original_path', '')
                for dataset in ['ACDC', 'BDD10k', 'BDD100k', 'IDD-AW', 'MapillaryVistas', 'OUTSIDE15k', 'Cityscapes']:
                    if dataset.lower() in original_path.lower():
                        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
                        break
        return dataset_counts
    
    def __len__(self):
        return len(self.entries)


# ============================================================================
# Custom Dataset for Generated Image Augmentation
# ============================================================================

if MMSEG_AVAILABLE:
    
    @DATASETS.register_module()
    class GeneratedAugmentedDataset(BaseSegDataset):
        """
        Dataset that combines original images with generated adversarial images.
        
        This dataset:
        1. Loads original training images and labels
        2. Adds generated images using the same labels as originals
        3. Provides 7x augmentation for clear_day images
        
        Args:
            data_root: Root path to original dataset
            generated_root: Root path to generated images
            manifest_path: Path to manifest CSV mapping generated to original
            conditions: List of conditions to include (default: all 6)
            include_original: Whether to include original clear_day images
            dataset_filter: Dataset name to filter by (e.g., 'BDD10k', 'ACDC').
                            CRITICAL: Set this to avoid cross-dataset contamination!
                            If not set, will load ALL datasets from manifest.
            img_suffix: Image file suffix
            seg_map_suffix: Segmentation map suffix
        """
        
        METAINFO = dict(
            classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                    'traffic light', 'traffic sign', 'vegetation', 'terrain',
                    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                    'motorcycle', 'bicycle'),
        )
        
        def __init__(
            self,
            data_root: str,
            generated_root: str,
            manifest_path: str,
            conditions: List[str] = None,
            include_original: bool = True,
            dataset_filter: str = None,
            img_suffix: str = '.png',
            seg_map_suffix: str = '.png',
            **kwargs
        ):
            self.generated_root = generated_root
            self.manifest_path = manifest_path
            self.conditions = conditions or [
                'cloudy', 'dawn_dusk', 'fog', 'night', 'rainy', 'snowy'
            ]
            self.include_original = include_original
            # Filter to only include generated images from a specific dataset
            # This is CRITICAL to avoid loading cross-dataset images
            self.dataset_filter = dataset_filter
            
            # Load manifest
            self.manifest = GeneratedImagesManifest(manifest_path)
            
            super().__init__(
                data_root=data_root,
                img_suffix=img_suffix,
                seg_map_suffix=seg_map_suffix,
                **kwargs
            )
        
        def load_data_list(self) -> List[dict]:
            """Load and combine original and generated image data"""
            
            data_list = []
            
            # Load original dataset entries
            if self.include_original:
                original_list = super().load_data_list()
                data_list.extend(original_list)
                print(f"Loaded {len(original_list)} original images")
            
            # Add generated images
            loaded_count_by_condition = {}
            skipped_count_by_condition = {}
            
            for condition in self.conditions:
                condition_entries = self.manifest.get_condition_paths(condition)
                loaded_count = 0
                skipped_count = 0
                
                for gen_path, original_path in condition_entries:
                    # CRITICAL: Filter by dataset if specified
                    # This prevents cross-dataset contamination
                    if self.dataset_filter and self.dataset_filter not in original_path:
                        skipped_count += 1
                        continue
                    
                    # Get label path from original
                    label_path = self._get_label_for_original(original_path)
                    
                    if label_path and os.path.exists(gen_path):
                        data_list.append({
                            'img_path': gen_path,
                            'seg_map_path': label_path,
                            'condition': condition,
                            'is_generated': True,
                        })
                        loaded_count += 1
                
                loaded_count_by_condition[condition] = loaded_count
                skipped_count_by_condition[condition] = skipped_count
                
                if self.dataset_filter:
                    print(f"Added {loaded_count} {self.dataset_filter} generated images for {condition} (skipped {skipped_count} from other datasets)")
                else:
                    print(f"Added {loaded_count} generated images for {condition}")
            
            total_loaded = sum(loaded_count_by_condition.values())
            total_skipped = sum(skipped_count_by_condition.values())
            
            if self.dataset_filter:
                print(f"Total dataset size: {len(data_list)} images (dataset_filter={self.dataset_filter}, skipped {total_skipped} cross-dataset images)")
            else:
                print(f"WARNING: No dataset_filter specified! Loading {len(data_list)} images from ALL datasets. This may cause cross-dataset contamination.")
            
            return data_list
        
        def _get_label_for_original(self, original_path: str) -> Optional[str]:
            """
            Get the label/segmentation map path for an original image.
            
            This maps from the image path to the corresponding label path
            based on the dataset structure.
            """
            # Convert image path to label path
            # Original: /path/to/images/dataset/condition/image.jpg
            # Label:    /path/to/labels/dataset/condition/image.png
            
            label_path = original_path.replace('/images/', '/labels/')
            
            # Handle different extensions
            for img_ext in ['.jpg', '.png', '.jpeg']:
                for label_ext in ['.png', '_labelTrainIds.png', '_gtFine_labelTrainIds.png']:
                    test_path = label_path.replace(img_ext, label_ext)
                    if os.path.exists(test_path):
                        return test_path
            
            return None


# ============================================================================
# Pipeline Transform for Mixing Generated Images
# ============================================================================

class GeneratedImageMixTransform:
    """
    Data augmentation transform that randomly replaces original images
    with their generated counterparts during training.
    
    This allows dynamic mixing of original and generated images within
    a training batch, providing diverse augmentation.
    
    Args:
        manifest_path: Path to manifest CSV
        mix_ratio: Probability of using generated image (default: 0.5)
        conditions: List of conditions to sample from
    """
    
    def __init__(
        self,
        manifest_path: str,
        mix_ratio: float = 0.5,
        conditions: List[str] = None
    ):
        self.manifest = GeneratedImagesManifest(manifest_path)
        self.mix_ratio = mix_ratio
        self.conditions = conditions or [
            'cloudy', 'dawn_dusk', 'fog', 'night', 'rainy', 'snowy'
        ]
    
    def __call__(self, results: dict) -> dict:
        """
        Transform that may replace original image with generated version.
        
        Args:
            results: Dict containing 'img_path' and other data
            
        Returns:
            Modified results dict
        """
        if np.random.random() > self.mix_ratio:
            return results  # Keep original
        
        img_path = results.get('img_path')
        if not img_path:
            return results
        
        # Get generated alternatives
        generated = self.manifest.get_generated_for_original(img_path)
        
        if not generated:
            return results  # No generated versions available
        
        # Filter by conditions
        valid_generated = [
            g for g in generated 
            if g['condition'] in self.conditions
        ]
        
        if not valid_generated:
            return results
        
        # Randomly select one
        selected = np.random.choice(valid_generated)
        gen_path = selected['gen_path']
        
        if os.path.exists(gen_path):
            results['img_path'] = gen_path
            results['is_generated'] = True
            results['condition'] = selected['condition']
        
        return results


# ============================================================================
# Utility Functions
# ============================================================================

def get_augmentation_multiplier(conditions: List[str] = None) -> int:
    """
    Calculate the augmentation multiplier based on conditions.
    
    Args:
        conditions: List of adverse conditions to include
        
    Returns:
        Multiplier (1 original + N conditions)
    """
    conditions = conditions or ['cloudy', 'dawn_dusk', 'fog', 'night', 'rainy', 'snowy']
    return 1 + len(conditions)  # Original + generated conditions


def estimate_augmented_dataset_size(
    original_clear_day_count: int,
    conditions: List[str] = None
) -> dict:
    """
    Estimate the augmented dataset size.
    
    Args:
        original_clear_day_count: Number of original clear_day images
        conditions: List of conditions to include
        
    Returns:
        Dict with size estimates
    """
    conditions = conditions or ['cloudy', 'dawn_dusk', 'fog', 'night', 'rainy', 'snowy']
    
    return {
        'original_clear_day': original_clear_day_count,
        'generated_per_condition': original_clear_day_count,
        'total_generated': original_clear_day_count * len(conditions),
        'total_augmented': original_clear_day_count * (1 + len(conditions)),
        'multiplier': 1 + len(conditions),
        'conditions': conditions,
    }


def list_available_generative_models(gen_root: str) -> List[Dict]:
    """
    List available generative models with their manifest info.
    
    Args:
        gen_root: Root directory of generated images
        
    Returns:
        List of dicts with model info
    """
    models = []
    
    for model_dir in os.listdir(gen_root):
        model_path = os.path.join(gen_root, model_dir)
        if not os.path.isdir(model_path):
            continue
        
        manifest_path = os.path.join(model_path, 'manifest.csv')
        manifest_json = os.path.join(model_path, 'manifest.json')
        
        info = {
            'name': model_dir,
            'path': model_path,
            'has_manifest': os.path.exists(manifest_path),
        }
        
        if os.path.exists(manifest_json):
            try:
                import json
                with open(manifest_json) as f:
                    meta = json.load(f)
                    info['total_generated'] = meta.get('total_generated', 0)
                    info['conditions'] = [d['target_domain'] for d in meta.get('domains', [])]
            except:
                pass
        
        models.append(info)
    
    return models


# ============================================================================
# Main - Demo and Testing
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generated Images Dataset Utility")
    parser.add_argument('--gen-root', default='${AWARE_DATA_ROOT}/GENERATED_IMAGES',
                       help='Root directory of generated images')
    parser.add_argument('--list-models', action='store_true',
                       help='List available generative models')
    parser.add_argument('--check-manifest', type=str,
                       help='Check a specific manifest file')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available Generative Models:")
        print("=" * 60)
        models = list_available_generative_models(args.gen_root)
        for model in models:
            status = "✓ has manifest" if model['has_manifest'] else "✗ no manifest"
            count = model.get('total_generated', 'N/A')
            print(f"{model['name']:40} | {status} | {count} images")
    
    if args.check_manifest:
        print(f"\nChecking manifest: {args.check_manifest}")
        print("=" * 60)
        manifest = GeneratedImagesManifest(args.check_manifest)
        print(f"Total entries: {len(manifest)}")
        
        # Sample entries
        print("\nSample entries:")
        for i, entry in enumerate(manifest.entries[:5]):
            print(f"  {i+1}. {entry['target_domain']}: {entry['name']}")
