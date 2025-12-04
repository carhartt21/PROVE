#!/usr/bin/env python3
"""
PROVE Augmentation Configuration Generator

This module generates training configurations for:
1. Baseline training (no augmentation)
2. PhotoMetricDistort augmentation
3. Generated images augmentation (from various generative models)

The generated images augment clear_day images with 6 adverse conditions:
- cloudy, dawn_dusk, fog, night, rainy, snowy

This results in 7x the original dataset size when using generated images.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional


# ============================================================================
# Configuration Constants
# ============================================================================

# Base paths
FINAL_SPLITS_ROOT = '/scratch/aaa_exchange/AWARE/FINAL_SPLITS'
GENERATED_IMAGES_ROOT = '/scratch/aaa_exchange/AWARE/GENERATED_IMAGES'
OUTPUT_DIR = '/scratch/aaa_exchange/AWARE/WEIGHTS'
CONFIG_OUTPUT_DIR = './multi_model_configs'

# Adverse weather conditions generated from clear_day images
ADVERSE_CONDITIONS = ['cloudy', 'dawn_dusk', 'fog', 'night', 'rainy', 'snowy']

# Datasets available for training
DATASETS = {
    'ACDC': {'task': 'segmentation', 'format': 'cityscapes'},
    'BDD10k': {'task': 'segmentation', 'format': 'cityscapes'},
    'BDD100k': {'task': 'detection', 'format': 'bdd100k_json'},
    'IDD-AW': {'task': 'segmentation', 'format': 'cityscapes'},
    'MapillaryVistas': {'task': 'segmentation', 'format': 'mapillary'},
    'OUTSIDE15k': {'task': 'segmentation', 'format': 'cityscapes'},
}

# Segmentation models
SEGMENTATION_MODELS = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']

# Detection models  
DETECTION_MODELS = ['faster_rcnn_r50_fpn_1x', 'yolox_l', 'rtmdet_l']

# Generative models available
GENERATIVE_MODELS = [
    'cycleGAN',
    'CUT', 
    'stargan_v2',
    'SUSTechGAN',
    'EDICT',
    'Img2Img',
    'IP2P',
    'UniControl',
    'step1x_new',
    'StyleID',
    'NST',
    'albumentations',
    'automold',
    'imgaug_weather',
    'Weather_Effect_Generator',
    'Attribute_Hallucination',
    'cnet_seg',
    '2stageMultipleAdverseWeatherRemoval',
    'maxim',
    'MPRNet',
    'weatherformer',
    'tunit',
]

# Training settings
TRAINING_CONFIG = {
    'max_iters': 40000,
    'batch_size': 2,
    'checkpoint_interval': 5000,
    'eval_interval': 3333,
    'log_interval': 50,
}


# ============================================================================
# Augmentation Strategies
# ============================================================================

class AugmentationStrategy:
    """Base class for augmentation strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    def get_pipeline_transforms(self) -> List[dict]:
        """Return the augmentation transforms for the training pipeline"""
        raise NotImplementedError
    
    def get_dataset_config(self, base_dataset: dict) -> dict:
        """Return modified dataset configuration"""
        return base_dataset


class BaselineStrategy(AugmentationStrategy):
    """No augmentation - baseline training"""
    
    def __init__(self):
        super().__init__('baseline')
    
    def get_pipeline_transforms(self) -> List[dict]:
        return []


class PhotoMetricDistortStrategy(AugmentationStrategy):
    """RandomPhotometricDistort augmentation"""
    
    def __init__(self):
        super().__init__('photometric_distort')
    
    def get_pipeline_transforms(self) -> List[dict]:
        return [
            dict(type='PhotoMetricDistortion'),
        ]


class GeneratedImagesStrategy(AugmentationStrategy):
    """Augmentation using generated images from a specific generative model"""
    
    def __init__(self, generative_model: str, conditions: List[str] = None):
        super().__init__(f'gen_{generative_model}')
        self.generative_model = generative_model
        self.conditions = conditions or ADVERSE_CONDITIONS
        self.gen_root = os.path.join(GENERATED_IMAGES_ROOT, generative_model)
    
    def get_pipeline_transforms(self) -> List[dict]:
        # No additional transforms - augmentation is in the dataset
        return []
    
    def get_generated_image_paths(self) -> Dict[str, str]:
        """Get paths to generated images for each condition"""
        paths = {}
        for condition in self.conditions:
            condition_dir = os.path.join(
                self.gen_root, 
                f'clear_day2{condition}',
                'test_latest',
                'images'
            )
            if os.path.exists(condition_dir):
                paths[condition] = condition_dir
        return paths
    
    def get_manifest_path(self) -> Optional[str]:
        """Get path to manifest CSV for image mapping"""
        manifest_path = os.path.join(self.gen_root, 'manifest.csv')
        if os.path.exists(manifest_path):
            return manifest_path
        return None


# ============================================================================
# Configuration Generator
# ============================================================================

class AugmentationConfigGenerator:
    """Generate training configurations for different augmentation strategies"""
    
    def __init__(self, output_dir: str = CONFIG_OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.strategies = {
            'baseline': BaselineStrategy(),
            'photometric_distort': PhotoMetricDistortStrategy(),
        }
        
        # Add generative model strategies
        for gen_model in GENERATIVE_MODELS:
            gen_path = os.path.join(GENERATED_IMAGES_ROOT, gen_model)
            if os.path.exists(gen_path):
                self.strategies[f'gen_{gen_model}'] = GeneratedImagesStrategy(gen_model)
    
    def list_available_strategies(self) -> List[str]:
        """List all available augmentation strategies"""
        return list(self.strategies.keys())
    
    def list_available_generative_models(self) -> List[str]:
        """List generative models with available generated images"""
        available = []
        for gen_model in GENERATIVE_MODELS:
            gen_path = os.path.join(GENERATED_IMAGES_ROOT, gen_model)
            if os.path.exists(gen_path):
                # Check if it has the expected structure
                manifest = os.path.join(gen_path, 'manifest.csv')
                if os.path.exists(manifest):
                    available.append(gen_model)
        return available
    
    def generate_config(
        self,
        dataset: str,
        model: str,
        strategy_name: str,
        output_subdir: str = None
    ) -> dict:
        """
        Generate a training configuration for a specific combination.
        
        Args:
            dataset: Dataset name (e.g., 'ACDC', 'BDD10k')
            model: Model name (e.g., 'deeplabv3plus_r50')
            strategy_name: Augmentation strategy name
            output_subdir: Optional subdirectory for output configs
            
        Returns:
            Configuration dictionary
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        dataset_info = DATASETS.get(dataset)
        
        if dataset_info is None:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # Build base config
        config = self._build_base_config(dataset, model, dataset_info)
        
        # Add augmentation transforms
        augmentation_transforms = strategy.get_pipeline_transforms()
        
        # Build training pipeline
        config['train_pipeline'] = self._build_train_pipeline(
            dataset_info['task'],
            augmentation_transforms
        )
        
        # For generated images, modify dataset to include generated data
        if isinstance(strategy, GeneratedImagesStrategy):
            config = self._add_generated_images_dataset(
                config, dataset, strategy
            )
        
        # Set output directory
        subdir = output_subdir or strategy_name
        config['work_dir'] = os.path.join(
            OUTPUT_DIR, subdir, dataset.lower(), model
        )
        
        return config
    
    def _build_base_config(self, dataset: str, model: str, dataset_info: dict) -> dict:
        """Build base configuration without augmentation"""
        
        task = dataset_info['task']
        
        # Model base configs
        if task == 'segmentation':
            model_configs = {
                'deeplabv3plus_r50': '../_base_/models/deeplabv3plus_r50-d8.py',
                'pspnet_r50': '../_base_/models/pspnet_r50-d8.py',
                'segformer_mit-b5': '../_base_/models/segformer_mit-b5.py',
            }
            metric = 'mIoU'
        else:  # detection
            model_configs = {
                'faster_rcnn_r50_fpn_1x': '../_base_/models/faster_rcnn_r50_fpn.py',
                'yolox_l': '../_base_/models/yolox_l.py',
                'rtmdet_l': '../_base_/models/rtmdet_l.py',
            }
            metric = 'bbox'
        
        config = {
            '_base_': [model_configs.get(model, model_configs[list(model_configs.keys())[0]])],
            'runner': dict(
                type='IterBasedRunner',
                max_iters=TRAINING_CONFIG['max_iters']
            ),
            'checkpoint_config': dict(interval=TRAINING_CONFIG['checkpoint_interval']),
            'evaluation': dict(
                interval=TRAINING_CONFIG['eval_interval'],
                metric=metric
            ),
            'log_config': dict(
                interval=TRAINING_CONFIG['log_interval'],
                hooks=[
                    dict(type='TextLoggerHook'),
                    dict(type='TensorboardLoggerHook'),
                ]
            ),
            'data': dict(
                samples_per_gpu=TRAINING_CONFIG['batch_size'],
                workers_per_gpu=4,
            ),
            'seed': 42,
            'deterministic': True,
        }
        
        # Add dataset-specific configuration
        config['data_root'] = FINAL_SPLITS_ROOT
        config['dataset'] = dataset
        
        return config
    
    def _build_train_pipeline(
        self, 
        task: str, 
        augmentation_transforms: List[dict]
    ) -> List[dict]:
        """Build training data pipeline"""
        
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', scale=(512, 512), keep_ratio=True),
        ]
        
        # Add augmentation transforms
        pipeline.extend(augmentation_transforms)
        
        # Add final packing
        if task == 'segmentation':
            pipeline.append(dict(type='PackSegInputs'))
        else:
            pipeline.append(dict(type='PackDetInputs'))
        
        return pipeline
    
    def _add_generated_images_dataset(
        self, 
        config: dict, 
        dataset: str,
        strategy: GeneratedImagesStrategy
    ) -> dict:
        """Add generated images to the training dataset"""
        
        # Get manifest for mapping generated to original images
        manifest_path = strategy.get_manifest_path()
        
        if manifest_path:
            config['generated_images'] = {
                'enabled': True,
                'model': strategy.generative_model,
                'manifest': manifest_path,
                'conditions': strategy.conditions,
                'root': strategy.gen_root,
            }
        
        return config
    
    def save_config(self, config: dict, filepath: str):
        """Save configuration to a Python file"""
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write("# PROVE Augmentation Configuration\n")
            f.write(f"# Strategy: {config.get('_strategy', 'unknown')}\n")
            f.write(f"# Dataset: {config.get('dataset', 'unknown')}\n")
            f.write("# Auto-generated - do not edit manually\n\n")
            
            for key, value in config.items():
                if key.startswith('_'):
                    continue
                f.write(f"{key} = {repr(value)}\n")
    
    def generate_all_configs(
        self,
        strategies: List[str] = None,
        datasets: List[str] = None,
        models: List[str] = None
    ):
        """
        Generate all configuration combinations.
        
        Args:
            strategies: List of strategy names (default: all)
            datasets: List of dataset names (default: all)
            models: List of model names (default: all per task)
        """
        strategies = strategies or list(self.strategies.keys())
        datasets = datasets or list(DATASETS.keys())
        
        total_configs = 0
        
        for strategy_name in strategies:
            print(f"\n{'='*60}")
            print(f"Generating configs for strategy: {strategy_name}")
            print(f"{'='*60}")
            
            for dataset in datasets:
                dataset_info = DATASETS[dataset]
                task = dataset_info['task']
                
                # Select appropriate models
                if models:
                    dataset_models = models
                elif task == 'segmentation':
                    dataset_models = SEGMENTATION_MODELS
                else:
                    dataset_models = DETECTION_MODELS
                
                for model in dataset_models:
                    try:
                        config = self.generate_config(
                            dataset=dataset,
                            model=model,
                            strategy_name=strategy_name
                        )
                        
                        # Add metadata
                        config['_strategy'] = strategy_name
                        
                        # Save config
                        config_dir = self.output_dir / strategy_name / dataset.upper()
                        config_path = config_dir / f"{dataset.lower()}_{model}_config.py"
                        
                        self.save_config(config, str(config_path))
                        
                        print(f"✓ Generated: {config_path}")
                        total_configs += 1
                        
                    except Exception as e:
                        print(f"✗ Failed: {dataset}/{model} - {str(e)}")
        
        print(f"\n{'='*60}")
        print(f"Total configs generated: {total_configs}")
        print(f"{'='*60}")


def main():
    """Main entry point for config generation"""
    
    generator = AugmentationConfigGenerator()
    
    print("PROVE Augmentation Configuration Generator")
    print("=" * 60)
    
    # List available strategies
    print("\nAvailable Augmentation Strategies:")
    for strategy in generator.list_available_strategies():
        print(f"  - {strategy}")
    
    print("\nAvailable Generative Models with Images:")
    for model in generator.list_available_generative_models():
        print(f"  - {model}")
    
    # Generate configs for baseline and photometric_distort
    print("\n" + "=" * 60)
    print("Generating configurations...")
    print("=" * 60)
    
    generator.generate_all_configs(
        strategies=['baseline', 'photometric_distort'],
        datasets=list(DATASETS.keys())
    )


if __name__ == "__main__":
    main()
