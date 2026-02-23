# MMDetection Universal Training Pipeline Configuration
# PROVE: Pipeline for Recognition & Object Vision Evaluation
# Compatible with Object Detection & Semantic Segmentation

import os
from mmengine import Config

# Import label unification module
try:
    from label_unification import (
        LabelUnificationManager,
        CITYSCAPES_CLASSES,
        MAPILLARY_CLASSES,
        UNIFIED_CLASSES,
        CityscapesTrainID,
        UnifiedTrainID,
    )
    LABEL_UNIFICATION_AVAILABLE = True
except ImportError:
    LABEL_UNIFICATION_AVAILABLE = False


class PROVEConfig:
    """
    Configuration class for PROVE pipeline supporting:
    - Object Detection with BDD100k JSON format
    - Semantic Segmentation with Cityscapes, Mapillary Vistas, OUTSIDE15k formats
    - Joint training with unified label spaces
    """
    
    def __init__(self):
        self.base_config = {
            # Pipeline metadata
            'pipeline_name': 'PROVE',
            'version': '1.1.0',
            'description': 'Pipeline for Recognition & Object Vision Evaluation',
            
            # Supported tasks and formats
            'supported_tasks': ['object_detection', 'semantic_segmentation'],
            'supported_formats': {
                'object_detection': ['bdd100k_json', 'coco_json'],
                'semantic_segmentation': ['cityscapes', 'mapillary_vistas', 'outside15k', 'joint_cityscapes_mapillary']
            },
            
            # Base paths
            'base_paths': {
                'data_root': './data/',
                'work_dir': './work_dirs/',
                'checkpoint_dir': './checkpoints/',
                'log_dir': './logs/',
                'results_dir': './results/'
            },
            
            # Training configuration
            'training': {
                'gpu_ids': [0],
                'seed': 42,
                'deterministic': True,
                'workers_per_gpu': 4,
                'samples_per_gpu': 2,
                'lr': 0.02,
                'max_epochs': 12,
                'evaluation_interval': 1,
                'checkpoint_interval': 1,
                'log_interval': 50
            },
            
            # Model configurations
            'models': {
                'object_detection': {
                    'default': 'faster_rcnn_r50_fpn_1x',
                    'available': [
                        'faster_rcnn_r50_fpn_1x',
                        'yolox_l',
                        'rtmdet_l',
                        'detr_r50',
                        'mask_rcnn_r50_fpn_1x'
                    ]
                },
                'semantic_segmentation': {
                    'default': 'deeplabv3plus_r50',
                    'available': [
                        'deeplabv3plus_r50',
                        'pspnet_r50',
                        'segformer_mit-b5',
                        'upernet_swin'
                    ]
                }
            },
            
            # Dataset format specifications
            'dataset_formats': {
                'bdd100k_json': {
                    'annotation_format': 'json',
                    'image_format': ['jpg', 'png'],
                    'required_fields': ['name', 'labels', 'attributes'],
                    'converter': 'bdd100k_to_coco'
                },
                'cityscapes': {
                    'annotation_format': 'png',
                    'image_format': ['png'],
                    'required_files': ['gtFine', 'leftImg8bit'],
                    'converter': 'cityscapes_to_coco',
                    'num_classes': 19
                },
                'mapillary_vistas': {
                    'annotation_format': 'png',
                    'image_format': ['jpg', 'png'],
                    'required_files': ['labels', 'images'],
                    'converter': 'mapillary_to_coco',
                    'num_classes': 66
                },
                'outside15k': {
                    'annotation_format': 'png',
                    'image_format': ['jpg', 'png'],
                    'required_files': ['labels', 'images'],
                    'converter': 'outside15k_to_coco'
                },
                'joint_cityscapes_mapillary': {
                    'annotation_format': 'png',
                    'image_format': ['jpg', 'png'],
                    'required_files': ['cityscapes', 'mapillary_vistas'],
                    'converter': 'joint_unified',
                    'label_spaces': ['cityscapes', 'unified'],
                    'default_label_space': 'cityscapes'
                }
            },
            
            # Label unification settings
            'label_unification': {
                'enabled': True,
                'strategies': ['cityscapes', 'unified'],
                'default_strategy': 'cityscapes',
                'cityscapes_num_classes': 19,
                'unified_num_classes': 42,
                'mapillary_num_classes': 66
            }
        }
        
        # Initialize label unification manager if available
        if LABEL_UNIFICATION_AVAILABLE:
            self.label_manager = LabelUnificationManager()
        else:
            self.label_manager = None
    
    def generate_multi_model_config(self, task_type, dataset_format, dataset_path, model_names=None):
        """
        Generate configurations for multiple models with the same dataset setup.

        Args:
            task_type: 'object_detection' or 'semantic_segmentation'
            dataset_format: Dataset format (e.g., 'bdd100k_json', 'cityscapes')
            dataset_path: Path to dataset
            model_names: List of model names to generate configs for

        Returns:
            Dictionary with model names as keys and their configs as values
        """
        if model_names is None:
            model_names = self.base_config['models'][task_type]['available']

        # Generate shared dataset and training config once
        shared_config = self._build_dataset_config(dataset_format, dataset_path)
        shared_config.update(self._build_training_config())

        # Generate individual configs for each model
        configs = {}
        for model_name in model_names:
            if model_name not in self.base_config['models'][task_type]['available']:
                print(f"Warning: {model_name} not in available models, skipping")
                continue

            config = self._build_base_config(task_type, model_name)
            config.update(shared_config.copy())  # Use shared dataset/training config
            configs[model_name] = config

        return configs
    
    def _build_base_config(self, task_type, model_name):
        """Build base configuration for the specified task and model"""
        
        if task_type == 'object_detection':
            return self._get_detection_config(model_name)
        elif task_type == 'semantic_segmentation':
            return self._get_segmentation_config(model_name)
    
    def _get_detection_config(self, model_name):
        """Get object detection model configuration"""
        
        configs = {
            'faster_rcnn_r50_fpn_1x': {
                '_base_': [
                    '../_base_/models/faster_rcnn_r50_fpn.py',
                    '../_base_/datasets/coco_detection.py',
                    '../_base_/schedules/schedule_1x.py',
                    '../_base_/default_runtime.py'
                ]
            },
            'yolox_l': {
                '_base_': [
                    '../_base_/models/yolox_l.py',
                    '../_base_/datasets/coco_detection.py',
                    '../_base_/schedules/schedule_1x.py',
                    '../_base_/default_runtime.py'
                ]
            },
            'rtmdet_l': {
                '_base_': [
                    '../_base_/models/rtmdet_l.py',
                    '../_base_/datasets/coco_detection.py',
                    '../_base_/schedules/schedule_1x.py',
                    '../_base_/default_runtime.py'
                ]
            }
        }
        
        return configs.get(model_name, configs['faster_rcnn_r50_fpn_1x'])
    
    def _get_segmentation_config(self, model_name):
        """Get semantic segmentation model configuration"""
        
        configs = {
            'deeplabv3plus_r50': {
                '_base_': [
                    '../_base_/models/deeplabv3plus_r50-d8.py',
                    '../_base_/datasets/cityscapes.py',
                    '../_base_/default_runtime.py',
                    '../_base_/schedules/schedule_80k.py'
                ]
            },
            'pspnet_r50': {
                '_base_': [
                    '../_base_/models/pspnet_r50-d8.py',
                    '../_base_/datasets/cityscapes.py',
                    '../_base_/default_runtime.py',
                    '../_base_/schedules/schedule_80k.py'
                ]
            }
        }
        
        return configs.get(model_name, configs['deeplabv3plus_r50'])
    
    def _build_dataset_config(self, dataset_format, dataset_path):
        """Build dataset configuration based on format"""
        
        base_dataset_config = {
            'data_root': dataset_path,
            'img_norm_cfg': dict(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True
            ),
            'train_pipeline': [],
            'test_pipeline': [],
            'data': {
                'samples_per_gpu': self.base_config['training']['samples_per_gpu'],
                'workers_per_gpu': self.base_config['training']['workers_per_gpu']
            }
        }
        
        if dataset_format == 'bdd100k_json':
            return self._get_bdd100k_config(base_dataset_config, dataset_path)
        elif dataset_format == 'cityscapes':
            return self._get_cityscapes_config(base_dataset_config, dataset_path)
        elif dataset_format == 'mapillary_vistas':
            return self._get_mapillary_config(base_dataset_config, dataset_path)
        elif dataset_format == 'outside15k':
            return self._get_outside15k_config(base_dataset_config, dataset_path)
        elif dataset_format == 'joint_cityscapes_mapillary':
            return self._get_joint_config(base_dataset_config, dataset_path)
        
        return base_dataset_config
    
    def _get_bdd100k_config(self, base_config, dataset_path):
        """BDD100k dataset configuration"""
        base_config.update({
            'dataset_type': 'CocoDataset',
            'classes': (
                'pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 
                'motorcycle', 'bicycle', 'traffic light', 'traffic sign'
            ),
            'data': {
                **base_config['data'],
                'train': dict(
                    type='CocoDataset',
                    ann_file=f'{dataset_path}/labels/bdd100k_labels_images_train.json',
                    img_prefix=f'{dataset_path}/images/100k/train/',
                    classes=base_config.get('classes')
                ),
                'val': dict(
                    type='CocoDataset',
                    ann_file=f'{dataset_path}/labels/bdd100k_labels_images_val.json',
                    img_prefix=f'{dataset_path}/images/100k/val/',
                    classes=base_config.get('classes')
                ),
                'test': dict(
                    type='CocoDataset',
                    ann_file=f'{dataset_path}/labels/bdd100k_labels_images_val.json',
                    img_prefix=f'{dataset_path}/images/100k/val/',
                    classes=base_config.get('classes')
                )
            }
        })
        return base_config
    
    def _get_cityscapes_config(self, base_config, dataset_path):
        """Cityscapes dataset configuration"""
        base_config.update({
            'dataset_type': 'CityscapesDataset',
            'classes': (
                'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                'traffic light', 'traffic sign', 'vegetation', 'terrain',
                'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                'motorcycle', 'bicycle'
            ),
            'data': {
                **base_config['data'],
                'train': dict(
                    type='CityscapesDataset',
                    data_root=dataset_path,
                    img_dir='leftImg8bit/train',
                    ann_dir='gtFine/train'
                ),
                'val': dict(
                    type='CityscapesDataset',
                    data_root=dataset_path,
                    img_dir='leftImg8bit/val',
                    ann_dir='gtFine/val'
                ),
                'test': dict(
                    type='CityscapesDataset',
                    data_root=dataset_path,
                    img_dir='leftImg8bit/test',
                    ann_dir='gtFine/test'
                )
            }
        })
        return base_config
    
    def _get_mapillary_config(self, base_config, dataset_path):
        """Mapillary Vistas dataset configuration"""
        base_config.update({
            'dataset_type': 'MapillaryUnifiedDataset',
            'num_classes': 66,
            'data': {
                **base_config['data'],
                'train': dict(
                    type='MapillaryUnifiedDataset',
                    data_root=dataset_path,
                    img_dir='training/images',
                    ann_dir='training/v1.2/labels',
                    target_space='cityscapes'
                ),
                'val': dict(
                    type='MapillaryUnifiedDataset',
                    data_root=dataset_path,
                    img_dir='validation/images',
                    ann_dir='validation/v1.2/labels',
                    target_space='cityscapes'
                ),
                'test': dict(
                    type='MapillaryUnifiedDataset',
                    data_root=dataset_path,
                    img_dir='testing/images',
                    ann_dir='testing/v1.2/labels',
                    target_space='cityscapes'
                )
            }
        })
        return base_config
    
    def _get_outside15k_config(self, base_config, dataset_path):
        """OUTSIDE15k dataset configuration"""
        base_config.update({
            'dataset_type': 'CustomDataset',
            'data': {
                **base_config['data'],
                'train': dict(
                    type='CustomDataset',
                    data_root=dataset_path,
                    img_dir='images/train',
                    ann_dir='labels/train'
                ),
                'val': dict(
                    type='CustomDataset',
                    data_root=dataset_path,
                    img_dir='images/val',
                    ann_dir='labels/val'
                ),
                'test': dict(
                    type='CustomDataset',
                    data_root=dataset_path,
                    img_dir='images/test',
                    ann_dir='labels/test'
                )
            }
        })
        return base_config
    
    def _get_joint_config(self, base_config, dataset_path, label_space='cityscapes'):
        """
        Joint Cityscapes + Mapillary Vistas dataset configuration.
        
        Args:
            base_config: Base configuration dictionary
            dataset_path: Dictionary with 'cityscapes' and 'mapillary_vistas' paths
                         or string path to parent directory containing both
            label_space: Target label space ('cityscapes' or 'unified')
            
        Returns:
            Updated configuration dictionary
        """
        # Handle dataset path input
        if isinstance(dataset_path, dict):
            cityscapes_path = dataset_path.get('cityscapes', './data/cityscapes')
            mapillary_path = dataset_path.get('mapillary_vistas', './data/mapillary_vistas')
        else:
            cityscapes_path = os.path.join(dataset_path, 'cityscapes')
            mapillary_path = os.path.join(dataset_path, 'mapillary_vistas')
        
        # Set number of classes based on label space
        if label_space == 'unified':
            num_classes = self.base_config['label_unification']['unified_num_classes']
            classes = (
                'road', 'sidewalk', 'parking', 'rail track', 'bike lane',
                'building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel', 'barrier',
                'pole', 'traffic light', 'traffic sign', 'street light', 'utility pole', 'other object',
                'vegetation', 'terrain', 'sky', 'water', 'snow', 'mountain',
                'person', 'bicyclist', 'motorcyclist', 'other rider',
                'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
                'caravan', 'trailer', 'boat', 'other vehicle', 'wheeled slow', 'animal',
                'lane marking', 'crosswalk'
            )
            cityscapes_type = 'UnifiedCityscapesDataset'
        else:
            num_classes = self.base_config['label_unification']['cityscapes_num_classes']
            classes = (
                'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                'traffic light', 'traffic sign', 'vegetation', 'terrain',
                'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                'motorcycle', 'bicycle'
            )
            cityscapes_type = 'CityscapesDataset'
        
        base_config.update({
            'dataset_type': 'ConcatDataset',
            'num_classes': num_classes,
            'classes': classes,
            'label_space': label_space,
            'data': {
                **base_config['data'],
                'train': dict(
                    type='ConcatDataset',
                    datasets=[
                        # Cityscapes training data
                        dict(
                            type=cityscapes_type,
                            data_root=cityscapes_path,
                            img_dir='leftImg8bit/train',
                            ann_dir='gtFine/train',
                            seg_map_suffix='_gtFine_labelTrainIds.png'
                        ),
                        # Mapillary Vistas training data
                        dict(
                            type='MapillaryUnifiedDataset',
                            data_root=mapillary_path,
                            img_dir='training/images',
                            ann_dir='training/v1.2/labels',
                            target_space=label_space
                        )
                    ]
                ),
                'val': dict(
                    type='CityscapesDataset',
                    data_root=cityscapes_path,
                    img_dir='leftImg8bit/val',
                    ann_dir='gtFine/val',
                    seg_map_suffix='_gtFine_labelTrainIds.png'
                ),
                'test': dict(
                    type='CityscapesDataset',
                    data_root=cityscapes_path,
                    img_dir='leftImg8bit/val',
                    ann_dir='gtFine/val',
                    seg_map_suffix='_gtFine_labelTrainIds.png'
                )
            }
        })
        return base_config
    
    def generate_joint_training_config(self, 
                                        cityscapes_path: str,
                                        mapillary_path: str,
                                        model_name: str = 'deeplabv3plus_r50',
                                        label_space: str = 'cityscapes',
                                        crop_size: tuple = (512, 512),
                                        batch_size: int = 8) -> dict:
        """
        Generate a complete configuration for joint Cityscapes + Mapillary training.
        
        Args:
            cityscapes_path: Path to Cityscapes dataset
            mapillary_path: Path to Mapillary Vistas dataset
            model_name: Model architecture name
            label_space: Target label space ('cityscapes' or 'unified')
            crop_size: Training crop size (height, width)
            batch_size: Batch size per GPU
            
        Returns:
            Complete configuration dictionary
        """
        # Get base model config
        config = self._get_segmentation_config(model_name)
        
        # Set up dataset paths
        dataset_paths = {
            'cityscapes': cityscapes_path,
            'mapillary_vistas': mapillary_path
        }
        
        # Build dataset config
        base_dataset_config = {
            'data_root': '',
            'img_norm_cfg': dict(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True
            ),
            'train_pipeline': [],
            'test_pipeline': [],
            'data': {
                'samples_per_gpu': batch_size,
                'workers_per_gpu': self.base_config['training']['workers_per_gpu']
            }
        }
        
        dataset_config = self._get_joint_config(base_dataset_config, dataset_paths, label_space)
        config.update(dataset_config)
        
        # Update model num_classes
        if label_space == 'unified':
            config['num_classes'] = self.base_config['label_unification']['unified_num_classes']
        else:
            config['num_classes'] = self.base_config['label_unification']['cityscapes_num_classes']
        
        # Add training config
        config.update(self._build_training_config())
        
        # Add pipeline configurations
        config['crop_size'] = crop_size
        config['train_pipeline'] = self._build_train_pipeline(crop_size, label_space)
        config['test_pipeline'] = self._build_test_pipeline()
        
        return config
    
    def _build_train_pipeline(self, crop_size, label_space='cityscapes'):
        """Build training data pipeline with label transformation"""
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
        ]
        
        # Add label transformation for joint training
        if label_space == 'unified':
            pipeline.append(dict(type='CityscapesLabelTransform', target_space='unified'))
        
        pipeline.extend([
            dict(
                type='RandomResize',
                scale=(2048, 1024),
                ratio_range=(0.5, 2.0),
                keep_ratio=True
            ),
            dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ])
        
        return pipeline
    
    def _build_test_pipeline(self):
        """Build test/validation data pipeline"""
        return [
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ]
    
    def _build_training_config(self):
        """Build training configuration"""
        training_cfg = self.base_config['training']
        
        return {
            'gpu_ids': training_cfg['gpu_ids'],
            'seed': training_cfg['seed'],
            'deterministic': training_cfg['deterministic'],
            
            'optimizer': dict(
                type='SGD',
                lr=training_cfg['lr'],
                momentum=0.9,
                weight_decay=0.0001
            ),
            
            'optimizer_config': dict(grad_clip=None),
            
            'lr_config': dict(
                policy='step',
                warmup='linear',
                warmup_iters=500,
                warmup_ratio=0.001,
                step=[8, 11]
            ),
            
            'runner': dict(
                type='EpochBasedRunner',
                max_epochs=training_cfg['max_epochs']
            ),
            
            'checkpoint_config': dict(
                interval=training_cfg['checkpoint_interval']
            ),
            
            'log_config': dict(
                interval=training_cfg['log_interval'],
                hooks=[
                    dict(type='TextLoggerHook'),
                    dict(type='TensorboardLoggerHook')
                ]
            ),
            
            'evaluation': dict(
                interval=training_cfg['evaluation_interval'],
                metric='bbox' if 'object_detection' else 'mIoU'
            ),
            
            'work_dir': self.base_config['base_paths']['work_dir']
        }


# Example usage
if __name__ == "__main__":
    # Initialize pipeline config
    pipeline = PROVEConfig()
    
    # Generate configs for multiple models with the same dataset
    multi_model_configs = pipeline.generate_multi_model_config(
        task_type='object_detection',
        dataset_format='bdd100k_json',
        dataset_path='./data/bdd100k/',
        model_names=['faster_rcnn_r50_fpn_1x', 'yolox_l', 'rtmdet_l']
    )
    
    # Generate config for single model (existing functionality)
    od_config = pipeline.generate_config(
        task_type='object_detection',
        dataset_format='bdd100k_json',
        dataset_path='./data/bdd100k/',
        model_name='faster_rcnn_r50_fpn_1x'
    )
    
    # Generate config for semantic segmentation with Cityscapes
    seg_config = pipeline.generate_config(
        task_type='semantic_segmentation',
        dataset_format='cityscapes',
        dataset_path='./data/cityscapes/',
        model_name='deeplabv3plus_r50'
    )
    
    # Generate config for joint training
    joint_config = pipeline.generate_joint_training_config(
        cityscapes_path='./data/cityscapes/',
        mapillary_path='./data/mapillary_vistas/',
        model_name='deeplabv3plus_r50',
        label_space='cityscapes'
    )
    
    print("PROVE Pipeline Configuration Generated Successfully!")
    print(f"Generated configs for {len(multi_model_configs)} models: {list(multi_model_configs.keys())}")
    print(f"Joint training config has {joint_config['num_classes']} classes")