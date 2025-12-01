# MMDetection Universal Training Pipeline Configuration
# PROVE: Pipeline for Recognition & Object Vision Evaluation
# Compatible with Object Detection & Semantic Segmentation

import os
from mmcv import Config

class PROVEConfig:
    """
    Configuration class for PROVE pipeline supporting:
    - Object Detection with BDD100k JSON format
    - Semantic Segmentation with Cityscapes, Mapillary Vistas, OUTSIDE15k formats
    """
    
    def __init__(self):
        self.base_config = {
            # Pipeline metadata
            'pipeline_name': 'PROVE',
            'version': '1.0.0',
            'description': 'Pipeline for Recognition & Object Vision Evaluation',
            
            # Supported tasks and formats
            'supported_tasks': ['object_detection', 'semantic_segmentation'],
            'supported_formats': {
                'object_detection': ['bdd100k_json', 'coco_json'],
                'semantic_segmentation': ['cityscapes', 'mapillary_vistas', 'outside15k']
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
                    'converter': 'cityscapes_to_coco'
                },
                'mapillary_vistas': {
                    'annotation_format': 'png',
                    'image_format': ['jpg', 'png'],
                    'required_files': ['labels', 'images'],
                    'converter': 'mapillary_to_coco'
                },
                'outside15k': {
                    'annotation_format': 'png',
                    'image_format': ['jpg', 'png'],
                    'required_files': ['labels', 'images'],
                    'converter': 'outside15k_to_coco'
                }
            }
        }
    
    def generate_config(self, task_type, dataset_format, dataset_path, model_name=None):
        """Generate MMDetection config for specific task and dataset"""
        
        if task_type not in self.base_config['supported_tasks']:
            raise ValueError(f"Unsupported task: {task_type}")
        
        if dataset_format not in self.base_config['supported_formats'][task_type]:
            raise ValueError(f"Unsupported format for {task_type}: {dataset_format}")
        
        # Select model
        if model_name is None:
            model_name = self.base_config['models'][task_type]['default']
        
        config = self._build_base_config(task_type, model_name)
        config.update(self._build_dataset_config(dataset_format, dataset_path))
        config.update(self._build_training_config())
        
        return config
    
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
    pipeline = MOVADETPipelineConfig()
    
    # Generate config for object detection with BDD100k
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
    
    print("MOVADET Pipeline Configuration Generated Successfully!")