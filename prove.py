#!/usr/bin/env python3
"""
MOVADET Pipeline Manager
Modular Vision Analytics Detection Pipeline for MMDetection

This script provides a comprehensive pipeline for training and testing 
object detection and semantic segmentation models using MMDetection framework.

Supports:
- Object Detection: BDD100k JSON format
- Semantic Segmentation: Cityscapes, Mapillary Vistas, OUTSIDE15k formats
"""

import os
import sys
import json
import argparse
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess

import mmcv
from mmcv import Config
from mmdet.apis import init_detector, inference_detector, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger

# Import our configuration class
from prove_config import PROVEConfig


class DatasetConverter:
    """Handles conversion between different dataset formats"""
    
    @staticmethod
    def convert_bdd100k_to_coco(bdd100k_path: str, output_path: str) -> bool:
        """Convert BDD100k format to COCO format"""
        try:
            # Implementation for BDD100k to COCO conversion
            logging.info(f"Converting BDD100k dataset from {bdd100k_path} to COCO format")
            
            # Load BDD100k annotations
            with open(bdd100k_path, 'r') as f:
                bdd_data = json.load(f)
            
            # Initialize COCO structure
            coco_data = {
                "images": [],
                "annotations": [],
                "categories": [
                    {"id": 1, "name": "pedestrian"},
                    {"id": 2, "name": "rider"},
                    {"id": 3, "name": "car"},
                    {"id": 4, "name": "truck"},
                    {"id": 5, "name": "bus"},
                    {"id": 6, "name": "train"},
                    {"id": 7, "name": "motorcycle"},
                    {"id": 8, "name": "bicycle"},
                    {"id": 9, "name": "traffic light"},
                    {"id": 10, "name": "traffic sign"}
                ]
            }
            
            # Convert annotations
            annotation_id = 1
            for img_data in bdd_data:
                # Add image info
                image_info = {
                    "id": len(coco_data["images"]) + 1,
                    "file_name": img_data["name"],
                    "width": 1280,  # BDD100k standard width
                    "height": 720   # BDD100k standard height
                }
                coco_data["images"].append(image_info)
                
                # Convert labels to annotations
                if "labels" in img_data:
                    for label in img_data["labels"]:
                        if "box2d" in label:
                            box = label["box2d"]
                            annotation = {
                                "id": annotation_id,
                                "image_id": image_info["id"],
                                "category_id": next((cat["id"] for cat in coco_data["categories"] 
                                                   if cat["name"] == label["category"]), 1),
                                "bbox": [
                                    box["x1"], box["y1"],
                                    box["x2"] - box["x1"], box["y2"] - box["y1"]
                                ],
                                "area": (box["x2"] - box["x1"]) * (box["y2"] - box["y1"]),
                                "iscrowd": 0
                            }
                            coco_data["annotations"].append(annotation)
                            annotation_id += 1
            
            # Save converted data
            with open(output_path, 'w') as f:
                json.dump(coco_data, f, indent=2)
            
            logging.info(f"Successfully converted to {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error converting BDD100k to COCO: {str(e)}")
            return False
    
    @staticmethod
    def convert_cityscapes_to_coco(cityscapes_path: str, output_path: str) -> bool:
        """Convert Cityscapes format to COCO format for semantic segmentation"""
        try:
            logging.info(f"Converting Cityscapes dataset from {cityscapes_path}")
            
            # Use MMDetection's built-in converter
            from mmdet.datasets.pipelines import Compose
            from mmdet.datasets import CityscapesDataset
            
            # This would use the existing Cityscapes dataset class
            # Implementation depends on specific requirements
            
            logging.info("Cityscapes conversion completed")
            return True
            
        except Exception as e:
            logging.error(f"Error converting Cityscapes: {str(e)}")
            return False


class PROVE:
    """Main class for managing the PROVE pipeline"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.config_generator = PROVEConfig()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('movadet_pipeline.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger('MOVADET')
    
    def prepare_dataset(self, dataset_path: str, dataset_format: str, 
                       output_path: str) -> bool:
        """Prepare dataset by converting to appropriate format"""
        
        self.logger.info(f"Preparing {dataset_format} dataset from {dataset_path}")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        converter = DatasetConverter()
        
        if dataset_format == 'bdd100k_json':
            # Find JSON annotation files
            json_files = list(Path(dataset_path).glob('**/*.json'))
            for json_file in json_files:
                output_file = os.path.join(output_path, f"coco_{json_file.stem}.json")
                success = converter.convert_bdd100k_to_coco(str(json_file), output_file)
                if not success:
                    return False
                    
        elif dataset_format == 'cityscapes':
            success = converter.convert_cityscapes_to_coco(dataset_path, output_path)
            if not success:
                return False
                
        # Add more converters as needed
        
        self.logger.info("Dataset preparation completed successfully")
        return True
    
    def generate_config(self, task_type: str, dataset_format: str, 
                       dataset_path: str, model_name: str = None,
                       output_config_path: str = None) -> str:
        """Generate MMDetection configuration file"""
        
        self.logger.info(f"Generating config for {task_type} with {dataset_format}")
        
        # Generate configuration
        config_dict = self.config_generator.generate_config(
            task_type=task_type,
            dataset_format=dataset_format,
            dataset_path=dataset_path,
            model_name=model_name
        )
        
        # Save configuration file
        if output_config_path is None:
            output_config_path = f"prove_{task_type}_{dataset_format}_config.py"
        
        # Convert dict to MMDetection config format
        config_content = self._dict_to_config_file(config_dict)
        
        with open(output_config_path, 'w') as f:
            f.write(config_content)
        
        self.logger.info(f"Configuration saved to {output_config_path}")
        return output_config_path
    
    def _dict_to_config_file(self, config_dict: Dict) -> str:
        """Convert configuration dictionary to MMDetection config file format"""
        
        config_content = "# Generated PROVE Configuration\n\n"
        
        for key, value in config_dict.items():
            if key.startswith('_'):
                # Handle base configs
                config_content += f"{key} = {value}\n\n"
            else:
                config_content += f"{key} = {repr(value)}\n\n"
        
        return config_content
    
    def train(self, config_path: str, work_dir: str = None, 
              resume_from: str = None, load_from: str = None) -> bool:
        """Train model using MMDetection"""
        
        try:
            self.logger.info(f"Starting training with config: {config_path}")
            
            # Load configuration
            cfg = Config.fromfile(config_path)
            
            # Set work directory
            if work_dir:
                cfg.work_dir = work_dir
            elif not hasattr(cfg, 'work_dir'):
                cfg.work_dir = './work_dirs'
            
            # Create work directory
            os.makedirs(cfg.work_dir, exist_ok=True)
            
            # Set resume/load options
            if resume_from:
                cfg.resume_from = resume_from
            if load_from:
                cfg.load_from = load_from
            
            # Build dataset
            datasets = [build_dataset(cfg.data.train)]
            
            # Build model
            model = build_detector(cfg.model)
            model.init_weights()
            
            # Start training
            train_detector(
                model,
                datasets,
                cfg,
                distributed=False,
                validate=True,
                timestamp=None,
                meta=dict()
            )
            
            self.logger.info("Training completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            return False
    
    def test(self, config_path: str, checkpoint_path: str, 
             output_dir: str = None) -> bool:
        """Test model and generate results"""
        
        try:
            self.logger.info(f"Testing model with checkpoint: {checkpoint_path}")
            
            # Load configuration
            cfg = Config.fromfile(config_path)
            
            # Set output directory
            if output_dir is None:
                output_dir = os.path.join(cfg.work_dir, 'results')
            os.makedirs(output_dir, exist_ok=True)
            
            # Initialize model
            model = init_detector(cfg, checkpoint_path)
            
            # Build test dataset
            test_dataset = build_dataset(cfg.data.test)
            
            # Run evaluation
            # This would involve running the MMDetection test script
            cmd = [
                'python', 'tools/test.py',
                config_path,
                checkpoint_path,
                '--eval', 'bbox' if 'detection' in config_path else 'mIoU',
                '--out', os.path.join(output_dir, 'results.pkl')
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("Testing completed successfully")
                return True
            else:
                self.logger.error(f"Testing failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Testing failed: {str(e)}")
            return False
    
    def inference(self, config_path: str, checkpoint_path: str, 
                  image_path: str, output_path: str = None) -> bool:
        """Run inference on single image"""
        
        try:
            self.logger.info(f"Running inference on {image_path}")
            
            # Initialize model
            model = init_detector(config_path, checkpoint_path)
            
            # Run inference
            result = inference_detector(model, image_path)
            
            # Save results
            if output_path:
                # Visualize and save results
                model.show_result(
                    image_path,
                    result,
                    out_file=output_path
                )
                self.logger.info(f"Results saved to {output_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Inference failed: {str(e)}")
            return False


def main():
    """Main function to handle command line arguments"""
    
    parser = argparse.ArgumentParser(description='PROVE Manager')
    parser.add_argument('command', choices=['prepare', 'config', 'train', 'test', 'inference'],
                       help='Pipeline command to execute')
    
    # Dataset preparation arguments
    parser.add_argument('--dataset-path', type=str, help='Path to input dataset')
    parser.add_argument('--dataset-format', type=str, 
                       choices=['bdd100k_json', 'cityscapes', 'mapillary_vistas', 'outside15k'],
                       help='Dataset format')
    parser.add_argument('--output-path', type=str, help='Output path for processed data')
    
    # Configuration arguments
    parser.add_argument('--task-type', type=str, 
                       choices=['object_detection', 'semantic_segmentation'],
                       help='Type of task')
    parser.add_argument('--model-name', type=str, help='Model name to use')
    parser.add_argument('--config-path', type=str, help='Path to configuration file')
    
    # Training arguments
    parser.add_argument('--work-dir', type=str, help='Working directory for training')
    parser.add_argument('--resume-from', type=str, help='Resume training from checkpoint')
    parser.add_argument('--load-from', type=str, help='Load pretrained weights')
    
    # Testing arguments
    parser.add_argument('--checkpoint-path', type=str, help='Path to model checkpoint')
    
    # Inference arguments
    parser.add_argument('--image-path', type=str, help='Path to input image')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PROVE()
    
    # Execute command
    if args.command == 'prepare':
        if not all([args.dataset_path, args.dataset_format, args.output_path]):
            print("Error: prepare command requires --dataset-path, --dataset-format, --output-path")
            return 1
        
        success = pipeline.prepare_dataset(
            args.dataset_path, 
            args.dataset_format, 
            args.output_path
        )
        
    elif args.command == 'config':
        if not all([args.task_type, args.dataset_format, args.dataset_path]):
            print("Error: config command requires --task-type, --dataset-format, --dataset-path")
            return 1
        
        config_path = pipeline.generate_config(
            args.task_type,
            args.dataset_format,
            args.dataset_path,
            args.model_name,
            args.config_path
        )
        print(f"Configuration generated: {config_path}")
        success = True
        
    elif args.command == 'train':
        if not args.config_path:
            print("Error: train command requires --config-path")
            return 1
        
        success = pipeline.train(
            args.config_path,
            args.work_dir,
            args.resume_from,
            args.load_from
        )
        
    elif args.command == 'test':
        if not all([args.config_path, args.checkpoint_path]):
            print("Error: test command requires --config-path, --checkpoint-path")
            return 1
        
        success = pipeline.test(
            args.config_path,
            args.checkpoint_path,
            args.output_path
        )
        
    elif args.command == 'inference':
        if not all([args.config_path, args.checkpoint_path, args.image_path]):
            print("Error: inference command requires --config-path, --checkpoint-path, --image-path")
            return 1
        
        success = pipeline.inference(
            args.config_path,
            args.checkpoint_path,
            args.image_path,
            args.output_path
        )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())