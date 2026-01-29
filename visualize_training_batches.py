#!/usr/bin/env python3
"""
Batch Visualization Hook for Training Verification

This script can be injected into training to visualize actual batches being fed to the model.
It saves sample images, labels, and metadata to verify correct data loading.

Usage:
    1. Import in unified_training.py: from visualize_training_batches import BatchVisualizer
    2. Create instance: visualizer = BatchVisualizer(save_dir="./batch_debug")
    3. Hook into dataloader: visualizer.save_batch(batch_data, batch_idx)
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import json
from typing import Dict, Any, Optional
import cv2


class BatchVisualizer:
    """Visualizes training batches to verify data loading"""
    
    def __init__(
        self,
        save_dir: str = "./batch_debug",
        save_first_n_batches: int = 5,
        save_every_n_iters: Optional[int] = None,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_first_n_batches = save_first_n_batches
        self.save_every_n_iters = save_every_n_iters
        self.batches_saved = 0
        
        # Create subdirectories
        (self.save_dir / "images").mkdir(exist_ok=True)
        (self.save_dir / "labels").mkdir(exist_ok=True)
        (self.save_dir / "overlays").mkdir(exist_ok=True)
        
        # Cityscapes color palette
        self.palette = self._get_cityscapes_palette()
    
    def _get_cityscapes_palette(self):
        """Get Cityscapes color palette for visualization"""
        palette = np.array([
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
        ], dtype=np.uint8)
        return palette
    
    def should_save_batch(self, batch_idx: int, iter_num: Optional[int] = None) -> bool:
        """Determine if this batch should be saved"""
        # Always save first N batches
        if self.batches_saved < self.save_first_n_batches:
            return True
        
        # Save every N iterations if specified
        if self.save_every_n_iters is not None and iter_num is not None:
            if iter_num % self.save_every_n_iters == 0:
                return True
        
        return False
    
    def denormalize_image(self, img: torch.Tensor) -> np.ndarray:
        """
        Denormalize image tensor to uint8 RGB
        
        Args:
            img: (C, H, W) tensor, normalized with ImageNet stats
        
        Returns:
            (H, W, 3) uint8 array
        """
        # ImageNet normalization stats
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        
        # Convert to numpy and denormalize
        img = img.cpu().numpy().transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)
        img = img * std + mean
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Convert BGR to RGB if needed (MMSeg uses BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    
    def colorize_label(self, label: np.ndarray) -> np.ndarray:
        """
        Convert label map to RGB visualization
        
        Args:
            label: (H, W) label map with class IDs
        
        Returns:
            (H, W, 3) uint8 RGB image
        """
        h, w = label.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id in range(len(self.palette)):
            mask = label == class_id
            colored[mask] = self.palette[class_id]
        
        return colored
    
    def create_overlay(self, image: np.ndarray, label: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Create overlay of image and label"""
        colored_label = self.colorize_label(label)
        overlay = (image * (1 - alpha) + colored_label * alpha).astype(np.uint8)
        return overlay
    
    def save_batch(
        self,
        batch_data: Dict[str, Any],
        batch_idx: int,
        iter_num: Optional[int] = None,
        dataset_name: str = "unknown",
        real_gen_ratio: float = 1.0,
    ):
        """
        Save batch images and labels for verification
        
        Args:
            batch_data: Dictionary containing 'inputs' and 'data_samples'
            batch_idx: Current batch index
            iter_num: Current iteration number (optional)
            dataset_name: Name of the dataset being used
            real_gen_ratio: Real/generated ratio being used
        """
        if not self.should_save_batch(batch_idx, iter_num):
            return
        
        # Extract data
        inputs = batch_data.get('inputs', None)
        data_samples = batch_data.get('data_samples', None)
        
        if inputs is None:
            print(f"Warning: No inputs in batch {batch_idx}")
            return
        
        batch_size = inputs.shape[0]
        
        # Save metadata
        metadata = {
            'batch_idx': batch_idx,
            'iter_num': iter_num,
            'batch_size': batch_size,
            'dataset': dataset_name,
            'real_gen_ratio': real_gen_ratio,
            'input_shape': list(inputs.shape),
        }
        
        if data_samples is not None:
            # Extract image paths if available (to verify real vs generated)
            img_paths = []
            for sample in data_samples:
                if hasattr(sample, 'img_path'):
                    img_paths.append(sample.img_path)
                elif hasattr(sample, 'metainfo') and 'img_path' in sample.metainfo:
                    img_paths.append(sample.metainfo['img_path'])
            
            metadata['img_paths'] = img_paths
            
            # Count real vs generated based on path patterns
            real_count = sum(1 for p in img_paths if 'generated' not in p.lower() and 'gen_' not in p.lower())
            gen_count = batch_size - real_count
            metadata['real_count'] = real_count
            metadata['gen_count'] = gen_count
            metadata['composition'] = f"{real_count} real + {gen_count} generated"
        
        # Save metadata
        metadata_path = self.save_dir / f"batch_{self.batches_saved:04d}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save individual samples
        for i in range(min(batch_size, 4)):  # Save up to 4 samples per batch
            # Denormalize image
            img = self.denormalize_image(inputs[i])
            
            # Save image
            img_path = self.save_dir / "images" / f"batch_{self.batches_saved:04d}_sample_{i:02d}.jpg"
            Image.fromarray(img).save(img_path, quality=95)
            
            # Save label if available
            if data_samples is not None and i < len(data_samples):
                sample = data_samples[i]
                
                # Extract label (different attributes depending on mmseg version)
                label = None
                if hasattr(sample, 'gt_sem_seg'):
                    label = sample.gt_sem_seg.data.cpu().numpy().squeeze()
                elif hasattr(sample, 'gt_semantic_seg'):
                    label = sample.gt_semantic_seg.data.cpu().numpy().squeeze()
                
                if label is not None:
                    # Save raw label
                    label_path = self.save_dir / "labels" / f"batch_{self.batches_saved:04d}_sample_{i:02d}.png"
                    Image.fromarray(label.astype(np.uint8)).save(label_path)
                    
                    # Save colored label
                    colored_path = self.save_dir / "labels" / f"batch_{self.batches_saved:04d}_sample_{i:02d}_colored.png"
                    colored = self.colorize_label(label)
                    Image.fromarray(colored).save(colored_path)
                    
                    # Save overlay
                    overlay_path = self.save_dir / "overlays" / f"batch_{self.batches_saved:04d}_sample_{i:02d}.jpg"
                    overlay = self.create_overlay(img, label)
                    Image.fromarray(overlay).save(overlay_path, quality=95)
        
        self.batches_saved += 1
        print(f"✓ Saved batch {batch_idx} visualization (total saved: {self.batches_saved})")
    
    def generate_summary(self):
        """Generate summary report of saved batches"""
        summary_path = self.save_dir / "summary.txt"
        
        # Load all metadata files
        metadata_files = sorted(self.save_dir.glob("batch_*_metadata.json"))
        
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BATCH VISUALIZATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            for meta_file in metadata_files:
                with open(meta_file, 'r') as mf:
                    meta = json.load(mf)
                
                f.write(f"Batch {meta['batch_idx']} (saved as {meta_file.stem}):\n")
                f.write(f"  Iteration: {meta.get('iter_num', 'N/A')}\n")
                f.write(f"  Composition: {meta.get('composition', 'Unknown')}\n")
                f.write(f"  Dataset: {meta['dataset']}\n")
                f.write(f"  Real/Gen Ratio: {meta['real_gen_ratio']}\n")
                
                if 'img_paths' in meta:
                    f.write(f"  Sample Paths:\n")
                    for path in meta['img_paths'][:4]:  # Show first 4
                        path_type = "GENERATED" if 'generated' in path.lower() or 'gen_' in path.lower() else "REAL"
                        f.write(f"    [{path_type}] {path}\n")
                
                f.write("\n")
        
        print(f"✓ Summary report saved to {summary_path}")


def add_batch_visualization_hook(runner, save_dir: str = "./batch_debug", save_first_n: int = 5):
    """
    Add batch visualization hook to MMSeg Runner
    
    Args:
        runner: MMSeg Runner instance
        save_dir: Directory to save visualizations
        save_first_n: Number of initial batches to save
    
    Returns:
        BatchVisualizer instance
    """
    visualizer = BatchVisualizer(save_dir=save_dir, save_first_n_batches=save_first_n)
    
    # Hook into runner's train loop
    # We'll need to monkey-patch the runner's training loop
    # This is a bit hacky but works for debugging
    
    original_run_iter = runner._train_loop.run_iter
    
    def wrapped_run_iter(idx, data_batch):
        # Save batch before processing
        visualizer.save_batch(
            batch_data=data_batch,
            batch_idx=idx,
            iter_num=runner.iter,
            dataset_name=getattr(runner, 'dataset_name', 'unknown'),
            real_gen_ratio=getattr(runner, 'real_gen_ratio', 1.0),
        )
        
        # Call original
        return original_run_iter(idx, data_batch)
    
    runner._train_loop.run_iter = wrapped_run_iter
    
    print(f"✓ Batch visualization hook added (saving to {save_dir})")
    print(f"  Will save first {save_first_n} batches")
    
    return visualizer
