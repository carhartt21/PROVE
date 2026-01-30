#!/usr/bin/env python3
"""
Validation Visualization Hook for MMEngine

This hook saves segmentation visualization outputs during validation with:
- Input image
- Ground truth
- Model prediction
Side by side in a single output image.

Usage:
    In training config, add to default_hooks:
    
    default_hooks = dict(
        ...
        visualization=dict(
            type='ValVisualizationHook',
            max_samples=5,  # Save up to 5 samples per validation
        ),
    )
"""

import os
import os.path as osp
from typing import Optional, Dict, Any, List
import numpy as np
import torch

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.registry import HOOKS


# Cityscapes color palette for 19 classes
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


def colorize_mask(mask: np.ndarray, num_classes: int = 19) -> np.ndarray:
    """Convert class indices to colored mask.
    
    Args:
        mask: 2D array of class indices (H, W)
        num_classes: Number of classes
        
    Returns:
        3D array of RGB colors (H, W, 3)
    """
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(num_classes):
        if class_id < len(CITYSCAPES_PALETTE):
            color = CITYSCAPES_PALETTE[class_id]
        else:
            # Generate color for classes beyond Cityscapes
            color = [(class_id * 67) % 256, (class_id * 113) % 256, (class_id * 179) % 256]
        colored[mask == class_id] = color
    
    return colored


@HOOKS.register_module()
class ValVisualizationHook(Hook):
    """
    Hook to save validation visualizations showing Input | GT | Prediction side by side.
    
    Args:
        max_samples: Maximum number of samples to save per validation (default: 5)
        output_dir: Subdirectory name within work_dir (default: 'val_visualizations')
    
    Example config:
        default_hooks = dict(
            ...
            visualization=dict(
                type='ValVisualizationHook',
                max_samples=5,
            ),
        )
    """
    
    def __init__(
        self,
        max_samples: int = 5,
        output_dir: str = 'val_visualizations',
    ):
        super().__init__()
        self.max_samples = max_samples
        self.output_dir = output_dir
        self._sample_count = 0
        self._current_iter = 0
        
        print(f"[ValVisualizationHook] Initialized: max_samples={max_samples}")
    
    def before_val_epoch(self, runner: Runner) -> None:
        """Reset sample counter before each validation epoch."""
        self._sample_count = 0
        self._current_iter = runner.iter
    
    def after_val_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: Optional[Dict[str, Any]] = None,
        outputs: Optional[List] = None,
    ) -> None:
        """
        Save visualization after validation iteration.
        
        Args:
            runner: The training runner
            batch_idx: Index of current batch
            data_batch: Dictionary containing 'inputs' and 'data_samples'
            outputs: Model outputs (SegDataSample list with pred_sem_seg)
        """
        if self._sample_count >= self.max_samples:
            return
        
        if data_batch is None or outputs is None:
            return
        
        try:
            # Lazy import to avoid circular imports
            from PIL import Image
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            # Create output directory
            save_dir = osp.join(runner.work_dir, self.output_dir, f'iter_{self._current_iter}')
            os.makedirs(save_dir, exist_ok=True)
            
            inputs = data_batch.get('inputs', [])
            data_samples = data_batch.get('data_samples', [])
            
            # Process each sample in the batch (up to max_samples total)
            for i in range(len(outputs)):
                if self._sample_count >= self.max_samples:
                    break
                
                # Get input image
                if isinstance(inputs, torch.Tensor):
                    if inputs.dim() == 4:
                        img = inputs[i].cpu().numpy()
                    else:
                        img = inputs.cpu().numpy()
                elif isinstance(inputs, list) and len(inputs) > i:
                    img = inputs[i].cpu().numpy()
                else:
                    continue
                
                # Denormalize image (MMSeg uses ImageNet normalization)
                img = self._denormalize_image(img)
                
                # Get ground truth
                gt_mask = None
                if i < len(data_samples):
                    sample = data_samples[i]
                    if hasattr(sample, 'gt_sem_seg') and hasattr(sample.gt_sem_seg, 'data'):
                        gt_mask = sample.gt_sem_seg.data.cpu().numpy().squeeze()
                
                # Get prediction
                pred_mask = None
                output = outputs[i]
                if hasattr(output, 'pred_sem_seg') and hasattr(output.pred_sem_seg, 'data'):
                    pred_mask = output.pred_sem_seg.data.cpu().numpy().squeeze()
                
                if gt_mask is None or pred_mask is None:
                    continue
                
                # Create side-by-side visualization
                self._save_visualization(
                    save_dir=save_dir,
                    sample_idx=self._sample_count,
                    image=img,
                    gt_mask=gt_mask,
                    pred_mask=pred_mask,
                )
                
                self._sample_count += 1
                
        except Exception as e:
            print(f"[ValVisualizationHook] Warning: Failed to save visualization: {e}")
    
    def _denormalize_image(self, img: np.ndarray) -> np.ndarray:
        """Denormalize image from ImageNet normalization.
        
        Args:
            img: Image tensor in CHW format, normalized
            
        Returns:
            Image in HWC format, uint8 [0, 255]
        """
        # ImageNet mean/std
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        
        # CHW -> HWC
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        
        # Denormalize
        img = img * std + mean
        
        # Clip and convert
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return img
    
    def _save_visualization(
        self,
        save_dir: str,
        sample_idx: int,
        image: np.ndarray,
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
    ) -> None:
        """Save Input | GT | Prediction side by side.
        
        Args:
            save_dir: Directory to save the visualization
            sample_idx: Sample index
            image: Input image (H, W, 3)
            gt_mask: Ground truth mask (H, W)
            pred_mask: Predicted mask (H, W)
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Determine number of classes from masks
        num_classes = max(gt_mask.max(), pred_mask.max()) + 1
        num_classes = max(num_classes, 19)  # At least Cityscapes 19 classes
        
        # Colorize masks
        gt_colored = colorize_mask(gt_mask.astype(np.int32), num_classes)
        pred_colored = colorize_mask(pred_mask.astype(np.int32), num_classes)
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Input image
        axes[0].imshow(image)
        axes[0].set_title('Input', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Ground truth
        axes[1].imshow(gt_colored)
        axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(pred_colored)
        axes[2].set_title('Prediction', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save
        save_path = osp.join(save_dir, f'sample_{sample_idx:03d}.png')
        fig.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        if sample_idx == 0:  # Only print for first sample
            print(f"[ValVisualizationHook] Saving visualizations to {save_dir}")


# For testing
if __name__ == '__main__':
    print("ValVisualizationHook module loaded successfully")
    print(f"  - Cityscapes palette: {len(CITYSCAPES_PALETTE)} colors")
