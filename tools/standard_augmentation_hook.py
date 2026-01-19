#!/usr/bin/env python3
"""
Standard Augmentation Hook for MMEngine

This hook applies batch-level standard augmentations (CutMix, MixUp, AutoAugment, RandAugment)
during training. These augmentations require batch-level operations and cannot be integrated
into the per-sample data pipeline.

The hook intercepts batches before the forward pass and applies the configured augmentation.

Usage:
    In training config, add to custom_hooks:
    
    custom_hooks = [
        dict(
            type='StandardAugmentationHook',
            method='cutmix',  # or 'mixup', 'autoaugment', 'randaugment'
            p_aug=0.5,
            priority='VERY_HIGH',  # Apply before training step
        ),
    ]
"""

import torch
from typing import Dict, Any, Optional
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

try:
    from tools.standard_augmentations import StandardAugmentationFamily
    STANDARD_AUG_AVAILABLE = True
except ImportError:
    try:
        from standard_augmentations import StandardAugmentationFamily
        STANDARD_AUG_AVAILABLE = True
    except ImportError:
        STANDARD_AUG_AVAILABLE = False
        print("Warning: StandardAugmentationFamily not available")


@HOOKS.register_module()
class StandardAugmentationHook(Hook):
    """
    Hook that applies standard augmentations (CutMix, MixUp, etc.) to batches during training.
    
    This hook modifies the input batch before the model forward pass, applying
    batch-level augmentations that require mixing operations across samples.
    
    Args:
        method: Augmentation method ('cutmix', 'mixup', 'autoaugment', 'randaugment')
        p_aug: Probability of applying augmentation (default: 0.5)
        priority: Hook priority (default: 'VERY_HIGH' to run before forward)
    
    Example config:
        custom_hooks = [
            dict(
                type='StandardAugmentationHook',
                method='cutmix',
                p_aug=0.5,
            ),
        ]
    """
    
    priority = 'VERY_HIGH'  # Run before training step
    
    def __init__(
        self,
        method: str = 'cutmix',
        p_aug: float = 0.5,
        **kwargs
    ):
        super().__init__()
        
        if not STANDARD_AUG_AVAILABLE:
            raise ImportError(
                "StandardAugmentationFamily not available. "
                "Please ensure tools/standard_augmentations.py is accessible."
            )
        
        self.method = method
        self.p_aug = p_aug
        
        # Initialize the augmentation
        self.augmentation = StandardAugmentationFamily(
            method=method,
            p_aug=p_aug,
        )
        
        print(f"[StandardAugmentationHook] Initialized with method='{method}', p_aug={p_aug}")
    
    def before_train_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Apply standard augmentation to the batch before training iteration.
        
        This hook intercepts the data batch and applies batch-level augmentation
        to both images and segmentation masks.
        
        Args:
            runner: The training runner
            batch_idx: Index of current batch
            data_batch: Dictionary containing 'inputs' and 'data_samples'
        """
        if data_batch is None:
            return
        
        # Extract inputs (images) from the batch
        # MMSeg format: data_batch['inputs'] is a list of tensors or a batched tensor
        inputs = data_batch.get('inputs')
        data_samples = data_batch.get('data_samples')
        
        if inputs is None or data_samples is None:
            return
        
        try:
            # Convert inputs to batched tensor if needed
            if isinstance(inputs, (list, tuple)):
                # Stack list of tensors into batch
                images = torch.stack([inp for inp in inputs], dim=0)
            else:
                images = inputs
            
            # Extract segmentation masks from data_samples
            labels = []
            for sample in data_samples:
                # MMSeg stores labels in gt_sem_seg.data
                if hasattr(sample, 'gt_sem_seg') and hasattr(sample.gt_sem_seg, 'data'):
                    labels.append(sample.gt_sem_seg.data)
                elif hasattr(sample, 'gt_semantic_seg'):
                    labels.append(sample.gt_semantic_seg)
                else:
                    # Cannot find labels, skip augmentation
                    return
            
            if len(labels) == 0:
                return
            
            # Stack labels into batch tensor
            labels = torch.stack(labels, dim=0)
            
            # Remove extra dimension if present (B, 1, H, W) -> (B, H, W)
            if labels.dim() == 4 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            
            # Ensure images are on same device as labels
            device = labels.device
            images = images.to(device)
            
            # Normalize images to [0, 1] range if needed (for proper augmentation)
            # MMSeg typically uses images in [0, 255] range after normalization
            # But standard augmentations expect [0, 1]
            img_min, img_max = images.min(), images.max()
            if img_max > 1.0:
                # Likely in [0, 255] range, normalize
                images_norm = images / 255.0
                was_normalized = False
            else:
                images_norm = images
                was_normalized = True
            
            # Apply augmentation
            aug_images, aug_labels = self.augmentation(images_norm, labels)
            
            # Denormalize back if needed
            if not was_normalized:
                aug_images = aug_images * 255.0
            
            # Update the batch in-place
            # Update inputs
            if isinstance(data_batch['inputs'], (list, tuple)):
                for i in range(len(data_batch['inputs'])):
                    data_batch['inputs'][i] = aug_images[i]
            else:
                data_batch['inputs'] = aug_images
            
            # Update labels in data_samples
            for i, sample in enumerate(data_samples):
                if hasattr(sample, 'gt_sem_seg') and hasattr(sample.gt_sem_seg, 'data'):
                    # Add dimension back if needed
                    new_label = aug_labels[i]
                    if sample.gt_sem_seg.data.dim() == 3:
                        new_label = new_label.unsqueeze(0)
                    sample.gt_sem_seg.data = new_label
                elif hasattr(sample, 'gt_semantic_seg'):
                    sample.gt_semantic_seg = aug_labels[i]
            
        except Exception as e:
            # Log error but don't crash training
            import logging
            logging.warning(f"[StandardAugmentationHook] Error applying augmentation: {e}")
            # Continue with original batch
            pass
    
    def after_train_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """No-op after training iteration."""
        pass


def build_standard_augmentation_hook_config(
    method: str,
    p_aug: float = 0.5,
) -> Dict[str, Any]:
    """
    Build configuration dict for StandardAugmentationHook.
    
    Args:
        method: Augmentation method ('cutmix', 'mixup', 'autoaugment', 'randaugment')
        p_aug: Probability of applying augmentation
        
    Returns:
        Configuration dictionary for custom_hooks
    """
    return dict(
        type='StandardAugmentationHook',
        method=method,
        p_aug=p_aug,
    )


# Test hook standalone
if __name__ == '__main__':
    print("Testing StandardAugmentationHook...")
    
    # Create hook
    hook = StandardAugmentationHook(method='cutmix', p_aug=1.0)
    
    # Create mock batch
    B, C, H, W = 4, 3, 512, 512
    images = torch.rand(B, C, H, W) * 255.0
    
    # Create mock data_samples (simplified)
    class MockSample:
        def __init__(self, label):
            self.gt_sem_seg = type('obj', (object,), {'data': label})()
    
    labels = torch.randint(0, 19, (B, 1, H, W))
    data_samples = [MockSample(labels[i]) for i in range(B)]
    
    data_batch = {
        'inputs': images,
        'data_samples': data_samples,
    }
    
    print(f"Before augmentation:")
    print(f"  images shape: {data_batch['inputs'].shape}")
    print(f"  labels shape: {data_samples[0].gt_sem_seg.data.shape}")
    
    # Apply hook
    hook.before_train_iter(None, 0, data_batch)
    
    print(f"After augmentation:")
    print(f"  images shape: {data_batch['inputs'].shape}")
    print(f"  labels shape: {data_samples[0].gt_sem_seg.data.shape}")
    print("✓ Hook test passed!")
