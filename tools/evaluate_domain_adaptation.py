#!/usr/bin/env python3
"""
Domain Adaptation Ablation: Evaluate models trained on BDD10k/IDD-AW/MapillaryVistas on ACDC

This script evaluates cross-dataset domain adaptation by testing models trained on
traffic-focused datasets against the ACDC adverse weather benchmark.

Key features:
1. Filters out ACDC reference images (_ref_ suffix) with mismatched labels
2. Excludes clear_day and dawn_dusk domains (no valid non-ref images)
3. Handles label unification (Cityscapes labelID → trainID for ACDC)
4. Reports per-domain (weather condition) metrics

Usage:
    python evaluate_domain_adaptation.py \
        --source-dataset BDD10k \
        --model deeplabv3plus_r50 \
        --checkpoint /path/to/checkpoint.pth

    python evaluate_domain_adaptation.py --all  # Evaluate all combinations
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Configuration
# ============================================================================

DATA_ROOT = Path(os.environ.get('PROVE_DATA_ROOT', '/scratch/aaa_exchange/AWARE/FINAL_SPLITS'))
WEIGHTS_ROOT = Path(os.environ.get('PROVE_WEIGHTS_ROOT', '/scratch/aaa_exchange/AWARE/WEIGHTS'))
OUTPUT_ROOT = WEIGHTS_ROOT / 'domain_adaptation_ablation'

# Source datasets for training
SOURCE_DATASETS = ['BDD10k', 'IDD-AW', 'MapillaryVistas']

# Models to evaluate
MODELS = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']

# ACDC domains to evaluate (excluding clear_day and dawn_dusk with no valid images)
ACDC_VALID_DOMAINS = ['foggy', 'rainy', 'snowy', 'night', 'cloudy']

# Cityscapes classes
CITYSCAPES_CLASSES = (
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle',
)

# Cityscapes labelID → trainID mapping
CITYSCAPES_ID_TO_TRAINID = {
    7: 0,   # road
    8: 1,   # sidewalk
    11: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    17: 5,  # pole
    19: 6,  # traffic light
    20: 7,  # traffic sign
    21: 8,  # vegetation
    22: 9,  # terrain
    23: 10, # sky
    24: 11, # person
    25: 12, # rider
    26: 13, # car
    27: 14, # truck
    28: 15, # bus
    31: 16, # train
    32: 17, # motorcycle
    33: 18, # bicycle
}


# ============================================================================
# Label Transformation
# ============================================================================

def create_cityscapes_lut():
    """Create lookup table for Cityscapes labelID → trainID conversion."""
    lut = np.full(256, 255, dtype=np.uint8)  # Default to ignore
    for label_id, train_id in CITYSCAPES_ID_TO_TRAINID.items():
        lut[label_id] = train_id
    return lut

CITYSCAPES_LUT = create_cityscapes_lut()


def transform_acdc_label(label: np.ndarray) -> np.ndarray:
    """Transform ACDC label from Cityscapes labelID to trainID format."""
    # Handle 3-channel labels (take first channel)
    if label.ndim == 3:
        label = label[:, :, 0]
    return CITYSCAPES_LUT[label]


# ============================================================================
# Data Loading
# ============================================================================

def get_acdc_file_pairs(split: str = 'test') -> Dict[str, List[Tuple[Path, Path]]]:
    """
    Get ACDC image-label pairs, filtering out reference images.
    
    Returns:
        Dict mapping domain name to list of (image_path, label_path) tuples
    """
    img_root = DATA_ROOT / split / 'images' / 'ACDC'
    label_root = DATA_ROOT / split / 'labels' / 'ACDC'
    
    domain_pairs = {}
    
    for domain in ACDC_VALID_DOMAINS:
        domain_img_dir = img_root / domain
        domain_label_dir = label_root / domain
        
        if not domain_img_dir.exists():
            print(f"Warning: Domain directory not found: {domain_img_dir}")
            continue
        
        pairs = []
        for img_file in sorted(domain_img_dir.glob('*.png')):
            # Filter out reference images
            if '_ref' in img_file.name:
                continue
            
            # Find corresponding label file
            label_file = domain_label_dir / img_file.name
            if label_file.exists():
                pairs.append((img_file, label_file))
            else:
                print(f"Warning: Label not found for {img_file.name}")
        
        if pairs:
            domain_pairs[domain] = pairs
            print(f"  {domain}: {len(pairs)} valid image-label pairs")
    
    return domain_pairs


def get_acdc_combined_files(split: str = 'test') -> List[Tuple[Path, Path]]:
    """Get all ACDC files combined (train + test) excluding reference images."""
    all_pairs = []
    
    for s in ['train', 'test'] if split == 'all' else [split]:
        domain_pairs = get_acdc_file_pairs(s)
        for pairs in domain_pairs.values():
            all_pairs.extend(pairs)
    
    return all_pairs


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_name: str, checkpoint_path: str, device: str = 'cuda'):
    """Load segmentation model from checkpoint."""
    try:
        from mmseg.apis import init_model
        from unified_training_config import UnifiedTrainingConfig
        
        # Build config for model
        config = UnifiedTrainingConfig()
        
        # We need a base config - use ACDC as placeholder since we're only loading weights
        cfg = config.build(
            dataset='ACDC',
            model=model_name,
            strategy='baseline',
            config_only=True
        )
        
        # Initialize model with checkpoint
        model = init_model(cfg, checkpoint_path, device=device)
        model.eval()
        
        return model
        
    except ImportError:
        print("Error: mmseg not available. Please install mmengine and mmseg.")
        sys.exit(1)


def load_model_torchvision(model_name: str, checkpoint_path: str, num_classes: int = 19, device: str = 'cuda'):
    """Load model using torchvision (fallback if mmseg not available)."""
    import torch
    import torchvision.models.segmentation as seg_models
    
    # Map model names to torchvision constructors
    model_map = {
        'deeplabv3plus_r50': lambda: seg_models.deeplabv3_resnet50(weights=None, num_classes=num_classes),
        'deeplabv3_r50': lambda: seg_models.deeplabv3_resnet50(weights=None, num_classes=num_classes),
        'fcn_r50': lambda: seg_models.fcn_resnet50(weights=None, num_classes=num_classes),
    }
    
    if model_name not in model_map:
        raise ValueError(f"Model {model_name} not supported in torchvision fallback mode")
    
    model = model_map[model_name]()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Try to load weights (may need key adaptation for mmseg checkpoints)
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Warning: Partial weight loading: {e}")
    
    model = model.to(device)
    model.eval()
    
    return model


# ============================================================================
# Inference & Evaluation
# ============================================================================

def predict_single(model, image: np.ndarray, device: str = 'cuda') -> np.ndarray:
    """Run inference on a single image."""
    import torch
    from torchvision import transforms
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Handle mmseg models vs torchvision models
    if hasattr(model, 'inference'):
        # MMSeg model
        from mmseg.apis import inference_model
        result = inference_model(model, image)
        pred = result.pred_sem_seg.data.cpu().numpy()[0]
    else:
        # Torchvision model
        with torch.no_grad():
            input_tensor = transform(image).unsqueeze(0).to(device)
            output = model(input_tensor)
            if isinstance(output, dict):
                output = output['out']
            pred = output.argmax(1).cpu().numpy()[0]
    
    return pred


def compute_iou(pred: np.ndarray, label: np.ndarray, num_classes: int = 19, ignore_index: int = 255):
    """Compute IoU for each class."""
    ious = []
    
    for cls_id in range(num_classes):
        pred_mask = pred == cls_id
        label_mask = label == cls_id
        
        # Exclude ignore pixels
        valid_mask = label != ignore_index
        pred_mask = pred_mask & valid_mask
        label_mask = label_mask & valid_mask
        
        intersection = (pred_mask & label_mask).sum()
        union = (pred_mask | label_mask).sum()
        
        if union == 0:
            iou = float('nan')  # Class not present
        else:
            iou = intersection / union
        
        ious.append(iou)
    
    return np.array(ious)


def evaluate_model_on_acdc(
    model,
    domain_pairs: Dict[str, List[Tuple[Path, Path]]],
    device: str = 'cuda',
    num_classes: int = 19
) -> Dict:
    """
    Evaluate model on ACDC dataset with per-domain breakdown.
    
    Returns:
        Dict with overall and per-domain metrics
    """
    results = {
        'overall': {'ious': [], 'per_class_iou': np.zeros(num_classes)},
        'per_domain': {}
    }
    
    total_intersection = np.zeros(num_classes)
    total_union = np.zeros(num_classes)
    
    for domain, pairs in domain_pairs.items():
        print(f"\nEvaluating {domain} ({len(pairs)} images)...")
        
        domain_intersection = np.zeros(num_classes)
        domain_union = np.zeros(num_classes)
        
        for img_path, label_path in tqdm(pairs, desc=domain):
            # Load image and label
            image = np.array(Image.open(img_path).convert('RGB'))
            label = np.array(Image.open(label_path))
            
            # Transform ACDC label from Cityscapes labelID to trainID
            label = transform_acdc_label(label)
            
            # Resize image if needed (model may expect specific size)
            # For now, assume model handles variable sizes
            
            # Run inference
            pred = predict_single(model, image, device)
            
            # Resize prediction to match label if needed
            if pred.shape != label.shape:
                from scipy.ndimage import zoom
                zoom_factors = (label.shape[0] / pred.shape[0], label.shape[1] / pred.shape[1])
                pred = zoom(pred, zoom_factors, order=0)  # Nearest neighbor for labels
            
            # Compute per-class IoU
            for cls_id in range(num_classes):
                pred_mask = pred == cls_id
                label_mask = label == cls_id
                valid_mask = label != 255
                
                pred_mask = pred_mask & valid_mask
                label_mask = label_mask & valid_mask
                
                intersection = (pred_mask & label_mask).sum()
                union = (pred_mask | label_mask).sum()
                
                domain_intersection[cls_id] += intersection
                domain_union[cls_id] += union
        
        # Compute domain mIoU
        valid_classes = domain_union > 0
        domain_iou = np.where(valid_classes, domain_intersection / domain_union, np.nan)
        domain_miou = np.nanmean(domain_iou)
        
        results['per_domain'][domain] = {
            'mIoU': float(domain_miou),
            'per_class_iou': domain_iou.tolist(),
            'num_images': len(pairs)
        }
        
        print(f"  {domain} mIoU: {domain_miou * 100:.2f}%")
        
        # Accumulate for overall
        total_intersection += domain_intersection
        total_union += domain_union
    
    # Compute overall mIoU
    valid_classes = total_union > 0
    overall_iou = np.where(valid_classes, total_intersection / total_union, np.nan)
    overall_miou = np.nanmean(overall_iou)
    
    results['overall'] = {
        'mIoU': float(overall_miou),
        'per_class_iou': overall_iou.tolist(),
        'class_names': CITYSCAPES_CLASSES
    }
    
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

def get_checkpoint_path(source_dataset: str, model: str) -> Optional[Path]:
    """Get checkpoint path for a source dataset and model."""
    checkpoint_dir = WEIGHTS_ROOT / 'baseline' / source_dataset.lower() / model
    
    # Try different checkpoint names
    candidates = [
        checkpoint_dir / 'iter_80000.pth',
        checkpoint_dir / 'latest.pth',
    ]
    
    # Also try best checkpoint
    for f in checkpoint_dir.glob('best_*.pth'):
        candidates.insert(0, f)
    
    for path in candidates:
        if path.exists():
            return path
    
    return None


def run_evaluation(source_dataset: str, model: str, checkpoint_path: str = None, device: str = 'cuda'):
    """Run domain adaptation evaluation for a single configuration."""
    
    print(f"\n{'='*70}")
    print(f"Domain Adaptation Evaluation")
    print(f"  Source Dataset: {source_dataset}")
    print(f"  Model: {model}")
    print(f"{'='*70}")
    
    # Get checkpoint path if not provided
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path(source_dataset, model)
        if checkpoint_path is None:
            print(f"ERROR: No checkpoint found for {source_dataset}/{model}")
            return None
    
    checkpoint_path = Path(checkpoint_path)
    print(f"  Checkpoint: {checkpoint_path}")
    
    # Load ACDC evaluation data (combining train + test, excluding _ref images)
    print("\nLoading ACDC evaluation data (train + test splits)...")
    
    # Get pairs from both splits and combine
    domain_pairs = {}
    for split in ['train', 'test']:
        split_pairs = get_acdc_file_pairs(split)
        for domain, pairs in split_pairs.items():
            if domain not in domain_pairs:
                domain_pairs[domain] = []
            domain_pairs[domain].extend(pairs)
    
    # Print summary
    print("\nCombined dataset summary:")
    total_images = 0
    for domain, pairs in domain_pairs.items():
        print(f"  {domain}: {len(pairs)} images")
        total_images += len(pairs)
    print(f"Total evaluation images: {total_images}")
    
    # Load model
    print("\nLoading model...")
    try:
        loaded_model = load_model(model, str(checkpoint_path), device)
    except Exception as e:
        print(f"MMSeg loading failed: {e}")
        print("Trying torchvision fallback...")
        loaded_model = load_model_torchvision(model, str(checkpoint_path), device=device)
    
    # Run evaluation
    results = evaluate_model_on_acdc(loaded_model, domain_pairs, device)
    
    # Add metadata
    results['metadata'] = {
        'source_dataset': source_dataset,
        'model': model,
        'checkpoint': str(checkpoint_path),
        'target_dataset': 'ACDC',
        'excluded_domains': ['clear_day', 'dawn_dusk'],
        'excluded_pattern': '_ref'
    }
    
    # Save results
    output_dir = OUTPUT_ROOT / source_dataset.lower() / model
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = output_dir / 'acdc_evaluation.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Overall mIoU: {results['overall']['mIoU'] * 100:.2f}%")
    print(f"\nPer-Domain mIoU:")
    for domain, metrics in results['per_domain'].items():
        print(f"  {domain:<10}: {metrics['mIoU'] * 100:.2f}% ({metrics['num_images']} images)")
    
    print(f"\nResults saved to: {result_file}")
    
    return results


def run_all_evaluations(device: str = 'cuda'):
    """Run evaluation for all source dataset / model combinations."""
    
    all_results = {}
    
    for source_dataset in SOURCE_DATASETS:
        for model in MODELS:
            key = f"{source_dataset}_{model}"
            
            results = run_evaluation(source_dataset, model, device=device)
            if results:
                all_results[key] = results
    
    # Save combined results
    combined_file = OUTPUT_ROOT / 'all_results.json'
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"ALL EVALUATIONS COMPLETE")
    print(f"Combined results saved to: {combined_file}")
    print(f"{'='*70}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Domain Adaptation Ablation: Evaluate cross-dataset generalization to ACDC"
    )
    
    parser.add_argument('--source-dataset', type=str, choices=SOURCE_DATASETS,
                        help='Source dataset the model was trained on')
    parser.add_argument('--model', type=str, choices=MODELS,
                        help='Model architecture')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (auto-detected if not provided)')
    parser.add_argument('--all', action='store_true',
                        help='Evaluate all source dataset / model combinations')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda or cpu)')
    parser.add_argument('--list-checkpoints', action='store_true',
                        help='List available checkpoints')
    
    args = parser.parse_args()
    
    if args.list_checkpoints:
        print("Available checkpoints:")
        for source in SOURCE_DATASETS:
            for model in MODELS:
                ckpt = get_checkpoint_path(source, model)
                status = "✓" if ckpt else "✗"
                print(f"  {status} {source}/{model}: {ckpt if ckpt else 'NOT FOUND'}")
        return
    
    if args.all:
        run_all_evaluations(device=args.device)
    elif args.source_dataset and args.model:
        run_evaluation(args.source_dataset, args.model, args.checkpoint, args.device)
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python evaluate_domain_adaptation.py --source-dataset BDD10k --model deeplabv3plus_r50")
        print("  python evaluate_domain_adaptation.py --all")


if __name__ == '__main__':
    main()
