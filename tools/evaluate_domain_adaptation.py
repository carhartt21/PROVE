#!/usr/bin/env python3
"""
Domain Adaptation Ablation: Evaluate models trained on BDD10k/IDD-AW/MapillaryVistas

This script evaluates cross-dataset domain adaptation by testing models trained on
traffic-focused datasets against:
- Cityscapes (clear_day condition)
- ACDC adverse weather conditions (foggy, night, rainy, snowy)

Key features:
1. Evaluates both full dataset models and clear_day baseline models (_clear_day variant)
2. Handles label unification (Cityscapes labelID → trainID for both datasets)
3. Reports per-domain (weather condition) metrics
4. Supports comparing full vs. clear_day training effect on domain adaptation

Usage:
    # Evaluate full dataset model
    python evaluate_domain_adaptation.py \
        --source-dataset BDD10k \
        --model deeplabv3plus_r50

    # Evaluate clear_day baseline model
    python evaluate_domain_adaptation.py \
        --source-dataset BDD10k \
        --model deeplabv3plus_r50 \
        --variant _clear_day

    # Evaluate all combinations (full + clear_day for all source/model pairs)
    python evaluate_domain_adaptation.py --all

    # Evaluate all without variants (full dataset models only)
    python evaluate_domain_adaptation.py --all --no-variants

    # List available checkpoints
    python evaluate_domain_adaptation.py --list-checkpoints
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

DATA_ROOT = Path(os.environ.get('PROVE_DATA_ROOT', '${AWARE_DATA_ROOT}/FINAL_SPLITS'))
WEIGHTS_ROOT = Path(os.environ.get('PROVE_WEIGHTS_ROOT', '${AWARE_DATA_ROOT}/WEIGHTS'))
OUTPUT_ROOT = WEIGHTS_ROOT / 'domain_adaptation_ablation'

# Source datasets for training
SOURCE_DATASETS = ['BDD10k', 'IDD-AW', 'MapillaryVistas']

# Base models to evaluate
# DeepLabV3+ excluded per DOMAIN_ADAPTATION_ABLATION.md update
BASE_MODELS = ['pspnet_r50', 'segformer_mit-b5']

# Model variants (including clear_day trained baseline)
MODELS = BASE_MODELS  # For backward compatibility
MODEL_VARIANTS = ['', '_clear_day']  # '' = full dataset, '_clear_day' = clear_day only

# ACDC domains to evaluate - adverse weather conditions
# Note: ACDC now has flat structure in test/images/ACDC/{domain}/ and test/labels/ACDC/{domain}/
ACDC_DOMAINS = ['foggy', 'night', 'rainy', 'snowy']

# Cityscapes represents "clear_day" condition for domain adaptation study
# Structure: test/images/Cityscapes/{city}/ and test/labels/Cityscapes/{city}/
# Note: Cityscapes TEST split contains only these 3 cities
CITYSCAPES_CITIES = ['frankfurt', 'lindau', 'munster']

# Combined domains for evaluation: clear_day (Cityscapes) + adverse (ACDC)
ALL_DOMAINS = ['clear_day'] + ACDC_DOMAINS

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
    Get ACDC image-label pairs from the flat folder structure.
    
    New structure (post-reorganization):
    - test/images/ACDC/{domain}/*.png - flat image files
    - test/labels/ACDC/{domain}/*.png - flat label files (labelID format)
    
    Returns:
        Dict mapping domain name to list of (image_path, label_path) tuples
    """
    img_root = DATA_ROOT / split / 'images' / 'ACDC'
    label_root = DATA_ROOT / split / 'labels' / 'ACDC'
    
    domain_pairs = {}
    
    for domain in ACDC_DOMAINS:
        domain_img_dir = img_root / domain
        domain_label_dir = label_root / domain
        
        if not domain_img_dir.exists():
            print(f"Warning: Domain directory not found: {domain_img_dir}")
            continue
        
        pairs = []
        # New flat structure: images are directly in domain folder with _rgb_anon.png suffix
        for img_file in sorted(domain_img_dir.glob('*_rgb_anon.png')):
            # Skip reference images (should not exist in new structure, but check anyway)
            if '_ref' in img_file.name:
                continue
            
            # Find corresponding label file
            # Image: GOPR0475_frame_000041_rgb_anon.png
            # Label: GOPR0475_frame_000041_gt_labelIds.png
            base_name = img_file.name.replace('_rgb_anon.png', '_gt_labelIds.png')
            label_file = domain_label_dir / base_name
            
            if label_file.exists():
                pairs.append((img_file, label_file))
            else:
                print(f"Warning: Label not found for {img_file.name}")
        
        if pairs:
            domain_pairs[domain] = pairs
            print(f"  {domain}: {len(pairs)} valid image-label pairs")
    
    return domain_pairs


def get_cityscapes_file_pairs(split: str = 'test') -> List[Tuple[Path, Path]]:
    """
    Get Cityscapes image-label pairs. Used as "clear_day" condition.
    
    Structure:
    - test/images/Cityscapes/{city}/*_leftImg8bit.png
    - test/labels/Cityscapes/{city}/*_gtFine_labelIds.png
    
    Returns:
        List of (image_path, label_path) tuples
    """
    img_root = DATA_ROOT / split / 'images' / 'Cityscapes'
    label_root = DATA_ROOT / split / 'labels' / 'Cityscapes'
    
    pairs = []
    
    for city in CITYSCAPES_CITIES:
        city_img_dir = img_root / city
        city_label_dir = label_root / city
        
        if not city_img_dir.exists():
            print(f"Warning: City directory not found: {city_img_dir}")
            continue
        
        for img_file in sorted(city_img_dir.glob('*_leftImg8bit.png')):
            # Find corresponding label file
            # Image: berlin_000000_000019_leftImg8bit.png
            # Label: berlin_000000_000019_gtFine_labelIds.png
            base_name = img_file.name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
            label_file = city_label_dir / base_name
            
            if label_file.exists():
                pairs.append((img_file, label_file))
            else:
                print(f"Warning: Label not found for {img_file.name}")
    
    if pairs:
        print(f"  clear_day (Cityscapes): {len(pairs)} valid image-label pairs")
    
    return pairs
    
    return domain_pairs


def get_all_domain_file_pairs(split: str = 'test') -> Dict[str, List[Tuple[Path, Path]]]:
    """
    Get all domain file pairs including Cityscapes (clear_day) and ACDC (adverse).
    
    Returns:
        Dict mapping domain name to list of (image_path, label_path) tuples
        Domains: clear_day, foggy, night, rainy, snowy
    """
    domain_pairs = {}
    
    # Add Cityscapes as clear_day
    cityscapes_pairs = get_cityscapes_file_pairs(split)
    if cityscapes_pairs:
        domain_pairs['clear_day'] = cityscapes_pairs
    
    # Add ACDC domains
    acdc_pairs = get_acdc_file_pairs(split)
    domain_pairs.update(acdc_pairs)
    
    return domain_pairs


def get_acdc_combined_files(split: str = 'test') -> List[Tuple[Path, Path]]:
    """Get all ACDC files combined (for backward compatibility)."""
    all_pairs = []
    
    acdc_pairs = get_acdc_file_pairs(split)
    for pairs in acdc_pairs.values():
        all_pairs.extend(pairs)
    
    return all_pairs


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_name: str, checkpoint_path: str, device: str = 'cuda'):
    """Load segmentation model from checkpoint."""
    try:
        from mmseg.apis import init_model
        from mmseg.utils import register_all_modules
        import mmseg
        
        # Register all modules
        register_all_modules()
        
        # Get mmseg config directory
        mmseg_path = Path(mmseg.__file__).parent
        mim_configs = mmseg_path / '.mim' / 'configs'
        
        # Map model names to config files
        config_map = {
            'deeplabv3plus_r50': ('deeplabv3plus', 'deeplabv3plus_r50-d8_4xb2-80k_cityscapes-512x1024.py'),
            'pspnet_r50': ('pspnet', 'pspnet_r50-d8_4xb2-80k_cityscapes-512x1024.py'), 
            'segformer_mit-b5': ('segformer', 'segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py'),
        }
        
        if model_name not in config_map:
            raise ValueError(f"Unknown model: {model_name}")
        
        subdir, config_name = config_map[model_name]
        config_path = mim_configs / subdir / config_name
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Initialize model with checkpoint
        model = init_model(str(config_path), checkpoint_path, device=device)
        model.eval()
        
        return model
        
    except ImportError as e:
        print(f"Error: mmseg not available: {e}")
        raise
    except Exception as e:
        print(f"Error loading model with mmseg: {e}")
        raise


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
            
            # All labels (ACDC and Cityscapes) are in Cityscapes labelID format
            # Convert to trainID format (0-18)
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

def get_checkpoint_path(source_dataset: str, model: str, variant: str = '', strategy: str = None) -> Optional[Path]:
    """
    Get checkpoint path for a source dataset and model.
    
    Args:
        source_dataset: Source dataset name (e.g., 'BDD10k')
        model: Base model name (e.g., 'deeplabv3plus_r50')
        variant: Model variant suffix (e.g., '' or '_clear_day')
        strategy: Augmentation strategy name (e.g., 'gen_cyclediffusion')
    
    Returns:
        Path to checkpoint or None if not found
    """
    if strategy:
        # Strategy checkpoint paths
        # Generative strategies: WEIGHTS/{strategy}/{dataset}/{model}_ratio0p50/
        # Standard strategies: WEIGHTS/{strategy}/{dataset}/{model}/
        candidates_dirs = [
            WEIGHTS_ROOT / strategy / source_dataset.lower() / f"{model}_ratio0p50",
            WEIGHTS_ROOT / strategy / source_dataset.lower() / model,
        ]
        
        for checkpoint_dir in candidates_dirs:
            candidates = [
                checkpoint_dir / 'iter_80000.pth',
                checkpoint_dir / 'latest.pth',
            ]
            for f in checkpoint_dir.glob('best_*.pth'):
                candidates.insert(0, f)
            for path in candidates:
                if path.exists():
                    return path
        return None
    else:
        # Baseline checkpoint paths
        # Build model directory name (e.g., 'deeplabv3plus_r50' or 'deeplabv3plus_r50_clear_day')
        model_dir_name = model + variant
        checkpoint_dir = WEIGHTS_ROOT / 'baseline' / source_dataset.lower() / model_dir_name
        
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


def run_evaluation(source_dataset: str, model: str, checkpoint_path: str = None, device: str = 'cuda', variant: str = '', strategy: str = None):
    """Run domain adaptation evaluation for a single configuration."""
    
    # Build full model name (includes strategy if specified)
    if strategy:
        full_model_name = f"{strategy}/{model}"
        display_name = f"{strategy} + {model}"
    else:
        full_model_name = model + variant
        display_name = full_model_name
    
    print(f"\n{'='*70}")
    print(f"Domain Adaptation Evaluation")
    print(f"  Source Dataset: {source_dataset}")
    print(f"  Model: {display_name}")
    if strategy:
        print(f"  Strategy: {strategy}")
    else:
        print(f"  Variant: {variant if variant else 'full_dataset (default)'}")
    print(f"{'='*70}")
    
    # Get checkpoint path if not provided
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path(source_dataset, model, variant, strategy)
        if checkpoint_path is None:
            print(f"ERROR: No checkpoint found for {source_dataset}/{display_name}")
            return None
    
    checkpoint_path = Path(checkpoint_path)
    print(f"  Checkpoint: {checkpoint_path}")
    
    # Load evaluation data: Cityscapes (clear_day) + ACDC (adverse conditions)
    print("\nLoading evaluation data...")
    print("  Cityscapes = clear_day condition")
    print("  ACDC = adverse weather conditions (foggy, night, rainy, snowy)")
    
    # Get all domain pairs from test split
    domain_pairs = get_all_domain_file_pairs('test')
    
    # Print summary
    print("\nDataset summary:")
    total_images = 0
    for domain, pairs in domain_pairs.items():
        print(f"  {domain}: {len(pairs)} images")
        total_images += len(pairs)
    print(f"Total evaluation images: {total_images}")
    
    # Load model (use base model name for config)
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
        'model_full': full_model_name,
        'strategy': strategy if strategy else None,
        'variant': variant if variant else 'full_dataset',
        'checkpoint': str(checkpoint_path),
        'target_datasets': 'ACDC + Cityscapes',
        'domains': {
            'clear_day': 'Cityscapes (6 cities)',
            'foggy': 'ACDC foggy',
            'night': 'ACDC night',
            'rainy': 'ACDC rainy',
            'snowy': 'ACDC snowy'
        },
        'label_format': 'Cityscapes labelID (0-33) converted to trainID (0-18)'
    }
    
    # Save results - use strategy path if specified, otherwise use full model name
    if strategy:
        # Check if checkpoint has _ratio0p50 suffix
        if '_ratio0p50' in str(checkpoint_path):
            output_dir = OUTPUT_ROOT / strategy / source_dataset.lower() / f"{model}_ratio0p50"
        else:
            output_dir = OUTPUT_ROOT / strategy / source_dataset.lower() / model
    else:
        output_dir = OUTPUT_ROOT / source_dataset.lower() / full_model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = output_dir / 'domain_adaptation_evaluation.json'
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


def run_all_evaluations(device: str = 'cuda', include_variants: bool = True):
    """Run evaluation for all source dataset / model combinations."""
    
    all_results = {}
    
    # Determine which variants to evaluate
    variants_to_eval = MODEL_VARIANTS if include_variants else ['']
    
    for source_dataset in SOURCE_DATASETS:
        for model in BASE_MODELS:
            for variant in variants_to_eval:
                full_model = model + variant
                key = f"{source_dataset}_{full_model}"
                
                results = run_evaluation(source_dataset, model, device=device, variant=variant)
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
    parser.add_argument('--model', type=str, choices=BASE_MODELS,
                        help='Base model architecture')
    parser.add_argument('--variant', type=str, default='', choices=['', '_clear_day'],
                        help='Model variant: "" for full dataset, "_clear_day" for clear_day only')
    parser.add_argument('--strategy', type=str, default=None,
                        help='Training strategy (e.g., baseline, gen_cycleGAN). If provided, outputs to ablation directory.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (auto-detected if not provided)')
    parser.add_argument('--all', action='store_true',
                        help='Evaluate all source dataset / model combinations')
    parser.add_argument('--include-variants', action='store_true', default=True,
                        help='Include _clear_day variants when using --all')
    parser.add_argument('--no-variants', action='store_true',
                        help='Exclude _clear_day variants when using --all')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda or cpu)')
    parser.add_argument('--list-checkpoints', action='store_true',
                        help='List available checkpoints')
    
    args = parser.parse_args()
    
    if args.list_checkpoints:
        print("Available checkpoints:")
        for source in SOURCE_DATASETS:
            for model in BASE_MODELS:
                for variant in MODEL_VARIANTS:
                    full_model = model + variant
                    ckpt = get_checkpoint_path(source, model, variant)
                    status = "✓" if ckpt else "✗"
                    print(f"  {status} {source}/{full_model}: {ckpt if ckpt else 'NOT FOUND'}")
        return
    
    if args.all:
        include_variants = not args.no_variants
        run_all_evaluations(device=args.device, include_variants=include_variants)
    elif args.source_dataset and args.model:
        run_evaluation(args.source_dataset, args.model, args.checkpoint, args.device, args.variant, args.strategy)
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python evaluate_domain_adaptation.py --source-dataset BDD10k --model deeplabv3plus_r50")
        print("  python evaluate_domain_adaptation.py --source-dataset BDD10k --model deeplabv3plus_r50 --variant _clear_day")
        print("  python evaluate_domain_adaptation.py --source-dataset BDD10k --model pspnet_r50 --strategy gen_cycleGAN --checkpoint /path/to/weights.pth")
        print("  python evaluate_domain_adaptation.py --all")
        print("  python evaluate_domain_adaptation.py --all --no-variants  # Skip _clear_day variants")


if __name__ == '__main__':
    main()
