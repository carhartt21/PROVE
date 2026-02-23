#!/usr/bin/env python3
"""
t-SNE Domain Gap Visualization Tool

Visualizes domain gap reduction between clear-day and adverse weather conditions
using t-SNE embeddings of semantic segmentation features.

Features:
- Feature extraction from DeepLabv3+, PSPNet, and SegFormer decoder layers
- Pixel subsampling for computational efficiency (50k-100k pixels)
- 3-panel visualization comparing baseline vs augmented models
- Silhouette score quantification for domain gap measurement

Usage:
    python tsne_domain_gap.py \\
        --checkpoint-baseline /path/to/baseline.pth \\
        --checkpoint-augmented /path/to/augmented.pth \\
        --data-root ${AWARE_DATA_ROOT}/FINAL_SPLITS \\
        --model-type deeplabv3plus \\
        --output ./tsne_plots
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


# ============================================================================
# Constants
# ============================================================================

# Weather conditions and their colors (7 domains)
WEATHER_COLORS = {
    'clear_day': '#1f77b4',   # blue
    'rainy': '#ff7f0e',       # orange
    'snowy': '#2ca02c',       # green
    'fog': '#d62728',         # red
    'night': '#9467bd',       # purple
    'cloudy': '#8c564b',      # brown
    'dawn_dusk': '#e377c2'    # pink
}

# t-SNE parameters (literature standard)
# Note: 'max_iter' is used for sklearn >= 1.4, replaces deprecated 'n_iter'
TSNE_PARAMS = {
    'perplexity': 30,         # Optimal for image features [van der Maaten 2008]
    'max_iter': 1000,         # Convergence guarantee (was 'n_iter' in sklearn < 1.4)
    'learning_rate': 200,     # Stable optimization
    'random_state': 42,       # Reproducibility
    'n_components': 2,        # 2D embedding
    'metric': 'euclidean',
    'init': 'pca',
}

# Feature extraction layer names by model type
# Support both MMSeg (decode_head.*) and torchvision (classifier.*) naming
FEATURE_LAYERS = {
    'deeplabv3plus': [
        # MMSeg naming
        'decode_head.bottleneck', 
        'decode_head.aspp',
        # Torchvision naming
        'classifier.0',  # ASPP module
        'classifier.4',  # final conv before upsampling
        'classifier',
        'backbone.layer4',  # Backbone final layer (fallback)
    ],
    'pspnet': ['decode_head.bottleneck'],
    'segformer': ['decode_head.linear_fuse'],
}


# ============================================================================
# Feature Extraction Hooks
# ============================================================================

class FeatureExtractor:
    """
    Hook-based feature extractor for segmentation models.
    
    Extracts intermediate features from decoder layers for t-SNE visualization.
    """
    
    def __init__(self, model: nn.Module, model_type: str = 'deeplabv3plus'):
        self.model = model
        self.model_type = model_type
        self.features = {}
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on target layers."""
        target_layers = FEATURE_LAYERS.get(self.model_type, [])
        registered = False
        
        # First, print all available layer names for debugging
        all_layers = [name for name, _ in self.model.named_modules()]
        print(f"[FeatureExtractor] Model has {len(all_layers)} layers")
        print(f"[FeatureExtractor] Target layers: {target_layers}")
        
        for name, module in self.model.named_modules():
            for target in target_layers:
                # Exact match or partial match
                if target == name or (target in name and len(name) < len(target) + 20):
                    hook = module.register_forward_hook(
                        self._create_hook(name)
                    )
                    self.hooks.append(hook)
                    print(f"[FeatureExtractor] Registered hook on: {name}")
                    registered = True
                    break
        
        if not registered:
            print(f"[WARNING] No hooks registered! Available layers with 'layer4' or 'classifier':")
            for name, _ in self.model.named_modules():
                if 'layer4' in name or 'classifier' in name or 'decode' in name:
                    print(f"  - {name}")
    
    def _create_hook(self, name: str):
        """Create a forward hook that stores features."""
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.features[name] = output.detach()
            elif isinstance(output, tuple):
                self.features[name] = output[0].detach()
        return hook
    
    def extract(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from images.
        
        Args:
            images: Input images (B, C, H, W)
            
        Returns:
            Dictionary of layer_name -> features (B, C, H', W')
        """
        self.features = {}
        with torch.no_grad():
            _ = self.model(images)
        return self.features.copy()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def flatten_and_subsample(
    features: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    predictions: Optional[torch.Tensor] = None,
    num_samples: int = 75000,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Flatten spatial dimensions and subsample pixels.
    
    Args:
        features: Feature tensor (B, C, H, W)
        labels: Ground truth labels (B, H, W) - optional
        predictions: Model predictions (B, H, W) - optional
        num_samples: Number of pixels to sample
        
    Returns:
        Tuple of (sampled_features, sampled_labels, sampled_correctness)
    """
    B, C, H, W = features.shape
    
    # Flatten: (B, C, H, W) -> (B*H*W, C)
    features_flat = features.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
    
    total_pixels = features_flat.shape[0]
    
    if total_pixels <= num_samples:
        indices = np.arange(total_pixels)
    else:
        indices = np.random.choice(total_pixels, num_samples, replace=False)
    
    sampled_features = features_flat[indices]
    
    sampled_labels = None
    if labels is not None:
        labels_flat = labels.reshape(-1).cpu().numpy()
        sampled_labels = labels_flat[indices]
    
    sampled_correctness = None
    if predictions is not None and labels is not None:
        preds_flat = predictions.reshape(-1).cpu().numpy()
        labels_flat = labels.reshape(-1).cpu().numpy()
        correctness = (preds_flat == labels_flat).astype(np.int32)
        sampled_correctness = correctness[indices]
    
    return sampled_features, sampled_labels, sampled_correctness


# ============================================================================
# Dataset Utilities
# ============================================================================

class WeatherDomainDataset:
    """
    Simple dataset loader for weather domain images.
    
    Loads images from domain-specific subdirectories.
    Supports multiple directory structures:
    1. Cityscapes style: {data_root}/leftImg8bit/{split}/{domain}/
    2. PROVE style: {data_root}/{split}/images/{dataset}/{domain}/
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'val',
        domains: List[str] = None,
        max_images_per_domain: int = 100,
        img_size: Tuple[int, int] = (512, 512),
        dataset_name: str = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.domains = domains or list(WEATHER_COLORS.keys())
        self.max_images_per_domain = max_images_per_domain
        self.img_size = img_size
        self.dataset_name = dataset_name or "ACDC"
        
        self.samples = self._collect_samples()
    
    def _collect_samples(self) -> List[Dict[str, Any]]:
        """Collect image samples from each domain."""
        samples = []
        
        # Try different directory structures
        possible_img_bases = [
            # Cityscapes style
            self.data_root / 'leftImg8bit' / self.split,
            # PROVE style
            self.data_root / self.split / 'images' / self.dataset_name,
            self.data_root / 'test' / 'images' / self.dataset_name,
            self.data_root / 'val' / 'images' / self.dataset_name,
            # Direct domain folders
            self.data_root,
        ]
        
        possible_ann_bases = [
            # Cityscapes style
            self.data_root / 'gtFine' / self.split,
            # PROVE style
            self.data_root / self.split / 'labels_adjusted' / self.dataset_name,
            self.data_root / 'test' / 'labels_adjusted' / self.dataset_name,
            self.data_root / 'val' / 'labels_adjusted' / self.dataset_name,
            # Direct domain folders
            self.data_root.parent / 'labels_adjusted' / self.dataset_name if 'images' in str(self.data_root) else None,
        ]
        
        def safe_exists(path):
            """Check if path exists, handling permission errors."""
            try:
                return path.exists()
            except (PermissionError, OSError):
                return False
        
        img_base = None
        ann_base = None
        
        for base in possible_img_bases:
            if base and safe_exists(base):
                # Check if domains exist
                test_domain = self.domains[0] if self.domains else 'foggy'
                if safe_exists(base / test_domain) or safe_exists(base / 'foggy'):
                    img_base = base
                    print(f"[WeatherDomainDataset] Found image base: {img_base}")
                    break
        
        for base in possible_ann_bases:
            if base and safe_exists(base):
                ann_base = base
                print(f"[WeatherDomainDataset] Found annotation base: {ann_base}")
                break
        
        if img_base is None:
            print(f"[WARNING] No valid image directory found in {self.data_root}")
            print(f"[DEBUG] Tried: {[str(p) for p in possible_img_bases if p]}")
            return samples
        
        for domain in self.domains:
            domain_img_dir = img_base / domain
            domain_ann_dir = ann_base / domain if ann_base else None
            
            if not safe_exists(domain_img_dir):
                print(f"[WARNING] Domain directory not found: {domain_img_dir}")
                continue
            
            # Find images
            image_files = list(domain_img_dir.glob('**/*.png'))
            image_files += list(domain_img_dir.glob('**/*.jpg'))
            
            # Limit per domain
            image_files = image_files[:self.max_images_per_domain]
            
            for img_path in image_files:
                ann_path = None
                if domain_ann_dir and domain_ann_dir.exists():
                    # Try to find corresponding annotation
                    rel_path = img_path.relative_to(domain_img_dir)
                    ann_path = domain_ann_dir / str(rel_path).replace(
                        '_leftImg8bit', '_gtFine_labelTrainIds'
                    ).replace('.jpg', '.png')
                    
                    if not ann_path.exists():
                        # Try alternative naming
                        ann_path = domain_ann_dir / str(rel_path).replace('.jpg', '.png')
                
                samples.append({
                    'img_path': str(img_path),
                    'ann_path': str(ann_path) if ann_path and ann_path.exists() else None,
                    'domain': domain,
                })
        
        print(f"[WeatherDomainDataset] Collected {len(samples)} samples from {len(self.domains)} domains")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def load_batch(
        self,
        indices: List[int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[str]]:
        """
        Load a batch of images and annotations.
        
        Returns:
            Tuple of (images, annotations, domains)
        """
        images = []
        annotations = []
        domains = []
        
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        
        for idx in indices:
            sample = self.samples[idx]
            
            # Load and preprocess image
            img = Image.open(sample['img_path']).convert('RGB')
            img = img.resize(self.img_size, Image.BILINEAR)
            img = np.array(img, dtype=np.float32)
            img = (img - mean) / std
            img = img.transpose(2, 0, 1)  # HWC -> CHW
            images.append(img)
            
            # Load annotation if available
            if sample['ann_path'] and os.path.exists(sample['ann_path']):
                ann = Image.open(sample['ann_path'])
                ann = ann.resize(self.img_size, Image.NEAREST)
                ann = np.array(ann, dtype=np.int64)
                annotations.append(ann)
            else:
                annotations.append(np.zeros(self.img_size, dtype=np.int64))
            
            domains.append(sample['domain'])
        
        images = torch.from_numpy(np.stack(images)).float().to(device)
        annotations = torch.from_numpy(np.stack(annotations)).long().to(device)
        
        return images, annotations, domains


# ============================================================================
# t-SNE Computation and Visualization
# ============================================================================

def compute_tsne(features: np.ndarray, **kwargs) -> np.ndarray:
    """
    Compute t-SNE embedding.
    
    Args:
        features: Feature array (N, D)
        **kwargs: Override default TSNE parameters
        
    Returns:
        2D embedding (N, 2)
    """
    params = TSNE_PARAMS.copy()
    params.update(kwargs)
    
    print(f"[t-SNE] Computing embedding for {features.shape[0]} samples...")
    print(f"[t-SNE] Parameters: perplexity={params['perplexity']}, max_iter={params['max_iter']}")
    
    tsne = TSNE(**params)
    embedding = tsne.fit_transform(features)
    
    print(f"[t-SNE] Complete. Final KL divergence: {tsne.kl_divergence_:.4f}")
    
    return embedding


def plot_tsne_panel(
    ax: plt.Axes,
    embedding: np.ndarray,
    domains: List[str],
    correctness: Optional[np.ndarray] = None,
    title: str = "",
    silhouette: Optional[float] = None,
):
    """
    Plot a single t-SNE panel.
    
    Args:
        ax: Matplotlib axes
        embedding: 2D embedding (N, 2)
        domains: List of domain labels for each point
        correctness: Optional array of 0/1 for incorrect/correct predictions
        title: Panel title
        silhouette: Optional silhouette score to display
    """
    unique_domains = list(WEATHER_COLORS.keys())
    
    for domain in unique_domains:
        mask = np.array([d == domain for d in domains])
        if not mask.any():
            continue
        
        color = WEATHER_COLORS[domain]
        
        if correctness is not None:
            # Correct predictions: circles, incorrect: x markers
            correct_mask = mask & (correctness == 1)
            incorrect_mask = mask & (correctness == 0)
            
            if correct_mask.any():
                ax.scatter(
                    embedding[correct_mask, 0],
                    embedding[correct_mask, 1],
                    c=color, marker='o', s=5, alpha=0.6, label=f'{domain} (correct)'
                )
            if incorrect_mask.any():
                ax.scatter(
                    embedding[incorrect_mask, 0],
                    embedding[incorrect_mask, 1],
                    c=color, marker='x', s=10, alpha=0.6
                )
        else:
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=color, marker='o', s=5, alpha=0.6, label=domain
            )
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    if silhouette is not None:
        ax.text(
            0.05, 0.95, f'Silhouette: {silhouette:.3f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )


def create_visualization(
    baseline_embedding: np.ndarray,
    baseline_domains: List[str],
    augmented_embedding: np.ndarray,
    augmented_domains: List[str],
    baseline_silhouette: float,
    augmented_silhouette: float,
    output_path: str,
    baseline_correctness: Optional[np.ndarray] = None,
    augmented_correctness: Optional[np.ndarray] = None,
):
    """
    Create 3-panel t-SNE visualization.
    
    Panel 1: Baseline model (weather clustering = domain gap)
    Panel 2: Augmented model (weather overlap = domain invariance)
    Panel 3: Silhouette score comparison bar chart
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: Baseline
    plot_tsne_panel(
        axes[0], baseline_embedding, baseline_domains,
        baseline_correctness,
        title="Baseline (Clear-Day Training)",
        silhouette=baseline_silhouette
    )
    
    # Panel 2: Augmented
    plot_tsne_panel(
        axes[1], augmented_embedding, augmented_domains,
        augmented_correctness,
        title="Augmented (p_aug=0.5)",
        silhouette=augmented_silhouette
    )
    
    # Panel 3: Silhouette comparison
    ax3 = axes[2]
    models = ['Baseline', 'Augmented']
    scores = [baseline_silhouette, augmented_silhouette]
    colors = ['#d62728', '#2ca02c']  # red for baseline, green for augmented
    
    bars = ax3.bar(models, scores, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Silhouette Score', fontsize=11)
    ax3.set_title('Domain Gap Quantification', fontsize=12, fontweight='bold')
    ax3.set_ylim(-0.1, 1.0)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax3.annotate(
            f'{score:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
    
    # Add interpretation text
    gap_reduction = baseline_silhouette - augmented_silhouette
    interpretation = (
        f"Domain gap reduction: {gap_reduction:.3f}\n"
        f"Lower silhouette = better invariance"
    )
    ax3.text(
        0.5, -0.15, interpretation,
        transform=ax3.transAxes, fontsize=9,
        ha='center', va='top', style='italic'
    )
    
    # Create legend
    legend_patches = [
        mpatches.Patch(color=color, label=domain)
        for domain, color in WEATHER_COLORS.items()
    ]
    fig.legend(
        handles=legend_patches,
        loc='lower center',
        ncol=7,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Visualization] Saved to: {output_path}")


# ============================================================================
# Model Loading Utilities
# ============================================================================

def load_segmentation_model(
    checkpoint_path: str,
    model_type: str = 'deeplabv3plus',
    device: torch.device = None,
) -> nn.Module:
    """
    Load a segmentation model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_type: Model architecture type
        device: Target device
        
    Returns:
        Loaded model in eval mode
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Try MMSegmentation loading first
        from mmseg.apis import init_model
        from mmengine.config import Config
        
        # Infer config from checkpoint if available
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'meta' in checkpoint and 'config' in checkpoint['meta']:
            # Config embedded in checkpoint
            cfg = Config.fromstring(checkpoint['meta']['config'], file_format='.py')
            model = init_model(cfg, checkpoint_path, device=str(device))
        else:
            # Use default config based on model type
            print(f"[WARNING] No config in checkpoint, using default for {model_type}")
            model = _load_default_model(model_type, checkpoint_path, device)
        
    except ImportError:
        print("[WARNING] MMSegmentation not available, using torch.load directly")
        model = _load_default_model(model_type, checkpoint_path, device)
    
    model.eval()
    return model


def _load_default_model(
    model_type: str,
    checkpoint_path: str,
    device: torch.device,
) -> nn.Module:
    """Load model with default architecture."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Build default model architecture
    if model_type == 'deeplabv3plus':
        try:
            from torchvision.models.segmentation import deeplabv3_resnet50
            # Use weights=None to avoid downloading pretrained weights
            # We will load from the checkpoint instead
            model = deeplabv3_resnet50(weights=None, num_classes=19)
        except:
            raise RuntimeError("Cannot load default DeepLabv3+ model")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load weights (with key adaptation if needed)
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"[WARNING] Partial weight loading: {e}")
    
    return model.to(device)


# ============================================================================
# Main Pipeline
# ============================================================================

def extract_features_from_model(
    model: nn.Module,
    dataset: WeatherDomainDataset,
    model_type: str,
    device: torch.device,
    batch_size: int = 4,
    num_samples: int = 75000,
) -> Tuple[np.ndarray, List[str], Optional[np.ndarray]]:
    """
    Extract features from all dataset images.
    
    Returns:
        Tuple of (features, domains, correctness)
    """
    extractor = FeatureExtractor(model, model_type)
    
    all_features = []
    all_domains = []
    all_correctness = []
    
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        indices = list(range(start_idx, end_idx))
        
        images, annotations, domains = dataset.load_batch(indices, device)
        
        # Extract features
        features_dict = extractor.extract(images)
        
        if not features_dict:
            print(f"[WARNING] No features extracted for batch {batch_idx}")
            continue
        
        # Use first available feature layer
        layer_name = list(features_dict.keys())[0]
        features = features_dict[layer_name]
        
        # Get predictions
        with torch.no_grad():
            output = model(images)
            if isinstance(output, dict):
                logits = output.get('out', output.get('seg_logits', None))
                if logits is None:
                    logits = list(output.values())[0]
            else:
                logits = output
            
            # Resize to feature map size if needed
            if logits.shape[-2:] != features.shape[-2:]:
                logits = nn.functional.interpolate(
                    logits, size=features.shape[-2:], mode='bilinear', align_corners=False
                )
            predictions = logits.argmax(dim=1)
            
            # Resize annotations to match
            if annotations.shape[-2:] != features.shape[-2:]:
                annotations = nn.functional.interpolate(
                    annotations.unsqueeze(1).float(),
                    size=features.shape[-2:],
                    mode='nearest'
                ).squeeze(1).long()
        
        # Flatten and subsample
        samples_per_batch = num_samples // num_batches
        feat_flat, _, correct_flat = flatten_and_subsample(
            features, annotations, predictions, samples_per_batch
        )
        
        all_features.append(feat_flat)
        all_correctness.append(correct_flat)
        
        # Expand domains to match sampled pixels
        batch_domains = []
        for i, domain in enumerate(domains):
            n_pixels = samples_per_batch // len(domains)
            batch_domains.extend([domain] * n_pixels)
        all_domains.extend(batch_domains[:len(feat_flat)])
        
        if (batch_idx + 1) % 10 == 0:
            print(f"[Feature Extraction] Processed {batch_idx + 1}/{num_batches} batches")
    
    extractor.remove_hooks()
    
    # Concatenate all
    features = np.concatenate(all_features, axis=0)[:num_samples]
    domains = all_domains[:num_samples]
    correctness = np.concatenate(all_correctness, axis=0)[:num_samples] if all_correctness[0] is not None else None
    
    print(f"[Feature Extraction] Final feature shape: {features.shape}")
    
    return features, domains, correctness


def run_domain_gap_analysis(
    checkpoint_baseline: str,
    checkpoint_augmented: str,
    data_root: str,
    model_type: str = 'deeplabv3plus',
    num_samples: int = 75000,
    output_dir: str = './tsne_output',
    split: str = 'val',
    max_images_per_domain: int = 50,
    dataset_name: str = 'ACDC',
):
    """
    Run complete domain gap analysis.
    
    Args:
        checkpoint_baseline: Path to baseline model checkpoint
        checkpoint_augmented: Path to augmented model checkpoint
        data_root: Path to dataset root
        model_type: Model architecture type
        num_samples: Number of pixels to sample for t-SNE
        output_dir: Output directory for plots
        split: Dataset split to use (train/val)
        max_images_per_domain: Max images per weather domain
        dataset_name: Name of the dataset (e.g., ACDC, BDD10k)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using: {device}")
    
    # Create dataset
    dataset = WeatherDomainDataset(
        data_root=data_root,
        split=split,
        max_images_per_domain=max_images_per_domain,
        dataset_name=dataset_name,
    )
    
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found in {data_root}")
    
    # Process baseline model
    print("\n" + "=" * 60)
    print("Processing BASELINE model")
    print("=" * 60)
    
    baseline_model = load_segmentation_model(checkpoint_baseline, model_type, device)
    baseline_features, baseline_domains, baseline_correctness = extract_features_from_model(
        baseline_model, dataset, model_type, device, num_samples=num_samples
    )
    del baseline_model
    torch.cuda.empty_cache()
    
    # Process augmented model
    print("\n" + "=" * 60)
    print("Processing AUGMENTED model")
    print("=" * 60)
    
    augmented_model = load_segmentation_model(checkpoint_augmented, model_type, device)
    augmented_features, augmented_domains, augmented_correctness = extract_features_from_model(
        augmented_model, dataset, model_type, device, num_samples=num_samples
    )
    del augmented_model
    torch.cuda.empty_cache()
    
    # Compute t-SNE embeddings
    print("\n" + "=" * 60)
    print("Computing t-SNE embeddings")
    print("=" * 60)
    
    baseline_embedding = compute_tsne(baseline_features)
    augmented_embedding = compute_tsne(augmented_features)
    
    # Compute silhouette scores
    print("\n" + "=" * 60)
    print("Computing silhouette scores")
    print("=" * 60)
    
    # Convert domains to numeric labels for silhouette score
    domain_to_idx = {d: i for i, d in enumerate(WEATHER_COLORS.keys())}
    baseline_labels = np.array([domain_to_idx.get(d, 0) for d in baseline_domains])
    augmented_labels = np.array([domain_to_idx.get(d, 0) for d in augmented_domains])
    
    baseline_silhouette = silhouette_score(baseline_embedding, baseline_labels)
    augmented_silhouette = silhouette_score(augmented_embedding, augmented_labels)
    
    print(f"Baseline silhouette score: {baseline_silhouette:.4f}")
    print(f"Augmented silhouette score: {augmented_silhouette:.4f}")
    print(f"Domain gap reduction: {baseline_silhouette - augmented_silhouette:.4f}")
    
    # Create visualization
    print("\n" + "=" * 60)
    print("Creating visualization")
    print("=" * 60)
    
    output_path = os.path.join(output_dir, f'tsne_domain_gap_{model_type}.png')
    
    create_visualization(
        baseline_embedding=baseline_embedding,
        baseline_domains=baseline_domains,
        augmented_embedding=augmented_embedding,
        augmented_domains=augmented_domains,
        baseline_silhouette=baseline_silhouette,
        augmented_silhouette=augmented_silhouette,
        output_path=output_path,
        baseline_correctness=baseline_correctness,
        augmented_correctness=augmented_correctness,
    )
    
    # Save numerical results
    results = {
        'baseline_silhouette': float(baseline_silhouette),
        'augmented_silhouette': float(augmented_silhouette),
        'domain_gap_reduction': float(baseline_silhouette - augmented_silhouette),
        'num_samples': num_samples,
        'model_type': model_type,
    }
    
    import json
    results_path = os.path.join(output_dir, f'tsne_results_{model_type}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[Complete] Results saved to {output_dir}")
    
    return results


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='t-SNE Domain Gap Visualization for Semantic Segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with two checkpoints
  python tsne_domain_gap.py \\
      --checkpoint-baseline ./checkpoints/baseline.pth \\
      --checkpoint-augmented ./checkpoints/augmented.pth \\
      --data-root ${AWARE_DATA_ROOT}/FINAL_SPLITS/ACDC

  # With custom parameters
  python tsne_domain_gap.py \\
      --checkpoint-baseline ./baseline.pth \\
      --checkpoint-augmented ./augmented.pth \\
      --data-root ./data \\
      --model-type segformer \\
      --num-samples 100000 \\
      --output ./visualizations
        """
    )
    
    parser.add_argument('--checkpoint-baseline', type=str, required=True,
                       help='Path to baseline model checkpoint')
    parser.add_argument('--checkpoint-augmented', type=str, required=True,
                       help='Path to augmented model checkpoint')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Path to dataset root (e.g., FINAL_SPLITS/ACDC)')
    parser.add_argument('--model-type', type=str, default='deeplabv3plus',
                       choices=['deeplabv3plus', 'pspnet', 'segformer'],
                       help='Model architecture type')
    parser.add_argument('--num-samples', type=int, default=75000,
                       help='Number of pixels to sample for t-SNE (default: 75000)')
    parser.add_argument('--output', type=str, default='./tsne_output',
                       help='Output directory for plots')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--max-images-per-domain', type=int, default=50,
                       help='Maximum images per weather domain')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--dataset', type=str, default='ACDC',
                       help='Dataset name (e.g., ACDC, BDD10k, IDD-AW)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set GPU
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    
    # Run analysis
    run_domain_gap_analysis(
        checkpoint_baseline=args.checkpoint_baseline,
        checkpoint_augmented=args.checkpoint_augmented,
        data_root=args.data_root,
        model_type=args.model_type,
        num_samples=args.num_samples,
        output_dir=args.output,
        split=args.split,
        max_images_per_domain=args.max_images_per_domain,
        dataset_name=args.dataset,
    )


if __name__ == '__main__':
    main()
