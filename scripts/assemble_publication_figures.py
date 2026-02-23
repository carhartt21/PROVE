#!/usr/bin/env python3
"""
Assemble Publication-Quality Figures for Generated Images & Segmentation Results

Three figure types:
    1. gen-gallery:    Grid showing generated images from each strategy × domain
    2. seg-comparison: Segmentation predictions across strategies for test images
    3. domain-strips:  Per-domain comparison of segmentation quality

Usage:
    # Generate all figures
    python scripts/assemble_publication_figures.py --figure all

    # Generate specific figure
    python scripts/assemble_publication_figures.py --figure gen-gallery
    python scripts/assemble_publication_figures.py --figure seg-comparison
    python scripts/assemble_publication_figures.py --figure domain-strips

    # Options
    --dataset BDD10k              # Only one dataset
    --output-dir /path/to/dir     # Custom output directory
    --group-by-family             # Group strategies by family (GAN, diffusion, etc.)
    --dpi 300                     # Output DPI (default: 300)
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict, OrderedDict
import json
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ============================================================================
# CONFIGURATION
# ============================================================================

SAMPLE_ROOT = Path('${AWARE_DATA_ROOT}/SAMPLE_EXTRACTION')
IEEE_FIGURES = Path('${HOME}/repositories/-IEEE-Access-01-26-Data-Augmentation/figures/exploration/qualitative')

DATASETS = ['BDD10k', 'IDD-AW', 'MapillaryVistas', 'OUTSIDE15k']

WEATHER_DOMAINS = ['cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy']
DOMAIN_LABELS = {
    'cloudy': 'Cloudy', 'dawn_dusk': 'Dawn/Dusk', 'foggy': 'Foggy',
    'night': 'Night', 'rainy': 'Rainy', 'snowy': 'Snowy', 'clear_day': 'Clear Day',
}

# Generative strategies (as they appear in training_samples/generated/)
GEN_STRATEGIES_DIR = [
    'albumentations_weather', 'Attribute_Hallucination', 'augmenters', 'automold',
    'CNetSeg', 'CUT', 'cyclediffusion', 'cycleGAN', 'flux_kontext',
    'Img2Img', 'IP2P', 'LANIT', 'Qwen-Image-Edit', 'stargan_v2',
    'step1x_new', 'SUSTechGAN', 'TSIT', 'UniControl', 'VisualCloze',
    'Weather_Effect_Generator',
]

# Nice display names for strategies
STRATEGY_DISPLAY = {
    'albumentations_weather': 'Albumentations',
    'Attribute_Hallucination': 'Attr. Hallucin.',
    'augmenters': 'Augmenters',
    'automold': 'Automold',
    'CNetSeg': 'CNetSeg',
    'CUT': 'CUT',
    'cyclediffusion': 'CycleDiffusion',
    'cycleGAN': 'CycleGAN',
    'flux_kontext': 'FLUX Kontext',
    'Img2Img': 'Img2Img',
    'IP2P': 'InstructPix2Pix',
    'LANIT': 'LANIT',
    'Qwen-Image-Edit': 'Qwen Edit',
    'stargan_v2': 'StarGAN v2',
    'step1x_new': 'Step1X',
    'SUSTechGAN': 'SUSTechGAN',
    'TSIT': 'TSIT',
    'UniControl': 'UniControl',
    'VisualCloze': 'VisualCloze',
    'Weather_Effect_Generator': 'Weather FX Gen',
}

# Strategy families for grouped visualization
STRATEGY_FAMILIES = OrderedDict([
    ('GAN-based', ['cycleGAN', 'CUT', 'TSIT', 'stargan_v2', 'LANIT', 'SUSTechGAN']),
    ('Diffusion-based', ['cyclediffusion', 'flux_kontext', 'step1x_new', 'IP2P', 'Img2Img', 'CNetSeg', 'UniControl']),
    ('VLM/Editing', ['Qwen-Image-Edit', 'Attribute_Hallucination', 'VisualCloze']),
    ('Heuristic/Classical', ['albumentations_weather', 'augmenters', 'automold', 'Weather_Effect_Generator']),
])

FAMILY_COLORS = {
    'GAN-based': '#e74c3c',
    'Diffusion-based': '#3498db',
    'VLM/Editing': '#2ecc71',
    'Heuristic/Classical': '#f39c12',
}

# Prediction strategies (as they appear in predictions/{stage}/)
PRED_STRATEGIES = [
    'baseline',
    'std_autoaugment', 'std_cutmix', 'std_mixup', 'std_randaugment',
    'gen_albumentations_weather', 'gen_Attribute_Hallucination', 'gen_augmenters', 'gen_automold',
    'gen_CNetSeg', 'gen_CUT', 'gen_cyclediffusion', 'gen_cycleGAN', 'gen_flux_kontext',
    'gen_Img2Img', 'gen_IP2P', 'gen_LANIT', 'gen_Qwen_Image_Edit', 'gen_stargan_v2',
    'gen_step1x_new', 'gen_step1x_v1p2', 'gen_SUSTechGAN', 'gen_TSIT', 'gen_UniControl',
    'gen_VisualCloze', 'gen_Weather_Effect_Generator',
]

PRED_DISPLAY = {
    'baseline': 'Baseline (no aug)',
    'std_autoaugment': 'AutoAugment',
    'std_cutmix': 'CutMix',
    'std_mixup': 'MixUp',
    'std_randaugment': 'RandAugment',
    'gen_albumentations_weather': 'Albumentations',
    'gen_Attribute_Hallucination': 'Attr. Hallucin.',
    'gen_augmenters': 'Augmenters',
    'gen_automold': 'Automold',
    'gen_CNetSeg': 'CNetSeg',
    'gen_CUT': 'CUT',
    'gen_cyclediffusion': 'CycleDiffusion',
    'gen_cycleGAN': 'CycleGAN',
    'gen_flux_kontext': 'FLUX Kontext',
    'gen_Img2Img': 'Img2Img',
    'gen_IP2P': 'InstructPix2Pix',
    'gen_LANIT': 'LANIT',
    'gen_Qwen_Image_Edit': 'Qwen Edit',
    'gen_stargan_v2': 'StarGAN v2',
    'gen_step1x_new': 'Step1X',
    'gen_step1x_v1p2': 'Step1X v1.2',
    'gen_SUSTechGAN': 'SUSTechGAN',
    'gen_TSIT': 'TSIT',
    'gen_UniControl': 'UniControl',
    'gen_VisualCloze': 'VisualCloze',
    'gen_Weather_Effect_Generator': 'Weather FX Gen',
}

PRED_FAMILIES = OrderedDict([
    ('No Augmentation', ['baseline']),
    ('Standard Aug.', ['std_autoaugment', 'std_cutmix', 'std_mixup', 'std_randaugment']),
    ('GAN-based Gen.', ['gen_cycleGAN', 'gen_CUT', 'gen_TSIT', 'gen_stargan_v2', 'gen_LANIT', 'gen_SUSTechGAN']),
    ('Diffusion-based Gen.', ['gen_cyclediffusion', 'gen_flux_kontext', 'gen_step1x_new', 'gen_step1x_v1p2',
                              'gen_IP2P', 'gen_Img2Img', 'gen_CNetSeg', 'gen_UniControl']),
    ('VLM/Editing Gen.', ['gen_Qwen_Image_Edit', 'gen_Attribute_Hallucination', 'gen_VisualCloze']),
    ('Heuristic Gen.', ['gen_albumentations_weather', 'gen_augmenters', 'gen_automold', 'gen_Weather_Effect_Generator']),
])

# Cityscapes class palette for GT labels
CITYSCAPES_PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
    [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
    [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
    [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32],
], dtype=np.uint8)

# MapillaryVistas palette (66 classes) — must match extract_samples_and_predictions.py
MAPILLARY_PALETTE = np.array(
    [CITYSCAPES_PALETTE[i].tolist() if i < 19
     else [(i * 67 + 37) % 256, (i * 113 + 59) % 256, (i * 179 + 83) % 256]
     for i in range(66)],
    dtype=np.uint8
)

# OUTSIDE15k palette (24 classes) — must match extract_samples_and_predictions.py
OUTSIDE15K_PALETTE = np.array(
    list(CITYSCAPES_PALETTE[:19].tolist()) + [
        [100, 100, 100], [165, 42, 42], [0, 170, 30], [140, 140, 140], [0, 0, 0]
    ],
    dtype=np.uint8
)

CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]


def get_palette(num_classes):
    """Get the correct color palette for a given number of classes."""
    if num_classes == 19:
        return CITYSCAPES_PALETTE
    elif num_classes == 66:
        return MAPILLARY_PALETTE
    elif num_classes == 24:
        return OUTSIDE15K_PALETTE
    else:
        # Procedural fallback
        return np.array(
            [[(i * 67 + 37) % 256, (i * 113 + 59) % 256, (i * 179 + 83) % 256]
             for i in range(num_classes)],
            dtype=np.uint8
        )


def get_font(size=12):
    """Try to load a nice font, fallback to default."""
    font_paths = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except:
                pass
    return ImageFont.load_default()


def center_crop_resize(img, size):
    """Center-crop an image to a square, then resize to (size, size).

    This avoids aspect-ratio distortion that plain resize causes.
    """
    w, h = img.size
    crop_dim = min(w, h)
    left = (w - crop_dim) // 2
    top = (h - crop_dim) // 2
    img_cropped = img.crop((left, top, left + crop_dim, top + crop_dim))
    return img_cropped.resize((size, size), Image.LANCZOS)


def compute_error_map(gt_array, pred_array, ignore_label=255):
    """Compute a per-pixel error map between GT and prediction.

    Args:
        gt_array: 2D numpy array of GT class IDs (may need resizing to match pred)
        pred_array: 2D numpy array of predicted class IDs
        ignore_label: Class ID to treat as void/ignore (dark grey)

    Returns:
        RGB numpy array: green=correct, red=incorrect, dark grey=ignore/void
    """
    # Resize GT to match prediction size if needed (nearest-neighbor to preserve class IDs)
    if gt_array.shape != pred_array.shape:
        gt_pil = Image.fromarray(gt_array)
        gt_pil = gt_pil.resize((pred_array.shape[1], pred_array.shape[0]), Image.NEAREST)
        gt_array = np.array(gt_pil)

    h, w = gt_array.shape
    error_map = np.zeros((h, w, 3), dtype=np.uint8)

    # Ignore/void regions (dark grey)
    ignore_mask = (gt_array == ignore_label)
    error_map[ignore_mask] = [80, 80, 80]

    # Correct predictions (green)
    correct_mask = (gt_array == pred_array) & ~ignore_mask
    error_map[correct_mask] = [0, 180, 0]

    # Incorrect predictions (red)
    error_mask = (gt_array != pred_array) & ~ignore_mask
    error_map[error_mask] = [220, 30, 30]

    return error_map


def load_gt_class_ids(label_path):
    """Load GT label as a 2D array of class IDs, handling both L and P mode images."""
    gt_img = Image.open(label_path)
    if gt_img.mode == 'P':
        # Palette-indexed: convert to array gives class IDs directly
        return np.array(gt_img)
    elif gt_img.mode == 'L':
        return np.array(gt_img)
    else:
        # RGB - shouldn't happen for GT but handle gracefully
        return np.array(gt_img.convert('L'))


def colorize_label(label_array, num_classes=19):
    """Colorize a single-channel label mask using the appropriate palette.

    Uses the same palettes as extract_samples_and_predictions.py to ensure
    GT and prediction colors match.
    """
    h, w = label_array.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    palette = get_palette(num_classes)
    for c in range(min(num_classes, len(palette))):
        colored[label_array == c] = palette[c]
    return colored


# ============================================================================
# FIGURE A: Generated Image Gallery
# ============================================================================

def build_gen_gallery(dataset, output_dir, group_by_family=False, dpi=300):
    """
    Create a grid of generated images: rows = strategies, cols = weather domains.
    Original image shown above the grid (centered).
    Images are square (512x512 source).
    """
    train_dir = SAMPLE_ROOT / 'training_samples' / dataset
    originals_dir = train_dir / 'originals'
    generated_dir = train_dir / 'generated'

    if not originals_dir.exists():
        print(f"  WARNING: No originals for {dataset}")
        return

    # Pick the first original image
    orig_images = sorted(originals_dir.iterdir())
    if not orig_images:
        print(f"  WARNING: No original images for {dataset}")
        return

    # Use first image for the main gallery
    orig_path = orig_images[0]
    orig_stem = orig_path.stem  # e.g., "0273a587-00000000"
    print(f"  Using original: {orig_path.name}")

    # Determine strategy order
    if group_by_family:
        strategy_order = []
        family_labels = {}  # strategy -> family
        for family, members in STRATEGY_FAMILIES.items():
            for s in members:
                if (generated_dir / s).exists():
                    strategy_order.append(s)
                    family_labels[s] = family
    else:
        strategy_order = [s for s in GEN_STRATEGIES_DIR if (generated_dir / s).exists()]

    if not strategy_order:
        print(f"  WARNING: No generated images found for {dataset}")
        return

    # Load original image
    orig_img = Image.open(orig_path).convert('RGB')

    # Square thumbnails (source images are 512x512)
    cell_size = 180  # Square cell size for grid
    orig_display_size = 256  # Larger display for original above grid

    # Layout: columns = 6 domains (no original column), rows = strategies
    n_cols = len(WEATHER_DOMAINS)
    n_rows = len(strategy_order)

    label_w = 160  # Left label column width
    header_h = 30  # Column header row height
    orig_area_h = orig_display_size + 40  # Original image area above grid (image + label + padding)
    pad = 2

    # Canvas size
    canvas_w = label_w + n_cols * (cell_size + pad) + pad
    canvas_h = orig_area_h + header_h + n_rows * (cell_size + pad) + pad

    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = get_font(11)
    font_header = get_font(13)
    font_orig = get_font(14)

    # Draw original image centered above grid (center-cropped to square)
    orig_thumb = center_crop_resize(orig_img, orig_display_size)
    orig_x = (canvas_w - orig_display_size) // 2
    orig_y = 5
    canvas.paste(orig_thumb, (orig_x, orig_y))
    # Label below original
    orig_label = f"Original (Clear Day) — {dataset}"
    bbox = draw.textbbox((0, 0), orig_label, font=font_orig)
    tw = bbox[2] - bbox[0]
    draw.text(((canvas_w - tw) // 2, orig_y + orig_display_size + 5), orig_label,
              fill=(0, 0, 0), font=font_orig)

    # Draw column headers for domains
    headers = [DOMAIN_LABELS[d] for d in WEATHER_DOMAINS]
    for col_idx, header in enumerate(headers):
        x = label_w + col_idx * (cell_size + pad) + pad
        bbox = draw.textbbox((0, 0), header, font=font_header)
        tw = bbox[2] - bbox[0]
        draw.text((x + (cell_size - tw) // 2, orig_area_h + 5), header, fill=(0, 0, 0), font=font_header)

    # Fill grid: rows = strategies, cols = domains
    grid_y_start = orig_area_h + header_h
    for row_idx, strategy in enumerate(strategy_order):
        y = grid_y_start + row_idx * (cell_size + pad) + pad

        # Strategy label
        display = STRATEGY_DISPLAY.get(strategy, strategy)
        if group_by_family and strategy in family_labels:
            family = family_labels[strategy]
            color = FAMILY_COLORS.get(family, '#000000')
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            draw.text((5, y + cell_size // 2 - 7), display, fill=(r, g, b), font=font)
        else:
            draw.text((5, y + cell_size // 2 - 7), display, fill=(0, 0, 0), font=font)

        for col_idx, domain in enumerate(WEATHER_DOMAINS):
            x = label_w + col_idx * (cell_size + pad) + pad
            gen_dir = generated_dir / strategy / domain

            # Try multiple extensions
            gen_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = gen_dir / f"{orig_stem}{ext}"
                if candidate.exists():
                    gen_path = candidate
                    break

            if gen_path and gen_path.exists():
                try:
                    gen_img = Image.open(gen_path).convert('RGB')
                    thumb = gen_img.resize((cell_size, cell_size), Image.LANCZOS)
                    canvas.paste(thumb, (x, y))
                except Exception as e:
                    draw.rectangle([x, y, x + cell_size, y + cell_size], fill=(200, 200, 200))
                    draw.text((x + 5, y + cell_size // 2 - 5), 'Error', fill=(255, 0, 0), font=font)
            else:
                draw.rectangle([x, y, x + cell_size, y + cell_size], fill=(240, 240, 240))
                draw.text((x + 5, y + cell_size // 2 - 5), 'N/A', fill=(150, 150, 150), font=font)

    # Add family group separators if grouped
    if group_by_family:
        row_counter = 0
        for family, members in STRATEGY_FAMILIES.items():
            active = [s for s in members if s in strategy_order]
            if active:
                if row_counter > 0:
                    sep_y = grid_y_start + row_counter * (cell_size + pad)
                    draw.line([(0, sep_y), (canvas_w, sep_y)], fill=(100, 100, 100), width=2)
                row_counter += len(active)

    # Save
    suffix = '_grouped' if group_by_family else ''
    out_path = output_dir / f'gen_gallery_{dataset}{suffix}.png'
    canvas.save(str(out_path), dpi=(dpi, dpi))
    print(f"  Saved: {out_path}")

    # Also save a high-res version with all 10 originals as a multi-page figure
    _build_multi_image_gallery(dataset, output_dir, strategy_order, group_by_family, dpi,
                               family_labels if group_by_family else None)


def _build_multi_image_gallery(dataset, output_dir, strategy_order, group_by_family, dpi, family_labels=None):
    """Create individual gallery pages for each of the 10 original images.
    Original shown above the grid, square thumbnails."""
    train_dir = SAMPLE_ROOT / 'training_samples' / dataset
    originals_dir = train_dir / 'originals'
    generated_dir = train_dir / 'generated'

    orig_images = sorted(originals_dir.iterdir())

    pages_dir = output_dir / f'gen_gallery_{dataset}_pages'
    pages_dir.mkdir(parents=True, exist_ok=True)

    cell_size = 150  # Square cells for grid
    orig_display_size = 200  # Larger original above grid

    for img_idx, orig_path in enumerate(orig_images):
        orig_stem = orig_path.stem
        orig_img = Image.open(orig_path).convert('RGB')

        n_cols = len(WEATHER_DOMAINS)
        n_rows = len(strategy_order)
        label_w = 140
        header_h = 25
        orig_area_h = orig_display_size + 35
        pad = 2

        canvas_w = label_w + n_cols * (cell_size + pad) + pad
        canvas_h = orig_area_h + header_h + n_rows * (cell_size + pad) + pad

        canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        font = get_font(10)
        font_header = get_font(11)
        font_orig = get_font(12)

        # Original image centered above grid (center-cropped to square)
        orig_thumb = center_crop_resize(orig_img, orig_display_size)
        orig_x = (canvas_w - orig_display_size) // 2
        canvas.paste(orig_thumb, (orig_x, 3))
        orig_label = f"Original (Clear Day)"
        bbox = draw.textbbox((0, 0), orig_label, font=font_orig)
        tw = bbox[2] - bbox[0]
        draw.text(((canvas_w - tw) // 2, orig_display_size + 6), orig_label, fill=(0, 0, 0), font=font_orig)

        # Domain column headers
        headers = [DOMAIN_LABELS[d] for d in WEATHER_DOMAINS]
        for col_idx, header in enumerate(headers):
            x = label_w + col_idx * (cell_size + pad) + pad
            bbox = draw.textbbox((0, 0), header, font=font_header)
            tw = bbox[2] - bbox[0]
            draw.text((x + (cell_size - tw) // 2, orig_area_h + 3), header, fill=(0, 0, 0), font=font_header)

        grid_y_start = orig_area_h + header_h
        for row_idx, strategy in enumerate(strategy_order):
            y = grid_y_start + row_idx * (cell_size + pad) + pad
            display = STRATEGY_DISPLAY.get(strategy, strategy)

            if group_by_family and family_labels and strategy in family_labels:
                family = family_labels[strategy]
                color = FAMILY_COLORS.get(family, '#000000')
                r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                draw.text((3, y + cell_size // 2 - 6), display, fill=(r, g, b), font=font)
            else:
                draw.text((3, y + cell_size // 2 - 6), display, fill=(0, 0, 0), font=font)

            for col_idx, domain in enumerate(WEATHER_DOMAINS):
                x = label_w + col_idx * (cell_size + pad) + pad
                gen_dir = generated_dir / strategy / domain
                gen_path = None
                for ext in ['.png', '.jpg', '.jpeg']:
                    candidate = gen_dir / f"{orig_stem}{ext}"
                    if candidate.exists():
                        gen_path = candidate
                        break
                if gen_path:
                    try:
                        gen_img = Image.open(gen_path).convert('RGB')
                        thumb = gen_img.resize((cell_size, cell_size), Image.LANCZOS)
                        canvas.paste(thumb, (x, y))
                    except:
                        draw.rectangle([x, y, x + cell_size, y + cell_size], fill=(200, 200, 200))
                else:
                    draw.rectangle([x, y, x + cell_size, y + cell_size], fill=(240, 240, 240))

        out_path = pages_dir / f'page_{img_idx:02d}_{orig_stem}.png'
        canvas.save(str(out_path), dpi=(dpi, dpi))

    print(f"  Saved {len(orig_images)} individual pages in {pages_dir}/")


# ============================================================================
# FIGURE A-alt: Generated Image Gallery — Transposed (domains × strategies)
# ============================================================================

def build_gen_gallery_transposed(dataset, output_dir, dpi=300):
    """
    Alternate gallery layout:
      - Rows   = weather domains (cloudy, dawn_dusk, foggy, night, rainy, snowy)
      - Cols   = selected generative strategies
      - Original clear-day image shown in the top-left corner
    One figure per original image (10 pages).
    """
    train_dir = SAMPLE_ROOT / 'training_samples' / dataset
    originals_dir = train_dir / 'originals'
    generated_dir = train_dir / 'generated'

    if not originals_dir.exists():
        print(f"  WARNING: No originals for {dataset}")
        return

    orig_images = sorted(originals_dir.iterdir())
    if not orig_images:
        print(f"  WARNING: No original images for {dataset}")
        return

    # Selected strategies (user-specified subset)
    selected_strategies = [
        'cycleGAN', 'CUT', 'SUSTechGAN', 'flux_kontext',
        'step1x_new', 'Qwen-Image-Edit', 'VisualCloze', 'IP2P',
    ]
    # Filter to strategies that actually exist
    available = [s for s in selected_strategies if (generated_dir / s).exists()]
    if not available:
        print(f"  WARNING: No selected strategies found for {dataset}")
        return

    cell_size = 180  # Square cell size
    label_w = 100    # Left label column for domain names
    header_h = 60    # Top header row for strategy names + original
    pad = 3

    n_rows = len(WEATHER_DOMAINS)
    n_cols = 1 + len(available)  # original column + strategy columns

    canvas_w = label_w + n_cols * (cell_size + pad) + pad
    canvas_h = header_h + n_rows * (cell_size + pad) + pad

    pages_dir = output_dir / f'gen_gallery_{dataset}_transposed'
    pages_dir.mkdir(parents=True, exist_ok=True)

    for img_idx, orig_path in enumerate(orig_images):
        orig_stem = orig_path.stem
        orig_img = Image.open(orig_path).convert('RGB')

        canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        font = get_font(11)
        font_header = get_font(12)

        # Column headers: "Original" + strategy names
        col_headers = ['Original'] + [STRATEGY_DISPLAY.get(s, s) for s in available]
        for col_idx, header in enumerate(col_headers):
            x = label_w + col_idx * (cell_size + pad) + pad
            # Center the text
            bbox = draw.textbbox((0, 0), header, font=font_header)
            tw = bbox[2] - bbox[0]
            draw.text((x + max(0, (cell_size - tw) // 2), 5), header,
                      fill=(0, 0, 0), font=font_header)

        # Also add dataset subtitle below headers
        subtitle = f"{dataset} — Image {img_idx + 1}/{len(orig_images)}"
        bbox = draw.textbbox((0, 0), subtitle, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((canvas_w - tw) // 2, 30), subtitle, fill=(100, 100, 100), font=font)

        # Fill grid: rows = domains, cols = original + strategies
        for row_idx, domain in enumerate(WEATHER_DOMAINS):
            y = header_h + row_idx * (cell_size + pad) + pad

            # Domain label
            domain_label = DOMAIN_LABELS.get(domain, domain)
            bbox = draw.textbbox((0, 0), domain_label, font=font)
            th = bbox[3] - bbox[1]
            draw.text((5, y + (cell_size - th) // 2), domain_label,
                      fill=(0, 0, 0), font=font)

            # Column 0: Original image (same for all rows, center-cropped)
            x = label_w + pad
            orig_thumb = center_crop_resize(orig_img, cell_size)
            canvas.paste(orig_thumb, (x, y))

            # Strategy columns
            for col_idx, strategy in enumerate(available):
                x = label_w + (1 + col_idx) * (cell_size + pad) + pad
                gen_dir = generated_dir / strategy / domain

                gen_path = None
                for ext in ['.png', '.jpg', '.jpeg']:
                    candidate = gen_dir / f"{orig_stem}{ext}"
                    if candidate.exists():
                        gen_path = candidate
                        break

                if gen_path:
                    try:
                        gen_img = Image.open(gen_path).convert('RGB')
                        thumb = center_crop_resize(gen_img, cell_size)
                        canvas.paste(thumb, (x, y))
                    except Exception:
                        draw.rectangle([x, y, x + cell_size, y + cell_size],
                                       fill=(200, 200, 200))
                        draw.text((x + 5, y + cell_size // 2 - 5), 'Error',
                                  fill=(255, 0, 0), font=font)
                else:
                    draw.rectangle([x, y, x + cell_size, y + cell_size],
                                   fill=(240, 240, 240))
                    draw.text((x + cell_size // 2 - 10, y + cell_size // 2 - 5),
                              'N/A', fill=(150, 150, 150), font=font)

        out_path = pages_dir / f'page_{img_idx:02d}_{orig_stem}.png'
        canvas.save(str(out_path), dpi=(dpi, dpi))

    # Also create a single summary using the first image
    first_page = pages_dir / f'page_00_{orig_images[0].stem}.png'
    if first_page.exists():
        import shutil
        summary_path = output_dir / f'gen_gallery_{dataset}_transposed.png'
        shutil.copy2(str(first_page), str(summary_path))
        print(f"  Saved summary: {summary_path}")

    print(f"  Saved {len(orig_images)} transposed pages in {pages_dir}/")


# ============================================================================
# FIGURE B: Segmentation Quality Comparison
# ============================================================================

def build_seg_comparison(dataset, output_dir, group_by_family=False, dpi=300):
    """
    For each test image: show Original | GT Label | predictions from all strategies.
    Two versions: Stage 1 and Stage 2.
    """
    test_dir = SAMPLE_ROOT / 'testing_samples' / dataset
    images_dir = test_dir / 'images'
    labels_dir = test_dir / 'labels'
    pred_base = test_dir / 'predictions'

    if not images_dir.exists():
        print(f"  WARNING: No test images for {dataset}")
        return

    test_images = sorted(images_dir.iterdir())
    if not test_images:
        print(f"  WARNING: No test images found for {dataset}")
        return

    # Determine num_classes for label colorization
    num_classes = {'BDD10k': 19, 'IDD-AW': 19, 'MapillaryVistas': 66, 'OUTSIDE15k': 24}.get(dataset, 19)

    for stage in ['stage1', 'stage2']:
        stage_pred_dir = pred_base / stage
        if not stage_pred_dir.exists():
            print(f"  WARNING: No {stage} predictions for {dataset}")
            continue

        # Discover available strategies for this stage
        if group_by_family:
            available_strategies = []
            for family_name, members in PRED_FAMILIES.items():
                for s in members:
                    s_dir = stage_pred_dir / s
                    if s_dir.exists() and any(s_dir.iterdir()):
                        available_strategies.append(s)
        else:
            available_strategies = []
            for s in PRED_STRATEGIES:
                s_dir = stage_pred_dir / s
                if s_dir.exists() and any(s_dir.iterdir()):
                    available_strategies.append(s)

        if not available_strategies:
            print(f"  WARNING: No available strategies for {dataset} {stage}")
            continue

        print(f"  {dataset} {stage}: {len(available_strategies)} strategies available")

        # Build a comparison figure for each test image
        pages_dir = output_dir / f'seg_comparison_{dataset}_{stage}'
        pages_dir.mkdir(parents=True, exist_ok=True)

        for img_idx, img_path in enumerate(test_images):
            img_name = img_path.stem  # e.g., "clear_day__0e33c3bd-01573a4f"
            safe_name = img_path.name.replace('.', '_')  # for prediction lookup
            domain = img_name.split('__')[0] if '__' in img_name else 'unknown'

            # Load test image
            test_img = Image.open(img_path).convert('RGB')

            # Load GT label
            label_path = labels_dir / f"{img_name}.png"
            gt_label = None
            if label_path.exists():
                gt_raw = np.array(Image.open(label_path))
                if gt_raw.ndim == 2:
                    gt_label = Image.fromarray(colorize_label(gt_raw, num_classes))
                else:
                    gt_label = Image.open(label_path).convert('RGB')

            # Layout: grid of strategy predictions with error maps
            # Square thumbnails (source predictions are 512x512)
            thumb_size = 160
            error_h = 80  # Error map height (half of thumb for compact layout)

            n_strategies = len(available_strategies)
            grid_cols = 6  # predictions per row
            grid_rows = (n_strategies + grid_cols - 1) // grid_cols

            label_col_w = 120
            header_h = 30
            title_h = 40
            pad = 3
            cell_h = thumb_size + 2 + error_h + 15  # prediction + gap + error map + label

            # Total canvas
            canvas_w = label_col_w + grid_cols * (thumb_size + pad) + pad
            canvas_h = title_h + (thumb_size + pad + 15) + pad + grid_rows * (cell_h + pad) + pad

            canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)
            font = get_font(10)
            font_title = get_font(14)
            font_label = get_font(9)

            # Title
            stage_label = 'Stage 1 (Clear-Only Training)' if stage == 'stage1' else 'Stage 2 (All-Domain Training)'
            title = f"{dataset} — {DOMAIN_LABELS.get(domain, domain)} — {stage_label}"
            draw.text((10, 5), title, fill=(0, 0, 0), font=font_title)

            # Load GT class IDs for error map computation
            gt_ids = None
            if label_path.exists():
                gt_ids = load_gt_class_ids(label_path)

            # Original image and GT
            y_header = title_h
            orig_thumb = test_img.resize((thumb_size, thumb_size), Image.LANCZOS)
            canvas.paste(orig_thumb, (label_col_w + pad, y_header))
            draw.text((label_col_w + pad + 5, y_header + thumb_size + 1), 'Input Image', fill=(0, 0, 0), font=font_label)

            if gt_label:
                gt_thumb = gt_label.resize((thumb_size, thumb_size), Image.LANCZOS)
                canvas.paste(gt_thumb, (label_col_w + thumb_size + 2 * pad, y_header))
                draw.text((label_col_w + thumb_size + 2 * pad + 5, y_header + thumb_size + 1),
                          'Ground Truth', fill=(0, 0, 0), font=font_label)

            # Strategy predictions in grid with error maps
            y_grid_start = y_header + thumb_size + pad + 15 + pad

            for strat_idx, strategy in enumerate(available_strategies):
                row = strat_idx // grid_cols
                col = strat_idx % grid_cols

                x = label_col_w + col * (thumb_size + pad) + pad
                y = y_grid_start + row * (cell_h + pad)

                # Find prediction image and raw prediction
                strat_dir = stage_pred_dir / strategy
                pred_img = None
                pred_raw_arr = None
                if strat_dir.exists():
                    for model_dir in strat_dir.iterdir():
                        pred_color = model_dir / f"{safe_name}_pred_color.png"
                        pred_raw_path = model_dir / f"{safe_name}_pred_raw.png"
                        if pred_color.exists():
                            pred_img = Image.open(pred_color).convert('RGB')
                            if pred_raw_path.exists() and gt_ids is not None:
                                pred_raw_arr = np.array(Image.open(pred_raw_path))
                            break

                if pred_img:
                    thumb = pred_img.resize((thumb_size, thumb_size), Image.LANCZOS)
                    canvas.paste(thumb, (x, y))

                    # Error map below prediction
                    if pred_raw_arr is not None and gt_ids is not None:
                        err = compute_error_map(gt_ids, pred_raw_arr)
                        err_thumb = Image.fromarray(err).resize((thumb_size, error_h), Image.NEAREST)
                        canvas.paste(err_thumb, (x, y + thumb_size + 2))
                else:
                    draw.rectangle([x, y, x + thumb_size, y + thumb_size], fill=(240, 240, 240))
                    draw.text((x + 5, y + thumb_size // 2 - 5), 'N/A', fill=(150, 150, 150), font=font)

                # Strategy label below
                display = PRED_DISPLAY.get(strategy, strategy)
                draw.text((x + 2, y + thumb_size + 2 + error_h + 1), display, fill=(0, 0, 0), font=font_label)

            out_path = pages_dir / f'{img_name}.png'
            canvas.save(str(out_path), dpi=(dpi, dpi))

        print(f"  Saved {len(test_images)} comparison pages in {pages_dir}/")

    # Also create a compact summary: one row per domain, columns = best/worst strategies
    _build_seg_summary(dataset, output_dir, dpi)


def _build_seg_summary(dataset, output_dir, dpi=300):
    """
    Compact summary figure: for each domain, show input + GT + baseline + top strategies.
    Side-by-side Stage 1 vs Stage 2.
    """
    test_dir = SAMPLE_ROOT / 'testing_samples' / dataset
    images_dir = test_dir / 'images'
    labels_dir = test_dir / 'labels'
    pred_base = test_dir / 'predictions'

    if not images_dir.exists():
        return

    test_images = sorted(images_dir.iterdir())
    num_classes = {'BDD10k': 19, 'IDD-AW': 19, 'MapillaryVistas': 66, 'OUTSIDE15k': 24}.get(dataset, 19)

    # Pick one representative image per domain
    domain_images = {}
    for img_path in test_images:
        domain = img_path.stem.split('__')[0] if '__' in img_path.stem else 'unknown'
        if domain not in domain_images:
            domain_images[domain] = img_path

    # Selected strategies for summary (representative subset)
    summary_strategies = ['baseline', 'gen_cycleGAN', 'gen_flux_kontext', 'gen_step1x_new',
                          'gen_IP2P', 'gen_CUT', 'gen_albumentations_weather', 'std_autoaugment']

    domains_to_show = [d for d in ['clear_day', 'foggy', 'night', 'rainy', 'snowy'] if d in domain_images]

    thumb_w = 140
    thumb_h = 140  # Square (source images are 512x512)

    for stage in ['stage1', 'stage2']:
        stage_pred_dir = pred_base / stage
        if not stage_pred_dir.exists():
            continue

        # Filter to available strategies
        avail = [s for s in summary_strategies if (stage_pred_dir / s).exists()]
        if not avail:
            continue

        # Columns: Domain label | Input | GT | strategy1 | strategy2 | ...
        n_cols = 2 + len(avail)  # input + GT + strategies
        n_rows = len(domains_to_show)

        label_w = 80
        header_h = 30
        pad = 2

        canvas_w = label_w + n_cols * (thumb_w + pad) + pad
        canvas_h = header_h + n_rows * (thumb_h + pad) + pad

        canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        font = get_font(10)
        font_header = get_font(11)

        # Header
        col_headers = ['Input', 'GT'] + [PRED_DISPLAY.get(s, s) for s in avail]
        for col_idx, header in enumerate(col_headers):
            x = label_w + col_idx * (thumb_w + pad) + pad
            bbox = draw.textbbox((0, 0), header, font=font_header)
            tw = bbox[2] - bbox[0]
            text_x = x + max(0, (thumb_w - tw) // 2)
            draw.text((text_x, 5), header, fill=(0, 0, 0), font=font_header)

        for row_idx, domain in enumerate(domains_to_show):
            y = header_h + row_idx * (thumb_h + pad) + pad
            img_path = domain_images[domain]
            img_name = img_path.stem
            safe_name = img_path.name.replace('.', '_')

            # Domain label
            draw.text((5, y + thumb_h // 2 - 6), DOMAIN_LABELS.get(domain, domain),
                      fill=(0, 0, 0), font=font)

            # Input image
            x = label_w + pad
            test_img = Image.open(img_path).convert('RGB').resize((thumb_w, thumb_h), Image.LANCZOS)
            canvas.paste(test_img, (x, y))

            # GT label
            x = label_w + (thumb_w + pad) + pad
            label_path = labels_dir / f"{img_name}.png"
            if label_path.exists():
                gt_raw = np.array(Image.open(label_path))
                if gt_raw.ndim == 2:
                    gt_colored = Image.fromarray(colorize_label(gt_raw, num_classes))
                else:
                    gt_colored = Image.open(label_path).convert('RGB')
                gt_thumb = gt_colored.resize((thumb_w, thumb_h), Image.LANCZOS)
                canvas.paste(gt_thumb, (x, y))

            # Strategy predictions
            for s_idx, strategy in enumerate(avail):
                x = label_w + (2 + s_idx) * (thumb_w + pad) + pad
                strat_dir = stage_pred_dir / strategy
                pred_found = False
                if strat_dir.exists():
                    for model_dir in strat_dir.iterdir():
                        pred_color = model_dir / f"{safe_name}_pred_color.png"
                        if pred_color.exists():
                            pred_img = Image.open(pred_color).convert('RGB').resize(
                                (thumb_w, thumb_h), Image.LANCZOS)
                            canvas.paste(pred_img, (x, y))
                            pred_found = True
                            break
                if not pred_found:
                    draw.rectangle([x, y, x + thumb_w, y + thumb_h], fill=(240, 240, 240))

        stage_label = 'Stage 1' if stage == 'stage1' else 'Stage 2'
        out_path = output_dir / f'seg_summary_{dataset}_{stage}.png'
        canvas.save(str(out_path), dpi=(dpi, dpi))
        print(f"  Saved summary: {out_path}")


# ============================================================================
# FIGURE C: Domain-Specific Strips
# ============================================================================

def build_domain_strips(dataset, output_dir, group_by_family=False, dpi=300):
    """
    For selected adverse domains, show all strategy predictions side by side.
    Layout: each strip = one strategy, showing Stage 1 vs Stage 2 prediction.
    """
    test_dir = SAMPLE_ROOT / 'testing_samples' / dataset
    images_dir = test_dir / 'images'
    labels_dir = test_dir / 'labels'
    pred_base = test_dir / 'predictions'

    if not images_dir.exists():
        print(f"  WARNING: No test images for {dataset}")
        return

    num_classes = {'BDD10k': 19, 'IDD-AW': 19, 'MapillaryVistas': 66, 'OUTSIDE15k': 24}.get(dataset, 19)

    # Pick test images from challenging domains
    focus_domains = ['foggy', 'night', 'snowy']
    test_images = sorted(images_dir.iterdir())
    domain_images = {}
    for img_path in test_images:
        domain = img_path.stem.split('__')[0] if '__' in img_path.stem else 'unknown'
        if domain in focus_domains and domain not in domain_images:
            domain_images[domain] = img_path

    if not domain_images:
        print(f"  WARNING: No images found for focus domains in {dataset}")
        return

    # Strategy order
    if group_by_family:
        strategy_order = []
        for family_name, members in PRED_FAMILIES.items():
            for s in members:
                strategy_order.append(s)
    else:
        strategy_order = PRED_STRATEGIES.copy()

    thumb_size = 160  # Square (source images are 512x512)

    for domain, img_path in domain_images.items():
        img_name = img_path.stem
        safe_name = img_path.name.replace('.', '_')

        # Filter to strategies that have predictions in at least one stage
        avail_strategies = []
        for s in strategy_order:
            for stage in ['stage1', 'stage2']:
                s_dir = pred_base / stage / s
                if s_dir.exists() and any(s_dir.iterdir()):
                    avail_strategies.append(s)
                    break

        if not avail_strategies:
            continue

        # Layout: rows = strategies, columns = [Name | Input | GT | S1 Pred | S1 Error | S2 Pred | S2 Error]
        n_rows = len(avail_strategies)
        label_w = 140
        header_h = 50  # Space for title + column headers
        pad = 2

        # Columns: input + GT + stage1_pred + stage1_error + stage2_pred + stage2_error
        n_cols = 6
        canvas_w = label_w + n_cols * (thumb_size + pad) + pad
        canvas_h = header_h + n_rows * (thumb_size + pad) + pad

        canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        font = get_font(10)
        font_header = get_font(12)
        font_title = get_font(14)

        # Title
        title = f"{dataset} — {DOMAIN_LABELS.get(domain, domain)} Domain"
        draw.text((10, 3), title, fill=(0, 0, 0), font=font_title)

        # Column headers
        col_headers = ['Input', 'Ground Truth', 'Stage 1 Pred.', 'S1 Error', 'Stage 2 Pred.', 'S2 Error']
        for col_idx, header in enumerate(col_headers):
            x = label_w + col_idx * (thumb_size + pad) + pad
            bbox = draw.textbbox((0, 0), header, font=font_header)
            tw = bbox[2] - bbox[0]
            draw.text((x + max(0, (thumb_size - tw) // 2), 25), header, fill=(0, 0, 0), font=font_header)

        # Load input image and GT once
        test_img = Image.open(img_path).convert('RGB').resize((thumb_size, thumb_size), Image.LANCZOS)
        label_path = labels_dir / f"{img_name}.png"
        gt_thumb = None
        gt_ids = None
        if label_path.exists():
            gt_ids = load_gt_class_ids(label_path)
            gt_raw = np.array(Image.open(label_path))
            if gt_raw.ndim == 2:
                gt_colored = Image.fromarray(colorize_label(gt_raw, num_classes))
            else:
                gt_colored = Image.open(label_path).convert('RGB')
            gt_thumb = gt_colored.resize((thumb_size, thumb_size), Image.LANCZOS)

        # Draw family separators if grouped
        current_family_idx = 0
        family_boundaries = []
        if group_by_family:
            row_count = 0
            for family_name, members in PRED_FAMILIES.items():
                active_in_family = [s for s in members if s in avail_strategies]
                if active_in_family and row_count > 0:
                    family_boundaries.append((row_count, family_name))
                row_count += len(active_in_family)

        for row_idx, strategy in enumerate(avail_strategies):
            y = header_h + row_idx * (thumb_size + pad) + pad

            # Strategy label
            display = PRED_DISPLAY.get(strategy, strategy)
            draw.text((3, y + thumb_size // 2 - 6), display, fill=(0, 0, 0), font=font)

            # Input image (same for all rows)
            x = label_w + pad
            canvas.paste(test_img, (x, y))

            # GT label
            x = label_w + (thumb_size + pad) + pad
            if gt_thumb:
                canvas.paste(gt_thumb, (x, y))

            # Stage 1 prediction + error map
            x_s1 = label_w + 2 * (thumb_size + pad) + pad
            x_s1_err = label_w + 3 * (thumb_size + pad) + pad
            s1_dir = pred_base / 'stage1' / strategy
            pred_found = False
            if s1_dir.exists():
                for model_dir in s1_dir.iterdir():
                    pred_path = model_dir / f"{safe_name}_pred_color.png"
                    pred_raw_path = model_dir / f"{safe_name}_pred_raw.png"
                    if pred_path.exists():
                        pred_img = Image.open(pred_path).convert('RGB').resize(
                            (thumb_size, thumb_size), Image.LANCZOS)
                        canvas.paste(pred_img, (x_s1, y))
                        # Error map
                        if pred_raw_path.exists() and gt_ids is not None:
                            pred_raw_arr = np.array(Image.open(pred_raw_path))
                            err = compute_error_map(gt_ids, pred_raw_arr)
                            err_thumb = Image.fromarray(err).resize(
                                (thumb_size, thumb_size), Image.NEAREST)
                            canvas.paste(err_thumb, (x_s1_err, y))
                        pred_found = True
                        break
            if not pred_found:
                draw.rectangle([x_s1, y, x_s1 + thumb_size, y + thumb_size], fill=(240, 240, 240))
                draw.rectangle([x_s1_err, y, x_s1_err + thumb_size, y + thumb_size], fill=(240, 240, 240))

            # Stage 2 prediction + error map
            x_s2 = label_w + 4 * (thumb_size + pad) + pad
            x_s2_err = label_w + 5 * (thumb_size + pad) + pad
            s2_dir = pred_base / 'stage2' / strategy
            pred_found = False
            if s2_dir.exists():
                for model_dir in s2_dir.iterdir():
                    pred_path = model_dir / f"{safe_name}_pred_color.png"
                    pred_raw_path = model_dir / f"{safe_name}_pred_raw.png"
                    if pred_path.exists():
                        pred_img = Image.open(pred_path).convert('RGB').resize(
                            (thumb_size, thumb_size), Image.LANCZOS)
                        canvas.paste(pred_img, (x_s2, y))
                        # Error map
                        if pred_raw_path.exists() and gt_ids is not None:
                            pred_raw_arr = np.array(Image.open(pred_raw_path))
                            err = compute_error_map(gt_ids, pred_raw_arr)
                            err_thumb = Image.fromarray(err).resize(
                                (thumb_size, thumb_size), Image.NEAREST)
                            canvas.paste(err_thumb, (x_s2_err, y))
                        pred_found = True
                        break
            if not pred_found:
                draw.rectangle([x_s2, y, x_s2 + thumb_size, y + thumb_size], fill=(240, 240, 240))
                draw.rectangle([x_s2_err, y, x_s2_err + thumb_size, y + thumb_size], fill=(240, 240, 240))

        # Draw family separation lines
        for boundary_row, family_name in family_boundaries:
            sep_y = header_h + boundary_row * (thumb_size + pad)
            draw.line([(0, sep_y), (canvas_w, sep_y)], fill=(150, 150, 150), width=1)

        out_path = output_dir / f'domain_strip_{dataset}_{domain}.png'
        canvas.save(str(out_path), dpi=(dpi, dpi))
        print(f"  Saved: {out_path}")


# ============================================================================
# CLASS LEGEND
# ============================================================================

def build_class_legend(output_dir, dpi=300):
    """Create a color legend for Cityscapes classes."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 5)
    ax.axis('off')

    cols = 4
    for i, (cls_name, color) in enumerate(zip(CITYSCAPES_CLASSES, CITYSCAPES_PALETTE)):
        row = i // cols
        col = i % cols
        x = col * 1.0
        y = 4.5 - row * 0.9
        rect = mpatches.FancyBboxPatch((x, y - 0.2), 0.3, 0.4,
                                        facecolor=np.array(color) / 255.0,
                                        edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        ax.text(x + 0.35, y, cls_name, fontsize=9, va='center')

    fig.tight_layout()
    out_path = output_dir / 'class_legend_cityscapes.png'
    fig.savefig(str(out_path), dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved legend: {out_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Assemble publication figures for generated images & segmentation')
    parser.add_argument('--figure', type=str, required=True,
                       choices=['gen-gallery', 'gen-gallery-transposed', 'seg-comparison', 'domain-strips', 'all'],
                       help='Which figure to generate')
    parser.add_argument('--dataset', type=str, default=None, choices=DATASETS,
                       help='Limit to one dataset')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Custom output directory')
    parser.add_argument('--group-by-family', action='store_true',
                       help='Group strategies by family (GAN, diffusion, etc.)')
    parser.add_argument('--dpi', type=int, default=300, help='Output DPI')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else IEEE_FIGURES
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    datasets = [args.dataset] if args.dataset else DATASETS

    figures_to_build = []
    if args.figure == 'all':
        figures_to_build = ['gen-gallery', 'gen-gallery-transposed', 'seg-comparison', 'domain-strips']
    else:
        figures_to_build = [args.figure]

    # Always generate class legend
    build_class_legend(output_dir, args.dpi)

    for fig_type in figures_to_build:
        print(f"\n{'='*60}")
        print(f"Building: {fig_type}")
        print(f"{'='*60}")

        for dataset in datasets:
            print(f"\n--- {dataset} ---")

            if fig_type == 'gen-gallery':
                # Build both ungrouped and grouped
                build_gen_gallery(dataset, output_dir, group_by_family=False, dpi=args.dpi)
                build_gen_gallery(dataset, output_dir, group_by_family=True, dpi=args.dpi)

            elif fig_type == 'gen-gallery-transposed':
                build_gen_gallery_transposed(dataset, output_dir, dpi=args.dpi)

            elif fig_type == 'seg-comparison':
                build_seg_comparison(dataset, output_dir, group_by_family=args.group_by_family, dpi=args.dpi)

            elif fig_type == 'domain-strips':
                # Build both ungrouped and grouped
                build_domain_strips(dataset, output_dir, group_by_family=False, dpi=args.dpi)
                build_domain_strips(dataset, output_dir, group_by_family=True, dpi=args.dpi)

    print(f"\n{'='*60}")
    print(f"All figures saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
