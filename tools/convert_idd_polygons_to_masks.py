#!/usr/bin/env python3
"""
Convert IDD polygon JSON annotations to segmentation masks with Cityscapes trainIds.

This script reads the original IDD polygon annotations (in JSON format) and generates
segmentation masks with proper Cityscapes trainId values for compatibility with
the 19-class Cityscapes evaluation protocol.
"""

import argparse
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


# IDD label name to Cityscapes trainId mapping (from idd_39.py)
IDD_NAME_TO_CSTRAINID: Dict[str, int] = {
    'road': 0,
    'parking': 255,
    'drivable fallback': 255,
    'sidewalk': 1,
    'rail track': 255,
    'non-drivable fallback': 9,  # terrain in Cityscapes
    'person': 11,
    'animal': 255,
    'rider': 12,
    'motorcycle': 17,
    'bicycle': 18,
    'autorickshaw': 255,
    'car': 13,
    'truck': 14,
    'bus': 15,
    'caravan': 255,
    'trailer': 255,
    'train': 16,
    'vehicle fallback': 255,
    'curb': 255,
    'wall': 3,
    'fence': 4,
    'guard rail': 255,
    'billboard': 255,
    'traffic sign': 7,
    'traffic light': 6,
    'pole': 5,
    'polegroup': 255,
    'obs-str-bar-fallback': 255,
    'building': 2,
    'bridge': 255,
    'tunnel': 255,
    'vegetation': 8,
    'sky': 10,
    'fallback background': 255,
    'unlabeled': 255,
    'ego vehicle': 255,
    'rectification border': 255,
    'out of roi': 255,
    'license plate': 255,
    'terrain': 9,  # Additional alias
}


def polygon_to_mask(polygon: List[List[float]], img_size: Tuple[int, int]) -> np.ndarray:
    """Convert polygon coordinates to a binary mask."""
    mask = Image.new('L', img_size, 0)
    if len(polygon) >= 3:
        # Convert to flat list of tuples for PIL
        flat_polygon = [(p[0], p[1]) for p in polygon]
        ImageDraw.Draw(mask).polygon(flat_polygon, fill=1)
    return np.array(mask)


def json_to_segmentation_mask(json_path: str, output_size: Tuple[int, int] = None) -> np.ndarray:
    """
    Convert IDD polygon JSON to segmentation mask with Cityscapes trainIds.
    
    Args:
        json_path: Path to the JSON polygon annotation file.
        output_size: Optional (width, height) to resize the mask. If None, uses original size.
    
    Returns:
        Segmentation mask as uint8 numpy array with Cityscapes trainId values.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    img_height = data['imgHeight']
    img_width = data['imgWidth']
    original_size = (img_width, img_height)
    
    # Create mask initialized with ignore index
    mask = np.full((img_height, img_width), 255, dtype=np.uint8)
    
    # Sort objects by size (larger first) so smaller objects render on top
    objects = data.get('objects', [])
    object_areas = []
    for obj in objects:
        polygon = obj.get('polygon', [])
        if len(polygon) >= 3:
            # Approximate area using shoelace formula
            n = len(polygon)
            area = 0.5 * abs(sum(polygon[i][0] * polygon[(i+1)%n][1] - polygon[(i+1)%n][0] * polygon[i][1] for i in range(n)))
            object_areas.append((area, obj))
        else:
            object_areas.append((0, obj))
    
    # Sort by area descending (large objects first, small objects overlay on top)
    object_areas.sort(key=lambda x: -x[0])
    
    # Render polygons
    for _, obj in object_areas:
        label = obj.get('label', '').lower().strip()
        polygon = obj.get('polygon', [])
        
        if len(polygon) < 3:
            continue
            
        # Get Cityscapes trainId
        train_id = IDD_NAME_TO_CSTRAINID.get(label, None)
        if train_id is None:
            # Try case-insensitive match
            for name, tid in IDD_NAME_TO_CSTRAINID.items():
                if name.lower() == label:
                    train_id = tid
                    break
        
        if train_id is None:
            print(f"Warning: Unknown label '{label}' in {json_path}, mapping to 255")
            train_id = 255
        
        # Create polygon mask
        poly_mask = polygon_to_mask(polygon, original_size)
        mask[poly_mask == 1] = train_id
    
    # Resize if needed
    if output_size is not None and output_size != original_size:
        mask_img = Image.fromarray(mask)
        mask_img = mask_img.resize(output_size, Image.NEAREST)
        mask = np.array(mask_img)
    
    return mask


def process_single_file(args: Tuple[str, str, Tuple[int, int]]) -> Tuple[str, bool, str]:
    """Process a single JSON file. Returns (filename, success, error_message)."""
    json_path, output_path, output_size = args
    try:
        mask = json_to_segmentation_mask(json_path, output_size)
        # Save as grayscale PNG
        Image.fromarray(mask).save(output_path)
        return (os.path.basename(json_path), True, "")
    except Exception as e:
        return (os.path.basename(json_path), False, str(e))


def find_corresponding_json(image_name: str, json_dir: str) -> str:
    """Find the JSON polygon file corresponding to an image."""
    # Extract base name (e.g., 000128 from 000128_leftImg8bit.png)
    base = image_name.replace('_leftImg8bit.png', '').replace('_leftImg8bit.jpg', '')
    base = base.replace('.png', '').replace('.jpg', '')
    
    # Try different naming conventions
    candidates = [
        f"{base}_gtFine_polygons.json",
        f"{base}_polygons.json",
        f"{base}.json",
    ]
    
    for candidate in candidates:
        json_path = os.path.join(json_dir, candidate)
        if os.path.exists(json_path):
            return json_path
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Convert IDD polygon JSON to segmentation masks')
    parser.add_argument('--json-dir', type=str, required=True,
                       help='Directory containing JSON polygon annotation files')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for segmentation masks')
    parser.add_argument('--image-list', type=str, default=None,
                       help='Optional: Text file with list of image names to process')
    parser.add_argument('--splits-dir', type=str, default=None,
                       help='Optional: FINAL_SPLITS directory to match existing masks')
    parser.add_argument('--output-size', type=str, default='512,512',
                       help='Output mask size as "width,height" (default: 512,512)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of parallel workers')
    parser.add_argument('--dry-run', action='store_true',
                       help='Just show what would be done without processing')
    args = parser.parse_args()
    
    # Parse output size
    output_size = tuple(map(int, args.output_size.split(',')))
    
    # Find JSON files to process
    json_files = []
    
    if args.splits_dir:
        # Match existing structure in FINAL_SPLITS
        for split in ['train', 'test']:
            labels_dir = os.path.join(args.splits_dir, split, 'labels', 'IDD-AW')
            if not os.path.exists(labels_dir):
                continue
            
            for weather in os.listdir(labels_dir):
                weather_dir = os.path.join(labels_dir, weather)
                if not os.path.isdir(weather_dir):
                    continue
                
                for mask_file in os.listdir(weather_dir):
                    if not mask_file.endswith('.png'):
                        continue
                    
                    json_path = find_corresponding_json(mask_file, args.json_dir)
                    if json_path:
                        output_path = os.path.join(args.output_dir, split, 'labels', 'IDD-AW', weather, mask_file)
                        json_files.append((json_path, output_path, output_size))
                    else:
                        print(f"Warning: No JSON found for {mask_file}")
    else:
        # Process all JSON files in json_dir
        for f in os.listdir(args.json_dir):
            if f.endswith('.json'):
                json_path = os.path.join(args.json_dir, f)
                base = f.replace('_gtFine_polygons.json', '').replace('.json', '')
                output_path = os.path.join(args.output_dir, f"{base}_leftImg8bit.png")
                json_files.append((json_path, output_path, output_size))
    
    print(f"Found {len(json_files)} files to process")
    
    if args.dry_run:
        for json_path, output_path, _ in json_files[:10]:
            print(f"  {os.path.basename(json_path)} -> {output_path}")
        if len(json_files) > 10:
            print(f"  ... and {len(json_files) - 10} more")
        return
    
    # Create output directories
    output_dirs = set(os.path.dirname(p) for _, p, _ in json_files)
    for d in output_dirs:
        os.makedirs(d, exist_ok=True)
    
    # Process files in parallel
    success_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_file, args): args for args in json_files}
        
        with tqdm(total=len(json_files), desc="Converting") as pbar:
            for future in as_completed(futures):
                filename, success, error = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    print(f"Error processing {filename}: {error}")
                pbar.update(1)
    
    print(f"\nDone! Processed {success_count} files successfully, {error_count} errors")


if __name__ == '__main__':
    main()
