#!/usr/bin/env python3
"""
Convert BDD100k JSON labels to COCO format for MMDetection.

BDD100k labels are stored as individual JSON files with box2d annotations.
This script converts them to a single COCO-format JSON file.

Usage:
    python convert_bdd100k_to_coco.py --input-dir /path/to/labels/BDD100k \
                                       --image-dir /path/to/images/BDD100k \
                                       --output /path/to/output.json
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from PIL import Image
from tqdm import tqdm


# BDD100k category mapping to standardized names
BDD100K_CATEGORY_MAP = {
    'person': 'pedestrian',
    'pedestrian': 'pedestrian',
    'rider': 'rider',
    'car': 'car',
    'truck': 'truck',
    'bus': 'bus',
    'train': 'train',
    'motor': 'motorcycle',
    'motorcycle': 'motorcycle',
    'bike': 'bicycle',
    'bicycle': 'bicycle',
    'traffic light': 'traffic light',
    'traffic sign': 'traffic sign',
}

# Categories for COCO format (with IDs)
COCO_CATEGORIES = [
    {'id': 1, 'name': 'pedestrian', 'supercategory': 'human'},
    {'id': 2, 'name': 'rider', 'supercategory': 'human'},
    {'id': 3, 'name': 'car', 'supercategory': 'vehicle'},
    {'id': 4, 'name': 'truck', 'supercategory': 'vehicle'},
    {'id': 5, 'name': 'bus', 'supercategory': 'vehicle'},
    {'id': 6, 'name': 'train', 'supercategory': 'vehicle'},
    {'id': 7, 'name': 'motorcycle', 'supercategory': 'vehicle'},
    {'id': 8, 'name': 'bicycle', 'supercategory': 'vehicle'},
    {'id': 9, 'name': 'traffic light', 'supercategory': 'object'},
    {'id': 10, 'name': 'traffic sign', 'supercategory': 'object'},
]

# Category name to ID mapping
CATEGORY_NAME_TO_ID = {cat['name']: cat['id'] for cat in COCO_CATEGORIES}


def parse_args():
    parser = argparse.ArgumentParser(description='Convert BDD100k to COCO format')
    parser.add_argument('--input-dir', required=True, 
                        help='Directory containing BDD100k JSON labels (with weather subdirs)')
    parser.add_argument('--image-dir', required=True,
                        help='Directory containing BDD100k images (with weather subdirs)')
    parser.add_argument('--output', required=True,
                        help='Output COCO JSON file path')
    parser.add_argument('--include-drivable', action='store_true',
                        help='Include drivable area annotations (poly2d)')
    return parser.parse_args()


def get_image_size(image_path):
    """Get image dimensions without loading full image."""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        print(f"Warning: Could not read image {image_path}: {e}")
        return (1280, 720)  # Default BDD100k size


def convert_bdd100k_to_coco(input_dir, image_dir, output_path, include_drivable=False):
    """Convert BDD100k annotations to COCO format."""
    
    input_dir = Path(input_dir)
    image_dir = Path(image_dir)
    
    coco_output = {
        'info': {
            'description': 'BDD100k converted to COCO format',
            'version': '1.0',
            'year': 2025,
            'contributor': 'PROVE',
            'date_created': datetime.now().isoformat()
        },
        'licenses': [{'id': 1, 'name': 'BDD100k License', 'url': ''}],
        'categories': COCO_CATEGORIES,
        'images': [],
        'annotations': []
    }
    
    image_id = 0
    annotation_id = 0
    
    # Collect all JSON files from weather subdirectories
    json_files = []
    for weather_dir in input_dir.iterdir():
        if weather_dir.is_dir():
            json_files.extend(list(weather_dir.glob('*.json')))
    
    # Also check for JSON files directly in input_dir
    json_files.extend(list(input_dir.glob('*.json')))
    
    print(f"Found {len(json_files)} annotation files")
    
    for json_file in tqdm(json_files, desc="Converting annotations"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not read {json_file}: {e}")
            continue
        
        # Get image filename
        image_name = data.get('name', json_file.stem + '.jpg')
        
        # Find image file (check in weather subdirs)
        weather_subdir = json_file.parent.name
        image_path = image_dir / weather_subdir / image_name
        
        if not image_path.exists():
            # Try without weather subdir
            image_path = image_dir / image_name
        
        if not image_path.exists():
            # Try with different extension
            for ext in ['.jpg', '.png', '.jpeg']:
                test_path = image_dir / weather_subdir / (json_file.stem + ext)
                if test_path.exists():
                    image_path = test_path
                    image_name = test_path.name
                    break
        
        # Get image dimensions
        if image_path.exists():
            width, height = get_image_size(image_path)
        else:
            width, height = 1280, 720  # Default BDD100k size
            print(f"Warning: Image not found: {image_path}")
        
        # Create relative path for image
        rel_image_path = f"{weather_subdir}/{image_name}"
        
        # Add image entry
        image_entry = {
            'id': image_id,
            'file_name': rel_image_path,
            'width': width,
            'height': height,
            'license': 1
        }
        
        # Add weather/scene attributes if available
        if 'attributes' in data:
            image_entry['weather'] = data['attributes'].get('weather', '')
            image_entry['scene'] = data['attributes'].get('scene', '')
            image_entry['timeofday'] = data['attributes'].get('timeofday', '')
        
        coco_output['images'].append(image_entry)
        
        # Process labels
        for label in data.get('labels', []):
            category = label.get('category', '')
            
            # Skip non-detection categories unless requested
            if category in ['drivable area', 'lane'] and not include_drivable:
                continue
            
            # Map category name
            mapped_category = BDD100K_CATEGORY_MAP.get(category.lower())
            if mapped_category is None:
                continue
            
            category_id = CATEGORY_NAME_TO_ID.get(mapped_category)
            if category_id is None:
                continue
            
            # Process box2d annotations
            if 'box2d' in label:
                box = label['box2d']
                x1, y1 = box['x1'], box['y1']
                x2, y2 = box['x2'], box['y2']
                
                # Convert to COCO format [x, y, width, height]
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                
                # Skip invalid boxes
                if bbox_width <= 0 or bbox_height <= 0:
                    continue
                
                annotation = {
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': category_id,
                    'bbox': [x1, y1, bbox_width, bbox_height],
                    'area': bbox_width * bbox_height,
                    'iscrowd': 0,
                    'segmentation': []
                }
                
                # Add attributes if available
                if 'attributes' in label:
                    annotation['occluded'] = label['attributes'].get('occluded', False)
                    annotation['truncated'] = label['attributes'].get('truncated', False)
                
                coco_output['annotations'].append(annotation)
                annotation_id += 1
        
        image_id += 1
    
    # Save COCO format JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(coco_output, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"  Images: {len(coco_output['images'])}")
    print(f"  Annotations: {len(coco_output['annotations'])}")
    print(f"  Output: {output_path}")


def main():
    args = parse_args()
    convert_bdd100k_to_coco(
        args.input_dir,
        args.image_dir,
        args.output,
        args.include_drivable
    )


if __name__ == '__main__':
    main()
