#!/usr/bin/env python3
"""
Prepare Cityscapes dataset by creating labelTrainIds from labelIds.

The standard Cityscapes ground truth uses labelIds (0-33) but mmsegmentation
expects labelTrainIds (0-18, 255 for ignore).

This script converts labelIds.png → labelTrainIds.png for all splits.

Usage:
    python prepare_cityscapes.py --data-root ${AWARE_DATA_ROOT}/CITYSCAPES
"""

import argparse
import os
from pathlib import Path
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Cityscapes label mapping: labelId -> trainId
# trainId 255 means ignore, trainId 0-18 are the 19 eval classes
LABEL_TO_TRAIN = {
    0: 255,   # unlabeled
    1: 255,   # ego vehicle
    2: 255,   # rectification border
    3: 255,   # out of roi
    4: 255,   # static
    5: 255,   # dynamic
    6: 255,   # ground
    7: 0,     # road
    8: 1,     # sidewalk
    9: 255,   # parking
    10: 255,  # rail track
    11: 2,    # building
    12: 3,    # wall
    13: 4,    # fence
    14: 255,  # guard rail
    15: 255,  # bridge
    16: 255,  # tunnel
    17: 5,    # pole
    18: 255,  # polegroup
    19: 6,    # traffic light
    20: 7,    # traffic sign
    21: 8,    # vegetation
    22: 9,    # terrain
    23: 10,   # sky
    24: 11,   # person
    25: 12,   # rider
    26: 13,   # car
    27: 14,   # truck
    28: 15,   # bus
    29: 255,  # caravan
    30: 255,  # trailer
    31: 16,   # train
    32: 17,   # motorcycle
    33: 18,   # bicycle
    -1: 255,  # license plate
}


def convert_label_to_train(label_path: Path) -> tuple:
    """Convert a single labelIds.png to labelTrainIds.png."""
    try:
        # Load label image
        label_img = np.array(Image.open(label_path))
        
        # Create train id image
        train_img = np.zeros_like(label_img, dtype=np.uint8)
        train_img.fill(255)  # Default to ignore
        
        # Apply mapping
        for label_id, train_id in LABEL_TO_TRAIN.items():
            if label_id >= 0:  # Skip -1 (license plate)
                train_img[label_img == label_id] = train_id
        
        # Generate output path
        output_path = str(label_path).replace('_gtFine_labelIds.png', '_gtFine_labelTrainIds.png')
        
        # Save
        Image.fromarray(train_img).save(output_path)
        
        return str(label_path), True, None
    except Exception as e:
        return str(label_path), False, str(e)


def main():
    parser = argparse.ArgumentParser(description='Prepare Cityscapes labelTrainIds')
    parser.add_argument('--data-root', type=str, 
                       default='${AWARE_DATA_ROOT}/CITYSCAPES',
                       help='Path to Cityscapes data root')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of parallel workers')
    parser.add_argument('--dry-run', action='store_true',
                       help='Only show what would be done')
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    gtfine_dir = data_root / 'gtFine'
    
    if not gtfine_dir.exists():
        print(f"Error: gtFine directory not found at {gtfine_dir}")
        return 1
    
    # Find all labelIds.png files
    label_files = list(gtfine_dir.rglob('*_gtFine_labelIds.png'))
    print(f"Found {len(label_files)} labelIds files")
    
    # Check for existing labelTrainIds
    existing = list(gtfine_dir.rglob('*_gtFine_labelTrainIds.png'))
    print(f"Found {len(existing)} existing labelTrainIds files")
    
    # Filter to only files that need conversion
    to_convert = []
    for lf in label_files:
        train_path = str(lf).replace('_gtFine_labelIds.png', '_gtFine_labelTrainIds.png')
        if not Path(train_path).exists():
            to_convert.append(lf)
    
    print(f"Need to convert {len(to_convert)} files")
    
    if args.dry_run:
        print("\nDry run - would convert:")
        for f in to_convert[:10]:
            print(f"  {f}")
        if len(to_convert) > 10:
            print(f"  ... and {len(to_convert) - 10} more")
        return 0
    
    if not to_convert:
        print("All files already converted!")
        return 0
    
    # Convert in parallel
    success = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(convert_label_to_train, f): f for f in to_convert}
        
        with tqdm(total=len(to_convert), desc="Converting") as pbar:
            for future in as_completed(futures):
                path, ok, error = future.result()
                if ok:
                    success += 1
                else:
                    failed += 1
                    print(f"\nFailed: {path}: {error}")
                pbar.update(1)
    
    print(f"\nDone! Success: {success}, Failed: {failed}")
    
    # Verify
    final_count = len(list(gtfine_dir.rglob('*_gtFine_labelTrainIds.png')))
    print(f"Total labelTrainIds files: {final_count}")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    exit(main())
