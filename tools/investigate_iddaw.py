#!/usr/bin/env python3
"""
Investigate IDD-AW label values and structure.
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image

def main():
    print("=" * 60)
    print("IDD-AW Label Investigation")
    print("=" * 60)
    
    # 1. Check FINAL_SPLITS structure
    base = '${AWARE_DATA_ROOT}/FINAL_SPLITS'
    print(f"\n1. FINAL_SPLITS structure ({base}):")
    print("-" * 40)
    
    try:
        for item in sorted(os.listdir(base)):
            path = os.path.join(base, item)
            if os.path.isdir(path):
                print(f"  DIR: {item}/")
                try:
                    subs = sorted(os.listdir(path))
                    for sub in subs[:5]:
                        print(f"    - {sub}")
                    if len(subs) > 5:
                        print(f"    ... and {len(subs) - 5} more")
                except Exception as e:
                    print(f"    Error: {e}")
    except Exception as e:
        print(f"  Error listing {base}: {e}")
    
    # 2. Check IDD-AW specific paths
    iddaw_paths = [
        'train/images/IDD-AW',
        'train/labels/IDD-AW',
        'test/images/IDD-AW',
        'test/labels/IDD-AW',
    ]
    
    print(f"\n2. IDD-AW paths in FINAL_SPLITS:")
    print("-" * 40)
    
    for rel_path in iddaw_paths:
        full_path = os.path.join(base, rel_path)
        if os.path.exists(full_path):
            count = len(os.listdir(full_path))
            print(f"  EXISTS: {rel_path} ({count} files)")
            # Show a few sample files
            samples = sorted(os.listdir(full_path))[:3]
            for s in samples:
                print(f"    - {s}")
        else:
            print(f"  MISSING: {rel_path}")
    
    # 3. Check if IDD-AW exists at root level
    iddaw_root = os.path.join(base, 'IDD-AW')
    if os.path.exists(iddaw_root):
        print(f"\n3. Found IDD-AW at root level: {iddaw_root}")
        print("-" * 40)
        for item in sorted(os.listdir(iddaw_root)):
            path = os.path.join(iddaw_root, item)
            if os.path.isdir(path):
                print(f"  DIR: {item}/")
                try:
                    subs = sorted(os.listdir(path))[:5]
                    for sub in subs:
                        print(f"    - {sub}")
                except:
                    pass
    
    # 4. Check label values in IDD-AW labels
    print(f"\n4. IDD-AW label value analysis:")
    print("-" * 40)
    
    # Try different possible label paths
    label_paths_to_try = [
        os.path.join(base, 'train/labels/IDD-AW'),
        os.path.join(base, 'IDD-AW/train/labels'),
        os.path.join(base, 'IDD-AW/leftImg8bit/train'),
        os.path.join(base, 'IDD-AW/gtFine/train'),
    ]
    
    for label_dir in label_paths_to_try:
        if os.path.exists(label_dir):
            print(f"\n  Checking: {label_dir}")
            
            # Find label files
            label_files = []
            for root, dirs, files in os.walk(label_dir):
                for f in files:
                    if f.endswith('.png'):
                        label_files.append(os.path.join(root, f))
            
            if not label_files:
                print(f"    No PNG files found")
                continue
            
            print(f"    Found {len(label_files)} PNG files")
            
            # Check first few label files
            all_unique = set()
            for lf in label_files[:5]:
                try:
                    img = Image.open(lf)
                    arr = np.array(img)
                    unique = np.unique(arr)
                    all_unique.update(unique.tolist())
                    print(f"    {os.path.basename(lf)}: shape={arr.shape}, unique={sorted(unique)[:10]}...")
                except Exception as e:
                    print(f"    Error reading {lf}: {e}")
            
            print(f"\n    All unique values found: {sorted(all_unique)}")
            print(f"    Max value: {max(all_unique) if all_unique else 'N/A'}")
            print(f"    Values >= 19: {[v for v in all_unique if v >= 19 and v != 255]}")
    
    # 5. Check training config
    print(f"\n5. Generated training config:")
    print("-" * 40)
    
    config_path = '${AWARE_DATA_ROOT}/WEIGHTS/baseline/idd-aw_cd/deeplabv3plus_r50/configs/training_config.py'
    if os.path.exists(config_path):
        print(f"  Config exists: {config_path}")
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Extract key paths
        import re
        
        # Find data_root
        match = re.search(r"data_root\s*=\s*['\"]([^'\"]+)['\"]", content)
        if match:
            print(f"  data_root: {match.group(1)}")
        
        # Find img_dir and ann_dir
        for pattern in ['img_dir', 'ann_dir', 'train_img_dir', 'train_ann_dir']:
            matches = re.findall(rf"{pattern}\s*=\s*['\"]([^'\"]+)['\"]", content)
            for m in matches[:2]:
                print(f"  {pattern}: {m}")
        
        # Check for transforms
        if 'CityscapesLabelIdToTrainId' in content:
            print(f"  Transform: CityscapesLabelIdToTrainId is used")
        if 'ReduceToSingleChannel' in content:
            print(f"  Transform: ReduceToSingleChannel is used")
    else:
        print(f"  Config not found: {config_path}")
    
    print("\n" + "=" * 60)
    print("Investigation complete")
    print("=" * 60)

if __name__ == '__main__':
    main()
