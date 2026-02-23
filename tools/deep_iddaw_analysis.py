#!/usr/bin/env python3
"""
Deep investigation of IDD-AW label values.
"""

import os
import numpy as np
from PIL import Image
from collections import Counter

def main():
    base = '${AWARE_DATA_ROOT}/FINAL_SPLITS'
    label_dir = os.path.join(base, 'train/labels/IDD-AW')
    
    print("=" * 60)
    print("Deep IDD-AW Label Analysis")
    print("=" * 60)
    
    # Find all label files
    label_files = []
    for root, dirs, files in os.walk(label_dir):
        for f in files:
            if f.endswith('.png'):
                label_files.append(os.path.join(root, f))
    
    print(f"Total label files: {len(label_files)}")
    
    # Analyze all labels
    all_values = Counter()
    files_with_19 = []
    files_with_gt19 = []
    
    print("\nAnalyzing all labels...")
    
    for i, lf in enumerate(label_files):
        if i % 500 == 0:
            print(f"  Progress: {i}/{len(label_files)}")
        
        try:
            img = Image.open(lf)
            arr = np.array(img)
            
            # Count values
            for val in np.unique(arr):
                all_values[val] += 1
            
            # Check for problematic values
            if 19 in arr:
                files_with_19.append(lf)
            
            # Check for values > 19 (excluding 255 which is ignore)
            problematic = [v for v in np.unique(arr) if v > 19 and v != 255]
            if problematic:
                files_with_gt19.append((lf, problematic))
        except Exception as e:
            print(f"Error reading {lf}: {e}")
    
    print(f"\n\nResults:")
    print("-" * 40)
    print(f"All unique values across dataset:")
    for val in sorted(all_values.keys()):
        count = all_values[val]
        print(f"  {val:3d}: appears in {count} files")
    
    print(f"\nFiles with value 19: {len(files_with_19)}")
    if files_with_19:
        print(f"  Examples:")
        for f in files_with_19[:5]:
            print(f"    - {f}")
    
    print(f"\nFiles with values > 19 (excluding 255): {len(files_with_gt19)}")
    if files_with_gt19:
        print(f"  Examples:")
        for f, vals in files_with_gt19[:5]:
            print(f"    - {f}: {vals}")
    
    # Check shape of labels
    print("\n\nLabel shape analysis:")
    print("-" * 40)
    sample = label_files[0]
    img = Image.open(sample)
    arr = np.array(img)
    print(f"  Sample: {sample}")
    print(f"  Shape: {arr.shape}")
    print(f"  Mode: {img.mode}")
    print(f"  Dtype: {arr.dtype}")
    
    if len(arr.shape) == 3:
        print(f"\n  Label is 3-channel (RGB)!")
        print(f"  Channel 0 unique: {np.unique(arr[:,:,0])[:15]}...")
        print(f"  Channel 1 unique: {np.unique(arr[:,:,1])[:15]}...")
        print(f"  Channel 2 unique: {np.unique(arr[:,:,2])[:15]}...")
        
        # Check if channels are identical
        if np.array_equal(arr[:,:,0], arr[:,:,1]) and np.array_equal(arr[:,:,1], arr[:,:,2]):
            print(f"  All channels are identical!")
        else:
            print(f"  Channels differ!")
    
    # Check test labels too
    print("\n\nTest labels analysis:")
    print("-" * 40)
    test_label_dir = os.path.join(base, 'test/labels/IDD-AW')
    test_files = []
    for root, dirs, files in os.walk(test_label_dir):
        for f in files:
            if f.endswith('.png'):
                test_files.append(os.path.join(root, f))
    
    print(f"Total test label files: {len(test_files)}")
    
    test_values = set()
    for lf in test_files[:100]:
        img = Image.open(lf)
        arr = np.array(img)
        test_values.update(np.unique(arr).tolist())
    
    print(f"Unique values in test (first 100 files): {sorted(test_values)}")
    print(f"Values >= 19 (excluding 255): {[v for v in test_values if v >= 19 and v != 255]}")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
