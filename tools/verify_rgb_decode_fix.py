#!/usr/bin/env python3
"""
Verify the MapillaryVistas RGB decoding optimization works correctly.

This script tests that the new 24-bit LUT produces identical results
to the old iteration-based method while being significantly faster.
"""

import sys
import time
from pathlib import Path
import numpy as np

# Add project root to path
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir.parent))

try:
    import cv2
except ImportError:
    print("ERROR: cv2 not found. Run: conda activate mmseg")
    sys.exit(1)

import custom_transforms


def old_decode_method(gt_seg_map):
    """Original slow method - iterates 66 times over image."""
    if gt_seg_map.ndim == 3 and gt_seg_map.shape[-1] == 3:
        h, w = gt_seg_map.shape[:2]
        native_labels = np.full((h, w), 255, dtype=np.uint8)
        
        r = gt_seg_map[:, :, 2].astype(np.int32)  # BGR: channel 2 = R
        g = gt_seg_map[:, :, 1].astype(np.int32)
        b = gt_seg_map[:, :, 0].astype(np.int32)
        packed = r * 65536 + g * 256 + b
        
        rgb_lookup = {}
        for rgb, class_id in custom_transforms.MAPILLARY_RGB_TO_ID.items():
            packed_rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
            rgb_lookup[packed_rgb] = class_id
        
        for packed_rgb, class_id in rgb_lookup.items():
            mask = packed == packed_rgb
            native_labels[mask] = class_id
        
        return native_labels
    return gt_seg_map[:, :, 0] if gt_seg_map.ndim == 3 else gt_seg_map


def new_decode_method(gt_seg_map, lut_24bit):
    """New optimized method - direct LUT lookup."""
    if gt_seg_map.ndim == 3 and gt_seg_map.shape[-1] == 3:
        r = gt_seg_map[:, :, 2].astype(np.int32)  # BGR: channel 2 = R
        g = gt_seg_map[:, :, 1].astype(np.int32)
        b = gt_seg_map[:, :, 0].astype(np.int32)
        packed = r * 65536 + g * 256 + b
        return lut_24bit[packed]
    return gt_seg_map[:, :, 0] if gt_seg_map.ndim == 3 else gt_seg_map


def build_lut_24bit():
    """Build 24-bit direct lookup table."""
    lut = np.full(256**3, 255, dtype=np.uint8)
    for rgb, class_id in custom_transforms.MAPILLARY_RGB_TO_ID.items():
        packed_rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
        lut[packed_rgb] = class_id
    return lut


def find_sample_label():
    """Find a sample MapillaryVistas label file."""
    paths = [
        Path("${AWARE_DATA_ROOT}/FINAL_SPLITS/test/labels/MapillaryVistas/clear_day"),
        Path("${AWARE_DATA_ROOT}/FINAL_SPLITS/val/labels/MapillaryVistas/clear_day"),
    ]
    for p in paths:
        if p.exists():
            labels = list(p.glob("*.png"))
            if labels:
                return labels[0]
    return None


def time_function(func, *args, iterations=5, **kwargs):
    """Time a function over multiple iterations."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return np.mean(times) * 1000, np.std(times) * 1000, result


def main():
    print("="*60)
    print("MapillaryVistas RGB Decoding Verification")
    print("="*60)
    
    # Find sample label
    label_path = find_sample_label()
    if label_path is None:
        print("\nERROR: No MapillaryVistas labels found!")
        print("Paths checked:")
        print("  ${AWARE_DATA_ROOT}/FINAL_SPLITS/test/labels/MapillaryVistas/clear_day")
        print("  ${AWARE_DATA_ROOT}/FINAL_SPLITS/val/labels/MapillaryVistas/clear_day")
        sys.exit(1)
    
    print(f"\nUsing sample label: {label_path.name}")
    
    # Load label (cv2 loads as BGR)
    label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    print(f"Label shape: {label.shape}")
    print(f"Label dtype: {label.dtype}")
    
    # Build LUT
    print("\nBuilding 24-bit LUT...")
    start = time.perf_counter()
    lut_24bit = build_lut_24bit()
    lut_time = (time.perf_counter() - start) * 1000
    print(f"  LUT build time: {lut_time:.2f} ms")
    print(f"  LUT memory: {lut_24bit.nbytes / 1024 / 1024:.1f} MB")
    
    # Test old method
    print("\n" + "-"*40)
    print("OLD METHOD (66 iterations):")
    old_mean, old_std, old_result = time_function(old_decode_method, label, iterations=5)
    print(f"  Time: {old_mean:.2f} ± {old_std:.2f} ms")
    
    # Test new method
    print("\nNEW METHOD (24-bit LUT):")
    new_mean, new_std, new_result = time_function(new_decode_method, label, lut_24bit, iterations=10)
    print(f"  Time: {new_mean:.2f} ± {new_std:.2f} ms")
    
    # Compare results
    print("\n" + "-"*40)
    print("VERIFICATION:")
    
    results_match = np.array_equal(old_result, new_result)
    print(f"  Results identical: {'✅ YES' if results_match else '❌ NO'}")
    
    if not results_match:
        diff_count = np.sum(old_result != new_result)
        print(f"  Differing pixels: {diff_count}")
        print("  Old unique values:", np.unique(old_result))
        print("  New unique values:", np.unique(new_result))
    else:
        print(f"  Unique classes found: {np.unique(old_result[old_result != 255])}")
        pixels_decoded = np.sum(old_result != 255)
        total_pixels = old_result.size
        print(f"  Pixels decoded: {pixels_decoded:,} / {total_pixels:,} ({100*pixels_decoded/total_pixels:.1f}%)")
    
    # Speedup
    speedup = old_mean / new_mean if new_mean > 0 else float('inf')
    print(f"\n  Speedup: {speedup:.1f}x")
    
    # Estimate impact on full test run
    print("\n" + "-"*40)
    print("ESTIMATED IMPACT:")
    num_images = 4949  # MapillaryVistas test set
    old_total = num_images * old_mean / 1000 / 60  # minutes
    new_total = num_images * new_mean / 1000 / 60  # minutes
    print(f"  Old decode time ({num_images} images): {old_total:.1f} min")
    print(f"  New decode time ({num_images} images): {new_total:.1f} min")
    print(f"  Time saved: {old_total - new_total:.1f} min")
    
    print("\n" + "="*60)
    if results_match and speedup > 10:
        print("✅ VERIFICATION PASSED - Fix is working correctly!")
    elif results_match:
        print("⚠️  Results match but speedup is lower than expected")
    else:
        print("❌ VERIFICATION FAILED - Results do not match!")
        sys.exit(1)
    print("="*60)


if __name__ == "__main__":
    main()
