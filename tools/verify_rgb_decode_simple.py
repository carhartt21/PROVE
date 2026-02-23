#!/usr/bin/env python3
"""
Simple verification of MapillaryVistas RGB decoding optimization.
No dependencies on cv2 - uses synthetic test data.
"""

import sys
import time
from pathlib import Path
import numpy as np

# Add project root to path
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir.parent))

import custom_transforms


def create_test_image():
    """Create a synthetic RGB label image with known Mapillary colors."""
    # Pick a few colors from the mapping
    colors = [
        ((165, 42, 42), 0),      # Bird
        ((128, 64, 128), 13),    # Road
        ((244, 35, 232), 15),    # Sidewalk
        ((70, 70, 70), 17),      # Building
        ((220, 20, 60), 19),     # Person
        ((0, 0, 142), 55),       # Car
        ((70, 130, 180), 27),    # Sky
    ]
    
    # Create 512x512 test image
    h, w = 512, 512
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Fill with stripes of different classes
    stripe_width = w // len(colors)
    expected = np.full((h, w), 255, dtype=np.uint8)
    
    for i, (rgb, class_id) in enumerate(colors):
        x1 = i * stripe_width
        x2 = (i + 1) * stripe_width if i < len(colors) - 1 else w
        # Note: img is in RGB order for this test
        img[:, x1:x2, 0] = rgb[0]  # R
        img[:, x1:x2, 1] = rgb[1]  # G
        img[:, x1:x2, 2] = rgb[2]  # B
        expected[:, x1:x2] = class_id
    
    return img, expected


def old_decode_method(gt_seg_map):
    """Original slow method - iterates 66 times."""
    h, w = gt_seg_map.shape[:2]
    native_labels = np.full((h, w), 255, dtype=np.uint8)
    
    # For this test, input is RGB order (not BGR)
    r = gt_seg_map[:, :, 0].astype(np.int32)
    g = gt_seg_map[:, :, 1].astype(np.int32)
    b = gt_seg_map[:, :, 2].astype(np.int32)
    packed = r * 65536 + g * 256 + b
    
    rgb_lookup = {}
    for rgb, class_id in custom_transforms.MAPILLARY_RGB_TO_ID.items():
        packed_rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
        rgb_lookup[packed_rgb] = class_id
    
    for packed_rgb, class_id in rgb_lookup.items():
        mask = packed == packed_rgb
        native_labels[mask] = class_id
    
    return native_labels


def new_decode_method(gt_seg_map, lut_24bit):
    """New optimized method - direct LUT."""
    r = gt_seg_map[:, :, 0].astype(np.int32)
    g = gt_seg_map[:, :, 1].astype(np.int32)
    b = gt_seg_map[:, :, 2].astype(np.int32)
    packed = r * 65536 + g * 256 + b
    return lut_24bit[packed]


def build_lut_24bit():
    """Build 24-bit direct lookup table."""
    lut = np.full(256**3, 255, dtype=np.uint8)
    for rgb, class_id in custom_transforms.MAPILLARY_RGB_TO_ID.items():
        packed_rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
        lut[packed_rgb] = class_id
    return lut


def time_function(func, *args, iterations=10, **kwargs):
    """Time a function."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return np.mean(times) * 1000, np.std(times) * 1000, result


def main():
    print("="*60)
    print("MapillaryVistas RGB Decoding Verification (Synthetic Data)")
    print("="*60)
    
    # Create test data
    print("\nCreating synthetic test image...")
    test_img, expected = create_test_image()
    print(f"Image shape: {test_img.shape}")
    print(f"Image size: {test_img.shape[0] * test_img.shape[1]:,} pixels")
    
    # Build LUT
    print("\nBuilding 24-bit LUT...")
    start = time.perf_counter()
    lut_24bit = build_lut_24bit()
    lut_time = (time.perf_counter() - start) * 1000
    print(f"  LUT build time: {lut_time:.2f} ms (one-time cost)")
    print(f"  LUT memory: {lut_24bit.nbytes / 1024 / 1024:.1f} MB")
    
    # Test old method
    print("\n" + "-"*40)
    print("OLD METHOD (66 iterations):")
    old_mean, old_std, old_result = time_function(old_decode_method, test_img, iterations=10)
    print(f"  Time: {old_mean:.2f} ± {old_std:.2f} ms")
    
    # Test new method
    print("\nNEW METHOD (24-bit LUT):")
    new_mean, new_std, new_result = time_function(new_decode_method, test_img, lut_24bit, iterations=20)
    print(f"  Time: {new_mean:.2f} ± {new_std:.2f} ms")
    
    # Verify correctness
    print("\n" + "-"*40)
    print("VERIFICATION:")
    
    # Check old method produces expected result
    old_correct = np.array_equal(old_result, expected)
    print(f"  Old method correct: {'✅ YES' if old_correct else '❌ NO'}")
    
    # Check new method produces expected result
    new_correct = np.array_equal(new_result, expected)
    print(f"  New method correct: {'✅ YES' if new_correct else '❌ NO'}")
    
    # Check both methods match
    methods_match = np.array_equal(old_result, new_result)
    print(f"  Methods match: {'✅ YES' if methods_match else '❌ NO'}")
    
    if not methods_match:
        diff_count = np.sum(old_result != new_result)
        print(f"  Differing pixels: {diff_count}")
    
    # Speedup
    speedup = old_mean / new_mean if new_mean > 0 else float('inf')
    print(f"\n  Speedup: {speedup:.1f}x")
    
    # Estimate real-world impact
    print("\n" + "-"*40)
    print("ESTIMATED REAL-WORLD IMPACT:")
    num_images = 4949  # MapillaryVistas test set
    old_total = num_images * old_mean / 1000  # seconds
    new_total = num_images * new_mean / 1000  # seconds
    print(f"  Old decode time ({num_images} images): {old_total:.1f}s ({old_total/60:.1f} min)")
    print(f"  New decode time ({num_images} images): {new_total:.1f}s ({new_total/60:.2f} min)")
    print(f"  Time saved per test run: {(old_total - new_total)/60:.1f} min")
    
    print("\n" + "="*60)
    if old_correct and new_correct and methods_match and speedup > 1.5:
        print("✅ VERIFICATION PASSED")
        print(f"   - Both methods produce correct results")
        print(f"   - New method is {speedup:.1f}x faster")
        print(f"   - Note: Actual speedup varies by CPU/cache (1.5-66x range)")
        return 0
    else:
        print("❌ VERIFICATION FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
