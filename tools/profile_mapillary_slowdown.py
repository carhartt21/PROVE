#!/usr/bin/env python3
"""
Profile script to identify the MapillaryVistas testing slowdown.

This script isolates and times individual components to find the bottleneck.

Usage:
    conda activate mmseg
    python tools/profile_mapillary_slowdown.py
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


def time_function(func, *args, iterations=10, **kwargs):
    """Time a function over multiple iterations."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return np.mean(times) * 1000, np.std(times) * 1000, result


def decode_rgb_current_method(gt_seg_map):
    """Current implementation - iterates over all 66 colors."""
    if gt_seg_map.ndim == 3 and gt_seg_map.shape[-1] == 3:
        h, w = gt_seg_map.shape[:2]
        native_labels = np.full((h, w), 255, dtype=np.uint8)
        
        r = gt_seg_map[:, :, 2].astype(np.int32)
        g = gt_seg_map[:, :, 1].astype(np.int32)
        b = gt_seg_map[:, :, 0].astype(np.int32)
        packed = r * 65536 + g * 256 + b
        
        # Build lookup
        rgb_lookup = {}
        for rgb, class_id in custom_transforms.MAPILLARY_RGB_TO_ID.items():
            packed_rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
            rgb_lookup[packed_rgb] = class_id
        
        # This is the slow part - iterates 66 times
        for packed_rgb, class_id in rgb_lookup.items():
            mask = packed == packed_rgb
            native_labels[mask] = class_id
        
        return native_labels
    return gt_seg_map[:, :, 0] if gt_seg_map.ndim == 3 else gt_seg_map


def decode_rgb_lut_method(gt_seg_map, lut_table):
    """Optimized implementation using pre-built lookup table (LUT).
    
    Uses a single array lookup instead of 66 iterations.
    Since packed RGB values can be large (up to 16M), we use a hash-based approach.
    """
    if gt_seg_map.ndim == 3 and gt_seg_map.shape[-1] == 3:
        h, w = gt_seg_map.shape[:2]
        
        r = gt_seg_map[:, :, 2].astype(np.int32)
        g = gt_seg_map[:, :, 1].astype(np.int32)
        b = gt_seg_map[:, :, 0].astype(np.int32)
        packed = r * 65536 + g * 256 + b
        
        # Use vectorized lookup via numpy unique + mapping
        unique_values, inverse = np.unique(packed, return_inverse=True)
        result_map = np.array([lut_table.get(v, 255) for v in unique_values], dtype=np.uint8)
        native_labels = result_map[inverse].reshape(h, w)
        
        return native_labels
    return gt_seg_map[:, :, 0] if gt_seg_map.ndim == 3 else gt_seg_map


def decode_rgb_vectorized_method(gt_seg_map, packed_to_class):
    """Vectorized method using pandas or custom vectorization."""
    if gt_seg_map.ndim == 3 and gt_seg_map.shape[-1] == 3:
        h, w = gt_seg_map.shape[:2]
        
        r = gt_seg_map[:, :, 2].astype(np.int64)
        g = gt_seg_map[:, :, 1].astype(np.int64)
        b = gt_seg_map[:, :, 0].astype(np.int64)
        packed = r * 65536 + g * 256 + b
        
        # Pre-compute all valid packed values and their class IDs
        # This creates a mapping array indexed by packed value
        native_labels = np.full((h, w), 255, dtype=np.uint8)
        
        # Vectorized: find which pixels match any of our colors
        flat_packed = packed.ravel()
        for packed_rgb, class_id in packed_to_class.items():
            mask_flat = flat_packed == packed_rgb
            native_labels.ravel()[mask_flat] = class_id
        
        return native_labels
    return gt_seg_map[:, :, 0] if gt_seg_map.ndim == 3 else gt_seg_map


def decode_rgb_direct_lookup(gt_seg_map, lut_24bit):
    """Direct 24-bit lookup table (most memory, fastest).
    
    Pre-allocates a 16M entry table mapping all possible 24-bit RGB values.
    Memory usage: 16MB for uint8 LUT
    """
    if gt_seg_map.ndim == 3 and gt_seg_map.shape[-1] == 3:
        # Pack RGB directly (assuming BGR from OpenCV)
        r = gt_seg_map[:, :, 2].astype(np.int32)
        g = gt_seg_map[:, :, 1].astype(np.int32)
        b = gt_seg_map[:, :, 0].astype(np.int32)
        packed = r * 65536 + g * 256 + b
        
        # Direct array indexing - O(1) per pixel
        native_labels = lut_24bit[packed]
        
        return native_labels
    return gt_seg_map[:, :, 0] if gt_seg_map.ndim == 3 else gt_seg_map


def build_lut_dict():
    """Build dictionary lookup table."""
    lut = {}
    for rgb, class_id in custom_transforms.MAPILLARY_RGB_TO_ID.items():
        packed_rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
        lut[packed_rgb] = class_id
    return lut


def build_lut_24bit():
    """Build 24-bit direct lookup table (16MB)."""
    lut = np.full(256**3, 255, dtype=np.uint8)
    for rgb, class_id in custom_transforms.MAPILLARY_RGB_TO_ID.items():
        packed_rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
        lut[packed_rgb] = class_id
    return lut


def load_bdd10k_label():
    """Load a sample BDD10k label."""
    label_path = Path("${AWARE_DATA_ROOT}/FINAL_SPLITS/test/labels/BDD10k/clear_day")
    labels = list(label_path.glob("*.png"))
    if labels:
        return cv2.imread(str(labels[0]), cv2.IMREAD_UNCHANGED)
    return None


def load_mapillary_label():
    """Load a sample MapillaryVistas label."""
    label_path = Path("${AWARE_DATA_ROOT}/FINAL_SPLITS/test/labels/MapillaryVistas/clear_day")
    labels = list(label_path.glob("*.png"))
    if labels:
        return cv2.imread(str(labels[0]), cv2.IMREAD_UNCHANGED)
    return None


def profile_io_operations():
    """Profile I/O operations for both datasets."""
    print("\n" + "="*60)
    print("I/O PROFILING")
    print("="*60)
    
    # BDD10k
    bdd_img_path = Path("${AWARE_DATA_ROOT}/FINAL_SPLITS/test/images/BDD10k/clear_day")
    bdd_label_path = Path("${AWARE_DATA_ROOT}/FINAL_SPLITS/test/labels/BDD10k/clear_day")
    
    # MapillaryVistas
    mv_img_path = Path("${AWARE_DATA_ROOT}/FINAL_SPLITS/test/images/MapillaryVistas/clear_day")
    mv_label_path = Path("${AWARE_DATA_ROOT}/FINAL_SPLITS/test/labels/MapillaryVistas/clear_day")
    
    bdd_imgs = list(bdd_img_path.glob("*.jpg")) + list(bdd_img_path.glob("*.png"))
    mv_imgs = list(mv_img_path.glob("*.jpg")) + list(mv_img_path.glob("*.png"))
    
    if bdd_imgs:
        sample_img = bdd_imgs[0]
        sample_label = bdd_label_path / (sample_img.stem + ".png")
        
        mean_ms, std_ms, _ = time_function(cv2.imread, str(sample_img))
        print(f"BDD10k image load: {mean_ms:.2f} ± {std_ms:.2f} ms")
        
        mean_ms, std_ms, _ = time_function(cv2.imread, str(sample_label), cv2.IMREAD_UNCHANGED)
        print(f"BDD10k label load: {mean_ms:.2f} ± {std_ms:.2f} ms")
    
    if mv_imgs:
        sample_img = mv_imgs[0]
        sample_label = mv_label_path / (sample_img.stem + ".png")
        
        mean_ms, std_ms, _ = time_function(cv2.imread, str(sample_img))
        print(f"MapillaryVistas image load: {mean_ms:.2f} ± {std_ms:.2f} ms")
        
        mean_ms, std_ms, _ = time_function(cv2.imread, str(sample_label), cv2.IMREAD_UNCHANGED)
        print(f"MapillaryVistas label load: {mean_ms:.2f} ± {std_ms:.2f} ms")


def main():
    print("="*60)
    print("MAPILLARY VISTAS TESTING SLOWDOWN PROFILER")
    print("="*60)
    
    # Profile I/O first
    profile_io_operations()
    
    # Load sample labels
    print("\n" + "="*60)
    print("LABEL DECODING PROFILING")
    print("="*60)
    
    bdd_label = load_bdd10k_label()
    mv_label = load_mapillary_label()
    
    if bdd_label is not None:
        print(f"\nBDD10k label shape: {bdd_label.shape}")
    
    if mv_label is not None:
        print(f"MapillaryVistas label shape: {mv_label.shape}")
    
    # Pre-build lookup tables
    print("\nBuilding lookup tables...")
    lut_dict = build_lut_dict()
    lut_24bit = build_lut_24bit()
    print(f"  Dictionary LUT: {len(lut_dict)} entries")
    print(f"  24-bit LUT: {lut_24bit.nbytes / 1024 / 1024:.1f} MB")
    
    # Profile different decode methods on MapillaryVistas
    if mv_label is not None:
        print("\n" + "-"*40)
        print("MapillaryVistas RGB decoding methods:")
        print("-"*40)
        
        # Current method (slow)
        mean_ms, std_ms, result1 = time_function(decode_rgb_current_method, mv_label, iterations=5)
        print(f"  Current (iterate 66x): {mean_ms:.2f} ± {std_ms:.2f} ms")
        
        # LUT method with np.unique
        mean_ms, std_ms, result2 = time_function(decode_rgb_lut_method, mv_label, lut_dict, iterations=10)
        print(f"  np.unique + lookup:    {mean_ms:.2f} ± {std_ms:.2f} ms")
        
        # Direct 24-bit LUT (fastest)
        mean_ms, std_ms, result3 = time_function(decode_rgb_direct_lookup, mv_label, lut_24bit, iterations=10)
        print(f"  Direct 24-bit LUT:     {mean_ms:.2f} ± {std_ms:.2f} ms")
        
        # Verify results match
        print("\n" + "-"*40)
        print("Verification (results should match):")
        print("-"*40)
        print(f"  Current vs np.unique: {np.array_equal(result1, result2)}")
        print(f"  Current vs 24-bit:    {np.array_equal(result1, result3)}")
        
        # Check unique classes found
        print(f"\n  Unique classes found: {np.unique(result1[result1 != 255])}")
    
    # Also profile BDD10k for comparison (simple decode)
    if bdd_label is not None:
        print("\n" + "-"*40)
        print("BDD10k label processing:")
        print("-"*40)
        
        def bdd_decode(label):
            return label[:, :, 0] if label.ndim == 3 else label
        
        mean_ms, std_ms, _ = time_function(bdd_decode, bdd_label, iterations=20)
        print(f"  Simple channel extract: {mean_ms:.2f} ± {std_ms:.2f} ms")
    
    print("\n" + "="*60)
    print("CONCLUSIONS")
    print("="*60)
    print("""
The current implementation iterates 66 times over the entire label image,
creating a boolean mask and assigning values for each class. This is O(66*n)
where n = number of pixels.

The 24-bit direct LUT method uses O(16MB) memory but provides O(n) decoding
with a single array lookup, which should be ~66x faster for the label
decoding step.
""")


if __name__ == "__main__":
    main()
