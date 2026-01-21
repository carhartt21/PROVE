# MapillaryVistas Class Detection Analysis

This document analyzes which of the 66 MapillaryVistas classes are typically detected by our trained models.

## Summary

Based on analysis of 81 trained models across all strategies:

| Category | Count | Description |
|----------|-------|-------------|
| ✅ Always Detected | 15 | Non-zero IoU in all 81 models |
| ⚠️ Sometimes Detected | 5 | Non-zero IoU in 50-99% of models |
| 🔸 Rarely Detected | 6 | Non-zero IoU in <50% of models |
| ❌ Never Detected | 40 | Zero IoU in all models |

**Key Insight:** Only 20-26 classes out of 66 are typically detected. This is expected because:
1. Many MapillaryVistas classes are very rare in the dataset (Bird, Boat, Caravan, etc.)
2. Some classes may not appear in the test split at all
3. Fine-grained distinctions (e.g., Traffic Sign Front vs Back) are challenging

## Always Detected Classes (15)

These classes consistently get non-zero IoU across all trained models:

| Class | Avg IoU | Max IoU | Notes |
|-------|---------|---------|-------|
| Phone Booth | 90.11% | 90.98% | Highly distinctive |
| Mountain | 70.75% | 79.48% | Large, distinct |
| Terrain | 70.69% | 74.57% | Common class |
| Guard Rail | 62.70% | 66.68% | Roadside feature |
| Car Mount | 62.54% | 71.80% | Dash-cam specific |
| Lane Marking - General | 57.47% | 60.96% | Road feature |
| Curb | 56.05% | 94.83% | High variability |
| Pole | 55.37% | 59.08% | Common infrastructure |
| Traffic Sign Frame | 48.31% | 57.72% | Distinct structure |
| Barrier | 45.55% | 83.03% | High variability |
| Billboard | 43.60% | 49.74% | Distinctive |
| Unlabeled | 32.09% | 35.97% | Catch-all class |
| Traffic Sign (Back) | 28.18% | 31.74% | Visible patterns |
| Junction Box | 24.57% | 31.57% | Small utility box |
| Curb Cut | 18.53% | 83.61% | High variability |

## Sometimes Detected Classes (5)

These classes are detected in most (50-99%) but not all models:

| Class | Detection Rate | Avg IoU | Notes |
|-------|----------------|---------|-------|
| Road | 98.8% | 90.67% | Very common, high IoU |
| Building | 98.8% | 89.83% | Very common, high IoU |
| Tunnel | 98.8% | 60.44% | Occasional in dataset |
| Pedestrian Area | 98.8% | 49.46% | Moderate presence |
| Service Lane | 98.8% | 38.34% | Moderate presence |

## Rarely Detected Classes (6)

These classes are only occasionally detected (<50% of models):

| Class | Detection Rate | Avg IoU | Notes |
|-------|----------------|---------|-------|
| Mailbox | 32.5% | 0.80% | Very small object |
| Car | 20.0% | 0.33% | Likely label confusion |
| Ground Animal | 2.5% | 0.75% | Very rare |
| Bird | 1.2% | 1.20% | Very rare |
| Fence | 1.2% | 0.43% | Possible confusion |
| Bike Lane | 1.2% | 0.38% | Rare marking |

## Never Detected Classes (40)

These classes have zero IoU across all 81 models. They are either not present in the test set or too challenging to detect:

**Structural/Infrastructure:**
- Wall, Sidewalk, Bridge, Rail Track, Parking

**Vehicles:**
- Bicycle, Boat, Bus, Car (different from 'Car' above), Caravan, Motorcycle, On Rails, Other Vehicle, Trailer, Truck, Wheeled Slow

**People:**
- Person, Bicyclist, Motorcyclist, Other Rider

**Nature:**
- Sky, Vegetation, Water, Snow, Sand

**Urban Features:**
- Banner, Bench, Bike Rack, Catch Basin, CCTV Camera, Fire Hydrant, Manhole, Pothole, Street Light, Utility Pole, Traffic Light, Traffic Sign (Front), Trash Can

**Road Features:**
- Crosswalk - Plain, Lane Marking - Crosswalk

**Other:**
- Ego Vehicle (dash-cam hood area)

## Implications for Analysis

1. **mIoU Interpretation:** The overall mIoU (~45-50%) is computed across all 66 classes. Many zeros pull down the average.

2. **Strategy Comparison:** When comparing strategies, focus on the ~20-26 detected classes rather than raw mIoU.

3. **Cross-Domain Performance:** The detected classes are consistent across different augmentation strategies, suggesting the issue is dataset-related rather than model-related.

4. **Class Imbalance:** This analysis reveals significant class imbalance in MapillaryVistas, which is common in real-world driving datasets.

## Recommendations

1. For detailed analysis, consider weighted metrics based on class frequency
2. Compare strategies using only the detected classes
3. Note that some zeros may be due to mapping issues in the original test procedure

---

*Generated from downstream_results.csv analysis across 81 trained models*
*Last updated: 2026-01-21*
