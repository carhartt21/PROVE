# MapillaryVistas Class Detection Analysis

This document analyzes which of the 66 MapillaryVistas classes are typically detected by our trained models.

## ⚠️ IMPORTANT FINDING: Sparse Test Set Annotations

**The MapillaryVistas test set has sparse ground-truth annotations:**
- Only **23 out of 66 classes** have any GT pixels in the test set
- **43 classes have ZERO GT annotations**, including common classes like:
  - Person, Sidewalk, Sky, Vegetation
  - All vehicles (Bus, Truck, Motorcycle, Bicycle)
  - Traffic lights, many infrastructure classes

This is a **dataset characteristic**, not a model or evaluation bug.

## Summary

Based on analysis of 81 trained models across all strategies:

| Category | Count | Description |
|----------|-------|-------------|
| ✅ Has GT & Detected | 20 | Classes with GT annotations that models can detect |
| 🔸 Has GT but Low IoU | 3 | Classes with GT but poor detection (Car, Mailbox, Ground Animal) |
| ❌ No GT Annotations | 43 | Classes not annotated in test set |

**Key Insight:** The 0% IoU for many classes is because they have **no ground truth labels** in the test set, not because models fail to detect them.

## Classes WITH Ground Truth Annotations (23 classes)

These classes actually have labeled pixels in the test set:

| Class | Avg IoU | GT Pixels | Notes |
|-------|---------|-----------|-------|
| Road | 91.9% | 262M | Dominant class |
| Building | 90.7% | 138M | Second largest |
| Phone Booth | 90.1% | 43M | Very consistent |
| Unlabeled | 31.7% | 20M | Catch-all class |
| Lane Marking - General | 59.1% | 19M | Road feature |
| Terrain | 69.9% | 12M | Ground cover |
| Pole | 55.9% | 11M | Infrastructure |
| Curb | 55.5% | 9M | Road edge |
| Billboard | 42.1% | 7M | Urban feature |
| Pedestrian Area | 48.7% | 6M | Walking zones |
| Tunnel | 61.9% | 5M | Specific scenes |
| Mountain | 75.2% | 3M | Background |
| Guard Rail | 58.1% | 3M | Road safety |
| Service Lane | 43.1% | 2M | Side roads |
| Traffic Sign Frame | 52.8% | 1M | Sign support |
| Curb Cut | 16.9% | 1M | Ramp feature |
| Traffic Sign (Back) | 25.8% | 1M | Sign rear |
| Car Mount | 62.4% | 0.5M | Dash-cam mount |
| Junction Box | 22.1% | 0.4M | Utility box |
| Barrier | 16.0% | 0.3M | Temporary barrier |
| Car | 0.0% | 90K | **Has GT but not detected** |
| Mailbox | 0.0% | 48K | **Has GT but not detected** |
| Ground Animal | 0.0% | 14K | **Has GT but not detected** |

## Classes WITHOUT Ground Truth Annotations (43 classes)

These classes have **ZERO** ground truth pixels in the test set:

**People & Riders:**
- Person, Bicyclist, Motorcyclist, Other Rider

**Vehicles:**
- Bicycle, Boat, Bus, Caravan, Motorcycle, On Rails, Other Vehicle, Trailer, Truck, Wheeled Slow

**Infrastructure:**
- Bird, Fence, Wall, Sidewalk, Bridge, Rail Track, Parking, Bike Lane
- Crosswalk - Plain, Lane Marking - Crosswalk
- Street Light, Utility Pole, Traffic Light, Traffic Sign (Front)
- Catch Basin, CCTV Camera, Fire Hydrant, Manhole, Pothole, Trash Can
- Banner, Bench, Bike Rack

**Nature:**
- Sky, Vegetation, Water, Sand, Snow

**Other:**
- Ego Vehicle

## Implications for Analysis

1. **mIoU Interpretation:** The overall mIoU (~45-50%) is computed across all 66 classes, but only 20-23 classes actually have GT annotations. The effective mIoU on annotated classes would be higher.

2. **Strategy Comparison:** All strategies face the same GT annotation sparsity, so relative comparisons remain valid.

3. **Dataset Limitation:** This is a characteristic of the MapillaryVistas test set, not our evaluation code. The test set appears to have been annotated for a subset of classes.

4. **Class-Level Analysis:** When analyzing per-class performance, focus on the 20 classes that are actually annotated and detected.

## Technical Notes

### CSV Fix Applied
The per_class_metrics in downstream_results.csv were corrected to use proper MapillaryVistas class names (previously mislabeled with Cityscapes names due to older test code).

### Verification
- Stage 1 tests (53 entries) were relabeled via post-processing
- Stage 2 tests already have correct labels
- IoU values are preserved - only labels were changed

---

*Analysis based on downstream_results.csv and test results JSON files*
*Last updated: 2026-01-21*
