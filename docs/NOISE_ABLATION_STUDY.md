# Noise Ablation Study

> **Purpose:** Determine whether segmentation models trained with generative data augmentation learn from actual image content or merely memorize spatial layouts of segmentation maps.

**Last Updated:** 2025-02-09

## Motivation

When generative augmentation strategies (e.g., CycleGAN, Flux Kontext, Step1X) produce synthetic weather images alongside their original segmentation labels, the training pipeline provides the model with:

1. **Novel pixel data** — generated images depicting adverse weather conditions
2. **Unchanged label masks** — segmentation maps copied from the original clear-day images

A critical question arises: *does the model actually learn from the visual content of generated images, or does the performance gain come solely from seeing the same segmentation layouts paired with any pixel data?*

If the model performs similarly when trained with **random noise** paired with the same labels, it would suggest:
- Generated image quality is irrelevant — the benefit comes from label layout diversity
- The additional segmentation maps alone act as a regularizer

If noise training produces **significantly worse** results than generative strategies, it confirms:
- Models genuinely learn weather-domain visual features from generated images
- Image quality and realism matter for generative data augmentation

## Method

### Training Protocol

| Parameter | Value |
|-----------|-------|
| Strategy | `gen_random_noise` |
| Reference manifest | `cycleGAN` (manifest.csv) |
| Real:Gen ratio | 0.5 (same as other gen_ strategies) |
| Noise type | Uniform random [0, 255] per pixel |
| Domain filter | `clear_day` (Stage 1 protocol) |
| Max iterations | 15,000 |
| Datasets | BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k |
| Models | pspnet_r50, segformer_mit-b3, segnext_mscan-b, mask2former_swin-b |
| Weights directory | `WEIGHTS_NOISE_ABLATION/` |

### Mechanism

1. **Entry enumeration:** Load entries from the CycleGAN manifest (CSV), filtered by target dataset. This provides the same number of additional training entries as a CycleGAN-augmented run.

2. **Image loading:** For each generated entry, load the **original clear-day image** (not the CycleGAN output). This ensures a valid image exists on disk for determining spatial dimensions.

3. **Noise replacement:** After `LoadImageFromFile`, the `ReplaceWithNoise` transform checks the `_replace_with_noise` flag. Flagged samples have their pixel data replaced with uniform random noise `U(0, 255)` of the same shape (H, W, C). Real images pass through unchanged.

4. **Label preservation:** Segmentation labels are loaded normally from the original label path — identical to what CycleGAN-augmented training uses.

### Key Design Decisions

- **Uniform noise** was chosen over Gaussian noise because it maximizes entropy and provides no structural information whatsoever. This creates the strongest possible "null" condition.
- **CycleGAN as reference** ensures the same entry count and label pool as a real generative strategy, making direct comparison valid.
- **Per-sample flagging** (`_replace_with_noise`) ensures only generated entries get noise replacement — real clear-day images in the batch remain untouched.

## Implementation

### Files Modified

| File | Change |
|------|--------|
| `custom_transforms.py` | Added `ReplaceWithNoise` transform (registered with MMSeg TRANSFORMS) |
| `unified_training_config.py` | Registered `gen_random_noise` strategy with type `noise_ablation`, added validation and config generation |
| `unified_training.py` | Modified mixed training script to handle noise ablation: inject transform, use original paths, set noise flags |
| `scripts/batch_training_submission.py` | Added `noise-ablation` stage, `WEIGHTS_NOISE_ABLATION` path, domain filter, manifest bypass |

### Batch Submission

```bash
# Always dry-run first
python scripts/batch_training_submission.py --stage noise-ablation --dry-run

# Submit all noise ablation jobs (32 total: 16 noise + 16 baseline)
python scripts/batch_training_submission.py --stage noise-ablation

# Submit only gen_random_noise (no baseline comparison)
python scripts/batch_training_submission.py --stage noise-ablation --strategies gen_random_noise

# Limit to specific dataset/model for testing
python scripts/batch_training_submission.py --stage noise-ablation \
    --datasets BDD10k --models pspnet_r50 --dry-run
```

### Job Matrix

| Strategy | Datasets | Models | Total Jobs |
|----------|----------|--------|-----------|
| `gen_random_noise` | 4 | 4 | 16 |
| `baseline` (comparison) | 4 | 4 | 16 |
| **Total** | | | **32** |

## Expected Analysis

### Comparison Framework

For each dataset × model combination, compare:

| Metric | gen_random_noise | gen_cycleGAN | baseline |
|--------|-----------------|--------------|----------|
| Overall mIoU | ? | reference | reference |
| Per-domain mIoU | ? | reference | reference |
| Per-class IoU | ? | reference | reference |

### Possible Outcomes

1. **noise ≈ baseline < gen_cycleGAN** → Generated images provide genuine visual features the model learns from. Image quality matters.

2. **noise ≈ gen_cycleGAN > baseline** → The benefit comes from label layout diversity, not image content. Generated image quality is irrelevant.

3. **noise > baseline but noise < gen_cycleGAN** → Partial benefit from label diversity exists, but real generated content adds additional value.

4. **noise < baseline** → Random noise is actively harmful, confirming models rely on image-label semantic coherence.

## Status

- [ ] Implementation complete
- [ ] Dry-run verified (32 jobs: 16 noise + 16 baseline)
- [ ] Jobs submitted
- [ ] Training complete
- [ ] Testing complete
- [ ] Analysis complete
