# Control Tests: Do Generated Images Actually Help?

**Date Submitted**: 2026-01-30 10:25  
**Status**: Test jobs submitted, awaiting results  
**Test Duration**: ~1-2 hours per job on GPU

---

## The Question

The fair leaderboard shows that generated images provide very modest improvements (~0.10-0.23 mIoU) on BDD10k + IDD-AW. But is this real or noise?

**Key Observation**: gen_stargan_v2 with ratio=0.5 (50% generated images) performs at 42.71 mIoU, only -0.12% below baseline (42.83). The improvement is negligible.

**Hypothesis**: Generated images may not be providing significant benefit on BDD10k/IDD-AW because:
1. The real images alone are already good (4000 clear-day BDD10k images)
2. The synthetic images may not be realistic enough
3. The mixing strategy (batch-level, 50-50) might not be optimal
4. Generated images only help cross-domain, not within-domain

---

## Submitted Control Tests

### Test 1: gen_stargan_v2 ratio=0.0 (Job 774921)
**Configuration**: BDD10k + DeepLabV3+ + gen_stargan_v2, domain-filter=clear_day, real-gen-ratio=0.0

**What it tests**: Training ONLY on synthetic (generated) images, NO real data
- Uses 100% starGAN generated images (~10,000 generated images)
- Does NOT use real BDD10k clear-day images
- Pure generative training as control

**Expected Result**:
- If similar to ratio=0.5 (42.71) → mixing real+synthetic is just as good as pure synthetic
- If much worse than ratio=0.5 → mixing real data improves performance (real data helps)
- If better than ratio=0.5 → synthetic-only training is best (real data hurts)

### Test 2: gen_stargan_v2 ratio=0.25 (Job 775022)
**Configuration**: BDD10k + DeepLabV3+ + gen_stargan_v2, real-gen-ratio=0.25

**What it tests**: Dosage effect of real vs generated data
- 25% real images (1,000 real images from 4,000)
- 75% synthetic (7,500 generated images from ~10,000)
- Total batch composition: 2 generated + 2 real per 8-sample batch

**Expected Result**: 
- Performance should vary with real-data ratio
- If linear: can identify optimal mix
- If non-linear: some optimal point exists

### Test 3: gen_cycleGAN ratio=0.0 (Job 775023)
**Configuration**: BDD10k + DeepLabV3+ + gen_cycleGAN, real-gen-ratio=0.0

**What it tests**: Whether effect is consistent across generators
- Control for gen_stargan_v2
- Pure real-image training with cycleGAN setup (but no generated images used)

### Test 4: gen_Attribute_Hallucination ratio=0.0 (Job 775024)
**Configuration**: BDD10k + DeepLabV3+ + gen_Attribute_Hallucination, real-gen-ratio=0.0

**What it tests**: Top performer in incomplete leaderboard with no generated images
- gen_Attribute_Hallucination was ranked high (42.93) in incomplete results
- If it was high ONLY because of good BDD10k performance, ratio=0.0 should still be good
- If ratio=0.0 is much worse → generated images were actually helping

---

## Expected Outcomes & Hypotheses

### Most Likely Scenario (Based on User Hypothesis)

**Hypothesis**: Synthetic images have low semantic correspondence with segmentation labels
- StarGAN generates style-transferred images (e.g., "foggy" version of clear-day scene)
- But segmentation labels come from ORIGINAL clear-day image
- Labels may not match the synthetic image content (e.g., fog obscures objects)

**Expected Result**: ratio=0.0 (pure synthetic) will perform MUCH WORSE (~20-30% drop)
```
baseline (real clear-day):    42.83 mIoU
gen_stargan_v2_r0.5 (mixed):  42.71 mIoU (-0.12%)
gen_stargan_v2_r0.0 (pure):   ~30-35 mIoU? (large drop expected)
```

**What this would prove**:
1. ✅ Bug fix IS working (synthetic images loaded)
2. ✅ BUT semantic mismatch between generated images and labels
3. ⚠️ Generated images only work when MIXED with real data (ratio=0.5)
4. ⚠️ Pure synthetic training fails due to label misalignment

### Alternative Scenarios

### Scenario A: All ratio=0.0 jobs are much WORSE than ratio=0.5
```
gen_stargan_v2_r0.0: ~35.0 mIoU (much worse than ratio=0.5: 42.71)
→ Interpretation: Real images are ESSENTIAL, synthetic alone is poor
→ Implication: Bug fix works, synthetic images alone insufficient
→ Recommendation: Keep mixing strategy, tune ratio
```

### Scenario B: ratio=0.0 similar to ratio=0.5
```
gen_stargan_v2_r0.0: ~42.7 mIoU (similar to ratio=0.5: 42.71)
→ Interpretation: Synthetic images can replace real images
→ Implication: Generated images are high quality
→ Recommendation: Could try even higher synthetic ratios
```

### Scenario C: ratio=0.0 is better than ratio=0.5
```
gen_stargan_v2_r0.0: ~45.0 mIoU (BETTER than ratio=0.5: 42.71)
→ Interpretation: Pure synthetic training outperforms mixing
→ Implication: Real data from one domain may be confusing synthetic from another
→ Recommendation: Consider synthetic-only training or soft mixing
```

### Scenario D: Linear dosage response (0.0 << 0.25 << 0.5)
```
gen_stargan_v2_r0.0:  35.0 mIoU
gen_stargan_v2_r0.25: 39.0 mIoU  
gen_stargan_v2_r0.5:  42.7 mIoU
→ Interpretation: More real data helps linearly
→ Implication: Bug fix is working, real images are valuable
→ Recommendation: Could try ratio=0.75 or ratio=1.0 (100% real)

---

## Why This Matters

Current findings on fair leaderboard (BDD10k + IDD-AW only):
- gen_stargan_v2: 42.71 mIoU (-0.12%)
- gen_cycleGAN: 42.99 mIoU (+0.15%)
- gen_Attribute_Hallucination: 42.93 mIoU (+0.10%)
- **baseline: 42.83 mIoU (reference)**

**These tiny differences (0.10-0.15%) COULD mean:**

1. **Regularization effect**: Mixing real+synthetic provides slight regularization benefit
   - Similar to data augmentation (adding noise helps generalization)
   - The synthetic images' semantic mismatch acts as noise
   - Small benefit but not from realistic synthetic scenes

2. **Label smoothing**: Misaligned labels force model to learn robust features
   - Model ignores small label errors, learns coarse scene understanding
   - Explains why all gen_* strategies perform similarly (all have mismatch)

3. **Just noise**: Standard deviation is ±3.2 mIoU, so 0.15% is within noise

### The Critical Test

If ratio=0.0 drops to ~30 mIoU as expected:
- **Proves**: Semantic mismatch between synthetic images and labels
- **Explains**: Why ratio=0.5 only marginally better than baseline
  - Half the training data is "noisy" (wrong labels for synthetic)
  - Acts like regularization, not true data augmentation
- **Implies**: Need better generation methods that preserve semantic structure

---

## Expected Completion Timeline

- **Submitted**: 2026-01-30 10:25
- **Expected completion**: 2026-01-30 12:30 (assuming 2 hours on GPU)
- **If GPU queue is slow**: Could take up to 4-6 hours

---

## What We've Already Verified

✅ **Bug fix IS working technically**:
- Generated images ARE being loaded
- Batch composition IS correct (4 real + 4 generated)
- serialize_data=False IS in place
- Training script properly injects generated images

✅ **Fair performance assessment shows small gains**:
- Best gen_* strategy: +0.23% (gen_step1x_v1p2)
- Most gen_* strategies: -0.12% to +0.15%
- Standard deviation within strategies: ±3.3 mIoU (large noise)

⚠️ **Open question**: Is the modest gain real or just noise?
- Control tests will answer this definitively
- If ratio=0.0 matches ratio=0.5 → generated images aren't helping
- If ratio=0.0 is much worse → generated images are valuable

---

## Next Steps After Control Tests

**If generated images DON'T help (Scenario A)**:
1. Analyze generated image quality
2. Check if they're realistic enough for the task
3. Consider trying different generation methods
4. OR accept that synthetic data isn't needed for clear-day training

**If generated images DO help (Scenario B)**:
1. Proceed with Stage 2 training (all domains, no domain filter)
2. Test on cross-domain generalization (key metric)
3. Analyze which generator types work best
4. Optimize mixing ratios

**If dosage response is clear (Scenario C)**:
1. Try ratio=0.75, ratio=1.0 (100% synthetic)
2. Find optimal mixing ratio for each generator
3. Could unlock even better performance

---

**Submitted Jobs**:
- 774921: gen_stargan_v2 ratio=0.0
- 775022: gen_stargan_v2 ratio=0.25
- 775023: gen_cycleGAN ratio=0.0
- 775024: gen_Attribute_Hallucination ratio=0.0

**Monitoring command**: `bjobs -u ${USER} | grep -E "774921|775022|775023|775024"`

**When ready to check results**: All should produce `results.json` in test_results_detailed/ directory
