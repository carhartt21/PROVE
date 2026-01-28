# Stage 1 vs Stage 2 Strategy Comparison Analysis

**Generated:** 2026-01-21
**Analysis Focus:** Why do some strategies excel in Stage 1 but underperform in Stage 2?

## Executive Summary

Training context matters significantly for augmentation strategy effectiveness:
- **Stage 1** (clear_day only): High-fidelity generative models excel by adding realistic weather diversity
- **Stage 2** (all domains): Real adverse weather data is already present, making synthetic images potentially harmful

## Key Findings

### 1. Gain Reduction in Stage 2

All strategies showed **reduced gain over baseline** when moving from Stage 1 to Stage 2:

| Statistic | Value |
|-----------|-------|
| Average gain reduction | -0.77 mIoU |
| Max gain reduction | -1.93 (gen_Qwen_Image_Edit) |

### 2. Ranking Changes (Stage 1 → Stage 2)

#### Biggest Winners (improved rank)

| Strategy | S1 Rank | S2 Rank | Change | S1 Gain | S2 Gain |
|----------|---------|---------|--------|---------|---------|
| gen_UniControl | 20 | 2 | **+18** | +0.85 | +0.51 |
| gen_augmenters | 24 | 7 | **+17** | +0.63 | +0.44 |
| gen_LANIT | 23 | 10 | **+13** | +0.68 | +0.34 |
| gen_CUT | 15 | 3 | **+12** | +1.02 | +0.48 |
| std_randaugment | 25 | 13 | **+12** | +0.50 | +0.29 |
| gen_VisualCloze | 18 | 8 | **+10** | +0.90 | +0.34 |

#### Biggest Losers (dropped rank)

| Strategy | S1 Rank | S2 Rank | Change | S1 Gain | S2 Gain |
|----------|---------|---------|--------|---------|---------|
| gen_step1x_new | 5 | 25 | **-20** | +1.28 | -0.03 |
| gen_Qwen_Image_Edit | 1 | 21 | **-20** | +1.97 | +0.04 |
| gen_Attribute_Hallucination | 2 | 20 | **-18** | +1.53 | +0.15 |
| gen_cyclediffusion | 8 | 22 | **-14** | +1.24 | +0.03 |
| gen_Weather_Effect_Generator | 12 | 26 | **-14** | +1.09 | -0.15 |
| std_autoaugment | 3 | 15 | **-12** | +1.38 | +0.27 |
| gen_flux_kontext | 6 | 18 | **-12** | +1.28 | +0.21 |

### 3. Below-Baseline Strategies in Stage 2

Three strategies actually **hurt performance** relative to baseline in Stage 2:

| Strategy | S2 Rank | S2 Gain | S1 Rank | S1 Gain | Notes |
|----------|---------|---------|---------|---------|-------|
| gen_step1x_new | 25 | -0.03 | 5 | +1.28 | Step1X image generator |
| gen_Weather_Effect_Generator | 26 | -0.15 | 12 | +1.09 | Weather-specific effects |
| std_mixup | 27 | -0.25 | 21 | +0.77 | Standard mixup augmentation |

## Pattern Analysis

### Strategies That Improved in Stage 2

**Common Characteristics:**
1. **Structure-preserving methods**: gen_CUT, gen_UniControl - These methods modify style without significantly altering image structure
2. **Simple augmentations**: std_cutmix, std_randaugment - Don't introduce potentially misleading weather patterns
3. **Lower fidelity generators**: gen_augmenters, gen_LANIT - Less likely to conflict with real weather patterns

### Strategies That Declined in Stage 2

**Common Characteristics:**
1. **High-fidelity generative models**: gen_Qwen_Image_Edit, gen_step1x_new - May produce images that conflict with real adverse weather
2. **Weather-specific generators**: gen_Weather_Effect_Generator - Synthetic weather may differ from real weather distribution
3. **Diffusion models**: gen_cyclediffusion - May introduce subtle artifacts

## Hypothesis: Why High-Fidelity Generators Decline

### Stage 1 (Limited Data)
- Model only sees clear_day images
- High-fidelity generated adverse weather images provide **valuable diversity**
- The generated images, even if imperfect, are **better than nothing**
- Result: Large gains over baseline

### Stage 2 (Diverse Real Data)
- Model sees real adverse weather images from all 7 domains
- High-fidelity synthetic images may:
  - Have **domain gap** with real adverse weather
  - Introduce **conflicting visual patterns**
  - Reduce effective training signal by diluting real data
- Result: Synthetic images provide **diminishing or negative returns**

## Per-Class Analysis: gen_step1x_new on BDD10k

Comparing per-class IoU between gen_step1x_new and baseline (Stage 2, deeplabv3plus):

| Class | gen_step1x_new | baseline | Difference | Notes |
|-------|---------------|----------|------------|-------|
| pole | 24.30 | 33.66 | **-9.36** | Small object |
| bicycle | 0.0 | 6.90 | **-6.90** | Small object |
| motorcycle | 0.0 | 5.17 | **-5.17** | Small object |
| traffic sign | 30.19 | 35.47 | **-5.28** | Small object |
| sidewalk | 42.23 | 46.23 | **-4.00** | - |
| building | 73.97 | 77.72 | **-3.75** | - |

**Key Insight:** Generated images appear to hurt **small object detection** most severely, possibly due to:
- Blurring or distortion of small objects in generated images
- Artifacts at object boundaries
- Inconsistent appearance of small objects across weather conditions

## Domain-Specific Analysis

### Per-Domain Performance (BDD10k deeplabv3plus, Stage 2)

| Domain | gen_step1x_new | baseline | Difference |
|--------|---------------|----------|------------|
| foggy | 62.81 | 60.83 | **+1.98** ✓ |
| cloudy | 45.01 | 46.53 | -1.52 |
| clear_day | 41.06 | 43.35 | -2.29 |
| dawn_dusk | 35.05 | 37.59 | -2.54 |

**Observation:** gen_step1x_new only outperforms baseline on the **foggy** domain, suggesting:
- The step1x_new fog generation may be closer to real fog than other weather effects
- Or foggy conditions are inherently easier to augment synthetically

## Recommendations

### For Future Research

1. **Evaluate augmentation strategies in context**: A strategy's effectiveness depends heavily on the training data diversity
2. **Consider stage-specific approaches**: Use different augmentation strategies for data-limited vs data-rich scenarios
3. **Monitor small object performance**: Generated images may disproportionately affect small object classes

### For Practitioners

1. **Stage 1 (limited domain data)**: High-fidelity generative augmentation is highly beneficial
2. **Stage 2 (diverse real data)**: Prefer:
   - Structure-preserving methods (gen_CUT, gen_UniControl)
   - Simple augmentations (std_cutmix)
   - Avoid weather-specific generators that may conflict with real data

## Full Comparison Table

| Strategy | S1 Rank | S1 Gain | S2 Rank | S2 Gain | Rank Δ | Gain Δ |
|----------|---------|---------|---------|---------|--------|--------|
| gen_Qwen_Image_Edit | 1 | +1.97 | 21 | +0.04 | -20 ↓ | -1.93 |
| gen_Attribute_Hallucination | 2 | +1.53 | 20 | +0.15 | -18 ↓ | -1.38 |
| std_autoaugment | 3 | +1.38 | 15 | +0.27 | -12 ↓ | -1.11 |
| gen_cycleGAN | 4 | +1.35 | 6 | +0.45 | -2 ↓ | -0.90 |
| gen_step1x_new | 5 | +1.28 | 25 | -0.03 | -20 ↓ | -1.31 |
| gen_flux_kontext | 6 | +1.28 | 18 | +0.21 | -12 ↓ | -1.07 |
| gen_stargan_v2 | 7 | +1.25 | 5 | +0.45 | +2 ↑ | -0.80 |
| gen_cyclediffusion | 8 | +1.24 | 22 | +0.03 | -14 ↓ | -1.21 |
| gen_automold | 9 | +1.20 | 19 | +0.17 | -10 ↓ | -1.03 |
| gen_CNetSeg | 10 | +1.14 | 4 | +0.46 | +6 ↑ | -0.68 |
| gen_albumentations_weather | 11 | +1.12 | 9 | +0.34 | +2 ↑ | -0.78 |
| gen_Weather_Effect_Generator | 12 | +1.09 | 26 | -0.15 | -14 ↓ | -1.24 |
| gen_IP2P | 13 | +1.08 | 11 | +0.30 | +2 ↑ | -0.78 |
| gen_SUSTechGAN | 14 | +1.06 | 14 | +0.29 | 0 = | -0.77 |
| gen_CUT | 15 | +1.02 | 3 | +0.48 | +12 ↑ | -0.54 |
| gen_Img2Img | 16 | +1.00 | 12 | +0.29 | +4 ↑ | -0.71 |
| gen_TSIT | 17 | +0.92 | 16 | +0.24 | +1 ↑ | -0.68 |
| gen_VisualCloze | 18 | +0.90 | 8 | +0.34 | +10 ↑ | -0.56 |
| gen_step1x_v1p2 | 19 | +0.89 | 17 | +0.23 | +2 ↑ | -0.66 |
| gen_UniControl | 20 | +0.85 | 2 | +0.51 | +18 ↑ | -0.34 |
| std_mixup | 21 | +0.77 | 27 | -0.25 | -6 ↓ | -1.02 |
| std_std_photometric_distort | 22 | +0.73 | 23 | +0.01 | -1 ↓ | -0.72 |
| gen_LANIT | 23 | +0.68 | 10 | +0.34 | +13 ↑ | -0.34 |
| gen_augmenters | 24 | +0.63 | 7 | +0.44 | +17 ↑ | -0.19 |
| std_randaugment | 25 | +0.50 | 13 | +0.29 | +12 ↑ | -0.21 |
| std_cutmix | 26 | +0.27 | 1 | +1.45 | +25 ↑ | +1.18 |
| baseline | 27 | +0.00 | 24 | +0.00 | +3 ↑ | 0.00 |

---

*This analysis is based on 324 Stage 1 test results and 317 Stage 2 test results across 27 augmentation strategies, 4 datasets, and 3 model architectures.*
