# PROVE Augmentation Study — Cross-Stage Analysis

**Generated**: 2026-02-10  
**Stages Analyzed**: Stage 1 (clear-day), Stage 2 (all-domain), Cityscapes-Gen (CS→ACDC)  
**Strategies**: 26 total (21 generative, 4 standard augmentation, 1 baseline)  
**Models**: deeplabv3plus_r50, hrnet_hr48, mask2former_swin-b, pspnet_r50, segformer_mit-b3, segnext_mscan-b

---

## 1. Executive Summary

| Stage | Baseline mIoU | Best Strategy | Best Gain | Strategies > Baseline |
|-------|--------------|---------------|-----------|----------------------|
| Stage 1 (clear-day → all) | 33.63% | gen_Img2Img | +6.35 | **25/25 (100%)** |
| Stage 2 (all → all) | 40.80% | gen_IP2P | +1.18 | 16/19 (84%) |
| CS-Gen (CS → ACDC) | 50.85% | gen_VisualCloze | +0.12 | 2/24 (8%) |

**Key Finding**: Augmentation is universally beneficial when training data is domain-limited (Stage 1), moderately helpful when training already includes diverse domains (Stage 2), and largely ineffective or harmful when training and test sets come from fundamentally different distributions (Cityscapes-Gen).

---

## 2. Cross-Stage Consistency

### 2.1 Gain Correlations Between Stages

| Comparison | Pearson r |
|-----------|----------|
| Stage 1 vs Stage 2 | **−0.17** |
| Stage 1 vs CS-Gen | **−0.40** |
| Stage 2 vs CS-Gen | **−0.35** |

All correlations are **negative**, meaning strategies that rank highly in one stage tend to rank differently (or even inversely) in other stages. This is a critical finding:

> **A strategy's effectiveness is highly context-dependent.** Selecting an augmentation strategy based on one evaluation protocol does not reliably predict performance under different training/testing conditions.

### 2.2 Notable Rank Reversals

| Strategy | S1 Rank | CS-Gen Rank | Interpretation |
|----------|---------|-------------|----------------|
| gen_Img2Img | **#1** (+6.35) | #14 (−1.79) | Best for domain gap bridging, but hurts clean performance |
| gen_UniControl | **#2** (+6.33) | #25 (−4.27) | Largest rank drop; generates unrealistic artifacts? |
| gen_VisualCloze | #11 (+6.03) | **#1** (+0.12) | Modest in S1 but most stable on Cityscapes |
| gen_flux_kontext | #17 (+5.71) | **#2** (+0.10) | Diffusion-based; minimal perturbation preserves quality |

---

## 3. Strategy Family Analysis

Strategies grouped by their generation methodology:

| Family | S1 Gain | S1 Gap | CS-Gen Gain | CS-Gen Gap | Members |
|--------|---------|--------|-------------|------------|---------|
| Image-editing | **+6.23** | 6.02 | −1.69 | 6.77 | 4 |
| Conditional-gen | +6.17 | 6.03 | −3.20 | **5.46** | 3 |
| Standard Aug | +6.02 | 5.97 | **−0.47** | 9.37 | 4 |
| GAN-based | +5.70 | 6.00 | −2.66 | 6.57 | 6 |
| Diffusion-based | +5.68 | 5.94 | −1.02 | 8.54 | 5 |
| Classical-augment | +5.66 | 5.97 | −1.08 | 8.99 | 3 |

### Key Observations:
- **Image-editing methods** (Img2Img, Qwen, VisualCloze, Attribute_Hallucination) achieve the highest S1 gains (+6.23 avg), suggesting they produce the most useful domain variations.
- **Conditional generation** (CNetSeg, UniControl, Weather_Effect_Generator) achieves the strongest domain gap reduction in CS-Gen (5.46 vs 9.54 baseline), but at the cost of overall mIoU.
- **Standard augmentation** causes the least harm in CS-Gen (−0.47 avg), confirming it's the safest but not the most powerful option.
- **Diffusion-based methods** are the most moderate across stages — not the best or worst anywhere.

---

## 4. The Normal–Adverse Tradeoff (CS-Gen)

In the Cityscapes-Gen stage, many generative strategies trade clean-domain (Cityscapes) performance for adverse-domain (ACDC) robustness:

| Strategy | CS mIoU Δ | ACDC mIoU Δ | Net | Domain Gap |
|----------|-----------|-------------|-----|-----------|
| gen_VisualCloze | +0.09 | +0.19 | ✅ Win-win | 9.44 |
| gen_flux_kontext | −0.05 | +0.40 | ✅ Near-neutral | 9.09 |
| std_mixup | −0.14 | +0.22 | ✅ Near-neutral | 9.17 |
| gen_Img2Img | −2.51 | **+0.87** | ⚠️ CS loss | 6.15 |
| gen_CUT | −5.52 | +0.62 | ❌ Heavy CS loss | **3.40** |
| gen_UniControl | −5.85 | +0.09 | ❌ Heavy CS loss | **3.59** |

**Insight**: Strategies with the smallest domain gaps (gen_CUT, gen_UniControl) achieve this by *degrading* clean performance rather than *improving* adverse performance. Only **gen_VisualCloze** and **gen_flux_kontext** achieve genuine win-win improvements.

---

## 5. Per-Model Analysis

### 5.1 Stage 1: Model Sensitivity to Augmentation

| Model | Baseline mIoU | Avg Augmented mIoU | Avg Gain | Best Strategy (Gain) |
|-------|--------------|-------------------|----------|---------------------|
| mask2former_swin-b | **46.58** | 44.87 | **−1.71** | gen_Img2Img (+0.02) |
| segformer_mit-b3 | 36.21 | 40.47 | **+4.26** | gen_Attribute_Hallucination (+5.02) |
| segnext_mscan-b | 37.43 | 40.58 | **+3.15** | std_randaugment (+3.94) |
| pspnet_r50 | 32.08 | 34.84 | **+2.76** | gen_albumentations_weather (+2.95) |

**Critical Finding**: mask2former_swin-b (the strongest baseline model) is **hurt** by almost all augmentation strategies. Augmentation primarily benefits weaker models. This is likely because mask2former already has sufficient capacity/inductive bias to generalize, and the noise from generated images degrades its learned representations.

### 5.2 Cross-Model Variance Reduction

| Metric | Baseline | Augmented (avg) |
|--------|----------|----------------|
| Stage 1 cross-model Std | **9.40** | **5.82** |

Augmentation reduces cross-model performance variance by **38%**, meaning it serves as a great equalizer — weaker architectures benefit more, bringing them closer to stronger ones.

### 5.3 CS-Gen: Model Vulnerability

| Model | Baseline | Most Vulnerable To |
|-------|----------|-------------------|
| mask2former_swin-b | 57.07 | gen_Weather_Effect_Generator (−6.84) |
| pspnet_r50 | 43.43 | gen_stargan_v2 (−8.09) |
| segnext_mscan-b | 51.37 | gen_Attribute_Hallucination (−6.84) |
| segformer_mit-b3 | 51.53 | gen_CNetSeg (−1.56) |

segformer_mit-b3 is the most **robust** model to generative augmentation — max loss is only 1.56 mIoU. This suggests transformer-based segmentation is more tolerant of training distribution shifts.

---

## 6. Stage 2: Diminishing Returns

When training already includes all weather domains, augmentation gains shrink dramatically:

| Family | S1 Gain | S2 Gain | Diminishing Factor |
|--------|---------|---------|-------------------|
| Generative (all) | +5.86 | +1.00 | 5.9× |
| Standard Aug | +6.02 | −0.55 | reversed |

- **Generative methods still help** in Stage 2 (avg +1.00), suggesting they add diversity beyond what's in the training set.
- **Standard augmentation becomes harmful** (−0.55), suggesting photometric/geometric transforms are redundant when real-world diversity is already present.

---

## 7. Conclusions & Recommendations

### 7.1 When to Use Generative Augmentation
1. **Strongly recommended** when training data is domain-limited (Stage 1 scenario): all 25 strategies beat baseline, average gain +5.9%.
2. **Mildly beneficial** when training data includes diverse domains (Stage 2): 15/15 generative strategies beat baseline.
3. **Use with extreme caution** for cross-dataset generalization (CS-Gen): only 2/24 strategies beat baseline.

### 7.2 Safest Strategies (Consistently Positive or Neutral)
- **gen_VisualCloze**: +6.03 (S1), +1.11 (S2), +0.12 (CS-Gen) — the only strategy with positive gains in all 3 stages.
- **gen_flux_kontext**: +5.71 (S1), +1.04 (S2), +0.10 (CS-Gen) — second-most consistent.
- **std_mixup**: +6.09 (S1), −1.22 (S2), −0.02 (CS-Gen) — least-harm standard augmentation.

### 7.3 Best Domain Gap Reducers (at the Cost of mIoU)
- **gen_CUT**: Reduces CS-Gen gap from 9.54 to 3.40 but loses 3.85 mIoU overall.
- **gen_UniControl**: Reduces CS-Gen gap from 9.54 to 3.59 but loses 4.27 mIoU overall.

### 7.4 Architecture Interaction
- **Strong models (mask2former)**: Often hurt by augmentation; already generalize well.
- **Weaker models (pspnet, segnext, segformer)**: Benefit substantially (+2.7 to +5.0 in Stage 1).
- **segformer_mit-b3**: Most robust to negative effects of generative augmentation.

### 7.5 Summary Verdict
> Generative augmentation is a powerful tool for bridging domain gaps in limited-data scenarios, but it introduces a **quality–diversity tradeoff**. The best approach depends on the deployment context: use aggressive generation (e.g., gen_Img2Img, gen_UniControl) when the training domain is narrow and test conditions are diverse; use conservative strategies (gen_VisualCloze, gen_flux_kontext) when maintaining clean-domain performance matters.

---

## 8. Data Coverage Caveats

### 8.1 Incomplete Coverage Overview

Not all strategies have full test coverage across all model/dataset combinations. Results for strategies with missing tests should be interpreted with caution, as incomplete coverage may bias aggregate metrics.

| Stage | Total Results | Full Coverage | Strategies with Missing Tests |
|-------|-------------|---------------|-------------------------------|
| Stage 1 | 364 | 14 tests/strategy | 3 strategies at 13/14 |
| Stage 2 | 140 | varies | Most gen strategies have only 1 model (segformer_mit-b3) |
| CS-Gen | 292 | 12 tests/strategy | 6 strategies below 12 |

### 8.2 Stage 1 — Nearly Complete

Three strategies are each missing 1 test result:
- **gen_automold**: 13/14 tests
- **gen_albumentations_weather**: 13/14 tests
- **gen_step1x_v1p2**: 13/14 tests

Impact: Minimal — these strategies are only missing a single model/dataset configuration each.

### 8.3 Stage 2 — Severely Limited Model Coverage

This is the most significant coverage limitation. Stage 2 generative strategies have been tested with **only 1 model** (segformer_mit-b3 with ratio 0.50) across 4 datasets, yielding just 4 test results each. In contrast:
- **baseline**: 37 tests across 7 model variants and 5 datasets
- **std_autoaugment**: 14 tests across 4 models
- **std_cutmix / std_mixup**: 11–12 tests across 3 models
- **std_randaugment**: 6 tests across 2 models

⚠️ **This means Stage 2 generative strategy rankings reflect single-model performance (segformer_mit-b3), not multi-model averages.** The +1.0 average gain for generative methods in Stage 2 may not generalize to other architectures. Cross-model validation is needed before drawing firm conclusions.

### 8.4 CS-Gen — Mostly Complete

Six strategies have incomplete test sets:

| Strategy | Tests | Missing |
|----------|-------|---------|
| gen_cycleGAN | 9/12 | 3 model/dataset configs |
| gen_augmenters | 9/12 | 3 model/dataset configs |
| gen_UniControl | 10/12 | 2 model/dataset configs |
| gen_SUSTechGAN | 11/12 | 1 config |
| gen_cyclediffusion | 11/12 | 1 config |
| gen_Attribute_Hallucination | 11/12 | 1 config |
| std_autoaugment | 15/12 | Over-coverage (has extra deeplabv3plus_r50 results) |

Note: std_autoaugment has 15 tests instead of 12 because it includes results for deeplabv3plus_r50, which other CS-Gen strategies lack. This gives it broader model coverage but makes direct comparison uneven.

### 8.5 Implications for Analysis

1. **Cross-stage comparisons** (Section 2) are most reliable for strategies with full coverage in all stages. The 18 strategies present in all 3 stages have varying completeness.
2. **Stage 2 conclusions** (Section 6) should be considered **preliminary** — generative strategy gains are based on a single model only.
3. **Per-model analysis** (Section 5) is limited by the fact that many strategies were only trained/tested on a subset of models. The finding that mask2former is hurt by augmentation is based on Stage 1 and CS-Gen data where it has coverage, but Stage 2 data for mask2former exists only for std_autoaugment.
4. The **strategy family analysis** (Section 3) averages over strategies with different coverage levels, which may introduce bias toward strategies with more favorable model/dataset combinations.

---

*Total data: Stage 1: 364 results, Stage 2: 140 results, CS-Gen: 292 results. Generated 2026-02-10.*
