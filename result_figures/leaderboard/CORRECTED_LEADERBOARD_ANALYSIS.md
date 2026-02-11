# Corrected Leaderboard Analysis

**Date:** 2026-02-12
**Context:** IDD-AW data was invisible in per-dataset breakdowns due to naming mismatch (`idd-aw` vs `iddaw`). Fixed in commit `4b3529c`. This analysis uses the corrected leaderboards with all four S1 datasets visible.

**Data Sources:**
- Stage 1 (S1): 364 test results, 26 strategies, 4 datasets × up to 5 models
- Cityscapes-Gen (CG): 372 test results, 25 strategies, Cityscapes + ACDC (cross-domain)
- Stage 2 (S2): 152 test results, 20 strategies (limited coverage, preliminary)

---

## 1. S1 Per-Dataset Analysis

### 1.1 Gain Distribution by Dataset

| Dataset | Baseline mIoU | Mean Gain (pp) | Min/Max Gain (pp) | Range | Interpretation |
|---------|:----:|:----:|:----:|:----:|---|
| **OUTSIDE15k** | 32.85% | **+7.79** | [+7.52, +8.14] | 0.62 | Largest gains, narrowest range — **all strategies help equally** |
| **MapillaryVistas** | 26.65% | +6.10 | [+3.44, +6.39] | 2.95 | High gains, widest range — **strategy choice matters** |
| **IDD-AW** | 32.93% | +4.79 | [+3.96, +5.20] | 1.24 | Moderate consistent gains |
| **BDD10k** | 42.28% | +3.48 | [+0.75, +4.94] | 4.19 | Lowest gains, **highest variance** |

**Key insights:**
- Weaker baselines see larger improvements (negative correlation between baseline mIoU and gain)
- OUTSIDE15k's uniformly high gains suggest the improvement may be driven by dataset-level factors rather than strategy quality
- BDD10k differentiates strategies most — the strongest dataset for evaluating strategy quality
- Strategy gains are NOT transferable across datasets: cross-dataset rank correlation is near zero (0.04–0.39)

### 1.2 Cross-Dataset Rank Correlation

|  | BDD10k | IDD-AW | MapVistas | OUTSIDE15k |
|--|:------:|:------:|:---------:|:----------:|
| **BDD10k** | 1.00 | 0.04 | 0.06 | 0.29 |
| **IDD-AW** | 0.04 | 1.00 | 0.39 | −0.18 |
| **MapVistas** | 0.06 | 0.39 | 1.00 | −0.06 |
| **OUTSIDE15k** | 0.29 | −0.18 | −0.06 | 1.00 |

Strategy effectiveness is **dataset-specific**. A strategy that ranks #1 on BDD10k has essentially no predictive power for its rank on IDD-AW (r=0.04). The highest correlation (0.39) is between IDD-AW and MapillaryVistas, both multi-class datasets with RGB-encoded labels.

### 1.3 Per-Dataset Winners

| Dataset | #1 Strategy | Gain | #2 Strategy | Gain |
|---------|------------|------|-------------|------|
| BDD10k | gen_UniControl | +4.94 | gen_Img2Img | +4.57 |
| IDD-AW | gen_Img2Img | +5.20 | gen_UniControl | +5.17 |
| MapVistas | gen_Weather_Effect_Gen | +6.39 | gen_UniControl | +6.37 |
| OUTSIDE15k | gen_stargan_v2 | +8.14 | gen_augmenters | +8.12 |

---

## 2. S1 Per-Model Analysis: The mask2former Paradox

### 2.1 Model-Level Gain Distribution

| Model | Baseline mIoU | Mean Gain (pp) | Range | Strategies > Baseline |
|-------|:---:|:---:|---|:---:|
| **mask2former_swin-b** | 46.58% | **−1.62** | [−3.30, +0.33] | **1/25** |
| pspnet_r50 | 32.08% | +2.76 | [+2.66, +2.95] | 25/25 |
| segformer_mit-b3 | 36.21% | +4.23 | [+2.58, +5.02] | 25/25 |
| segnext_mscan-b | 37.43% | +3.15 | [+1.19, +3.94] | 25/25 |

> **Finding: mask2former gets WORSE with every augmentation strategy in S1.**
> Only gen_UniControl (+0.33 pp) is marginally above baseline. All others degrade performance by 1–3 pp.

### 2.2 Interpretation

The pattern follows a clear capacity hypothesis:
- **Low-capacity models** (PSPNet, SegFormer, SegNeXt) benefit greatly from augmented training data
- **High-capacity models** (Mask2Former with Swin-B backbone) are already near the data efficiency ceiling — additional augmented data introduces noise that hurts

Note: HRNet has only 1 baseline result (18.58% mIoU — abnormally low, likely incomplete training or eval). Excluded from analysis.

### 2.3 pspnet's Remarkably Consistent Gains

PSPNet shows the most consistent behavior: all 25 strategies improve it by +2.66 to +2.95 pp (range of only 0.29 pp). This suggests PSPNet benefits from ANY additional training data regardless of quality — the bottleneck is data quantity, not data quality.

### 2.4 segformer: Best Strategy Differentiator

SegFormer has the widest gain range (+2.58 to +5.02 pp = 2.44 pp spread), making it the best model for evaluating augmentation strategy quality.

---

## 3. S1 vs CG: Complete Ranking Divergence

### 3.1 Rank Correlation

**Spearman rank correlation: r = 0.206, p = 0.33** (24 overlapping strategies)

Strategy rankings between S1 and CG remain statistically indistinguishable from random ($p > 0.05$). The correlation doubled from the earlier estimate (r=0.101 with incomplete CG data) but is still **not significant** — performance in one stage has weak predictive power for the other.

> **Note**: The previous r=0.101 was computed when gen_TSIT was missing its deeplabv3plus model, artificially inflating gen_TSIT's CG rank from #20 to #2. With complete data, the correlation increases because gen_TSIT is now stable at #20 in both stages.

### 3.2 Largest Rank Swings

| Strategy | S1 Rank | CG Rank | Δ | Direction |
|----------|:-------:|:-------:|:---:|:---------:|
| gen_IP2P | 7 | **24** | +17 | ↓↓ Top→Bottom |
| gen_stargan_v2 | 5 | **22** | +17 | ↓↓ Top→Bottom |
| gen_UniControl | **1** | 16 | +15 | ❄️ Champion→Mid |
| gen_CNetSeg | 6 | 19 | +13 | ↓ |
| gen_step1x_v1p2 | 25 | **11** | −14 | 🔥 Bottom→Mid |
| gen_automold | 21 | **8** | −13 | ↑ |
| gen_cycleGAN | 23 | 12 | −11 | ↑ |
| gen_flux_kontext | 17 | 7 | −10 | ↑ |

> **Important correction**: In the previous analysis with incomplete CG data, gen_TSIT appeared as the biggest rank swing (+18, from S1 #20 to CG #2). This was entirely an artifact of gen_TSIT's weakest model (deeplabv3plus, −1.93pp) being missing from the CG average. With complete data, gen_TSIT is perfectly stable at **rank #20 in both stages** (Δ=0).

### 3.3 Consistently Ranked Strategies

**5 strategies have identical ranks** across both stages (Δ=0):

| Strategy | S1 Rank | CG Rank | Δ | Note |
|----------|:-------:|:-------:|:---:|------|
| **gen_Img2Img** | **2** | **2** | 0 | Top-2 in both stages |
| **gen_Qwen_Image_Edit** | **4** | **4** | 0 | Top-4 in both stages |
| gen_Weather_Effect_Generator | 14 | 14 | 0 | Mid-pack in both |
| gen_SUSTechGAN | 18 | 18 | 0 | Low-mid in both |
| gen_TSIT | 20 | 20 | 0 | Low in both |

Additionally, **gen_Attribute_Hallucination** (S1 #3 → CG #1, Δ=−2) is the only strategy in the top-3 of both stages.

### 3.4 Why Do Rankings Diverge?

Hypothesized factors:
1. **Training domain**: S1 trains on clear-day only (large domain gap); CG trains on mixed Cityscapes (moderate gap)
2. **Dataset diversity**: S1 averages over 4 diverse datasets; CG is Cityscapes-specific
3. **Model composition**: S1 fair comparison uses 4 models (mask2former, pspnet, segformer, segnext); CG uses 5 models (adds deeplabv3plus)
4. **Augmentation style match**: Some generators produce images more suited to cross-domain robustness (e.g., gen_UniControl, S1 #1 but CG #16), while others are more neutral (e.g., gen_Img2Img, stably #2 in both)

> **Data quality warning**: CG rankings are sensitive to model completeness. gen_TSIT's apparent +18 rank swing (the headline finding of the earlier analysis) was entirely due to a missing model. With complete data, gen_TSIT has zero rank difference. Researchers should verify all strategies have equal model coverage before drawing conclusions from rank comparisons.

---

## 4. CG Per-Model Analysis

### 4.1 Model-Level Gains in CG

| Model | Baseline mIoU | Mean Gain | Above Baseline | Note |
|-------|:---:|:---:|:---:|---|
| deeplabv3plus_r50 | 43.74% | −0.95 pp | 1/21 | **Hurt by augmentation** |
| **mask2former_swin-b** | 57.07% | **+0.03 pp** | **14/24** | **Benefits in CG** (opposite of S1!) |
| **pspnet_r50** | 43.43% | **+0.21 pp** | **17/24** | Consistent beneficiary |
| segformer_mit-b3 | 51.53% | −0.23 pp | 11/24 | Slightly hurt |
| segnext_mscan-b | 51.37% | −0.44 pp | 5/24 | Most hurt |

### 4.2 The mask2former Reversal

| Stage | mask2former Mean Gain | Strategies > Baseline |
|-------|:----:|:----:|
| **S1** | **−1.62 pp** | 1/25 (hurt) |
| **CG** | **+0.03 pp** | 14/24 (benefits) |

This reversal is striking. Possible explanation:
- In S1, mask2former is the strongest model on diverse datasets and is already near the performance ceiling — augmentation adds noise
- In CG, mask2former is also strongest but the Cityscapes→ACDC domain shift is a different challenge where the model's capacity can leverage augmented data
- The style of augmentation (Cityscapes-specific weather generation) may be more aligned with mask2former's feature representations than the multi-dataset S1 augmentation

---

## 5. Domain Gap Analysis (S1)

### 5.1 Overall Domain Gap

| Metric | Baseline | Best Strategy | Improvement |
|--------|:---:|:---:|:---:|
| Normal mIoU (clear_day + cloudy) | 34.24% | gen_Img2Img (40.74%) | +6.50 pp |
| Adverse mIoU (foggy, night, rainy, snowy) | 28.89% | gen_UniControl (34.90%) | +6.01 pp |
| **Domain Gap** | 5.35 pp | gen_UniControl (5.72 pp) | **+0.37 pp (wider)** |

All strategies raise both Normal and Adverse mIoU, but Normal benefits slightly more → domain gap slightly **increases** (5.72–6.16 pp vs baseline 5.35 pp).

### 5.2 Per-Domain Improvements (vs baseline)

| Domain | Baseline | Best Strategy | Max Gain |
|--------|:---:|---|:---:|
| clear_day | 35.42% | gen_Attribute_Hallucination | +6.55 pp |
| cloudy | 33.05% | gen_Img2Img | +6.56 pp |
| dawn_dusk | 28.25% | gen_Weather_Effect_Gen | +5.61 pp |
| foggy | 32.55% | gen_VisualCloze | +6.15 pp |
| **night** | **23.80%** | gen_cyclediffusion | **+4.93 pp** |
| rainy | 30.02% | gen_Img2Img | +6.47 pp |
| snowy | 29.17% | gen_augmenters | +6.90 pp |

Night remains the hardest domain with the smallest gains. The best night-domain strategy (gen_cyclediffusion) achieves only 28.73%, still below the clear_day baseline (35.42%).

### 5.3 Smallest Domain Gaps (most robust)

| Strategy | Domain Gap | Type |
|----------|:---:|---|
| gen_UniControl | 5.72 pp | Generative |
| **std_autoaugment** | **5.79 pp** | Standard Aug |
| gen_IP2P | 5.90 pp | Generative |
| gen_SUSTechGAN | 5.88 pp | Generative |
| gen_augmenters | 5.92 pp | Generative |

std_autoaugment has the 2nd smallest gap despite being a simple standard augmentation strategy.

---

## 6. Recommendations for S2 Strategy Selection

Based on the analysis, these strategies are recommended for S2 curated training:

### Tier 1: Consistently Strong (must include)
| Strategy | S1 Rank | CG Rank | Rationale |
|----------|:---:|:---:|---|
| **gen_Attribute_Hallucination** | 3 | 1 | Only strategy in top-3 of BOTH stages |
| **gen_Img2Img** | 2 | **2** | Perfectly stable top-2 in both stages |
| **gen_Qwen_Image_Edit** | 4 | **4** | Perfectly stable top-4 in both |

### Tier 2: Stage Champions (include for coverage)
| Strategy | S1 Rank | CG Rank | Rationale |
|----------|:---:|:---:|---|
| **gen_UniControl** | **1** | 16 | S1 champion, lowest domain gap |
| **gen_CUT** | 13 | **5** | CG top-5, GAN-based unpaired translation |
| gen_augmenters | 8 | 3 | CG top-3, process-based augmentation |

### Tier 3: Standard Aug Representatives (include 1–2)
| Strategy | S1 Rank | CG Rank | Rationale |
|----------|:---:|:---:|---|
| std_autoaugment | 15 | 10 | Best std_* in S2 preliminary, smallest gap |
| std_cutmix | 16 | 9 | Most consistent std_* across stages |

### Tier 4: Diversity (include for breadth)
| Strategy | S1 Rank | CG Rank | Rationale |
|----------|:---:|:---:|---|
| gen_flux_kontext | 17 | 7 | Diffusion-based, big CG improvement (+10↑) |
| gen_step1x_v1p2 | 25 | 11 | Biggest CG gainer (+14↑), worth investigating |

**Recommended S2 subset:** 8–10 strategies (Tier 1–3 mandatory = 6, plus 2–3 from Tier 4)

---

## Appendix A: Mask2Former Per-Class Deep Dive

### A.1 The Vehicle Class Collapse in S1

The mask2former degradation (−1.62pp mean) is **not uniform across classes**. Vehicle classes account for almost all of the loss:

| Class | Group | S1 Mean Δ | Std | #Positive/25 |
|-------|-------|:---------:|:---:|:-----:|
| motorcycle | Vehicle | **−12.16** | 5.70 | 2/25 |
| bus | Vehicle | **−5.35** | 8.50 | 7/25 |
| truck | Vehicle | **−3.18** | 2.80 | 2/25 |
| wall | Construction | −2.45 | 1.63 | 1/25 |
| traffic sign | Object | −1.56 | 1.58 | 3/25 |
| fence | Construction | −1.24 | 0.69 | 1/25 |
| bicycle | Vehicle | −1.10 | 0.49 | 1/25 |
| sidewalk | Flat | −1.01 | 0.68 | 2/25 |
| person | Human | −0.89 | 0.45 | 2/25 |
| ... | ... | ... | ... | ... |
| rider | Human | **+0.59** | 1.55 | 18/25 |
| pole | Object | +0.12 | 0.70 | 16/25 |

Group averages: **Vehicle −3.68pp**, Construction −1.36pp, Flat −0.72pp, Object −0.54pp, Nature −0.28pp, Human −0.15pp.

### A.2 Root Cause: Rare Class Memorization

BDD10k class areas reveal the mechanism. Motorcycle is the 2nd rarest class (107k pixels, after train at 64k):

| Domain | Motorcycle Label Area | Motorcycle IoU (Baseline) | IoU (Most Strategies) |
|--------|:----:|:----:|:----:|
| **rainy** | **48,979** (46% of total) | **85.4%** | **0–10%** |
| clear_day | 47,606 | 23.9% | ~20% |
| snowy | 3,991 | 18.6% | ~15% |
| night | 3,477 | 0.0% | ~0% |
| cloudy | 2,252 | 0.3% | ~0% |
| foggy | 0 | 0.0% | 0% |

**46% of all motorcycle pixels are concentrated in the rainy domain.** The baseline mask2former memorizes these few instances, achieving 85.4% IoU. Augmentation disrupts this memorization, causing collapse to near-zero. Only 3 strategies maintain BDD10k rainy motorcycle IoU:
- gen_UniControl: 86.4% (slightly improved)
- std_autoaugment: 86.2%
- std_randaugment: 53.6%
- gen_Img2Img: 53.2%
- All others: 0–27%

**IDD-AW motorcycle** is much more evenly distributed (27–38% IoU across all domains, 33.4% rainy baseline) and shows minimal degradation across strategies (28–35%), confirming this is a BDD10k-specific rare-class artifact.

### A.3 The S1→CG Reversal: Per-Class Evidence

Comparing which classes reverse between S1 (hurt) and CG (helped):

| Class | Group | S1 Δ | CG Δ | Gap | Pattern |
|-------|-------|:----:|:----:|:---:|---------|
| **truck** | Vehicle | −3.18 | **+2.79** | +5.97 | REVERSAL ↑ |
| **building** | Construction | −0.38 | **+1.65** | +2.03 | REVERSAL ↑ |
| **wall** | Construction | −2.45 | **+0.94** | +3.39 | REVERSAL ↑ |
| **bicycle** | Vehicle | −1.10 | **+0.57** | +1.67 | REVERSAL ↑ |
| **sky** | Nature | −0.33 | **+0.62** | +0.95 | REVERSAL ↑ |
| **terrain** | Flat | −0.67 | **+0.14** | +0.81 | REVERSAL ↑ |
| **car** | Vehicle | −0.27 | **+0.41** | +0.68 | REVERSAL ↑ |
| **person** | Human | −0.89 | **+0.07** | +0.96 | REVERSAL ↑ |
| motorcycle | Vehicle | −12.16 | −1.12 | +11.05 | Both ↓ (much less hurt) |
| bus | Vehicle | −5.35 | −0.21 | +5.14 | Both ↓ (much less hurt) |
| train | Vehicle | +0.00 | −4.84 | −4.84 | ↓ in CG only |

**9 of 19 classes reverse** from S1 (hurt) to CG (helped). The key insight is that **Construction group reverses completely**: S1 −1.36pp → CG +0.70pp. This is because Cityscapes has abundant building/wall pixels, preventing the rare-class memorization issue.

### A.4 Per-Domain Effects in CG

In CG, night and snowy show improvements while foggy degrades:

| CG Domain | Mean Δ | Neg Classes | Worst Class | Best Class |
|-----------|:------:|:-----------:|-------------|------------|
| **night** | **+1.48** | 7/19 | traffic sign (−2.13) | building (+6.56) |
| **snowy** | **+0.79** | 9/19 | sidewalk (−4.56) | bus (+12.49) |
| rainy | −0.45 | 12/19 | sidewalk (−5.33) | truck (+3.55) |
| foggy | −1.16 | 12/19 | train (−16.76) | rider (+5.87) |

The night domain benefits most from augmentation in CG — consistent with the generated images providing night-condition training data that the original Cityscapes training set lacks entirely.

### A.5 Which CG Strategies Help mask2former Most?

| CG Strategy | mask2former mIoU | Δ vs Baseline |
|-------------|:---:|:---:|
| **gen_CUT** | 58.48 | **+1.41** |
| gen_cyclediffusion | 58.10 | +1.03 |
| gen_augmenters | 58.08 | +1.01 |
| std_autoaugment | 57.86 | +0.79 |
| gen_Attribute_Hallucination | 57.75 | +0.68 |
| gen_TSIT | 57.74 | +0.67 |
| gen_cycleGAN | 57.71 | +0.64 |
| gen_VisualCloze | 57.67 | +0.60 |
| ... | ... | ... |
| gen_Weather_Effect_Generator | 55.43 | **−1.64** |
| gen_automold | 55.53 | −1.54 |

The best CG mask2former strategy (gen_CUT, +1.41pp) is completely different from the best S1 strategy (gen_UniControl, +0.33pp — the only positive one).

### A.6 Conclusions

1. **mask2former's S1 degradation is a rare-class memorization effect**, not a fundamental incompatibility with augmentation
2. **Vehicle classes (motorcycle, bus, truck) drive >80% of the S1 loss**, caused by class rarity + domain concentration in BDD10k
3. **CG reverses the pattern** because Cityscapes has more balanced class distributions — augmentation adds genuine diversity rather than disrupting memorization
4. **Construction classes reverse completely** (S1: −1.36pp → CG: +0.70pp), with building improving +1.65pp in CG across 24/24 strategies
5. **Night domain benefits most** in CG (+1.48pp mean), validating that weather-augmented training data addresses genuine gaps

**Recommendation**: When using mask2former, either ensure training data has sufficient rare-class coverage, or use class-aware augmentation strategies that protect rare instances.

---

## Appendix B: Cross-Model Rare-Class Comparison

### B.1 The Mask2Former Effect is Model-Specific

Comparing per-class mean deltas across all 4 S1 models for the three most-affected classes:

| Class | PSPNet | SegFormer | SegNeXt | mask2former |
|-------|:------:|:---------:|:-------:|:-----------:|
| motorcycle | **+1.55** | −1.23 | **+9.48** | **−12.16** |
| bus | **+23.77** | **+6.23** | **+9.70** | **−5.35** |
| truck | **+12.97** | **+4.22** | **+10.50** | **−3.18** |

Vehicle group averages:

| Model | Vehicle Δ | All Groups Positive? |
|-------|:---------:|:---:|
| PSPNet | **+6.62** | Yes (all 6 groups positive) |
| SegNeXt | **+5.82** | Yes |
| SegFormer | **+2.16** | Yes |
| **mask2former** | **−3.68** | **No (all 6 groups negative)** |

mask2former is the **only model** where augmentation degrades Vehicle classes. All other models show massive improvements on the same rare classes — PSPNet gains +23.77pp on bus alone.

### B.2 Why Other Models Don't Memorize

- **PSPNet** (weakest model): Has the most headroom for improvement. Bus goes from ~37% → ~61% with augmentation (+24pp). The model's limited capacity forces it to learn generalizable features rather than memorize.
- **SegNeXt**: Strong gains on motorcycle (+9.48pp) and bus (+9.70pp). Like PSPNet, augmentation teaches new features rather than disrupting memorized ones.
- **SegFormer**: More moderate gains. motorcycle still slightly negative (−1.23pp) suggesting it's closer to the memorization threshold, but not crossing it like mask2former.

### B.3 Impact on Overall Rankings

Excluding motorcycle, bus, and train from mIoU (16-class mIoU):

**Spearman r = 0.954** — rankings are almost identical. The rare-class effect does NOT significantly distort the overall strategy rankings because:
1. It affects only 1 of 4 models (mask2former)
2. The 3 rare classes contribute only ~3/19 = 16% of the mIoU calculation
3. The vehicle losses in mask2former are partially offset by vehicle gains in other models

| Biggest Gainers (move UP) | Std Rank → RX Rank |
|--------------------------|:---:|
| gen_cyclediffusion | 9 → 4 (+5) |
| gen_Qwen_Image_Edit | 7 → 3 (+4) |
| gen_IP2P | 10 → 7 (+3) |

| Biggest Losers (move DOWN) | Std Rank → RX Rank |
|---------------------------|:---:|
| gen_stargan_v2 | 6 → 10 (−4) |
| gen_CNetSeg | 4 → 8 (−4) |
| std_randaugment | 11 → 15 (−4) |

### B.4 Mask2Former: Rare-Excluded Still Negative

Even excluding rare classes, mask2former performance remains mostly negative:
- Standard mIoU: 2/25 strategies above baseline
- Rare-excluded mIoU: 3/25 strategies above baseline
- Average "salvage" from excluding rare classes: +0.75pp (reduces loss from −1.62 to −0.87)

The rarest classes account for about **half** of mask2former's degradation (0.75pp of 1.62pp), but the other half comes from smaller losses across all 16 common classes.

---

## Appendix C: Data Completeness Notes

- **S1**: IDD-AW has all 14 expected fair-comparison test results. MapillaryVistas and OUTSIDE15k are missing mask2former (submitted to 80GB GPUs, awaiting completion).
- **CG**: **124/124 configurations complete** (372 test results). All 25 strategies have all 5 models tested on both Cityscapes and ACDC.
- **S2**: Only 20 strategies with limited model coverage (mostly SegFormer). Rankings are preliminary.
- **S1 per-model**: HRNet excluded (only 1 baseline result at 18.58% — likely incomplete).

> **Revision note** (2026-02-12): The original analysis was written with 123/124 CG configurations (gen_TSIT/deeplabv3plus missing). This caused gen_TSIT's CG rank to be artificially inflated from #20 to #2, which produced a misleading +18 rank swing that was highlighted as a major finding. All CG rankings, Spearman correlation, rank swing tables, and S2 recommendations in this document have been corrected with complete data.
