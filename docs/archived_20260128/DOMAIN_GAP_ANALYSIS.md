# Domain Gap Analysis: Comprehensive Findings

This document summarizes the comprehensive analysis of domain gaps between normal and adverse weather conditions across all augmentation strategies.

## Executive Summary

### Key Metrics

- **Primary Metric**: mIoU (mean Intersection over Union)
  - Equal weight to all classes
  - Recommended for domain robustness analysis
  
- **Secondary Metric**: fwIoU (Frequency-Weighted IoU)
  - **NOT recommended** for cross-domain comparison
  - Class frequency shifts in adverse weather inflate fwIoU artificially

### Baseline Performance (Clear-Day Trained Models)

| Condition | mIoU | fwIoU |
|-----------|------|-------|
| Normal (clear_day, cloudy) | 54.96% | 81.76% |
| Adverse (foggy, night, rainy, snowy, dawn_dusk) | 47.49% | 81.65% |
| **Domain Gap** | **7.46%** | 0.11% |

The fwIoU shows almost no gap (0.11%) while mIoU shows a significant 7.46% gap. This is because:
- Adverse weather naturally occludes small objects (people, cyclists, signs)
- This shifts class distribution toward "easy" classes (road, sky)
- fwIoU weights by class frequency, artificially inflating scores

## Domain Gap Calculation

### Definition

The **domain gap** measures the performance difference between models tested on "normal" vs "adverse" weather conditions:

```
Domain Gap = Normal_mIoU - Adverse_mIoU
```

Where:
- **Positive gap** (> 0): Model performs better on normal conditions (expected for clear-day trained models)
- **Negative gap** (< 0): Model performs better on adverse conditions (indicates training on adverse data or domain shift artifacts)
- **Zero gap** (≈ 0): Model is domain-invariant (ideal for robust deployment)

### Step-by-Step Calculation

#### 1. Collect Per-Domain Results

For each strategy, collect mIoU results across all test domains:

```python
# Weather domains
normal_domains = ['clear_day', 'cloudy']
adverse_domains = ['foggy', 'night', 'rainy', 'snowy', 'dawn_dusk']
```

#### 2. Filter Unreliable Domains

Exclude domains with fewer than 50 test images:

```python
reliable_results = results[results['num_images'] >= 50]
```

This ensures statistical reliability and prevents small sample sizes from skewing results.

#### 3. Calculate Per-Group Averages

**Option A: Simple Average** (equal weight to each result)
```python
normal_mIoU = reliable_results[reliable_results['domain'].isin(normal_domains)]['mIoU'].mean()
adverse_mIoU = reliable_results[reliable_results['domain'].isin(adverse_domains)]['mIoU'].mean()
```

**Option B: Frequency-Weighted Average** (recommended, weights by sample size)
```python
normal_data = reliable_results[reliable_results['domain'].isin(normal_domains)]
adverse_data = reliable_results[reliable_results['domain'].isin(adverse_domains)]

# Weighted by number of test images
normal_mIoU = np.average(normal_data['mIoU'], weights=normal_data['num_images'])
adverse_mIoU = np.average(adverse_data['mIoU'], weights=adverse_data['num_images'])
```

#### 4. Calculate Domain Gap

```python
domain_gap = normal_mIoU - adverse_mIoU
gap_reduction = baseline_gap - strategy_gap  # vs baseline
```

### Example Calculation

For **gen_StyleID**:

| Domain | mIoU | # Images | Category |
|--------|------|----------|----------|
| clear_day | 54.8% | 2,450 | Normal |
| cloudy | 58.2% | 1,890 | Normal |
| foggy | 74.7% | 520 | Adverse |
| night | 52.5% | 1,650 | Adverse |
| rainy | 58.2% | 2,100 | Adverse |
| snowy | 59.7% | 1,430 | Adverse |
| dawn_dusk | 58.7% | 890 | Adverse |

**Normal Average** (weighted):
```
Normal_mIoU = (54.8 × 2450 + 58.2 × 1890) / (2450 + 1890) = 56.3%
```

**Adverse Average** (weighted):
```
Adverse_mIoU = (74.7 × 520 + 52.5 × 1650 + 58.2 × 2100 + 59.7 × 1430 + 58.7 × 890) / (520 + 1650 + 2100 + 1430 + 890) = 58.5%
```

**Domain Gap**:
```
Domain Gap = 56.3% - 58.5% = -2.2%
```

**Interpretation**: gen_StyleID has a small negative domain gap, meaning it performs slightly better on adverse conditions. Compared to baseline (+7.46% gap), this represents a gap reduction of ~9.7 percentage points.

### Important Considerations

1. **Dataset Coverage Matters**: Strategies with incomplete coverage (e.g., only ACDC clear_day) will show biased gaps
2. **ACDC Artifact**: ACDC's clear_day split is exceptionally difficult (~25% mIoU), causing artificial negative gaps
3. **Sample Size**: Always filter by minimum sample size (≥50 images)
4. **Metric Choice**: Use mIoU, not fwIoU, for domain gap analysis

## Methodology

### Data Quality Filtering

- **Minimum sample size**: 50 images per domain
- **Unreliable domains excluded**:
  - BDD10k foggy (4 images)
  - MapillaryVistas foggy (27 images)
  - Other domains with < 50 images

### Domain Classification

- **Normal conditions**: clear_day, cloudy
- **Adverse conditions**: foggy, night, rainy, snowy, dawn_dusk

### Averaging Method

When calculating aggregate metrics:
- Frequency-weighted averaging based on number of reliable test samples
- This prevents small datasets from disproportionately affecting overall metrics

## Coverage Limitations

### Why Not All Strategies?

The domain gap analysis requires **per-domain test results**, which are stored in:
```
<strategy>/<dataset>/<model>/test_results_detailed/<date>/test_report.txt
```

**Strategies with DETAILED results (9, included in analysis):**
- baseline, gen_CUT, gen_StyleID, gen_cycleGAN, std_std_photometric_distort
- std_autoaugment, std_cutmix, std_mixup, std_randaugment

**Strategies with BASIC results only (16, NOT included):**
- gen_Attribute_Hallucination, gen_EDICT, gen_IP2P, gen_Img2Img, gen_LANIT
- gen_NST, gen_Qwen_Image_Edit, gen_SUSTechGAN, gen_TSIT, gen_UniControl
- gen_Weather_Effect_Generator, gen_automold, gen_flux1_kontext
- gen_imgaug_weather, gen_stargan_v2, gen_step1x_new

**Why this matters:**
- **Basic testing** (`metrics.json`): Only reports overall mIoU (single number)
- **Detailed testing** (`test_report.txt`): Reports per-domain mIoU (required for gap analysis)

**To include missing strategies**, run detailed testing:
```bash
python fine_grained_test.py --strategy <strategy_name> --all-datasets --all-models
```

443 trained models are ready for detailed testing but haven't been processed yet.

### Why Not All Datasets for Some Strategies?

Some strategies have incomplete dataset coverage:

| Strategy | Datasets with Results |
|----------|----------------------|
| baseline | ACDC, BDD10k, IDD-AW only (3/5) |
| gen_CUT | ACDC, BDD10k, IDD-AW only (3/5) |
| std_autoaugment | ACDC, BDD10k, IDD-AW only (3/5) |
| Others | All 5 datasets |

This creates **biased domain gap estimates** for strategies with limited coverage, as:
- ACDC's clear_day is exceptionally difficult (~25% mIoU)
- Missing results from BDD10k, MapillaryVistas, Outside15k skew averages

**Recommendation**: Compare strategies at same coverage level, or wait for complete testing.


## Strategy Comparison

### Fair Comparison Group

Strategies with comprehensive dataset coverage (results from 5+ datasets across all weather conditions):

| Strategy | Overall mIoU | Normal mIoU | Adverse mIoU | Domain Gap | Δ vs Baseline |
|----------|--------------|-------------|--------------|------------|---------------|
| **gen_StyleID** | 59.3% | 58.7% | 60.7% | -2.0% | +6.0% |
| std_std_photometric_distort | 57.3% | 56.5% | 59.9% | -3.5% | +4.0% |
| std_cutmix | 56.6% | 55.7% | 59.6% | -3.9% | +3.4% |
| std_randaugment | 56.5% | 55.4% | 60.3% | -4.8% | +3.3% |
| std_mixup | 56.5% | 55.5% | 59.8% | -4.3% | +3.2% |
| gen_cycleGAN | 55.8% | 54.7% | 59.3% | -4.5% | +2.6% |
| baseline_clear_day | 53.2% | 55.0% | 47.5% | **+7.5%** | - |

**Key Observations:**
- All augmentation strategies improve overall mIoU
- All strategies reduce the domain gap (baseline has +7.5% gap, augmented strategies show negative gaps)
- **gen_StyleID achieves the best balance** of high overall performance and low domain gap

### Incomplete Data Group

These strategies have limited dataset coverage (only ACDC for clear_day) and should not be compared directly:

| Strategy | Overall mIoU | Normal mIoU | Adverse mIoU | Domain Gap | Issue |
|----------|--------------|-------------|--------------|------------|-------|
| baseline | 57.2% | 33.7% | 64.9% | -31.1% | Only ACDC clear_day data |
| gen_CUT | 57.1% | 33.9% | 64.7% | -30.7% | Only ACDC clear_day data |
| std_autoaugment | 57.1% | 33.6% | 64.7% | -31.2% | Only ACDC clear_day data |

The extreme negative domain gaps (-30%) are due to missing clear_day results from other datasets. ACDC's clear_day split is notoriously difficult, creating artificial low normal_mIoU scores.

## Per-Dataset Analysis

### Dataset Characteristics

| Dataset | Clear_Day mIoU (avg) | Notes |
|---------|---------------------|-------|
| IDD-AW | 70-80% | High performance, diverse scenes |
| Outside15k | 60-70% | Good representation |
| BDD10k | 55-65% | Urban driving focus |
| MapillaryVistas | 40-50% | Complex global scenes |
| ACDC | 25-35% | Difficult Swiss conditions |

### Per-Dataset Domain Gaps

| Dataset | Domain Gap Pattern |
|---------|-------------------|
| ACDC | -24% (adverse > normal) - ACDC's clear_day is exceptionally difficult |
| BDD10k | +5-6% (normal > adverse) - typical expected pattern |
| IDD-AW | +6-7% (normal > adverse) - typical expected pattern |
| MapillaryVistas | ~0% (balanced) - diverse training data |
| Outside15k | +11-12% (normal > adverse) - high gap |

**Insight**: ACDC's negative domain gap is an artifact of its difficult clear_day conditions, not an indication that adverse weather is easier overall.

## Recommendations

### For Model Training

1. **Best Overall Strategy**: gen_StyleID
   - Highest overall mIoU (59.3%)
   - Excellent adverse weather performance (60.7%)
   - Minimal domain gap (-2.0%)

2. **Best Gap Reduction**: std_randaugment
   - Strong adverse improvement (+12.8%)
   - Gap reduced from +7.5% to -4.8%
   - Good overall performance (56.5%)

3. **Budget-Friendly Option**: std_std_photometric_distort
   - Good performance (57.3%)
   - No additional data generation required
   - +4.0% improvement over baseline

### For Evaluation

1. **Always use mIoU** for cross-domain comparison
2. **Filter domains** with < 50 test images
3. **Report per-dataset results** alongside aggregates
4. **Include dataset coverage** when comparing strategies

## Generated Outputs

All analysis outputs are saved in `result_figures/unified_domain_gap/`:

- `strategy_summary.csv` - Overall strategy comparison
- `all_domain_results.csv` - Detailed per-domain results
- `analysis_report.txt` - Full text report
- `fair_comparison.png` - Visual comparison of fair strategies
- `domain_gap_reduction.png` - Domain gap reduction visualization
- `per_dataset/` - Per-dataset breakdowns

## Scripts

```bash
# Run unified domain gap analysis
python analyze_unified_domain_gap.py

# Run baseline clear_day analysis
python analyze_baseline_clear_day.py

# Run class distribution analysis (explains mIoU vs fwIoU)
python analyze_class_distribution.py
```

## Citation

If you use these analysis findings, please cite the PROVE project.
