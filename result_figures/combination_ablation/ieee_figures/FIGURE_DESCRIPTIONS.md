# Combination Ablation Study - IEEE Figure Descriptions

## Overview

This document describes the 8 IEEE publication-ready figures generated for the 
combination ablation study, which analyzes the effects of combining different 
data augmentation strategies for semantic segmentation.

## Combination Types

- **Gen+Std**: Generative augmentation combined with standard augmentation
  - Examples: gen_CUT+std_mixup, gen_cycleGAN+std_randaugment
- **Std+Std**: Two standard augmentation techniques combined
  - Examples: std_cutmix+std_autoaugment, std_randaugment+std_mixup

---

## Figure 1: Combination Type Comparison

**File**: `fig1_combination_type_comparison.png`

**Description**: Grouped bar chart comparing the performance of Gen+Std vs Std+Std 
combination types across all datasets, with baseline performance for reference.

**Key Insights**:
- Shows whether generative+standard combinations outperform standard+standard
- Highlights dataset-specific differences in combination effectiveness
- Includes baseline reference for absolute performance context

**Recommended Caption**: "Comparison of combination strategy types across datasets. 
Gen+Std combinations (blue) vs Std+Std combinations (coral), with baseline reference 
(dark). Error bars show standard deviation across models."

---

## Figure 2: Component Synergy Heatmap

**File**: `fig2_synergy_heatmap.png`

**Description**: Heatmap showing the synergy between different component pairs, 
where synergy is defined as Combined_mIoU - max(Individual_mIoU).

**Key Insights**:
- Positive values (green) indicate synergistic combinations
- Negative values (red) indicate redundant or conflicting combinations
- Helps identify which component pairs work well together

**Recommended Caption**: "Synergy analysis between augmentation components. 
Values show the difference between combined performance and the best individual 
component (positive = synergistic, negative = redundant)."

---

## Figure 3: Per-Dataset Performance

**File**: `fig3_per_dataset_performance.png`

**Description**: Two-panel figure showing performance breakdown by dataset for 
Gen+Std (left) and Std+Std (right) combinations.

**Key Insights**:
- Shows dataset-specific effectiveness of each combination
- Helps identify which combinations work best for specific domains
- Allows comparison within each combination type category

**Recommended Caption**: "Per-dataset performance of combination strategies. 
Left: Generative+Standard combinations. Right: Standard+Standard combinations."

---

## Figure 4: Model Architecture Comparison

**File**: `fig4_model_comparison.png`

**Description**: Bar chart comparing combination type effectiveness across different 
segmentation model architectures (DeepLabV3+, PSPNet, SegFormer).

**Key Insights**:
- Shows whether certain models benefit more from specific combination types
- Highlights architecture-specific combination preferences
- Values annotated on bars for precise comparison

**Recommended Caption**: "Combination strategy performance by model architecture. 
Gen+Std (blue) and Std+Std (coral) performance across DeepLabV3+, PSPNet, and SegFormer."

---

## Figure 5: Combination Strategy Ranking

**File**: `fig5_combination_ranking.png`

**Description**: Horizontal bar chart ranking all combination strategies by their 
mean mIoU performance, with baseline reference line.

**Key Insights**:
- Provides overall ranking of all combinations
- Color-coded by combination type (Gen+Std vs Std+Std)
- Baseline reference shows absolute improvement

**Recommended Caption**: "Ranking of combination strategies by mean mIoU. 
Gen+Std combinations (blue) vs Std+Std combinations (coral). 
Dashed line indicates baseline performance."

---

## Figure 6: Gains Over Baseline

**File**: `fig6_gains_over_baseline.png`

**Description**: Horizontal bar chart showing the mean mIoU gain over baseline 
for each combination strategy. Positive gains in teal, negative in orange.

**Key Insights**:
- Quantifies the benefit of each combination over baseline
- Immediately shows which combinations improve/hurt performance
- Sorted by gain magnitude for easy identification of best combinations

**Recommended Caption**: "mIoU gains over baseline for each combination strategy. 
Positive gains (teal) indicate improvement, negative gains (orange) indicate degradation."

---

## Figure 7: Normal vs Adverse Conditions

**File**: `fig7_normal_vs_adverse.png`

**Description**: Scatter plot comparing combination performance under normal 
(clear day) conditions vs full test (including adverse weather).

**Key Insights**:
- Points above diagonal: better in adverse than normal (unexpected)
- Points below diagonal: better in normal than adverse (expected)
- Clustering by combination type shows type-specific robustness patterns

**Recommended Caption**: "Combination strategy performance under normal (clear day) 
vs adverse weather conditions. Points colored by combination type. 
Diagonal line indicates equal performance."

---

## Figure 8: Component Contribution Analysis

**File**: `fig8_component_contribution.png`

**Description**: Horizontal bar chart showing combined performance with synergy 
annotations indicating the difference between combined and best individual component.

**Key Insights**:
- Shows actual combined mIoU achieved
- Synergy annotations (+/-Xpp) indicate synergistic or redundant effects
- Helps understand whether combinations provide additive or super-additive benefits

**Recommended Caption**: "Combined performance with synergy effect annotations. 
Numbers indicate synergy (Combined - max(Individual)): positive = synergistic, 
negative = redundant combination."

---

## LaTeX Tables

The file `tables_booktabs.tex` contains 4 publication-ready tables:

1. **Table 1**: Combination Type Summary - overall statistics by type
2. **Table 2**: Per-Strategy Performance - detailed results across all datasets
3. **Table 3**: Model Architecture Comparison - type effectiveness by model
4. **Table 4**: Top 5 Combinations - best performing strategies ranked

---

## Technical Notes

- **IEEE Formatting**: All figures follow IEEE single/double column specifications
- **Resolution**: 300 DPI for publication quality
- **Font**: Times New Roman, 7-10pt as per IEEE guidelines
- **Colors**: Colorblind-friendly palette used throughout

## Data Source

Results extracted from `test_results_summary.csv` containing 311 combination 
strategy evaluations across 12 unique combinations.
