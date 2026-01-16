# IEEE Publication Figures - Strategy Leaderboard Analysis

**Generated:** January 8, 2026  
**Source Data:** PROVE Strategy Leaderboard  
**Target:** IEEE Journal Publication

---

## Executive Summary

This document accompanies the 8 publication-ready figures and 10 LaTeX booktabs tables generated from the strategy leaderboard analysis. The visualizations compare various data augmentation strategies for improving semantic segmentation robustness under domain shift.

### Key Findings

1. **AutoMold leads overall**: The AutoMold generative augmentation achieves the highest overall mIoU improvement of **+2.7%** over the clear-day baseline.

2. **Generative methods dominate**: 7 of the top 10 strategies are generative augmentation methods, outperforming traditional augmentation on average.

3. **Adverse weather improvement is substantial**: Top strategies show **+5-6%** improvement on adverse weather conditions while maintaining normal performance.

4. **Training data diversity amplifies augmentation effects**: Strategies show **2.1%** higher gains when applied with full (diverse) training data compared to clear-day-only training.

5. **Trade-offs exist**: Some strategies (e.g., CUT) excel at normal conditions but underperform on adverse weather, highlighting the importance of balanced evaluation.

---

## Generated Files

### Figures (PNG format, 300 DPI)
| File | Description | Size |
|------|-------------|------|
| `fig1_strategy_leaderboard.png` | Top 15 strategies by overall gain | Single column |
| `fig2_normal_adverse_scatter.png` | Normal vs adverse performance scatter | Single column |
| `fig3_domain_gains_heatmap.png` | Per-domain gains heatmap | Double column |
| `fig4_dataset_gains.png` | Per-dataset performance gains | Double column |
| `fig5_training_type_comparison.png` | Full vs clear day training | Single column |
| `fig6_gap_reduction.png` | Domain gap reduction ranking | Single column |
| `fig7_strategy_type_summary.png` | Strategy type comparison | Double column |
| `fig8_top5_comparison.png` | Top 5 detailed comparison | Single column |

### Tables (LaTeX booktabs)
| Table | Description |
|-------|-------------|
| Table I | Strategy Leaderboard (Top 15) |
| Table II | Normal vs Adverse Performance |
| Table III | Per-Domain Performance Gains |
| Table IV | Per-Dataset Performance Gains |
| Table V | Strategy Type Summary |
| Table VI | Training Strategy Comparison |
| Table VII | Leaderboard Comparison |
| Table VIII | Top 5 Detailed Metrics |
| Table IX | Domain-Specific Best Strategies |
| Table X | Key Findings Summary |

---

## Figure Descriptions

### Figure 1: Strategy Leaderboard
**File:** `fig1_strategy_leaderboard.png`  
**Size:** Single column (3.5")  
**Corresponding Table:** Table I

Horizontal bar chart showing the top 15 augmentation strategies ranked by overall mIoU gain relative to the clear-day baseline. Strategies are color-coded by type:
- **Blue**: Generative augmentation
- **Orange**: Standard augmentation
- **Dark gray**: Baseline

**Key observations:**
- AutoMold leads with +2.7% gain
- Top 5 strategies are all generative methods
- Most strategies show positive gains (+1-3%)

**Caption suggestion:**  
*Strategy leaderboard showing mIoU improvement over clear-day baseline. Generative augmentation methods (blue) dominate the top rankings, with AutoMold achieving the highest gain of +2.7%.*

---

### Figure 2: Normal vs Adverse Scatter
**File:** `fig2_normal_adverse_scatter.png`  
**Size:** Single column (3.5" × 3.5")  
**Corresponding Table:** Table II

Scatter plot comparing performance gains on normal conditions (x-axis) vs adverse conditions (y-axis). Each point represents a strategy, colored by type.

**Key observations:**
- Most strategies fall above the x-axis (better adverse than normal improvement)
- CUT is an outlier with high normal gain but low adverse gain
- Strategies clustered around (0-2%, 4-6%) for (normal, adverse) gains

**Caption suggestion:**  
*Comparison of strategy performance on normal vs adverse weather conditions. Most strategies show greater improvements on adverse conditions, indicating effective domain adaptation. CUT is an outlier with strong normal but weak adverse performance.*

---

### Figure 3: Domain Gains Heatmap
**File:** `fig3_domain_gains_heatmap.png`  
**Size:** Double column (7.16")  
**Corresponding Table:** Table III

Heatmap showing per-domain mIoU gains for top strategies across 7 weather domains. Green indicates improvement, red indicates degradation.

**Key observations:**
- Foggy domain shows largest gains (+6-8%) across most strategies
- Snowy domain also shows consistent improvement
- Clear day gains are minimal (0-2%)
- Night domain shows variable performance

**Caption suggestion:**  
*Per-domain performance gains for top augmentation strategies. Foggy and snowy conditions show the largest improvements, while clear day performance remains stable. Color scale: red (degradation) to green (improvement).*

---

### Figure 4: Dataset Gains
**File:** `fig4_dataset_gains.png`  
**Size:** Double column (7.16")  
**Corresponding Table:** Table IV

Grouped bar chart showing per-dataset performance gains for top strategies. Each strategy has bars for BDD10K, IDD-AW, Mapillary, and Outside15K.

**Key observations:**
- AutoMold excels on Mapillary (+5.6%)
- RandAugment shows variable performance across datasets
- IDD-AW benefits consistently from most strategies

**Caption suggestion:**  
*Per-dataset performance gains showing strategy effectiveness varies by dataset. AutoMold achieves the highest gain on Mapillary (+5.6%), while most strategies consistently improve IDD-AW performance.*

---

### Figure 5: Training Type Comparison
**File:** `fig5_training_type_comparison.png`  
**Size:** Single column (3.5")  
**Corresponding Table:** Table VI

Horizontal grouped bar chart comparing strategy effectiveness under full training vs clear-day-only training regimes.

**Key observations:**
- Full training amplifies augmentation benefits substantially
- Most strategies show negative gains with clear-day-only training
- TSIT and LANIT are exceptions that work better with clear-day training

**Caption suggestion:**  
*Impact of training data diversity on augmentation effectiveness. Full training (including adverse weather data) enables strategies to achieve significant gains, while clear-day-only training limits effectiveness.*

---

### Figure 6: Gap Reduction
**File:** `fig6_gap_reduction.png`  
**Size:** Single column (3.5")  
**Corresponding Table:** Table I (Gap Reduction column)

Horizontal bar chart showing domain gap reduction (Normal-Adverse difference reduction) for each strategy.

**Key observations:**
- TSIT achieves largest gap reduction (+5.5%)
- Most generative strategies reduce gap by 3-5%
- CUT increases gap (-4.4%), indicating worse adverse performance

**Caption suggestion:**  
*Domain gap reduction showing strategy effectiveness at closing the performance gap between normal and adverse conditions. TSIT achieves the largest reduction (+5.5%), while CUT increases the gap.*

---

### Figure 7: Strategy Type Summary
**File:** `fig7_strategy_type_summary.png`  
**Size:** Double column (7.16")  
**Corresponding Table:** Table V

Two-panel figure comparing strategy types:
- **(a)** Overall mIoU by strategy type
- **(b)** Adverse weather gain by strategy type

**Key observations:**
- Generative and standard augmentation perform similarly on average
- Photometric augmentation shows slightly higher overall mIoU
- All types show positive adverse weather gains

**Caption suggestion:**  
*Comparison of strategy types. (a) Overall mIoU shows similar performance across types. (b) All strategy types improve adverse weather performance, with generative methods showing highest variance.*

---

### Figure 8: Top 5 Comparison
**File:** `fig8_top5_comparison.png`  
**Size:** Single column (3.5")  
**Corresponding Table:** Table VIII

Grouped bar chart showing detailed metrics (Normal Gain, Adverse Gain, Gap Reduction) for the top 5 strategies.

**Key observations:**
- All top 5 strategies show balanced improvements
- NST achieves highest adverse gain (+5.7%)
- AutoMold has best overall balance

**Caption suggestion:**  
*Detailed comparison of top 5 strategies across key metrics. NST achieves highest adverse gain while AutoMold provides the best overall balance between normal performance and gap reduction.*

---

## Technical Specifications

### Figure Format
- **Format:** PNG at 300 DPI
- **Color mode:** RGB
- **Background:** White

### Sizing (IEEE compliant)
- Single column figures: 3.5 inches (88.9 mm)
- Double column figures: 7.16 inches (181.9 mm)

### Typography
- Font family: Times New Roman (serif)
- Title: 10 pt
- Axis labels: 9 pt
- Tick labels: 8 pt
- Annotations: 7 pt

### Color Scheme
**Strategy Types:**
- Generative: #2E86AB (Blue)
- Standard Aug: #E07A5F (Orange)
- Photometric: #3D405B (Dark slate)
- Baseline: #333333 (Dark gray)

### Legend Positioning
All legends positioned outside data areas to prevent overlap.

---

## Table-Figure Correspondence

| Figure | Complementary Table(s) |
|--------|------------------------|
| Fig. 1 | Table I |
| Fig. 2 | Table II |
| Fig. 3 | Table III |
| Fig. 4 | Table IV |
| Fig. 5 | Table VI |
| Fig. 6 | Table I |
| Fig. 7 | Table V |
| Fig. 8 | Table VIII |

---

## Recommended Selection for Publication

### Space-Limited Papers (6-8 pages)
**Essential figures:**
1. Figure 1 (Strategy ranking) + Table I
2. Figure 2 (Normal vs Adverse) + Table II
3. Figure 3 (Domain heatmap) + Table III

### Extended Papers (10+ pages)
Add:
- Figure 5 (Training comparison)
- Figure 7 (Strategy types)
- Tables IV, VI

### Supplementary Material
- Remaining figures and tables
- Full strategy-by-strategy CSV data

---

*Generated by PROVE Strategy Leaderboard Analysis Pipeline*
*Last updated: January 8, 2026*
