# IEEE Publication Figures - Key Insights and Descriptions

**Generated:** January 8, 2026  
**Source Data:** Baseline Consolidated Analysis  
**Target:** IEEE Journal Publication

---

## Executive Summary

This document accompanies the 8 publication-ready figures and 8 LaTeX booktabs tables generated from the baseline semantic segmentation analysis. The visualizations illustrate key findings about model performance under domain shift across different weather conditions.

### Key Findings

1. **SegFormer dominates**: SegFormer-MiT-B5 consistently outperforms CNN-based architectures (DeepLabV3+, PSPNet) with an overall mIoU of **69.7%** vs. 55.1% and 50.7%.

2. **Significant domain gap exists**: All models show performance degradation in adverse weather conditions, with an average gap of **6.24%** between normal and adverse domains.

3. **Training diversity matters**: Models trained on all domains show **3.25%** higher overall mIoU and **2.97%** smaller domain gap compared to models trained only on clear day images.

4. **Dataset difficulty varies significantly**: IDD-AW achieves the highest mIoU (73.6%) while Mapillary Vistas is most challenging (44.6%).

5. **Night conditions are most challenging**: Night domain consistently shows the largest performance drop across all architectures.

---

## Generated Files

### Figures (PNG format, 300 DPI)
| File | Description | Size |
|------|-------------|------|
| `fig1_model_comparison.png` | Model architecture comparison | Single column |
| `fig2_domain_gap_heatmap.png` | Domain gap heatmap | Single column |
| `fig3_domain_radar.png` | Weather domain radar chart | Single column |
| `fig4_training_comparison.png` | Training strategy comparison | Double column |
| `fig5_dataset_difficulty.png` | Dataset difficulty analysis | Single column |
| `fig6_performance_matrix.png` | Comprehensive performance matrix | Double column |
| `fig7_model_ranking.png` | Model ranking summary | Single column |
| `fig8_domain_shift.png` | Domain shift impact | Single column |

### Tables (LaTeX booktabs)
| Table | Description |
|-------|-------------|
| Table I | Overall Model Performance Comparison |
| Table II | Domain Gap Analysis |
| Table III | Per-Domain Performance |
| Table IV | Training Strategy Comparison |
| Table V | Model Architecture Summary |
| Table VI | Dataset Characteristics |
| Table VII | Comprehensive Per-Domain Results |
| Table VIII | Performance Drop Analysis |

---

## Figure Descriptions

### Figure 1: Model Architecture Comparison
**File:** `fig1_model_comparison.png`  
**Size:** Single column (3.5")  
**Corresponding Table:** Table I

Grouped bar chart comparing three model architectures (DeepLabV3+, PSPNet, SegFormer) across four benchmark datasets. Legend positioned at top center to avoid data overlap. SegFormer achieves the highest mIoU on all datasets.

**Caption suggestion:**  
*Comparison of semantic segmentation model architectures across benchmark datasets. SegFormer-MiT-B5 achieves superior performance across all datasets, with improvements of 14.6 and 19.0 percentage points over PSPNet and DeepLabV3+, respectively.*

---

### Figure 2: Domain Gap Heatmap
**File:** `fig2_domain_gap_heatmap.png`  
**Size:** Single column (3.5")  
**Corresponding Table:** Table II

Heatmap visualization showing the performance gap (Normal mIoU - Adverse mIoU) for each dataset-model combination. Color scale: green (small gap) → red (large gap). Key observations:
- Outside15K + DeepLabV3+ shows the largest gap (16.0%)
- Mapillary + PSPNet shows a negative gap (-4.8%)
- SegFormer maintains consistent cross-domain performance

**Caption suggestion:**  
*Domain gap heatmap showing performance difference between normal and adverse weather conditions. Larger values (red) indicate higher sensitivity to domain shift. SegFormer demonstrates superior robustness with consistently smaller gaps.*

---

### Figure 3: Weather Domain Radar Chart
**File:** `fig3_domain_radar.png`  
**Size:** Single column (3.5" × 3.5")  
**Corresponding Table:** Table III

Multi-axis radar chart showing model performance across 6 weather domains. Legend positioned below the chart. Observations:
- SegFormer maintains high performance across all domains
- All models show a performance dip in night and dawn/dusk conditions
- Clear day and cloudy domains show the highest performance

**Caption suggestion:**  
*Radar chart of model performance across weather domains. SegFormer maintains consistently high performance, while all models exhibit degradation in nighttime and twilight conditions.*

---

### Figure 4: Training Strategy Comparison
**File:** `fig4_training_comparison.png`  
**Size:** Double column (7.16")  
**Corresponding Table:** Table IV

Two-panel figure with shared legend at top:
- **Panel (a):** Per-domain performance comparison between full training and clear-day-only training
- **Panel (b):** Aggregated metrics showing overall, normal, adverse, and domain gap

Key insight: Full training provides +3.8% improvement on adverse conditions and reduces domain gap by ~3 percentage points.

**Caption suggestion:**  
*Impact of training data diversity on domain generalization. (a) Per-domain performance shows improvements from diverse weather training. (b) Full training reduces domain gap by 2.97 percentage points while improving adverse weather performance by 3.8%.*

---

### Figure 5: Dataset Difficulty Analysis
**File:** `fig5_dataset_difficulty.png`  
**Size:** Single column (3.5")  
**Corresponding Table:** Table VI

Horizontal bar chart showing dataset difficulty with normal/adverse breakdown. Legend at top center. Datasets ranked from easiest (IDD-AW) to hardest (Mapillary). Gap annotations (Δ) show domain shift impact.

**Caption suggestion:**  
*Dataset difficulty analysis showing performance under normal (green) and adverse (orange) conditions. IDD-AW achieves highest performance while Mapillary Vistas proves most challenging due to scene diversity.*

---

### Figure 6: Performance Matrix
**File:** `fig6_performance_matrix.png`  
**Size:** Double column (7.16")  
**Corresponding Table:** Table VII

Three-panel heatmap showing per-domain mIoU for each model across all datasets. Colorbar indicates mIoU scale (25-85%). Observations:
- SegFormer shows the "greenest" (highest) overall performance
- Night column is consistently challenging (more red)
- Dataset-specific patterns are visible

**Caption suggestion:**  
*Comprehensive performance matrix showing mIoU (%) across all dataset-domain combinations. SegFormer demonstrates superior and more consistent performance, particularly in challenging domains.*

---

### Figure 7: Model Ranking Summary
**File:** `fig7_model_ranking.png`  
**Size:** Single column (3.5")  
**Corresponding Table:** Table V

Horizontal bar chart with error bars showing model ranking by overall mIoU. Includes mean ± standard deviation:

| Model | mIoU |
|-------|------|
| SegFormer | 69.7% ± 7.3 |
| PSPNet | 55.1% ± 16.5 |
| DeepLabV3+ | 50.7% ± 13.2 |

**Caption suggestion:**  
*Model architecture ranking by mIoU with cross-dataset variance. SegFormer achieves highest mean (69.7%) and lowest variance, indicating both superior accuracy and consistency.*

---

### Figure 8: Domain Shift Impact
**File:** `fig8_domain_shift.png`  
**Size:** Single column (3.5")  
**Corresponding Table:** Table VIII

Bar chart showing performance drop from clear day to each adverse domain. Legend at top center. Domain shift ranking:
1. Night: Largest drop (10-20%)
2. Dawn/Dusk: Second most challenging
3. Rainy/Snowy: More moderate drops

**Caption suggestion:**  
*Performance degradation from clear day to adverse conditions. Night conditions cause the most significant drop (15-20 pp), suggesting temporal illumination changes are more challenging than precipitation phenomena.*

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

### Color Scheme (Colorblind-friendly)
**Models:**
- DeepLabV3+: #2E86AB (Deep blue)
- PSPNet: #E07A5F (Terracotta)
- SegFormer: #3D405B (Dark slate)

**Conditions:**
- Normal: #2A9D8F (Teal)
- Adverse: #E76F51 (Coral)

### Legend Positioning
All figures have legends positioned **outside the data area** to prevent overlap:
- Top center (horizontal): Figures 1, 4, 5, 8
- Below chart: Figure 3
- No legend needed: Figures 2, 6, 7 (colorbars or single series)

---

## LaTeX Table Usage

### Required Packages
```latex
\usepackage{booktabs}   % For professional table rules
\usepackage{multirow}   % For merged rows
\usepackage{siunitx}    % Optional: number alignment
```

### Including Tables
```latex
\input{tables_booktabs.tex}
% Or copy individual table environments from the file
```

### Table-Figure Correspondence
| Figure | Complementary Table(s) |
|--------|------------------------|
| Fig. 1 | Table I |
| Fig. 2 | Table II |
| Fig. 3 | Table III |
| Fig. 4 | Table IV |
| Fig. 5 | Table VI |
| Fig. 6 | Table VII |
| Fig. 7 | Table V |
| Fig. 8 | Table VIII |

---

## Recommended Selection for Publication

### Space-Limited Papers (6-8 pages)
**Essential figures:**
1. Figure 1 (Model comparison) + Table I
2. Figure 2 or 6 (Domain gap) + Table II
3. Figure 4 (Training impact) + Table IV

### Extended Papers (10+ pages)
Add:
- Figure 3 (Radar chart)
- Figure 8 (Domain shift)
- Table VII (Comprehensive results)

### Supplementary Material
- Remaining figures and tables
- Full per-domain CSV data

---

*Generated by PROVE Baseline Analysis Pipeline*
*Last updated: January 8, 2026*
