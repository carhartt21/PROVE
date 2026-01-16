# Ratio Ablation Study - Figure Descriptions

## Figure 1: Effect of Synthetic Data Ratio on mIoU
Shows how varying the proportion of synthetic training data affects segmentation 
performance (mIoU) on two adverse weather datasets. Error bars indicate standard 
deviation across model architectures. The optimal ratio is marked with a star.

**Key findings:** Performance peaks at moderate synthetic ratios (12-38%), 
with diminishing returns at higher ratios.

## Figure 2: Model Architecture Comparison Across Ratios
Compares how different architectures (DeepLabV3+, PSPNet, SegFormer) respond 
to varying synthetic data ratios. Each line represents a different model's 
performance trajectory.

**Key findings:** SegFormer shows highest absolute performance but similar 
ratio sensitivity patterns to CNN-based models.

## Figure 3: Optimal Ratio Heatmap
Heatmap visualization showing the optimal synthetic data ratio for each 
dataset-model combination. Color intensity indicates the ratio value.

**Key findings:** Optimal ratios vary by configuration, ranging from 0% to 38%.

## Figure 4: Performance Gain Analysis
Shows the relative performance change (in percentage points) compared to 
the baseline (0% synthetic data) for each architecture.

**Key findings:** Maximum gains of 2-3 percentage points achievable with 
optimal ratio selection.

## Figure 5: Per-Domain Ratio Effects
Examines how synthetic data affects performance on challenging weather 
domains (foggy, night) specifically.

**Key findings:** Challenging domains show more variability and may benefit 
more from synthetic data augmentation.

## Figure 6: Training Convergence Curves
Shows training dynamics over iterations for different synthetic data ratios.

**Key findings:** Different ratios converge at similar rates but to different 
final performance levels.

## Figure 7: Performance Variance Analysis
Analyzes the stability of model performance at different mixing ratios using 
standard deviation across architectures.

**Key findings:** Moderate ratios (25-38%) tend to produce more consistent 
results across different model architectures.

## Figure 8: Baseline vs Optimal Ratio Summary
Direct comparison of baseline (0% synthetic) versus optimal ratio performance 
for all configurations. Annotations show the optimal ratio for each case.

**Key findings:** Optimal ratio selection provides consistent but modest 
improvements over pure real data training.

---

## LaTeX Tables

### Table 1: Best Performance per Configuration
Lists the optimal synthetic data ratio and corresponding mIoU for each 
dataset-model combination.

### Table 2: Mean Performance Across Ratios  
Shows average mIoU across all models for each ratio value, enabling 
comparison of general ratio effectiveness.

### Table 3: Per-Domain Performance
Detailed breakdown of performance on each weather domain at the optimal ratio.

---

## Usage in Paper

Include figures using:
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\columnwidth]{figures/ratio_ablation/fig1_ratio_vs_miou_by_dataset.png}
    \caption{Effect of synthetic data ratio on segmentation performance.}
    \label{fig:ratio_effect}
\end{figure}
```

For double-column figures:
```latex
\begin{figure*}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/ratio_ablation/fig2_ratio_vs_miou_by_model.png}
    \caption{Comparison of model architectures across synthetic data ratios.}
    \label{fig:model_ratio_comparison}
\end{figure*}
```
